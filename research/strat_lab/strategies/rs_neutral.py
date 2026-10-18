"""Relative-strength market-neutral (rs_neutral).

Thesis: the baseline loses largely because it is LONG MARKET BETA (+0.45 corr with NIFTY) in a
flat/down tape. A dollar-neutral long/short book built from a cross-sectional ranking removes the
market factor, so P&L depends only on whether relative strength persists (momentum) or mean-reverts
(reversal) over the rest of the session.

Signal (per stock, causal, computed in prepare()):
    ret   = close / anchor - 1            anchor = today's 09:15 open ("open") or previous close ("pc")
    resid = ret - beta * NIFTY_ret        beta  = rolling 5-day (375-bar) beta of 5-min returns to NIFTY
    z     = resid / (sigma_idio * sqrt(n)) sigma_idio = rolling 5-day std of 5-min residual returns,
                                           n = number of 5-min bars elapsed since the anchor
At each decision time the stocks are ranked by z; with direction "mom" the top K are bought and the
bottom K shorted (direction "rev" flips the legs). Orders are interleaved long/short so a partial
fill stays balanced. Stops are wide catastrophe stops (<= 2.5%) so the 20%-of-capital notional cap
binds on every leg -> equal rupee notional (Rs 2L) per leg -> the book is ~dollar-neutral and, because
ranking is on beta-adjusted residuals, ~beta-neutral too. Each session has a fixed time exit.

TRAIN search (19 configs, all net of charges + 1bp slippage):
  * direction: residual MOMENTUM has positive gross in every timing tried; REVERSAL is the mirror
    image and loses heavily (-87k .. -171k) -> momentum.
  * anchor since-open beats since-prev-close (the gap component hurt: -43.5k vs +4.0k at K=5).
  * K (names/side) at 10:15->15:15: K=5 +4.0k, K=4 +34.2k, K=3 +31.3k, K=2 +56.4k -> the edge sits in
    the extremes; with K=5 the ~0.06% gross/leg is eaten by ~0.06% charges.
  * decision time with K=2: 10:00 +18.0k, 10:15 +56.4k, 10:30 +28.2k (all positive, 10:15 is the
    lucky peak - a fair expectation is the neighbourhood, not the peak).
  * 2 sessions/day doubles charges and lost; 1% stops cut winners (gross -> 32.6k from 42.7k at K=5).
Final: momentum, since-open, K=2, 10:15 entry -> 15:15 exit, 2% catastrophe stop, no target/trail.
"""
import numpy as np
import pandas as pd

WIN = 375          # 5 trading days of 5-min bars
MINP = 150         # need >= 2 days of history before a score is produced


class Strategy:
    name = "rs_neutral"
    entry_start = "09:30"
    entry_end = "14:30"
    squareoff = "15:15"
    max_positions = 10
    max_trades_per_day = None
    profit_lock_pct = None

    # ---- parameters (final config chosen on TRAIN only; 19 configs tried - see notes below)
    direction = "mom"                # "mom": long strongest / short weakest ; "rev": the opposite
    anchor = "open"                  # "open": since today's open ; "pc": since previous close
    k = 2                            # names per side (2k <= 10)
    stop_pct = 2.0                   # catastrophe stop per leg (<= 2.5 keeps notional cap binding)
    sessions = (("10:15", "15:15"),)  # (decision time = entry bar open, exit time)

    def prepare(self, mkt):
        n = mkt.df("NIFTY")
        n_c = n["Close"].astype(float)
        n_day = pd.Series(n.index.date, index=n.index)
        n_open = n["Open"].astype(float).groupby(n_day.values).transform("first")
        n_pc = n_c.groupby(n_day.values).last().shift(1)
        n_pc = pd.Series(n_day.map(n_pc).values, index=n.index, dtype=float)
        n_r = n_c.pct_change()
        n_r[(n_day != n_day.shift(1)).values] = np.nan          # drop overnight gap returns
        n_ret_open = n_c / n_open - 1
        n_ret_pc = n_c / n_pc - 1

        self.F = {}
        for sym in mkt.symbols:
            df = mkt.df(sym)
            idx = df.index
            c = df["Close"].astype(float)
            day = pd.Series(idx.date, index=idx)
            first = (day != day.shift(1)).values
            d_open = df["Open"].astype(float).groupby(day.values).transform("first")
            closes = c.groupby(day.values).last()
            pc = pd.Series(day.map(closes.shift(1)).values, index=idx, dtype=float)

            r = c.pct_change()
            r[first] = np.nan
            nr = n_r.reindex(idx)
            both = r.notna() & nr.notna()
            rr, nn = r.where(both), nr.where(both)
            cov = rr.rolling(WIN, min_periods=MINP).cov(nn)
            var = nn.rolling(WIN, min_periods=MINP).var()
            beta = (cov / var).clip(0.2, 2.5)
            idio = (rr - beta * nn).rolling(WIN, min_periods=MINP).std()

            nb = pd.Series(1, index=idx).groupby(day.values).cumsum()   # bars completed today
            ro = c / d_open - 1
            rp = c / pc - 1
            zo = (ro - beta * n_ret_open.reindex(idx)) / (idio * np.sqrt(nb))
            zp = (rp - beta * n_ret_pc.reindex(idx)) / (idio * np.sqrt(nb + 1))  # +1 for the gap
            self.F[sym] = pd.DataFrame({"z_open": zo, "z_pc": zp, "beta": beta}, index=idx)

    def on_bar(self, ctx):
        sess = [s for s in self.sessions if s[0] == ctx.time]
        if not sess:
            return []
        exit_t = sess[0][1]
        col = "z_open" if self.anchor == "open" else "z_pc"
        scores = []
        for sym in ctx.symbols:
            if sym in ctx.positions or not ctx.has_bar(sym):
                continue
            z = ctx.feat(sym, col)
            if z is None or not np.isfinite(z):
                continue
            scores.append((float(z), sym))
        if len(scores) < 4 * self.k:
            return []
        scores.sort()
        weak = [s for _, s in scores[: self.k]]
        strong = [s for _, s in scores[::-1][: self.k]]
        longs, shorts = (strong, weak) if self.direction == "mom" else (weak, strong)
        room = (self.max_positions - ctx.n_open) // 2
        orders = []
        for lg, sh in list(zip(longs, shorts))[:room]:
            orders.append(dict(sym=lg, side="long", stop_pct=self.stop_pct, exit_time=exit_t, tag="L"))
            orders.append(dict(sym=sh, side="short", stop_pct=self.stop_pct, exit_time=exit_t, tag="S"))
        return orders
