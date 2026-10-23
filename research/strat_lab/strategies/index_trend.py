"""Index-direction following (family: index_trend).

Idea: decide the day's regime from the NIFTY index in the first hour, and only on a CLEAR trend
day trade WITH the index, using the stocks that are leading it (relative strength); stand aside
on unclear days.

Regime (evaluated on each completed 5-min bar from the end of the opening range until `trigger_end`):
  * UP   : NIFTY close > opening-range high  AND  breadth (share of the NIFTY-100 stocks trading
           above their own day open) >= `breadth_min`  AND  NIFTY above its intraday TWAP
  * DOWN : mirror (close < OR low, breadth <= 1 - breadth_min, below TWAP)
  The first bar that satisfies UP or DOWN fires the day's ONE signal; later bars are ignored.
Selection: the `k` stocks with the highest (UP) / lowest (DOWN) return-from-open minus NIFTY's
  return-from-open, that are also on the right side of their own VWAP and moving with the index.
Orders: long on UP days, short on DOWN days; fixed % stop; exit at `exit_time` or square-off.

All features are causal: row i uses bars <= i only (opening range is NaN until it is complete,
cumulative VWAP/TWAP/breadth use bars up to and including i, within the same day).
"""
import numpy as np
import pandas as pd


def _day_key(idx):
    return np.asarray(idx.date)


class Strategy:
    name = "index_trend"
    entry_start = "10:15"
    entry_end = "13:00"        # last bar whose open we may enter at
    squareoff = "15:15"
    max_positions = 10

    # ---- parameters (tuned on TRAIN only, small grid)
    or_minutes = 60            # opening range length (from 09:15)
    breadth_min = 0.60         # share of stocks above their day open required for an UP day
    use_twap = True            # NIFTY must also be on the trend side of its intraday TWAP
    min_index_move = 0.0       # |NIFTY return from open| (%) required at the trigger bar
    k = 5                      # names traded on a signal day
    stop_pct = 1.0
    target_pct = None
    trail = None               # (breakeven_arm, trail_arm, trail_dist) in %
    hold_minutes = None        # None = hold to square-off
    sides = ("long", "short")

    def __init__(self):
        self.F = {}
        self._fired = set()

    # ------------------------------------------------------------------ features
    def prepare(self, mkt):
        syms = mkt.symbols
        # union timeline of every symbol (incl. NIFTY), so cross-sectional values line up by clock
        closes = {s: mkt.df(s)["Close"] for s in syms + ["NIFTY"]}
        opens = {s: mkt.df(s)["Open"] for s in syms + ["NIFTY"]}
        C = pd.DataFrame(closes).sort_index()
        dk = _day_key(C.index)
        Cf = C.groupby(dk).ffill()                         # causal, within-day forward fill
        Oday = pd.DataFrame(opens).sort_index().groupby(dk).transform("first")
        # NOTE: transform('first') takes the first NON-NaN open of the day per column; a stock whose
        # first bar is missing gets the open of its first available bar (known at that bar), and
        # rows before that bar have NaN close -> excluded from breadth. Still causal.
        Oday = Oday.where(C.notna().groupby(dk).cummax())  # do not expose a day-open before the bar exists
        ret_open = Cf / Oday - 1.0
        stock_ret = ret_open[syms]
        valid = stock_ret.notna()
        breadth = (stock_ret > 0).where(valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
        n_ret = ret_open["NIFTY"]

        # ---- NIFTY regime features
        N = mkt.df("NIFTY")
        nd = _day_key(N.index)
        mins = N.index.hour * 60 + N.index.minute
        or_end = 9 * 60 + 15 + self.or_minutes            # bars starting before this form the range
        in_or = pd.Series(mins < or_end, index=N.index)
        hi = N["High"].where(in_or).groupby(nd).cummax()
        lo = N["Low"].where(in_or).groupby(nd).cummin()
        or_done = pd.Series(mins >= or_end - 5, index=N.index)   # last OR bar is complete at its own row
        hi = hi.groupby(nd).ffill().where(or_done)
        lo = lo.groupby(nd).ffill().where(or_done)
        tp = (N["High"] + N["Low"] + N["Close"]) / 3.0
        twap = tp.groupby(nd).cumsum() / tp.groupby(nd).cumcount().add(1)
        fN = pd.DataFrame(index=N.index)
        fN["or_hi"], fN["or_lo"], fN["twap"] = hi, lo, twap
        fN["breadth"] = breadth.reindex(N.index)
        fN["n_ret"] = n_ret.reindex(N.index)
        self.F["NIFTY"] = fN

        # ---- per-stock features
        for s in syms:
            df = mkt.df(s)
            sd = _day_key(df.index)
            v = df["Volume"].astype(float)
            tps = (df["High"] + df["Low"] + df["Close"]) / 3.0
            cv = v.groupby(sd).cumsum()
            vwap = (tps * v).groupby(sd).cumsum() / cv.replace(0, np.nan)
            f = pd.DataFrame(index=df.index)
            f["ret_open"] = df["Close"] / df["Open"].groupby(sd).transform("first") - 1.0
            f["rs"] = f["ret_open"] - n_ret.reindex(df.index)
            f["vwap"] = vwap
            self.F[s] = f

    # ------------------------------------------------------------------ decisions
    def _regime(self, ctx):
        g = lambda c: ctx.feat("NIFTY", c)
        hi, lo, tw, br, nr = g("or_hi"), g("or_lo"), g("twap"), g("breadth"), g("n_ret")
        c = ctx.last("NIFTY", "c")
        if None in (hi, lo, tw, br, nr, c):
            return 0
        up = c > hi and br >= self.breadth_min and (not self.use_twap or c > tw) and nr * 100 >= self.min_index_move
        dn = c < lo and br <= 1 - self.breadth_min and (not self.use_twap or c < tw) and -nr * 100 >= self.min_index_move
        return 1 if up else (-1 if dn else 0)

    def on_bar(self, ctx):
        if ctx.date in self._fired:
            return []
        reg = self._regime(ctx)
        if reg == 0:
            return []
        side = "long" if reg > 0 else "short"
        self._fired.add(ctx.date)          # one signal per day, whether or not we trade it
        if side not in self.sides:
            return []
        cands = []
        for s in ctx.symbols:
            if s in ctx.positions or not ctx.has_bar(s):
                continue
            rs, ro, vw = ctx.feat(s, "rs"), ctx.feat(s, "ret_open"), ctx.feat(s, "vwap")
            px = ctx.last(s, "c")
            if None in (rs, ro, vw, px):
                continue
            if reg > 0 and ro > 0 and px > vw:
                cands.append((rs, s))
            elif reg < 0 and ro < 0 and px < vw:
                cands.append((-rs, s))
        cands.sort(reverse=True)
        ex = None
        if self.hold_minutes:
            m = min(ctx.minutes + self.hold_minutes, 15 * 60 + 15)
            ex = f"{m // 60:02d}:{m % 60:02d}"
        orders = []
        for _, s in cands[: self.k]:
            od = dict(sym=s, side=side, stop_pct=self.stop_pct, tag=f"idx_{side}")
            if self.target_pct:
                od["target_pct"] = self.target_pct
            if self.trail:
                od["trail"] = self.trail
            if ex:
                od["exit_time"] = ex
            orders.append(od)
        return orders
