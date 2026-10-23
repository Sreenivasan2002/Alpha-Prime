"""Low-frequency, cost-aware previous-day-range breakout (longs AND shorts).

Thesis: the baseline loses because it trades ~13x/day on a signal with no edge, so charges
(~0.06% + ~0.02% slippage per round trip) and market beta eat it. A breakout of an obvious
level (yesterday's high / low) only means something when real participation comes with it,
so we trade it rarely: at most `max_trades_per_day` entries, only on a volume-confirmed first
break of the level, and hold toward the close (measured excursions on TRAIN: median MFE
~0.6%, MFE > 2% in under 10% of events, so no tight target - the close is the exit).

Rules (all features causal - row i uses bars <= i):
  * level   : previous session's high (long) / low (short)              [lookback_days=1]
  * trigger : the FIRST 5-min close of today beyond the level (a gap that opens / first bar
              closes beyond it does not count - that move already happened)
  * volume  : today's cumulative volume up to the breakout bar >= `min_rvol` x the average
              cumulative volume at the same clock time over the previous 10 sessions
  * index   : optional - NIFTY must be on the same side of its own open (`index_filter`)
  * window  : enter at the open of the bar after the breakout, 09:35..`entry_end`
  * exit    : stop `stop_pct` from fill (optional breakeven/trail), optional target,
              otherwise square-off at 15:15
  * ranking : if several symbols break at the same bar, highest relative volume first
  * sides   : LONG ONLY in the chosen config. The mirrored short (first close below the previous
              day's low on volume) was tested on TRAIN and lost (42 trades, net -Rs 19k, PF 0.49):
              downside breaks of large caps tended to get bought back in this period. Set
              sides=("long", "short") to re-enable it.

TRAIN tuning (21 configs, one-at-a-time around a centre fixed before any run): long-only
result is positive at every relative-volume threshold tried (1.0/1.25/1.5/2.0) but its SIZE is
very sensitive to it (+25k/+38k/+61k/+0.5k), so the headline TRAIN number is partly luck.
"""
import numpy as np
import pandas as pd


def _cum_rvol(df: pd.DataFrame, days: int = 10, min_days: int = 3) -> np.ndarray:
    """Today's cumulative volume / mean cumulative volume at the same clock time on prior days."""
    dates = df.index.date
    tkey = df.index.strftime("%H:%M")
    cv = df["Volume"].groupby(dates).cumsum().to_numpy(float)
    piv = pd.DataFrame({"d": dates, "t": tkey, "cv": cv}).pivot_table(index="d", columns="t", values="cv")
    piv = piv.sort_index(axis=1).ffill(axis=1)  # missing bar -> cum volume so far (earlier times only)
    base = piv.shift(1).rolling(days, min_periods=min_days).mean()  # PRIOR sessions only
    b = base.stack()
    ref = b.reindex(pd.MultiIndex.from_arrays([dates, tkey])).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(ref > 0, cv / ref, np.nan)
    return out


class Strategy:
    name = "low_freq_breakout"
    entry_start = "09:35"
    entry_end = "13:00"
    squareoff = "15:15"
    max_positions = 2
    max_trades_per_day = 2
    profit_lock_pct = None

    # tunables
    lookback_days = 1        # range = high/low of the previous N sessions
    min_rvol = 1.5           # relative cumulative volume at the breakout bar
    index_filter = False     # require NIFTY on the same side of its day open
    stop_pct = 1.5
    target_pct = None
    trail = None             # e.g. (1.0, 1.5, 1.0)
    sides = ("long",)

    def prepare(self, mkt):
        self.F = {}
        nifty = mkt.df("NIFTY")
        nd = nifty.index.date
        n_open = nifty["Open"].groupby(nd).transform("first")
        n_ret = (nifty["Close"] / n_open - 1) * 100  # NIFTY at bar close vs its day open
        for sym in mkt.symbols:
            df = mkt.df(sym)
            dates = pd.Series(df.index.date, index=df.index)
            k = self.lookback_days
            dh = df["High"].groupby(dates.values).max()
            dl = df["Low"].groupby(dates.values).min()
            hi = dates.map(dh.rolling(k, min_periods=k).max().shift(1)).to_numpy(float)
            lo = dates.map(dl.rolling(k, min_periods=k).min().shift(1)).to_numpy(float)
            c = df["Close"]
            prior_max = c.groupby(dates.values).cummax().groupby(dates.values).shift(1).to_numpy(float)
            prior_min = c.groupby(dates.values).cummin().groupby(dates.values).shift(1).to_numpy(float)
            cv = c.to_numpy(float)
            first = np.isnan(prior_max)  # first bar of the day
            # first close beyond the level today; the day's first bar closing beyond = gap, never counts
            brk_up = (cv > hi) & ~first & (prior_max <= hi)
            brk_dn = (cv < lo) & ~first & (prior_min >= lo)
            self.F[sym] = pd.DataFrame({
                "brk_up": brk_up.astype(float),
                "brk_dn": brk_dn.astype(float),
                "rvol": _cum_rvol(df),
                "n_ret": n_ret.reindex(df.index).to_numpy(float),
            }, index=df.index)

    def on_bar(self, ctx):
        cands = []
        for sym in ctx.symbols:
            if sym in ctx.positions or not ctx.has_bar(sym):
                continue
            up, dn = ctx.feat(sym, "brk_up"), ctx.feat(sym, "brk_dn")
            if not up and not dn:
                continue
            rv = ctx.feat(sym, "rvol")
            if rv is None or rv < self.min_rvol:
                continue
            side = "long" if up else "short"
            if side not in self.sides:
                continue
            if self.index_filter:
                nr = ctx.feat(sym, "n_ret")
                if nr is None or (side == "long" and nr <= 0) or (side == "short" and nr >= 0):
                    continue
            cands.append((rv, sym, side))
        cands.sort(reverse=True)
        orders = []
        for rv, sym, side in cands:
            od = dict(sym=sym, side=side, stop_pct=self.stop_pct, tag=f"pd_{side}")
            if self.target_pct:
                od["target_pct"] = self.target_pct
            if self.trail:
                od["trail"] = self.trail
            orders.append(od)
        return orders
