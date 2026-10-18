"""Opening Range Breakout (ORB), long AND short, NIFTY 100 5-min bars.

Thesis: the first 15/30 minutes of NSE trading absorb the overnight order imbalance and set a
range. A decisive CLOSE beyond that range, on above-normal volume and in the same direction as
the NIFTY index, signals that one side has won the opening auction; intraday momentum
(continuation into the close) is the documented effect behind ORB. Being able to short removes
the long-market-beta problem of the baseline (which was long-only into a flat/down market).

Rules (all features causal, computed in prepare(); on_bar only reads ctx):
  * Opening range = high/low of the bars starting before 09:15 + or_min.
  * Signal on bar i (only bars after the range is complete): the FIRST close of the day beyond
    the range, on either side. Close > OR high -> long; close < OR low -> short. Later breaks
    the same day are ignored, so at most ONE trade per stock per day by construction.
  * Volume confirmation: the breakout bar's volume >= vol_mult x that symbol's average volume
    at the same time slot over the previous 10 sessions (09:15 bars have ~0 volume on this feed,
    so slot-normalised volume is used rather than a raw bar average).
  * "Stock in play" filter: opening-range volume >= or_rvol_min x its previous-10-session average
    (Zarattini & Aziz style). On TRAIN, plain ORB across all NIFTY-100 names LOSES (first breakouts
    mean-revert slightly in this period); only high-relative-volume names show continuation.
  * Index alignment (optional, OFF in the final config): long only if NIFTY's last close is above
    its day open, short only if below. On TRAIN it halved the trade count without improving net.
  * Skip abnormal ranges: OR width outside [min_width, max_width] % of price.
  * Entry window: after the range until entry_end. Ranked by opening-range relative volume
    ("stocks in play") when there are more signals than free slots.
  * Stop: other side of the range ("range") or the range midpoint ("mid"), clipped to
    [stop_lo, stop_hi] % from the fill. Optional target at target_r x stop distance.
  * Time exit at exit_time (square-off 15:15 by the harness at the latest).
"""
import numpy as np
import pandas as pd

OPEN_MIN = 9 * 60 + 15


class Strategy:
    name = "orb"
    # ---- parameters (final values chosen on TRAIN only; see research notes in the module result)
    or_min = 15            # opening-range length in minutes (15 or 30)
    vol_mult = 1.5         # breakout-bar volume vs same-slot 10-session average
    or_rvol_min = 2.0      # "stock in play": opening-range volume vs its 10-session average
    index_align = False    # require NIFTY direction since open to agree (tested: halves trades, no gain)
    stop_mode = "range"    # "range" = other side of the OR, "mid" = OR midpoint
    stop_lo, stop_hi = 0.3, 2.5
    min_width, max_width = 0.3, 5.0
    target_r = None        # e.g. 2.0 -> target at 2x the stop distance; None = no target
    trail = None           # (be_arm_pct, trail_arm_pct, trail_dist_pct) or None
    exit_time = "15:15"
    entry_end = "11:30"
    sides = ("long", "short")

    entry_start = "09:30"
    squareoff = "15:15"
    max_positions = 10
    max_trades_per_day = None
    profit_lock_pct = None

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self._traded = set()

    # ------------------------------------------------------------------ features (causal)
    def prepare(self, mkt):
        self.F = {}
        or_end = OPEN_MIN + self.or_min          # first bar start AFTER the range
        for sym in mkt.symbols:
            df = mkt.df(sym)
            if df.empty:
                continue
            idx = df.index
            dates = pd.Series(idx.date, index=idx)
            mins = pd.Series(idx.hour * 60 + idx.minute, index=idx)
            h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"].astype(float)

            in_or = mins < or_end
            after = mins >= or_end               # rows strictly after the range is complete
            # day-level OR high/low/volume; only ever read on rows after the range => causal
            or_h_day = h[in_or].groupby(dates[in_or]).max()
            or_l_day = l[in_or].groupby(dates[in_or]).min()
            or_v_day = v[in_or].groupby(dates[in_or]).sum()
            or_h = dates.map(or_h_day).where(after)
            or_l = dates.map(or_l_day).where(after)

            # opening-range relative volume vs previous 10 sessions (shifted => past days only)
            or_v_base = or_v_day.rolling(10, min_periods=5).mean().shift(1)
            or_rvol = (dates.map(or_v_day) / dates.map(or_v_base)).where(after)

            # breakout bar volume vs same-slot average of the previous 10 sessions
            piv = pd.DataFrame({"d": dates.values, "m": mins.values, "v": v.values}).pivot_table(
                index="d", columns="m", values="v", aggfunc="sum")
            base = piv.rolling(10, min_periods=5).mean().shift(1)
            key = pd.MultiIndex.from_arrays([dates.values, mins.values])
            slot_base = base.stack().reindex(key).to_numpy()
            bar_rvol = pd.Series(v.to_numpy() / np.where(slot_base > 0, slot_base, np.nan), index=idx)

            above = (c > or_h) & after
            below = (c < or_l) & after
            beyond = above | below
            n_beyond = beyond.astype(int).groupby(dates).cumsum()
            first = beyond & (n_beyond == 1)
            sig = np.where(first & above, 1.0, np.where(first & below, -1.0, 0.0))

            self.F[sym] = pd.DataFrame({
                "sig": sig, "or_h": or_h, "or_l": or_l, "or_rvol": or_rvol, "bar_rvol": bar_rvol,
            }, index=idx)

    # ------------------------------------------------------------------ decisions
    def on_bar(self, ctx):
        free = self.max_positions - ctx.n_open
        if free <= 0:
            return []
        nc, no = ctx.last("NIFTY", "c"), ctx.day_open("NIFTY")
        n_dir = 0 if (nc is None or no is None) else (1 if nc > no else (-1 if nc < no else 0))
        cands = []
        for sym in ctx.symbols:
            if sym in ctx.positions or (ctx.date, sym) in self._traded:
                continue
            s = ctx.feat(sym, "sig")
            if not s:
                continue
            side = "long" if s > 0 else "short"
            if side not in self.sides:
                continue
            if self.index_align and n_dir != (1 if s > 0 else -1):
                continue
            rv = ctx.feat(sym, "bar_rvol")
            if rv is None or rv < self.vol_mult:
                continue
            orv = ctx.feat(sym, "or_rvol")
            if self.or_rvol_min is not None and (orv is None or orv < self.or_rvol_min):
                continue
            hi, lo, px = ctx.feat(sym, "or_h"), ctx.feat(sym, "or_l"), ctx.last(sym, "c")
            if hi is None or lo is None or not px:
                continue
            width = (hi - lo) / px * 100
            if not (self.min_width <= width <= self.max_width):
                continue
            ref = lo if self.stop_mode == "range" else (hi + lo) / 2
            if s < 0:
                ref = hi if self.stop_mode == "range" else (hi + lo) / 2
            dist = abs(px - ref) / px * 100
            stop_pct = float(min(max(dist, self.stop_lo), self.stop_hi))
            od = dict(sym=sym, side=side, stop_pct=stop_pct, exit_time=self.exit_time,
                      tag=f"orb{self.or_min}_{side}")
            if self.target_r:
                od["target_pct"] = stop_pct * self.target_r
            if self.trail:
                od["trail"] = tuple(self.trail)
            cands.append((ctx.feat(sym, "or_rvol") or 0.0, od))
        cands.sort(key=lambda x: -x[0])
        out = [od for _, od in cands[:free]]
        for od in out:
            self._traded.add((ctx.date, od["sym"]))
        return out
