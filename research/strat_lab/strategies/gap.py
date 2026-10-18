"""Gap family: fade opening gaps vs the previous close that FAIL TO EXTEND.

Thesis (measured on TRAIN only, 2026-07-23..09-11, NIFTY-100 stock-days):
  * Moderate gap-UPs (+0.5%..+2% at the 09:15 open) that are NOT holding near the top of their
    30-minute opening range at 09:45 drift back toward the previous close for the rest of the day.
    Event study, short from the 09:45 open to 15:15: n=329 stock-days / 35 days, mean +0.27%,
    day-clustered t ~ 5. The fade accrues gradually and is largest into the close (15:15).
  * Gap-ups still pinned at the opening-range high ("holding") faded much less (+0.12%).
  * Gap-DOWN fades (buying the dip) had NO day-level edge (t ~ 0): the mean is carried by a few
    market-wide rebound days, so the long side is OFF by default (`trade_down=False`).
  * Gap-and-go (|gap| > 2% continuing) had positive means but only 9-37 cases with t < 1.1, so it
    is available as an option (`go_gap`) but not used by default. Gaps > 2% do NOT fade.

Rules (one decision per day at the 09:45 bar, i.e. after the 09:15-09:40 opening range is complete):
  1. gap = today's open / previous close - 1, in [gap_min, gap_max] (default +0.5%..+2.0%)
  2. idiosyncratic: stock gap - NIFTY gap >= `min_rgap` (0.5%) -> not just a market-wide gap
     (market-wide gap-ups faded much less on TRAIN; this also lowers the short-beta exposure)
  3. "fails to extend": the 09:40 close sits in the lower `max_or_pos` (2/3) of the 30-min opening
     range -> the gap did not keep pushing up
  4. SHORT at the 09:45 open; 2% stop; no target; exit at 15:15 (time square-off)
  5. rank candidates by gap size (largest first), at most `max_positions` (10) per day

TRAIN search: 14 configs (one-at-a-time changes from a 1.5%-stop base, then 2 combinations).
A wider stop helped (1.0 < 1.5 < 2.0 ~ 2.5); a gap-fill target, trailing, 15-min OR, both-sides and
no-OR-filter variants were all worse or no better. It is structurally NET SHORT (daily P&L corr
-0.39 with NIFTY): it lost on the 5 NIFTY-up days in TRAIN, so a rising market is its main risk.

All features are causal: row i uses bars <= i of the same day plus the previous day's close.
"""
import numpy as np
import pandas as pd


class Strategy:
    name = "gap"
    entry_start = "09:45"      # single decision: first bar after the 30-min opening range
    entry_end = "09:45"
    squareoff = "15:15"
    max_positions = 10
    max_trades_per_day = None
    profit_lock_pct = None

    # ---- parameters (defaults = chosen TRAIN config)
    or_minutes = 30            # opening range 09:15 .. 09:15+or_minutes (exclusive)
    gap_min = 0.5              # % gap-up lower bound
    gap_max = 2.0              # % gap-up upper bound
    max_or_pos = 0.67          # fade only if last close is in the lower 2/3 of the opening range
    min_rgap = 0.5             # require stock gap - NIFTY gap >= this (the gap is the stock's own, not beta)
    stop_pct = 2.0             # notional cap (20%) binds below 2.5%, so a wider stop does not shrink size
    target_mode = None         # None | "prev_close" (gap fill)
    exit_time = None           # None -> squareoff
    trail = None
    trade_down = False         # also fade gap-downs (long) - no edge on TRAIN, off
    go_gap = None              # optional gap-and-go for |gap| > go_gap holding at the OR extreme

    def prepare(self, mkt):
        orm = 555 + int(self.or_minutes)  # minutes-of-day where the OR ends (exclusive)
        na = mkt.arr.get("NIFTY")
        ngap_by_date = {}
        if na is not None:
            g = na["day_open"] / na["prev_close"] - 1
            for d, v in zip(na["date"], g):
                ngap_by_date[d] = v * 100
        self.F = {}
        for sym in mkt.symbols:
            df = mkt.df(sym)
            a = mkt.arr[sym]
            dates = pd.Series(a["date"], index=df.index)
            in_or = pd.Series(a["mins"] < orm, index=df.index)
            hi = df["High"].where(in_or)
            lo = df["Low"].where(in_or)
            or_hi = hi.groupby(dates.values).cummax().groupby(dates.values).ffill()
            or_lo = lo.groupby(dates.values).cummin().groupby(dates.values).ffill()
            rng = (or_hi - or_lo)
            pos = ((df["Close"] - or_lo) / rng).where(rng > 0, 0.5)
            gap = (a["day_open"] / a["prev_close"] - 1) * 100
            ngap = np.array([ngap_by_date.get(d, np.nan) for d in a["date"]], dtype=float)
            self.F[sym] = pd.DataFrame(dict(
                gap=gap, ngap=ngap, rgap=gap - ngap,
                or_hi=or_hi.to_numpy(float), or_lo=or_lo.to_numpy(float), or_pos=pos.to_numpy(float),
                or_done=(a["mins"] >= orm - 5).astype(float),   # 1 once the last OR bar is complete
            ), index=df.index)

    def _order(self, ctx, sym, side, tag):
        od = dict(sym=sym, side=side, stop_pct=self.stop_pct, tag=tag)
        if self.target_mode == "prev_close":
            pc = ctx.prev_close(sym)
            px = ctx.last(sym, "c")
            # target only if it is on the profitable side of the expected fill
            if pc and px and ((side == "short" and pc < px) or (side == "long" and pc > px)):
                od["target"] = pc
        if self.exit_time:
            od["exit_time"] = self.exit_time
        if self.trail:
            od["trail"] = self.trail
        return od

    def on_bar(self, ctx):
        cands = []
        for sym in ctx.symbols:
            if sym in ctx.positions or not ctx.has_bar(sym):
                continue
            if not ctx.feat(sym, "or_done"):
                continue
            gap, pos = ctx.feat(sym, "gap"), ctx.feat(sym, "or_pos")
            if gap is None or pos is None:
                continue
            rgap = ctx.feat(sym, "rgap")
            # gap-and-go (optional): big gap holding at the opening-range extreme -> continue
            if self.go_gap is not None and abs(gap) > self.go_gap:
                if gap > 0 and pos >= 0.9:
                    cands.append((abs(gap), self._order(ctx, sym, "long", "go_up")))
                elif gap < 0 and pos <= 0.1:
                    cands.append((abs(gap), self._order(ctx, sym, "short", "go_dn")))
                continue
            # gap-fade: moderate gap that failed to extend
            if self.gap_min <= gap <= self.gap_max and pos <= self.max_or_pos:
                if self.min_rgap is not None and (rgap is None or rgap < self.min_rgap):
                    continue
                cands.append((gap, self._order(ctx, sym, "short", "fade_up")))
            elif self.trade_down and -self.gap_max <= gap <= -self.gap_min and pos >= 1 - self.max_or_pos:
                if self.min_rgap is not None and (rgap is None or -rgap < self.min_rgap):
                    continue
                cands.append((-gap, self._order(ctx, sym, "long", "fade_dn")))
        cands.sort(key=lambda x: -x[0])
        room = self.max_positions - ctx.n_open
        return [od for _, od in cands[:max(room, 0)]]
