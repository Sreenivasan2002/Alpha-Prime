"""VWAP mean reversion (both directions) - fade stocks stretched far from their intraday VWAP.

Thesis
    Intraday VWAP is the benchmark most institutional execution is measured against, and late in the
    session VWAP/TWAP algos and intraday position-squaring pull prices that have run far from VWAP back
    toward it. A stock stretched > Z "VWAP-band sigmas" away is faded (short above, long below) with
    the VWAP itself as the target and a tight stop + time stop. Being long AND short, the book is
    roughly market-neutral (the baseline's problem was pure long beta).

Features (all causal, computed per day from bars <= the row's bar):
    vwap   cumulative volume-weighted typical price since the day's first bar
    sd     volume-weighted standard deviation of typical price around vwap (the "VWAP band")
    zb     (close - vwap) / sd
    rsi    RSI(14) on 5-min closes (Wilder)
    n_abs  |NIFTY close / NIFTY day open - 1| in %  (index regime: small = range-bound)

Rules (defaults = the chosen TRAIN config, 16 configs tried)
    entry window 13:30 .. 14:45 (signal = last completed bar)
    SHORT if zb >= 2.5 and RSI >= 75 ; LONG if zb <= -2.5 and RSI <= 25
    target = VWAP at the signal bar (absolute "VWAP touch"), stop 1.0% from fill,
    time stop 60 min after entry, squareoff 15:15
    at most one entry per stock per day, most-stretched candidates first.
    Optional (off by default): NIFTY range-bound filter (nifty_max), exhaustion-volume filter
    (vol_mult), reversal "hook" confirmation (hook/Zkeep/look), min VWAP distance (min_dist_pct).

TRAIN finding (honest): the whole-day version of this idea has ~zero GROSS edge and loses heavily
after charges (10:00-14:30, Z 2, RSI 70/30, NIFTY flat: -17.7% on TRAIN). Large VWAP stretches before
~13:00 tended to CONTINUE, not revert. Only a weak late-session reversion exists; with RSI 75/25 it
is about breakeven net of charges (+0.11%, PF 1.02, t 0.08), i.e. not distinguishable from zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _hm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


class Strategy:
    name = "vwap_reversion"
    entry_start = "13:30"
    entry_end = "14:45"
    squareoff = "15:15"
    max_positions = 10
    max_trades_per_day = None
    profit_lock_pct = None

    # parameters (defaults = chosen config)
    Z = 2.5            # band-z threshold
    stop_pct = 1.0     # % from fill
    rsi_hi = 75        # require RSI >= 75 for shorts (and <= 25 for longs); None = off
    nifty_max = None   # e.g. 0.5 -> only trade when |NIFTY day return so far| <= 0.5%
    hold_min = 60      # time stop in minutes after entry (None = hold to squareoff)
    min_dist_pct = 0.0  # require |close/vwap-1| >= this (target must clear costs)
    hook = False       # exhaustion hook: |zb| peaked >= Z within the last `look` bars and the last bar
                       # turned back toward VWAP (reversal candle, |zb| falling) while |zb| >= Zkeep
    Zkeep = 2.0
    look = 3
    vol_mult = None    # optional: the stretch had an exhaustion-volume bar (vol >= vol_mult x 20-bar avg)

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)
        self._done = set()  # (date, sym) already entered today

    # ------------------------------------------------------------------ features
    def _stock_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        g = np.asarray(idx.date)
        gs = pd.Series(g, index=idx)
        o, h, l, c, v = (df[k].astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
        tp = (h + l + c) / 3.0
        cv = v.groupby(g).cumsum()
        cpv = (tp * v).groupby(g).cumsum()
        cp2v = (tp * tp * v).groupby(g).cumsum()
        ok = cv > 0
        vwap = (cpv / cv).where(ok)
        var = (cp2v / cv - vwap * vwap).where(ok)
        sd = np.sqrt(var.clip(lower=0.0))
        zb = ((c - vwap) / sd).where(sd > 1e-9 * c)
        dc = c.diff()
        up = dc.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-dc.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        vrel = v / v.shift(1).rolling(20, min_periods=10).mean()
        # extremes of zb / vrel over the last `look` bars of the SAME day (rows <= current row)
        zmax, zmin, vmax = zb.copy(), zb.copy(), vrel.copy()
        for j in range(1, int(self.look)):
            same = gs.shift(j).to_numpy() == g
            zmax = np.fmax(zmax, zb.shift(j).where(same))
            zmin = np.fmin(zmin, zb.shift(j).where(same))
            vmax = np.fmax(vmax, vrel.shift(j).where(same))
        zprev = zb.shift(1).where(gs.shift(1).to_numpy() == g)
        return pd.DataFrame(dict(vwap=vwap, zb=zb, rsi=rsi, dist=(c / vwap - 1) * 100, zprev=zprev,
                                 zmax=zmax, zmin=zmin, vmax=vmax, bar=np.sign(c - o)), index=idx)

    def prepare(self, mkt):
        self.F = {}
        data = getattr(mkt, "_data", {})
        n_abs = None
        if "NIFTY" in data and len(data["NIFTY"]):
            n = data["NIFTY"]
            ng = np.asarray(n.index.date)
            nopen = n["Open"].astype(float).groupby(ng).transform("first")
            n_abs = ((n["Close"].astype(float) / nopen - 1) * 100).abs()
        for sym in mkt.symbols:
            df = mkt.df(sym)
            F = self._stock_feats(df)
            F["n_abs"] = n_abs.reindex(df.index) if n_abs is not None else np.nan
            self.F[sym] = F

    # ------------------------------------------------------------------ decisions
    def _signal(self, ctx, sym):
        """Return (score, side) or None, using only ctx (last completed bar)."""
        f = ctx.feat
        z = f(sym, "zb")
        if z is None:
            return None
        if self.hook:
            zmax, zmin, zp, bar = f(sym, "zmax"), f(sym, "zmin"), f(sym, "zprev"), f(sym, "bar")
            if zp is None or bar is None:
                return None
            if zmax is not None and zmax >= self.Z and self.Zkeep <= z < zp and bar < 0:
                side = "short"
            elif zmin is not None and zmin <= -self.Z and -self.Zkeep >= z > zp and bar > 0:
                side = "long"
            else:
                return None
            score = abs(zmax if side == "short" else zmin)
        else:
            if abs(z) < self.Z:
                return None
            side, score = ("short" if z > 0 else "long"), abs(z)
        if self.vol_mult is not None:
            vm = f(sym, "vmax")
            if vm is None or vm < self.vol_mult:
                return None
        if self.rsi_hi is not None:
            r = f(sym, "rsi")
            if r is None or (side == "short" and r < self.rsi_hi) or (side == "long" and r > 100 - self.rsi_hi):
                return None
        if self.nifty_max is not None:
            na = f(sym, "n_abs")
            if na is None or na > self.nifty_max:
                return None
        return score, side

    def on_bar(self, ctx):
        room = self.max_positions - ctx.n_open
        if room <= 0:
            return []
        cands = []
        for sym in ctx.symbols:
            if sym in ctx.positions or (ctx.date, sym) in self._done:
                continue
            sig = self._signal(ctx, sym)
            if sig is None:
                continue
            vw, dist = ctx.feat(sym, "vwap"), ctx.feat(sym, "dist")
            if vw is None or dist is None or abs(dist) < self.min_dist_pct:
                continue
            cands.append((sig[0], sym, sig[1], vw))
        cands.sort(reverse=True)
        orders = []
        for _, sym, side, vw in cands[:room]:
            od = dict(sym=sym, side=side, stop_pct=self.stop_pct, target=vw, tag=f"vwap_{side}")
            if self.hold_min:
                od["exit_time"] = _hm(min(ctx.minutes + int(self.hold_min), 15 * 60 + 15))
            orders.append(od)
            self._done.add((ctx.date, sym))
        return orders
