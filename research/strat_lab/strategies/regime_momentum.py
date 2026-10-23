"""regime_momentum - the MINIMAL FIX of the live app's "buy early gainers" idea.

Keeps the core idea (go long stocks that are already up on the day and leading the index) but fixes every
flaw the baseline diagnosis found:

  1. NIFTY REGIME FILTER - the baseline was pure long market beta and bled on flat/down days. We only go
     long while the index itself is up on the day AND trending intraday, measured on COMPLETED NIFTY bars:
       NIFTY last close >= prev session close * (1 + nifty_min_ret%)
       NIFTY last close >  NIFTY day open
       NIFTY last close >  NIFTY 5-min EMA20
  2. NO 10:00 BURST / NO CHASING - instead of buying whatever is up at the window open, a stock must
     PULL BACK to its intraday EMA20 (a low within pb_tol% of it in the last pb_bars bars) while staying
     above session VWAP, and then RECLAIM: the last completed bar closes above the previous bar's high,
     above EMA9, green, on volume >= vol_min x its 20-bar average. Entries are staggered: at most
     max_new_per_bar new names per bar and a hard daily cap (max_trades_per_day).
  3. RELATIVE STRENGTH - stock day return (vs prev close) in [gain_lo, gain_hi]% and at least rs_min%
     above NIFTY's day return; candidates ranked by relative strength.
  4. REALISTIC EXITS - stop / target sized to the ~0.5-0.7% typical excursion (not 1.2% / 2.5%), and an
     early time exit (exit_time) instead of riding to 15:15.

All features are vectorised and causal (row i uses bars <= i only; per-day groupings use cumulative ops).
No TechnicalAnalyzer calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _stock_feats(df: pd.DataFrame, pb_bars: int) -> pd.DataFrame:
    o, h, l, c, v = (df[k].astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    day = pd.Series(df.index.date, index=df.index)
    # previous session's last close (known all day)
    last_close = c.groupby(day).last()
    prev_close = day.map(dict(zip(last_close.index[1:], last_close.values[:-1]))).astype(float)
    day_open = o.groupby(day).transform("first")
    tp = (h + l + c) / 3.0
    cum_v = v.groupby(day).cumsum()
    vwap = ((tp * v).groupby(day).cumsum() / cum_v.where(cum_v > 0)).fillna(c)
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    vol_avg = v.shift(1).rolling(20, min_periods=10).mean()
    vol_ratio = v / vol_avg.where(vol_avg > 0)
    # lowest low of the last pb_bars bars, within the same day only
    low_n = l.groupby(day).transform(lambda x: x.rolling(pb_bars, min_periods=1).min())
    # was the EMA20 "touched" in the window: min over window of (low / ema20 - 1)
    touch = (l / ema20 - 1.0).groupby(day).transform(lambda x: x.rolling(pb_bars, min_periods=1).min())
    prev_high = h.groupby(day).shift(1)
    return pd.DataFrame(dict(
        ret_pc=(c / prev_close - 1.0) * 100.0,
        ret_open=(c / day_open - 1.0) * 100.0,
        above_vwap=(c > vwap).astype(float),
        vwap_dist=(c / vwap - 1.0) * 100.0,
        ema_touch=touch * 100.0,                       # % ; <= pb_tol means it pulled back to EMA20
        reclaim=((c > prev_high) & (c > ema9) & (c > o)).astype(float),
        vol_ratio=vol_ratio,
        swing_low_pct=(c / low_n - 1.0) * 100.0,       # distance from close to the pullback low
    ), index=df.index)


def _nifty_feats(df: pd.DataFrame) -> pd.DataFrame:
    o, c = df["Open"].astype(float), df["Close"].astype(float)
    day = pd.Series(df.index.date, index=df.index)
    last_close = c.groupby(day).last()
    prev_close = day.map(dict(zip(last_close.index[1:], last_close.values[:-1]))).astype(float)
    day_open = o.groupby(day).transform("first")
    ema20 = c.ewm(span=20, adjust=False).mean()
    return pd.DataFrame(dict(
        n_ret_pc=(c / prev_close - 1.0) * 100.0,
        n_above_open=(c > day_open).astype(float),
        n_above_ema=(c > ema20).astype(float),
    ), index=df.index)


class Strategy:
    name = "regime_momentum"
    entry_start = "10:15"
    entry_end = "12:30"
    squareoff = "15:15"
    max_positions = 5
    max_trades_per_day = 5

    # regime
    nifty_min_ret = 0.15     # NIFTY vs prev close, %
    # stock selection
    gain_lo, gain_hi = 1.0, 4.0
    rs_min = 0.5             # stock day return minus NIFTY day return, %
    pb_bars = 6              # pullback look-back window (bars)
    pb_tol = 0.15            # low came within this % of EMA20 (or below it)
    vol_min = 1.0
    max_new_per_bar = 1
    # exits
    stop_pct = 0.7
    target_pct = 1.0
    exit_time = "13:00"
    trail = None             # e.g. (0.5, 0.8, 0.4)

    def prepare(self, mkt):
        self.F = {}
        nf = _nifty_feats(mkt.df("NIFTY"))
        self.F["NIFTY"] = nf
        for s in mkt.symbols:
            df = mkt.df(s)
            f = _stock_feats(df, self.pb_bars)
            n = nf.reindex(df.index)  # same timestamp => NIFTY bar completes with the stock bar
            regime = ((n.n_ret_pc >= self.nifty_min_ret) & (n.n_above_open > 0) & (n.n_above_ema > 0))
            rs = f.ret_pc - n.n_ret_pc
            sig = (regime
                   & f.ret_pc.between(self.gain_lo, self.gain_hi)
                   & (rs >= self.rs_min)
                   & (f.above_vwap > 0)
                   & (f.ema_touch <= self.pb_tol)
                   & (f.reclaim > 0)
                   & (f.vol_ratio >= self.vol_min))
            f["rs"] = rs
            f["sig"] = sig.astype(float)
            self.F[s] = f

    def on_bar(self, ctx):
        room = min(self.max_new_per_bar, self.max_positions - ctx.n_open,
                   self.max_trades_per_day - ctx.trades_today)
        if room <= 0:
            return []
        cands = []
        for sym in ctx.symbols:
            if sym in ctx.positions:
                continue
            if ctx.feat(sym, "sig"):
                cands.append((ctx.feat(sym, "rs") or 0.0, sym))
        cands.sort(reverse=True)
        out = []
        for _, sym in cands[:room]:
            od = dict(sym=sym, side="long", stop_pct=self.stop_pct, exit_time=self.exit_time, tag="regime_mom")
            if self.target_pct:
                od["target_pct"] = self.target_pct
            if self.trail:
                od["trail"] = self.trail
            out.append(od)
        return out
