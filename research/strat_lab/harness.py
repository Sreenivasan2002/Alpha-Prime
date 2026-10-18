"""
Strategy lab harness - one honest, shared simulator for comparing intraday strategies.

Every strategy is judged by the SAME rules the live app uses:
  * NIFTY 100 5-min bars (cached in data_5m.pkl, 23 Jul - 16 Oct 2026), plus the NIFTY index as "NIFTY"
  * decisions see only COMPLETED bars; orders fill at the NEXT bar's open (+ slippage)
  * sizing from the live code: risk 0.5% of capital per trade, max 20% of capital per stock,
    5x margin buying power, max 10 open positions
  * stop / target / trailing checked intrabar; if a bar touches both, the STOP fills (pessimistic);
    a gap through the stop fills at the open
  * time exits fill at the open of the exit bar (= the price at that clock time)
  * daily loss limit -2%: new entries blocked AND (by default) everything flattened
  * Indian intraday charges on every trade (alpha_prime.core.costs), longs and shorts

Periods (fixed - do not tune on TEST):
  TRAIN = 2026-07-23 .. 2026-09-11      TEST = 2026-09-14 .. 2026-10-16

Strategy API - see strategies/app_current.py for a complete example:

    class Strategy:
        name = "my_strategy"
        entry_start = "09:30"     # earliest bar whose OPEN we may enter at
        entry_end = "14:30"       # latest
        squareoff = "15:15"       # all positions closed at the open of this bar
        max_positions = 10        # <= 10
        max_trades_per_day = None # optional cap on entries per day
        profit_lock_pct = None    # optional: stop new entries once day P&L >= this % of capital

        def prepare(self, mkt):   # optional, vectorised, CAUSAL features only
            self.F = {sym: DataFrame(index=mkt.df(sym).index, ...)}
        def on_bar(self, ctx):    # return a list of orders to fill at this bar's open
            return [dict(sym="TCS", side="long", stop_pct=1.0, target_pct=2.0,
                         exit_time="12:00", trail=(0.6, 1.0, 1.0), risk_mult=1.0, tag="x")]

    Order fields: sym, side ("long"|"short"), stop_pct (required, % from fill) or stop (absolute
    price), target_pct / target (optional), exit_time "HH:MM" (optional, <= squareoff),
    trail (be_arm_pct, trail_arm_pct, trail_dist_pct) optional, risk_mult in (0, 1] optional.

    CAUSALITY RULE: a feature value stored at row i may use bars <= i only (bar i is complete when
    the decision for bar i+1 is made). ctx.feat()/ctx.last()/ctx.bars() already return row i-1
    relative to the bar being entered. Never index mkt data directly inside on_bar.
    leak_check() verifies stored features by recomputing them on truncated data.

CLI:
    venv\\Scripts\\python research\\strat_lab\\harness.py strategies.app_current --period train
    venv\\Scripts\\python research\\strat_lab\\harness.py strategies.app_current --period test --leak-check
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

LAB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(LAB))
sys.path.insert(0, ROOT)
sys.path.insert(0, LAB)

from alpha_prime.core.costs import intraday_round_trip_cost  # noqa: E402
from alpha_prime.core.risk import compute_max_quantity, compute_trailed_stop  # noqa: E402

TRAIN = (date(2026, 7, 23), date(2026, 9, 11))
TEST = (date(2026, 9, 14), date(2026, 10, 16))
ALL = (date(2026, 7, 23), date(2026, 10, 16))
PERIODS = {"train": TRAIN, "test": TEST, "all": ALL}

CAPITAL = 1_000_000.0
MARGIN = 5.0
RISK_PCT = 0.5
NOTIONAL_PCT = 20.0
MAX_POS_CAP = 10
DAILY_LOSS_PCT = 2.0
DEFAULT_SLIPPAGE = 0.0001  # 1 bp per side, adverse


def hm(s: str) -> int:
    return int(s[:2]) * 60 + int(s[3:5])


_DATA_CACHE = None


def load_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        with open(os.path.join(LAB, "data_5m.pkl"), "rb") as f:
            _DATA_CACHE = pickle.load(f)
    return _DATA_CACHE


class Market:
    """Read-only container of per-symbol 5-min OHLCV (IST index) with day helpers."""

    def __init__(self, data: dict):
        self._data = data
        self.symbols = sorted(s for s in data if s != "NIFTY")
        self.arr = {}
        for s, df in data.items():
            idx = df.index
            dates = np.array(idx.date)
            mins = idx.hour * 60 + idx.minute
            first_of_day = np.r_[True, dates[1:] != dates[:-1]]
            self.arr[s] = dict(
                o=df["Open"].to_numpy(float), h=df["High"].to_numpy(float), l=df["Low"].to_numpy(float),
                c=df["Close"].to_numpy(float), v=df["Volume"].to_numpy(float),
                date=dates, mins=np.asarray(mins), first=first_of_day,
                pos={ts: i for i, ts in enumerate(idx)},
            )
            # previous session close and today's open per row (today's open only valid after bar 0)
            closes = df["Close"].groupby(dates).last()
            prev_map = dict(zip(closes.index[1:], closes.values[:-1]))
            opens = df["Open"].groupby(dates).first()
            self.arr[s]["prev_close"] = np.array([prev_map.get(d, np.nan) for d in dates])
            self.arr[s]["day_open"] = np.array([opens.get(d, np.nan) for d in dates])

    def df(self, sym: str) -> pd.DataFrame:
        return self._data[sym]

    def truncated(self, ts) -> "Market":
        """Market with only bars strictly before ts (for leak checks)."""
        return Market({s: d[d.index < ts] for s, d in self._data.items()})


class Ctx:
    """What a strategy may see when deciding orders for the bar starting at `ts`."""

    def __init__(self, sim, ts):
        self._sim = sim
        self.ts = ts
        self.date = ts.date()
        self.minutes = ts.hour * 60 + ts.minute
        self.time = ts.strftime("%H:%M")
        self.symbols = sim.mkt.symbols
        self.positions = {s: dict(side=p.side, entry=p.entry, qty=p.qty, tag=p.tag) for s, p in sim.open.items()}
        self.n_open = len(sim.open)
        self.trades_today = sim.trades_today
        self.day_pnl = sim.day_realized + sim.unrealized(ts)

    def _i(self, sym):
        """Index of the last COMPLETED bar of `sym` for today (None if none yet)."""
        a = self._sim.mkt.arr.get(sym)
        if a is None:
            return None
        i = a["pos"].get(self.ts)
        if i is None or i == 0 or a["date"][i - 1] != self.date:
            return None
        return i - 1

    def has_bar(self, sym) -> bool:
        a = self._sim.mkt.arr.get(sym)
        return a is not None and self.ts in a["pos"] and self._i(sym) is not None

    def last(self, sym, col: str):
        """Value of o/h/l/c/v from the last completed bar today."""
        i = self._i(sym)
        return None if i is None else float(self._sim.mkt.arr[sym][col][i])

    def prev_close(self, sym):
        i = self._i(sym)
        return None if i is None else float(self._sim.mkt.arr[sym]["prev_close"][i])

    def day_open(self, sym):
        i = self._i(sym)
        return None if i is None else float(self._sim.mkt.arr[sym]["day_open"][i])

    def feat(self, sym, name: str):
        i = self._i(sym)
        if i is None:
            return None
        F = getattr(self._sim.strategy, "F", {}).get(sym)
        if F is None or name not in F.columns:
            return None
        v = F[name].iat[i]
        return None if (isinstance(v, float) and math.isnan(v)) else v

    def bars(self, sym, n: int) -> pd.DataFrame:
        """Last n completed bars (may include previous days)."""
        a = self._sim.mkt.arr.get(sym)
        if a is None or self.ts not in a["pos"]:
            return pd.DataFrame()
        i = a["pos"][self.ts]
        return self._sim.mkt.df(sym).iloc[max(0, i - n):i]


@dataclass
class _Pos:
    sym: str
    side: str
    entry: float
    qty: int
    stop: float
    target: float | None
    exit_min: int
    trail: tuple | None
    tag: str
    entry_ts: object
    peak: float  # best favourable price seen (high for long, low for short)


@dataclass
class Result:
    name: str
    period: tuple
    trades: pd.DataFrame
    daily: pd.Series
    metrics: dict = field(default_factory=dict)


class Sim:
    def __init__(self, mkt: Market, strategy, start: date, end: date, slippage=DEFAULT_SLIPPAGE,
                 flatten_on_loss_limit=True, capital=CAPITAL):
        self.mkt, self.strategy, self.start, self.end = mkt, strategy, start, end
        self.slip = slippage
        self.flatten = flatten_on_loss_limit
        self.capital = capital
        self.open: dict[str, _Pos] = {}
        self.trades = []
        self.day_realized = 0.0
        self.trades_today = 0
        self.cash_used = 0.0

    # ---- helpers
    def _bar(self, sym, ts):
        a = self.mkt.arr[sym]
        i = a["pos"].get(ts)
        return None if i is None else (a["o"][i], a["h"][i], a["l"][i], a["c"][i], i)

    def unrealized(self, ts):
        u = 0.0
        for s, p in self.open.items():
            b = self._bar(s, ts)
            if b is None:
                continue
            i = b[4]
            if i == 0:
                continue
            px = self.mkt.arr[s]["c"][i - 1]  # last completed close
            u += (px - p.entry) * p.qty * (1 if p.side == "long" else -1)
        return u

    def _close(self, p: _Pos, px: float, ts, reason: str):
        side = 1 if p.side == "long" else -1
        fill = px * (1 - self.slip * side)  # adverse slippage
        gross = (fill - p.entry) * p.qty * side
        buy_v, sell_v = (p.entry * p.qty, fill * p.qty) if p.side == "long" else (fill * p.qty, p.entry * p.qty)
        cost = intraday_round_trip_cost(buy_v, sell_v)
        net = gross - cost
        self.day_realized += net
        self.cash_used -= p.entry * p.qty
        self.trades.append(dict(sym=p.sym, side=p.side, tag=p.tag, entry_time=p.entry_ts, exit_time=ts,
                                entry=round(p.entry, 2), exit=round(fill, 2), qty=p.qty, gross=round(gross, 2),
                                cost=round(cost, 2), pnl=round(net, 2), reason=reason,
                                pnl_pct=round((fill / p.entry - 1) * 100 * side, 3)))
        del self.open[p.sym]

    def _update_trail(self, p: _Pos, h, l, c):
        if not p.trail:
            return
        be, arm, dist = p.trail
        if p.side == "long":
            p.peak = max(p.peak, h)
            p.stop = compute_trailed_stop(p.entry, p.peak, c, p.stop, be, arm, dist)
        else:  # mirror for shorts: stop only ever moves DOWN
            p.peak = min(p.peak, l)
            gain = (p.entry - c) / p.entry
            new = p.stop
            if gain >= be / 100:
                new = min(new, p.entry)
            if gain >= arm / 100:
                new = min(new, p.peak * (1 + dist / 100))
            p.stop = new

    # ---- main loop
    def run(self) -> Result:
        st = self.strategy
        e_start, e_end = hm(getattr(st, "entry_start", "09:30")), hm(getattr(st, "entry_end", "14:30"))
        sq = hm(getattr(st, "squareoff", "15:15"))
        max_pos = min(int(getattr(st, "max_positions", 10)), MAX_POS_CAP)
        max_tpd = getattr(st, "max_trades_per_day", None)
        lock = getattr(st, "profit_lock_pct", None)
        timeline = sorted({ts for s in self.mkt.symbols for ts in self.mkt.df(s).index
                           if self.start <= ts.date() <= self.end})
        cur, blocked, daily = None, False, {}
        for ts in timeline:
            d, m = ts.date(), ts.hour * 60 + ts.minute
            if d != cur:
                if cur is not None:
                    daily[cur] = self.day_realized
                # anything still open from a previous day (its last bars were missing) is closed at
                # the last close we have for it on that day
                for p in list(self.open.values()):
                    a = self.mkt.arr[p.sym]
                    k = max(k for t, k in a["pos"].items() if t.date() == cur)
                    self._close(p, a["c"][k], ts, "STALE")
                cur, blocked, self.day_realized, self.trades_today = d, False, 0.0, 0
            # A. exits
            for p in list(self.open.values()):
                b = self._bar(p.sym, ts)
                if b is None:
                    continue
                o, h, l, c, i = b
                last_today = (i + 1 >= len(self.mkt.arr[p.sym]["date"])) or self.mkt.arr[p.sym]["date"][i + 1] != d
                long = p.side == "long"
                if m >= min(p.exit_min, sq):
                    self._close(p, o, ts, "TIME")
                elif (o <= p.stop) if long else (o >= p.stop):
                    self._close(p, o, ts, "STOP_GAP")
                elif (l <= p.stop) if long else (h >= p.stop):
                    self._close(p, p.stop, ts, "STOP")
                elif p.target and ((o >= p.target) if long else (o <= p.target)):
                    self._close(p, o, ts, "TARGET_GAP")
                elif p.target and ((h >= p.target) if long else (l <= p.target)):
                    self._close(p, p.target, ts, "TARGET")
                elif last_today:
                    self._close(p, c, ts, "TIME")
                else:
                    self._update_trail(p, h, l, c)
            # B. daily limits
            if not blocked:
                dp = self.day_realized + self.unrealized(ts)
                if dp <= -DAILY_LOSS_PCT / 100 * self.capital:
                    blocked = True
                    if self.flatten:
                        for p in list(self.open.values()):
                            b = self._bar(p.sym, ts)
                            if b:
                                self._close(p, b[0], ts, "LOSS_LIMIT")
                elif lock is not None and dp >= lock / 100 * self.capital:
                    blocked = True
            # C. entries
            if blocked or not (e_start <= m <= e_end) or m >= sq:
                continue
            if len(self.open) >= max_pos or (max_tpd is not None and self.trades_today >= max_tpd):
                continue
            orders = st.on_bar(Ctx(self, ts)) or []
            for od in orders:
                if len(self.open) >= max_pos or (max_tpd is not None and self.trades_today >= max_tpd):
                    break
                sym = od["sym"]
                if sym in self.open or sym not in self.mkt.arr:
                    continue
                b = self._bar(sym, ts)
                if b is None:
                    continue
                side = od.get("side", "long")
                sgn = 1 if side == "long" else -1
                fill = b[0] * (1 + self.slip * sgn)
                if "stop" in od and od["stop"]:
                    stop = float(od["stop"])
                else:
                    stop = fill * (1 - sgn * float(od["stop_pct"]) / 100)
                if (side == "long" and stop >= fill) or (side == "short" and stop <= fill):
                    continue
                tgt = od.get("target")
                if tgt is None and od.get("target_pct"):
                    tgt = fill * (1 + sgn * float(od["target_pct"]) / 100)
                # sizing: same pure function as the live app (risk measured on |entry - stop|)
                ref_stop = stop if side == "long" else fill - (stop - fill)
                max_q, _ = compute_max_quantity(self.capital, fill, ref_stop, RISK_PCT, NOTIONAL_PCT)
                rm = min(max(float(od.get("risk_mult", 1.0)), 0.0), 1.0)
                q = int(min(max_q * rm, (self.capital * MARGIN - self.cash_used) // fill))
                if q < 1:
                    continue
                ex = min(hm(od["exit_time"]), sq) if od.get("exit_time") else sq
                self.open[sym] = _Pos(sym, side, fill, q, stop, tgt, ex, od.get("trail"), od.get("tag", ""),
                                      ts, fill)
                self.cash_used += fill * q
                self.trades_today += 1
        if cur is not None:
            for p in list(self.open.values()):
                a = self.mkt.arr[p.sym]
                self._close(p, a["c"][max(k for t, k in a["pos"].items() if t.date() == cur)], cur, "END")
            daily[cur] = self.day_realized
        trades = pd.DataFrame(self.trades)
        days = sorted({ts.date() for ts in timeline})
        dser = pd.Series({dd: daily.get(dd, 0.0) for dd in days}, dtype=float)
        return Result(getattr(st, "name", type(st).__name__), (self.start, self.end), trades, dser,
                      metrics(trades, dser, self.capital))


def metrics(trades: pd.DataFrame, daily: pd.Series, capital=CAPITAL) -> dict:
    n = len(trades)
    out = dict(trades=n, days=len(daily), net=round(float(daily.sum()), 2),
               ret_pct=round(float(daily.sum()) / capital * 100, 3))
    if n:
        w, lo = trades[trades.pnl > 0], trades[trades.pnl < 0]
        out.update(gross=round(float(trades.gross.sum()), 2), costs=round(float(trades.cost.sum()), 2),
                   win_rate=round(len(w) / n * 100, 1),
                   profit_factor=round(float(w.pnl.sum() / -lo.pnl.sum()), 2) if len(lo) else None,
                   expectancy=round(float(trades.pnl.mean()), 1), trades_per_day=round(n / max(len(daily), 1), 1),
                   exits=trades.reason.value_counts().to_dict())
    eq = daily.cumsum() + capital
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100 if len(eq) else 0
    sd = daily.std(ddof=1) if len(daily) > 1 else 0
    out.update(max_dd_pct=round(float(dd), 2), up_days=int((daily > 0).sum()), down_days=int((daily < 0).sum()),
               daily_sharpe=round(float(daily.mean() / sd), 3) if sd else None,
               t_stat=round(float(daily.mean() / (sd / math.sqrt(len(daily)))), 2) if sd else None)
    return out


def run(strategy, period="train", slippage=DEFAULT_SLIPPAGE, flatten_on_loss_limit=True, capital=CAPITAL) -> Result:
    mkt = Market(load_data())
    if hasattr(strategy, "prepare"):
        strategy.prepare(mkt)
    start, end = PERIODS[period] if isinstance(period, str) else period
    return Sim(mkt, strategy, start, end, slippage, flatten_on_loss_limit, capital).run()


def leak_check(strategy_cls, n_samples=8, tol=1e-6) -> dict:
    """Recompute features on data truncated just after each sampled bar; any mismatch = lookahead."""
    full = Market(load_data())
    s_full = strategy_cls()
    if not hasattr(s_full, "prepare"):
        return dict(checked=0, mismatches=[], note="no prepare(): nothing to check")
    s_full.prepare(full)
    F = getattr(s_full, "F", None)
    if not F:
        return dict(checked=0, mismatches=[], note="prepare() stored no self.F features")
    ref = full.df("RELIANCE").index
    test_idx = [t for t in ref if TEST[0] <= t.date() <= TEST[1]]
    picks = [test_idx[int(k * (len(test_idx) - 1) / (n_samples - 1))] for k in range(n_samples)]
    mism, checked = [], 0
    for ts in picks:
        trunc = full.truncated(ts + pd.Timedelta(minutes=1))  # keep the bar starting at ts (complete)
        s_t = strategy_cls()
        s_t.prepare(trunc)
        for sym, Ff in F.items():
            Ft = getattr(s_t, "F", {}).get(sym)
            if Ft is None or ts not in Ft.index or ts not in Ff.index:
                continue
            for col in Ff.columns:
                a, b = Ff.at[ts, col], Ft.at[ts, col]
                checked += 1
                try:
                    fa, fb = float(a), float(b)
                    na, nb = math.isnan(fa), math.isnan(fb)
                    if na and nb:
                        continue
                    # a value that only exists with future data (or vanishes with it) is a leak
                    bad = (na != nb) or abs(fa - fb) > tol * max(1.0, abs(fa))
                except (TypeError, ValueError):
                    bad = a != b
                if bad:
                    mism.append(dict(ts=str(ts), sym=sym, col=col, full=a, truncated=b))
    return dict(checked=checked, mismatches=mism[:20], n_mismatches=len(mism))


def weekly(daily: pd.Series) -> pd.Series:
    s = daily.copy()
    s.index = pd.to_datetime(s.index)
    return s.groupby(s.index - pd.to_timedelta(s.index.weekday, unit="D")).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", help="e.g. strategies.app_current (class must be named Strategy)")
    ap.add_argument("--period", default="train", choices=list(PERIODS))
    ap.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE)
    ap.add_argument("--leak-check", action="store_true")
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--save", help="write trades CSV here")
    a = ap.parse_args()
    mod = importlib.import_module(a.module)
    res = run(mod.Strategy(), a.period, a.slippage)
    print(f"{res.name} | {a.period} {res.period[0]}..{res.period[1]} | slippage {a.slippage * 1e4:.1f}bp/side")
    for k, v in res.metrics.items():
        print(f"  {k}: {v}")
    if a.weekly:
        for wk, v in weekly(res.daily).items():
            print(f"  week {wk.date()}: {v:+,.0f}")
    if a.save:
        res.trades.to_csv(a.save, index=False)
    if a.leak_check:
        lc = leak_check(mod.Strategy)
        print("  leak_check:", {k: v for k, v in lc.items() if k != "mismatches"})
        for mm in lc.get("mismatches", [])[:5]:
            print("   MISMATCH", mm)


if __name__ == "__main__":
    main()
