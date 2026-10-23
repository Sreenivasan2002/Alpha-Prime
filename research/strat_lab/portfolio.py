"""
Multi-strategy portfolio construction on top of the strategy-lab harness.

What this script does (all P&L is NET of Indian intraday charges + 1 bp/side slippage, via harness.Sim):
  1. Baseline: strategies.app_current (the live app's rules) on TRAIN and TEST.
     Slow (~3 min/period), so the result is cached in results/app_current_baseline.json
     (pass --rerun-baseline to recompute).
  2. Stand-alone runs of every audited sub-strategy at full capital on TRAIN and TEST, and the TRAIN
     daily-P&L correlation matrix (plus the baseline and the NIFTY day return for reference).
  3. Candidate portfolios. Two ways to combine the same sub-strategies:
       * "slices"  - each sub-strategy runs alone with capital = CAPITAL * w (separate sub-accounts:
                     its own 0.5%-risk / 20%-notional sizing, 10-position cap and -2% loss limit on
                     its slice); daily P&L is summed. This is what the task specifies. NOTE: a smaller
                     slice means a smaller order, and brokerage is min(0.1%, Rs 20) per order, so
                     per-rupee costs rise as the slice shrinks (Rs 28.6k orders at w=1/7 pay ~0.07%
                     per side vs ~0.01% for Rs 2L orders).
       * "shared"  - ONE Rs 10L account. A wrapper Strategy (Multi) delegates to every sub-strategy's
                     on_bar; orders are full size; the 10-position cap and the -2% daily loss limit
                     are shared; each sub keeps its own entry window, max_positions and
                     max_trades_per_day. Slots are first-come-first-served in list order.
     Candidates
       a) EW_all        - all 7 audited strategies, equal weight
       b) TOP_T         - strategies with TRAIN daily t-stat >= 2 (selected on TRAIN only;
                          "TRAIN profit factor > 1" would select all 7 and duplicate (a))
       c) regime books  - each day a gate decides which sub-strategies may trade, using ONLY
                          information available before that sub's first possible entry:
            RG_beta     thesis rule, 0 fitted thresholds: long-only subs (low_freq_breakout,
                        regime_momentum) trade only if NIFTY is above its day open at their gate
                        time; the short-only gap fade only if NIFTY is NOT above its open;
                        direction-neutral subs always trade.
            RG_trend    trend/range rule, 1 threshold: trend subs (orb, index_trend, rs_neutral,
                        low_freq_breakout, regime_momentum) trade only if |NIFTY ret from open| >=
                        0.15% at their gate time; reversion subs (gap, vwap_reversion) only if < 0.15%.
            RG_fit      data-led rule, derived programmatically from the TRAIN regime table (section 2):
                        six market-state buckets at each sub's gate time (NIFTY up/down from open,
                        trend |ret| >= 0.15% / range, breadth >= 0.5 / < 0.5, breadth = share of
                        NIFTY-100 stocks above their day open). A sub is switched OFF in every bucket
                        where its TRAIN mean stand-alone daily net P&L was <= 0 (buckets with >= 5
                        TRAIN days only). Fitted on 37 days -> expect overfit; it is a test of whether
                        TRAIN regime dependence persists, not a validated rule.
            PERF5/10    strategy-momentum switch: a sub trades on day d only if its own stand-alone
                        (shadow, full-capital) net P&L over the previous 5 / 10 sessions is > 0
                        (on by default when there is no history). Known before the open.
     The recommended portfolio is the candidate with the highest TRAIN daily t-stat (TRAIN only);
     TEST is reported for every candidate but never used to choose.

Causality: the Multi wrapper reads market data only through ctx (sub-strategies get a proxy ctx that
serves THEIR OWN causal features). Gate decisions are stored as features on the NIFTY rows (value at
row i uses bars <= i, or only previous sessions) and are checked with harness.leak_check (--leak-check).
For per-sub position / trade-count bookkeeping the wrapper reads the simulator's own fill records
(ctx._sim.open / ctx._sim.trades) - this is our own past order history, not market data.

Usage:
    venv\\Scripts\\python research\\strat_lab\\portfolio.py              # everything
    venv\\Scripts\\python research\\strat_lab\\portfolio.py --leak-check # also leak-check the gates
    venv\\Scripts\\python research\\strat_lab\\portfolio.py --self-check # wrapper == stand-alone checks
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
import harness as H  # noqa: E402

RESULTS = os.path.join(LAB, "results")
CAPITAL = H.CAPITAL

# audited, valid sub-strategies (module names under strategies/)
SUBS = ["regime_momentum", "orb", "vwap_reversion", "gap", "rs_neutral", "index_trend", "low_freq_breakout"]

# first clock time at which each sub can actually enter (its gate is decided from the bar before)
GATE_TIME = {"rs_neutral": "10:15"}  # rs_neutral only acts at its 10:15 session; others: entry_start

SIDE_BIAS = {"low_freq_breakout": "long", "regime_momentum": "long", "gap": "short"}  # others neutral
TREND_SUBS = {"orb", "index_trend", "rs_neutral", "low_freq_breakout", "regime_momentum"}
REVERSION_SUBS = {"gap", "vwap_reversion"}
TREND_T = 0.15  # % |NIFTY return from open| separating trend from range at the gate time


def sub_cls(name):
    return importlib.import_module("strategies." + name).Strategy


def gate_min(name) -> int:
    return H.hm(GATE_TIME.get(name, getattr(sub_cls(name), "entry_start", "09:30")))


# ---------------------------------------------------------------------------------------------- market
_MKT = None


def market():
    global _MKT
    if _MKT is None:
        _MKT = H.Market(H.load_data())
    return _MKT


def run(strategy, period, capital=CAPITAL) -> H.Result:
    """Same as harness.run() but reuses one Market object (identical results, ~4 s faster per run)."""
    mkt = market()
    if hasattr(strategy, "prepare"):
        strategy.prepare(mkt)
    start, end = H.PERIODS[period]
    return H.Sim(mkt, strategy, start, end, H.DEFAULT_SLIPPAGE, True, capital).run()


# ---------------------------------------------------------------------------------------------- gates
_REGIME = {}


def _nifty_regime(mkt) -> pd.DataFrame:
    """Causal NIFTY-row features: return from day open and NIFTY-100 breadth (share above day open)."""
    key = (id(mkt), len(mkt.df("NIFTY")))
    if key not in _REGIME:
        _REGIME[key] = _nifty_regime_calc(mkt)
    return _REGIME[key]


def _nifty_regime_calc(mkt) -> pd.DataFrame:
    N = mkt.df("NIFTY")
    nd = np.asarray(N.index.date)
    n_open = N["Open"].astype(float).groupby(nd).transform("first")
    f = pd.DataFrame(index=N.index)
    f["n_ret"] = (N["Close"].astype(float) / n_open - 1.0) * 100.0
    syms = mkt.symbols
    C = pd.DataFrame({s: mkt.df(s)["Close"] for s in syms}).sort_index()
    O = pd.DataFrame({s: mkt.df(s)["Open"] for s in syms}).sort_index()
    dk = np.asarray(C.index.date)
    Cf = C.groupby(dk).ffill()                                         # within-day, causal
    Od = O.groupby(dk).transform("first").where(C.notna().groupby(dk).cummax())
    r = Cf / Od - 1.0
    valid = r.notna()
    br = (r > 0).where(valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    f["breadth"] = br.reindex(N.index)
    return f


def _gate_series(f: pd.DataFrame, raw: np.ndarray, gm: int) -> np.ndarray:
    """Running-then-frozen gate value. On rows whose bar starts at or before gm-5 (bars that are
    complete when the gate-time bar opens) the value is the rule evaluated on THAT row; on later rows
    of the day it stays frozen at the value of the last such row. Every row therefore depends only on
    rows <= itself (leak-check clean), and any read at/after the gate time sees the decision taken
    from the last completed bar before the gate time."""
    idx = f.index
    mins = np.asarray(idx.hour * 60 + idx.minute)
    days = np.asarray(idx.date)
    pre = pd.Series(np.where(mins <= gm - 5, raw, np.nan), index=idx)
    return pre.groupby(days).ffill().to_numpy(float)


class MarketGate:
    """Market-state gate. rule(sub, f) -> boolean array over NIFTY rows; f has n_ret (%, NIFTY close
    vs its day open) and breadth (0..1, share of NIFTY-100 stocks above their day open)."""

    def __init__(self, label, rule):
        self.label, self.rule = label, rule

    def columns(self, mkt, subs, capital):
        f = _nifty_regime(mkt)
        ok = np.isfinite(f["n_ret"].to_numpy(float)) & np.isfinite(f["breadth"].to_numpy(float))
        cols = {"n_ret": f["n_ret"].to_numpy(), "breadth": f["breadth"].to_numpy()}
        for s in subs:
            v = np.broadcast_to(np.asarray(self.rule(s, f), dtype=float), (len(f),))
            raw = np.where(ok, v, np.nan)
            cols["g_" + s] = _gate_series(f, raw, gate_min(s))
        return pd.DataFrame(cols, index=f.index)


def rule_beta(sub, f):
    bias = SIDE_BIAS.get(sub)
    if bias == "long":
        return (f["n_ret"] > 0).to_numpy()
    if bias == "short":
        return (f["n_ret"] <= 0).to_numpy()
    return True


def rule_trend(sub, f):
    trending = (f["n_ret"].abs() >= TREND_T).to_numpy()
    return trending if sub in TREND_SUBS else ~trending


BUCKETS = {  # market-state buckets at a sub's gate time; used by the TRAIN table AND by RG_fit
    "nifty_up": lambda f: f["n_ret"] > 0,
    "nifty_dn": lambda f: f["n_ret"] <= 0,
    "trend": lambda f: f["n_ret"].abs() >= TREND_T,
    "range": lambda f: f["n_ret"].abs() < TREND_T,
    "breadth_hi": lambda f: f["breadth"] >= 0.5,
    "breadth_lo": lambda f: f["breadth"] < 0.5,
}
FIT_MIN_DAYS = 5


def make_rule_fit(off: dict):
    """off = {sub: [bucket names in which the sub is switched off]} (derived on TRAIN)."""
    def rule_fit(sub, f):
        on = np.ones(len(f), dtype=bool)
        for b in off.get(sub, []):
            on &= ~BUCKETS[b](f).to_numpy(bool)
        return on
    return rule_fit


def derive_fit_off(stats: pd.DataFrame) -> dict:
    """From the numeric TRAIN regime stats (columns '<bucket>_mean', '<bucket>_n')."""
    off = {}
    for s in stats.index:
        off[s] = [b for b in BUCKETS
                  if stats.at[s, f"{b}_n"] >= FIT_MIN_DAYS and stats.at[s, f"{b}_mean"] <= 0]
    return off


_SHADOW = {}


def shadow_daily(mkt, sub) -> pd.Series:
    """Stand-alone full-capital daily net P&L of `sub` over every day in `mkt` (paper-tracking).
    Day d's value depends only on bars of day d and earlier (the simulator is causal)."""
    key = (sub, id(mkt), len(mkt.df("NIFTY")))
    if key not in _SHADOW:
        days = sorted(set(mkt.df("NIFTY").index.date))
        s = sub_cls(sub)()
        s.prepare(mkt)
        _SHADOW[key] = H.Sim(mkt, s, days[0], days[-1], H.DEFAULT_SLIPPAGE, True, CAPITAL).run().daily
    return _SHADOW[key]


class PerfGate:
    """Strategy-momentum gate: sub may trade on day d iff its shadow P&L over the previous n
    sessions is > 0 (on when there is no previous session). Known before the open."""

    def __init__(self, n):
        self.n, self.label = n, f"perf{n}"

    def columns(self, mkt, subs, capital):
        N = mkt.df("NIFTY")
        days = np.asarray(N.index.date)
        cols = {}
        for s in subs:
            dd = shadow_daily(mkt, s)
            trail = dd.rolling(self.n, min_periods=1).sum().shift(1)     # previous sessions only
            on = {d: (True if not np.isfinite(v) else v > 0) for d, v in trail.items()}
            cols["g_" + s] = np.array([1.0 if on.get(d, True) else 0.0 for d in days])
        return pd.DataFrame(cols, index=N.index)


# ---------------------------------------------------------------------------------------------- wrapper
class SubCtx:
    """What one sub-strategy sees: the real ctx, but its OWN features, its own position count and its
    own trades-today count. `positions` is the whole account (so it skips names other subs hold)."""

    def __init__(self, ctx, sub, n_open, trades_today):
        self._c, self._sub = ctx, sub
        self.ts, self.date, self.minutes, self.time = ctx.ts, ctx.date, ctx.minutes, ctx.time
        self.symbols, self.positions, self.day_pnl = ctx.symbols, ctx.positions, ctx.day_pnl
        self.n_open, self.trades_today = n_open, trades_today

    def has_bar(self, sym):
        return self._c.has_bar(sym)

    def last(self, sym, col):
        return self._c.last(sym, col)

    def prev_close(self, sym):
        return self._c.prev_close(sym)

    def day_open(self, sym):
        return self._c.day_open(sym)

    def bars(self, sym, n):
        return self._c.bars(sym, n)

    def feat(self, sym, name):
        i = self._c._i(sym)
        if i is None:
            return None
        F = getattr(self._sub, "F", {}).get(sym)
        if F is None or name not in F.columns:
            return None
        v = F[name].iat[i]
        return None if (isinstance(v, float) and math.isnan(v)) else v


class Multi:
    """Runs several sub-strategies in ONE simulated account (or one sub alone, for capital slices)."""
    name = "multi"
    squareoff = "15:15"
    max_positions = 10
    max_trades_per_day = None
    profit_lock_pct = None

    def __init__(self, subs=tuple(SUBS), gate=None, capital=CAPITAL, label=None):
        self.sub_names = list(subs)
        self.subs = [(n, sub_cls(n)()) for n in self.sub_names]
        self.gate, self.capital = gate, capital
        self.name = label or ("multi_" + "+".join(self.sub_names))
        self.entry_start = min((getattr(s, "entry_start", "09:30") for _, s in self.subs), key=H.hm)
        self.entry_end = max((getattr(s, "entry_end", "14:30") for _, s in self.subs), key=H.hm)
        self.F = {}

    def prepare(self, mkt):
        for _, s in self.subs:
            if hasattr(s, "prepare"):
                s.prepare(mkt)
        if self.gate is not None:
            self.F = {"NIFTY": self.gate.columns(mkt, self.sub_names, self.capital)}

    def _sub_counts(self, ctx, n):
        pre = n + "|"
        sim = ctx._sim
        n_open = sum(1 for p in sim.open.values() if p.tag.startswith(pre))
        done = 0
        for t in reversed(sim.trades):                 # closed trades, chronological by exit
            et = t["entry_time"]
            if et.date() != ctx.date:
                break
            if t["tag"].startswith(pre):
                done += 1
        return n_open, n_open + done

    def on_bar(self, ctx):
        orders, taken = [], set(ctx.positions)
        for n, s in self.subs:
            m = ctx.minutes
            if not (H.hm(getattr(s, "entry_start", "09:30")) <= m <= H.hm(getattr(s, "entry_end", "14:30"))):
                continue
            if m >= H.hm(getattr(s, "squareoff", "15:15")):
                continue
            if self.gate is not None and not ctx.feat("NIFTY", "g_" + n):
                continue
            n_open, tt = self._sub_counts(ctx, n)
            room = min(int(getattr(s, "max_positions", 10)), H.MAX_POS_CAP) - n_open
            mtpd = getattr(s, "max_trades_per_day", None)
            if mtpd is not None:
                room = min(room, mtpd - tt)
            if room <= 0:
                continue
            got = s.on_bar(SubCtx(ctx, s, n_open, tt)) or []
            for od in got:
                if room <= 0:
                    break
                sym = od["sym"]
                if sym in taken:
                    continue
                px = ctx.last(sym, "c")       # affordability proxy: the sim skips qty < 1 orders
                if px and px > self.capital * H.NOTIONAL_PCT / 100:
                    continue
                od = dict(od)
                od["tag"] = f"{n}|{od.get('tag', '')}"
                orders.append(od)
                taken.add(sym)
                room -= 1
        return orders


# ---------------------------------------------------------------------------------------------- books
def run_slices(subs, period, gate=None, weights=None):
    """Each sub alone on capital CAPITAL*w; returns (daily, trades, per-sub metrics)."""
    w = weights or {s: 1.0 / len(subs) for s in subs}
    dailies, trades, per = [], [], {}
    for s in subs:
        cap = CAPITAL * w[s]
        r = run(Multi([s], gate=gate, capital=cap, label=s), period, capital=cap)
        dailies.append(r.daily.rename(s))
        if len(r.trades):
            trades.append(r.trades.assign(book=s))
        per[s] = r.metrics
    daily = pd.concat(dailies, axis=1).fillna(0.0).sum(axis=1)
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["pnl", "gross", "cost", "reason"])
    return daily, tr, per


def run_shared(subs, period, gate=None, label="shared"):
    r = run(Multi(subs, gate=gate, capital=CAPITAL, label=label), period, capital=CAPITAL)
    return r.daily, r.trades, None


def book_metrics(daily, trades):
    m = H.metrics(trades, daily, CAPITAL)
    keep = ["net", "ret_pct", "trades", "days", "win_rate", "profit_factor", "max_dd_pct", "t_stat",
            "up_days", "gross", "costs"]
    return {k: m.get(k) for k in keep}


def short(m):
    return {k: m.get(k) for k in ["net", "ret_pct", "trades", "days", "win_rate", "profit_factor",
                                  "max_dd_pct", "t_stat", "up_days"]}


# ---------------------------------------------------------------------------------------------- checks
def self_check():
    """Wrapper fidelity: Multi([sub]) must reproduce the stand-alone sub exactly (full capital and a
    1/7 slice), and run() must equal harness.run()."""
    ok = True
    r0 = H.run(sub_cls("gap")(), "train")
    r1 = run(sub_cls("gap")(), "train")
    print(f"  harness.run == run (gap/train): {abs(r0.metrics['net'] - r1.metrics['net']) < 1e-6}")
    for s in SUBS:
        for cap in (CAPITAL, CAPITAL / 7):
            a = run(sub_cls(s)(), "train", capital=cap)
            b = run(Multi([s], capital=cap, label=s), "train", capital=cap)
            same = abs(a.metrics["net"] - b.metrics["net"]) < 1e-6 and a.metrics["trades"] == b.metrics["trades"]
            ok &= same
            print(f"  {s:18s} cap={cap:>10,.0f}  stand-alone {a.metrics['net']:>11,.2f} ({a.metrics['trades']})"
                  f"  wrapper {b.metrics['net']:>11,.2f} ({b.metrics['trades']})  {'OK' if same else 'MISMATCH'}")
    return ok


def gate_leak_checks():
    out = {}
    for label, gate in [("RG_beta", MarketGate("beta", rule_beta)),
                        ("RG_fit", MarketGate("fit", make_rule_fit(fit_off_from_train()))),
                        ("PERF5", PerfGate(5))]:
        t = time.time()
        lc = H.leak_check(functools.partial(Multi, SUBS, gate, CAPITAL, label))
        out[label] = {k: v for k, v in lc.items() if k != "mismatches"}
        print(f"  leak_check {label}: {out[label]}  ({time.time() - t:.0f}s)")
        for mm in lc.get("mismatches", [])[:3]:
            print("    MISMATCH", mm)
    return out


# ---------------------------------------------------------------------------------------------- main
def baseline(rerun=False):
    path = os.path.join(RESULTS, "app_current_baseline.json")
    if os.path.exists(path) and not rerun:
        with open(path) as f:
            return json.load(f)
    out = {}
    for p in ("train", "test"):
        r = H.run(sub_cls("app_current")(), p)
        out[p] = dict(metrics=r.metrics, daily={str(k): v for k, v in r.daily.items()})
    with open(path, "w") as f:
        json.dump(out, f, default=str, indent=1)
    return out


def regime_stats(solo_daily_train: dict) -> pd.DataFrame:
    """TRAIN-only conditional table used to define the switch rules: each sub's stand-alone daily net
    P&L split by the market state at ITS gate time (last completed bar before its first entry).
    Returns numeric columns '<bucket>_mean' and '<bucket>_n'."""
    f = _nifty_regime(market())
    mins = np.asarray(f.index.hour * 60 + f.index.minute)
    rows = []
    for s in SUBS:
        at = f[mins == gate_min(s) - 5].copy()
        at.index = at.index.date
        d = pd.DataFrame({"pnl": solo_daily_train[s]}).join(at[["n_ret", "breadth"]], how="left")
        row = {"sub": s}
        for b, fn in BUCKETS.items():
            msk = fn(d).to_numpy(bool)
            row[f"{b}_mean"] = float(d.pnl[msk].mean()) if msk.any() else np.nan
            row[f"{b}_n"] = int(msk.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("sub")


def regime_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Printable version of regime_stats: 'mean (n days)' per bucket."""
    out = pd.DataFrame(index=stats.index)
    out["gate"] = [f"{(gate_min(s) - 5) // 60:02d}:{(gate_min(s) - 5) % 60:02d}" for s in stats.index]
    for b in BUCKETS:
        out[b] = [f"{m:+,.0f} ({n})" if n else "-" for m, n in zip(stats[f"{b}_mean"], stats[f"{b}_n"])]
    return out


_FIT_OFF = None


def fit_off_from_train() -> dict:
    """RG_fit switch-off buckets derived from TRAIN stand-alone runs (cached)."""
    global _FIT_OFF
    if _FIT_OFF is None:
        sd = {s: run(sub_cls(s)(), "train").daily for s in SUBS}
        _FIT_OFF = derive_fit_off(regime_stats(sd))
    return _FIT_OFF


def checks_only():
    """Wrapper self-check + gate leak checks only; writes results/portfolio_checks.json."""
    t = time.time()
    print("== self-check: wrapper reproduces stand-alone strategies ==")
    ok = self_check()
    print("  all OK" if ok else "  SELF-CHECK FAILED")
    print("== gate leak checks ==")
    lc = gate_leak_checks()
    with open(os.path.join(RESULTS, "portfolio_checks.json"), "w") as fh:
        json.dump(dict(self_check_ok=bool(ok), leak_check=lc), fh, indent=1)
    print(f"wrote results/portfolio_checks.json ({time.time() - t:.0f}s)")


def nifty_day_returns():
    N = market().df("NIFTY")
    d = np.asarray(N.index.date)
    o = N["Open"].groupby(d).first()
    c = N["Close"].groupby(d).last()
    return ((c / c.shift(1) - 1) * 100).rename("NIFTY_cc%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-baseline", action="store_true")
    ap.add_argument("--leak-check", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--checks-only", action="store_true", help="only self-check + leak checks")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    T0 = time.time()
    if a.checks_only:
        checks_only()
        return

    # 1. baseline
    print("== 1. Baseline (strategies.app_current) ==")
    base = baseline(a.rerun_baseline)
    for p in ("train", "test"):
        print(f"  {p}: {short(base[p]['metrics'])}")

    if a.self_check:
        print("== self-check: wrapper reproduces stand-alone strategies ==")
        print("  all OK" if self_check() else "  SELF-CHECK FAILED")

    # 2. stand-alone runs + correlation
    print("== 2. Stand-alone sub-strategies (full capital) ==")
    solo = {p: {} for p in ("train", "test")}
    solo_daily = {p: {} for p in ("train", "test")}
    for s in SUBS:
        for p in ("train", "test"):
            r = run(sub_cls(s)(), p)
            solo[p][s] = short(r.metrics)
            solo_daily[p][s] = r.daily
        print(f"  {s:18s} TRAIN {solo['train'][s]}\n  {'':18s} TEST  {solo['test'][s]}")
    nifty = nifty_day_returns()
    corr = {}
    for p in ("train", "test"):
        D = pd.DataFrame(solo_daily[p])
        D["app_current"] = pd.Series({pd.Timestamp(k).date(): v for k, v in base[p]["daily"].items()})
        D["NIFTY_cc%"] = nifty.reindex(D.index)
        corr[p] = D.corr().round(2)
    print("  TRAIN daily P&L correlation:")
    print(corr["train"].to_string())
    print("  (TEST correlation, for reference only):")
    print(corr["test"].to_string())
    C = corr["train"].loc[SUBS, SUBS].to_numpy()
    off = C[~np.eye(len(SUBS), dtype=bool)]
    print(f"  TRAIN mean pairwise corr among subs: {off.mean():+.3f} (min {off.min():+.2f}, max {off.max():+.2f})")
    rstats = regime_stats(solo_daily["train"])
    rtab = regime_table(rstats)
    print("  TRAIN regime table (mean stand-alone daily net P&L, (n days), by market state at each sub's gate):")
    print(rtab.to_string())
    global _FIT_OFF
    _FIT_OFF = derive_fit_off(rstats)
    print(f"  RG_fit switch-off buckets (TRAIN mean <= 0, >= {FIT_MIN_DAYS} days): {_FIT_OFF}")

    # linear-scaling diagnostic: what EW_all would earn if costs scaled with size (they do not)
    lin = {p: pd.DataFrame(solo_daily[p]).sum(axis=1) / len(SUBS) for p in ("train", "test")}

    # 3. candidates
    tr_t = {s: (solo["train"][s]["t_stat"] or -9) for s in SUBS}
    top_t = [s for s in SUBS if tr_t[s] >= 2.0]
    print(f"== 3. Candidates ==  TOP_T (TRAIN t >= 2): {top_t}")
    gates = {
        "RG_beta": MarketGate("beta", rule_beta),
        "RG_trend": MarketGate("trend", rule_trend),
        "RG_fit": MarketGate(f"fit off={_FIT_OFF}", make_rule_fit(_FIT_OFF)),
        "PERF5": PerfGate(5),
        "PERF10": PerfGate(10),
    }
    specs = [
        ("EW_all", SUBS, None, "a) equal weight, all 7 audited strategies"),
        ("TOP_T", top_t, None, "b) equal weight, TRAIN daily t-stat >= 2"),
    ] + [(g, SUBS, gates[g], "c) regime book: " + gates[g].label) for g in gates]
    cands, dailies = [], {}
    for name, subs, gate, desc in specs:
        for mode in ("slices", "shared"):
            t = time.time()
            row = dict(name=f"{name}_{mode}", components=", ".join(subs), method=f"{desc}; {mode}")
            for p in ("train", "test"):
                if mode == "slices":
                    d, trd, per = run_slices(subs, p, gate)
                else:
                    d, trd, per = run_shared(subs, p, gate, label=name)
                row[p] = book_metrics(d, trd)
                dailies[(row["name"], p)] = d
                if gate is not None and len(trd):
                    tags = trd["tag"] if mode == "shared" else trd["book"]
                    row.setdefault("active_trades_by_sub", {})[p] = (
                        tags.astype(str).str.split("|").str[0].value_counts().to_dict())
            cands.append(row)
            print(f"  {row['name']:20s} TRAIN net {row['train']['net']:>11,.0f} t {row['train']['t_stat']}"
                  f" PF {row['train']['profit_factor']} | TEST net {row['test']['net']:>11,.0f}"
                  f" t {row['test']['t_stat']} PF {row['test']['profit_factor']}  ({time.time() - t:.0f}s)")

    # 4. selection on TRAIN only
    elig = [c for c in cands if (c["train"]["t_stat"] or -9) > 0 and (c["train"]["profit_factor"] or 0) > 1]
    pick = max(elig, key=lambda c: (c["train"]["t_stat"], -c["train"]["max_dd_pct"])) if elig else None
    pos_test = [c["name"] for c in cands if c["test"]["net"] > 0]
    print("== 4. Selection (TRAIN daily t-stat, PF > 1) ==")
    print(f"  TRAIN-selected: {pick['name'] if pick else None}")
    print(f"  candidates positive on TEST: {pos_test}")
    tbl = pd.DataFrame([{**{"name": c["name"]}, **{f"tr_{k}": c['train'][k] for k in ('net', 't_stat', 'profit_factor', 'max_dd_pct', 'costs')},
                         **{f"te_{k}": c['test'][k] for k in ('net', 't_stat', 'profit_factor', 'max_dd_pct', 'costs')}}
                        for c in cands]).set_index("name")
    print(tbl.to_string())
    print(f"  linear-scaled EW_all (cost-free scaling, NOT achievable): TRAIN {lin['train'].sum():,.0f}"
          f"  TEST {lin['test'].sum():,.0f}")

    lc = gate_leak_checks() if a.leak_check else None

    out = dict(baseline={p: short(base[p]["metrics"]) for p in ("train", "test")},
               solo=solo, corr_train=corr["train"].to_dict(), corr_test=corr["test"].to_dict(),
               regime_table_train=rtab.to_dict(orient="index"), rg_fit_off=_FIT_OFF,
               top_t=top_t, candidates=cands, selected=pick["name"] if pick else None,
               positive_on_test=pos_test, leak_check=lc,
               linear_scaled_ew={p: float(lin[p].sum()) for p in lin})
    with open(os.path.join(RESULTS, "portfolio_results.json"), "w") as f:
        json.dump(out, f, default=str, indent=1)
    dd = pd.DataFrame({f"{n}|{p}": v for (n, p), v in dailies.items()})
    dd.to_csv(os.path.join(RESULTS, "portfolio_daily.csv"))
    print(f"wrote results/portfolio_results.json and results/portfolio_daily.csv  ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()
