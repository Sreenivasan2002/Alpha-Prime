"""
Forward (out-of-sample) paper test - Tracks B and C of the 4-week validation plan in
results/critique.md. Run it each evening after 15:35 IST:

    venv\\Scripts\\python research\\strat_lab\\forward_test.py

What it does
  1. Checks that the strategy code is FROZEN: the SHA-256 of every file that decides a trade must
     match forward/FROZEN.json, written once by `--freeze` before the first forward day. If any of
     them changed, it refuses to run (a changed strategy is a new strategy and starts a new test).
  2. Downloads the latest 5-min bars (same NIFTY-100 universe as data_5m.pkl, plus NIFTY) and
     appends COMPLETED days after the study window to forward/data_forward.pkl. data_5m.pkl is
     never modified; it is only used as warm-up history. yfinance keeps ~60 days of 5-min bars,
     so run this at least once a week or days are lost for good.
  3. Replays, on forward days only (from FORWARD_START):
       B  index_trend  at 1bp and 2bp/side slippage       - the one strategy still in the running
       C  app_current  (the unchanged baseline) at 1bp    - the control
       A' app_capped   (baseline + the live caps: 3 entries/day, 3 open) - the replay the live
          paper trades are compared against (P1 replay fidelity)
  4. Prints the running scoreboard and the Track B / C pass rules, and writes
     forward/forward_daily.csv and forward/trades_<book>.csv.

Pass rules for B (keep forward-testing; this is NOT a go-live signal), at 2bp:
  net >= 0 after charges, PF >= 1.0, >= 8 signal days with both long and short days, and the stock
  basket beats trading NIFTY in the signal direction on >= 50% of signal days.
C: expected to keep losing ~0.3%/day; if it turns profitable the regime changed - re-check everything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sqlite3
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import pytz

LAB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(LAB))
sys.path.insert(0, ROOT)
sys.path.insert(0, LAB)

import harness  # noqa: E402

IST = pytz.timezone("Asia/Kolkata")
FWD = os.path.join(LAB, "forward")
FROZEN = os.path.join(FWD, "FROZEN.json")
FWD_DATA = os.path.join(FWD, "data_forward.pkl")
HIST_END = harness.ALL[1]            # last day of the study data (2026-10-16)
FORWARD_START = date(2026, 10, 19)   # first forward day (Monday after the study)

# Every file whose content can change a backtest trade
FROZEN_FILES = [
    "research/strat_lab/harness.py",
    "research/strat_lab/strategies/index_trend.py",
    "research/strat_lab/strategies/app_current.py",
    "alpha_prime/core/costs.py",
    "alpha_prime/core/risk.py",
    "alpha_prime/data/technical_analysis.py",
]


def sha(path):
    # Line endings are normalised so a git checkout that rewrites LF <-> CRLF is not a "change".
    with open(os.path.join(ROOT, path), "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def freeze():
    if os.path.exists(FROZEN):
        sys.exit(f"Already frozen ({FROZEN}). Delete it only if you are deliberately starting a NEW test.")
    os.makedirs(FWD, exist_ok=True)
    rec = dict(frozen_at=datetime.now(IST).isoformat(timespec="seconds"), forward_start=str(FORWARD_START),
               files={p: sha(p) for p in FROZEN_FILES})
    with open(FROZEN, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Frozen {len(FROZEN_FILES)} files -> {os.path.relpath(FROZEN, ROOT)}")


def verify_frozen():
    if not os.path.exists(FROZEN):
        sys.exit("Not frozen yet: run with --freeze once before the first forward day.")
    with open(FROZEN) as f:
        rec = json.load(f)
    changed = [p for p, h in rec["files"].items() if sha(p) != h]
    if changed:
        sys.exit("REFUSING: strategy code changed since it was frozen on "
                 f"{rec['frozen_at']}:\n  " + "\n  ".join(changed) +
                 "\nRevert these files, or start a new forward test (new freeze, new start date).")
    return rec


# ----------------------------------------------------------------------------- data
def _download(symbols, start):
    import yfinance as yf
    tick = {s: ("^NSEI" if s == "NIFTY" else f"{s}.NS") for s in symbols}
    raw = yf.download(list(tick.values()), start=start.strftime("%Y-%m-%d"), interval="5m",
                      progress=False, group_by="ticker", auto_adjust=False, threads=True)
    out = {}
    for s, t in tick.items():
        try:
            df = raw[t][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])
        except KeyError:
            continue
        if df.empty:
            continue
        df.index = df.index.tz_convert(IST)
        m = df.index.hour * 60 + df.index.minute
        out[s] = df[(m >= 9 * 60 + 15) & (m <= 15 * 60 + 25)]
    return out


def update_forward_data():
    hist = harness.load_data()
    symbols = sorted(hist)
    fwd = {}
    if os.path.exists(FWD_DATA):
        with open(FWD_DATA, "rb") as f:
            fwd = pickle.load(f)
    now = datetime.now(IST)
    # a day is complete once the market has closed
    last_complete = now.date() if (now.hour * 60 + now.minute) >= 15 * 60 + 35 else now.date() - timedelta(days=1)
    have = max((df.index[-1].date() for df in fwd.values() if len(df)), default=HIST_END)
    start = max(have + timedelta(days=1), FORWARD_START, now.date() - timedelta(days=55))
    if start > last_complete:
        return fwd, last_complete, []
    if start > max(have + timedelta(days=1), FORWARD_START):
        print(f"  WARNING: days from {max(have + timedelta(days=1), FORWARD_START)} to {start} are older than "
              "yfinance's ~60-day 5-min window and are lost.")
    new = _download(symbols, start)
    added = set()
    for s, df in new.items():
        df = df[(df.index.date > HIST_END) & (df.index.date <= last_complete)]
        if df.empty:
            continue
        old = fwd.get(s)
        comb = df if old is None else pd.concat([old, df])
        fwd[s] = comb[~comb.index.duplicated(keep="last")].sort_index()
        added |= set(df.index.date)
    with open(FWD_DATA, "wb") as f:
        pickle.dump(fwd, f)
    return fwd, last_complete, sorted(added)


def combined_market(fwd):
    hist = harness.load_data()
    data = {}
    for s, h in hist.items():
        f = fwd.get(s)
        h = h[h.index.date <= HIST_END]
        data[s] = h if f is None or f.empty else pd.concat([h, f[f.index.date > HIST_END]]).sort_index()
    return harness.Market(data)


# ----------------------------------------------------------------------------- books
def make_books():
    from strategies import app_current, index_trend

    class AppCapped(app_current.Strategy):
        """Baseline + the caps now enforced in the live app (3 entries/day, 3 open; with
        Rs 2L per name the 1x gross cap cannot bind). One-entry-per-15-min-cycle and the LLM
        layer are NOT modelled."""
        name = "app_capped"
        max_positions = 3
        max_trades_per_day = 3

    return [("B_index_trend_2bp", index_trend.Strategy, 0.0002),
            ("B_index_trend_1bp", index_trend.Strategy, 0.0001),
            ("C_app_current_1bp", app_current.Strategy, 0.0001),
            ("A_app_capped_1bp", AppCapped, 0.0001)]


def run_book(mkt, cls, slip, end):
    st = cls()
    if hasattr(st, "prepare"):
        st.prepare(mkt)
    return harness.Sim(mkt, st, FORWARD_START, end, slip).run()


def nifty_vs_basket(mkt, trades):
    """Per signal day: basket return (bp, after slippage, before charges) vs trading NIFTY in the
    signal direction over the same window (first entry bar open -> 15:15 bar open)."""
    if trades.empty:
        return pd.DataFrame()
    N = mkt.df("NIFTY")
    rows = []
    for d, g in trades.groupby(trades.entry_time.map(lambda t: t.date())):
        side = 1 if g.side.iloc[0] == "long" else -1
        t0 = g.entry_time.min()
        day = N[N.index.date == d]
        n0 = day.loc[day.index >= t0, "Open"]
        n1 = day.loc[day.index.strftime("%H:%M") >= "15:15", "Open"]
        if n0.empty:
            continue
        exit_px = n1.iloc[0] if len(n1) else day["Close"].iloc[-1]
        rows.append(dict(date=d, side=g.side.iloc[0], names=len(g),
                         basket_bp=round(float(g.pnl_pct.mean()) * 100, 1),
                         nifty_bp=round((exit_px / n0.iloc[0] - 1) * 1e4 * side, 1),
                         net=round(float(g.pnl.sum()), 0)))
    df = pd.DataFrame(rows)
    if len(df):
        df["basket_beats_nifty"] = df.basket_bp > df.nifty_bp
    return df


def replay_fidelity(capped_trades, start, end):
    """P1: share of live paper BUYs matched by the capped harness replay (same symbol and side,
    entry within +-1 bar = 5 min)."""
    sys.path.insert(0, ROOT)
    from alpha_prime.core.config import settings
    c = sqlite3.connect(str(settings.db_path))
    rows = c.execute("SELECT timestamp, symbol FROM trades WHERE trading_mode='paper' AND action='BUY' "
                     "AND UPPER(COALESCE(status,'')) NOT IN ('REJECTED','CANCELLED','FAILED') "
                     "AND date(timestamp) BETWEEN ? AND ?", (str(start), str(end))).fetchall()
    if not rows:
        return None, 0
    utc = pytz.utc
    matched = 0
    for ts, sym in rows:
        t = utc.localize(datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")).astimezone(IST)
        if not capped_trades.empty:
            cand = capped_trades[(capped_trades.sym == sym) & (capped_trades.side == "long")]
            if any(abs((et - t).total_seconds()) <= 5 * 60 + 59 for et in cand.entry_time):
                matched += 1
    return matched / len(rows) * 100, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="record the strategy code hashes (once)")
    ap.add_argument("--no-download", action="store_true", help="use only already-saved forward data")
    a = ap.parse_args()
    if a.freeze:
        freeze()
        return
    rec = verify_frozen()
    print(f"\nForward test - code frozen {rec['frozen_at']} (hashes verified), forward start {FORWARD_START}\n")

    if a.no_download:
        fwd = pickle.load(open(FWD_DATA, "rb")) if os.path.exists(FWD_DATA) else {}
        added = []
    else:
        fwd, _, added = update_forward_data()
    days = sorted({d for df in fwd.values() for d in df.index.date if d >= FORWARD_START})
    if added:
        print(f"  Added {len(added)} new day(s): {', '.join(map(str, added))}")
    if not days:
        print("  No completed forward trading days yet. Run again after 15:35 IST on a trading day.\n")
        return
    end = days[-1]
    print(f"  Forward days available: {len(days)} ({days[0]} .. {end})\n")
    mkt = combined_market(fwd)

    daily_rows, results = [], {}
    print(f"  {'book':<20}{'days':>5}{'trades':>8}{'net Rs':>11}{'PF':>6}{'win%':>7}{'t':>7}{'maxDD%':>8}")
    for name, cls, slip in make_books():
        res = run_book(mkt, cls, slip, end)
        results[name] = res
        m = res.metrics
        print(f"  {name:<20}{m['days']:>5}{m['trades']:>8}{m['net']:>11,.0f}{(m.get('profit_factor') or 0):>6.2f}"
              f"{(m.get('win_rate') or 0):>7.1f}{(m.get('t_stat') or 0):>7.2f}{m['max_dd_pct']:>8.2f}")
        res.trades.to_csv(os.path.join(FWD, f"trades_{name}.csv"), index=False)
        for d, v in res.daily.items():
            n_day = res.trades[res.trades.entry_time.map(lambda t: t.date()) == d] if len(res.trades) else res.trades
            daily_rows.append(dict(date=d, book=name, net=round(v, 2), trades=len(n_day)))
    pd.DataFrame(daily_rows).to_csv(os.path.join(FWD, "forward_daily.csv"), index=False)

    # ---- Track B rules at 2bp
    b = results["B_index_trend_2bp"]
    bm = b.metrics
    cmp = nifty_vs_basket(mkt, b.trades)
    sig_days = len(cmp)
    sides = set(cmp.side) if sig_days else set()
    beats = float(cmp.basket_beats_nifty.mean() * 100) if sig_days else 0.0
    print("\n  Track B - index_trend (2bp), signal days:")
    if sig_days:
        print(cmp.to_string(index=False, columns=["date", "side", "names", "basket_bp", "nifty_bp", "net"]))
    checks = [
        ("net >= 0 after charges", bm["net"] >= 0, f"Rs {bm['net']:,.0f}"),
        ("PF >= 1.0", (bm.get("profit_factor") or 0) >= 1.0, f"{bm.get('profit_factor')}"),
        (">= 8 signal days", sig_days >= 8, f"{sig_days}"),
        ("both long and short days", sides == {"long", "short"}, ", ".join(sorted(sides)) or "none"),
        ("basket beats NIFTY-in-signal-direction on >= 50% of days", beats >= 50, f"{beats:.0f}%"),
    ]
    for n, ok, d in checks:
        print(f"  [{'PASS' if ok else 'not yet' if n.startswith('>=') or 'both' in n else 'FAIL'}] {n}: {d}")

    # ---- Track C control
    cm = results["C_app_current_1bp"].metrics
    per_day = cm["net"] / max(cm["days"], 1) / harness.CAPITAL * 100
    print(f"\n  Track C - unchanged baseline: Rs {cm['net']:,.0f} ({per_day:+.2f}%/day; study average was -0.31%/day)")
    if cm["net"] > 0:
        print("  ! The baseline is PROFITABLE forward - the market regime may have changed. Re-check conclusions.")

    # ---- P1 replay fidelity (live paper vs capped replay)
    pct, n = replay_fidelity(results["A_app_capped_1bp"].trades, FORWARD_START, end)
    if pct is None:
        print("\n  P1 replay fidelity: no live paper BUYs in the forward window yet.")
    else:
        print(f"\n  P1 replay fidelity: {pct:.0f}% of {n} paper BUYs matched by the capped replay "
              f"(pass >= 90%; the LLM layer and 15-min cycle are not modelled, so a gap is expected)")
    print(f"\n  Files: {os.path.join(FWD, 'forward_daily.csv')} and trades_<book>.csv\n")


if __name__ == "__main__":
    main()
