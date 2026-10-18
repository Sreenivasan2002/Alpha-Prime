"""
Track A daily validation report for the paper experiment (see research/strat_lab/results/critique.md).

Run after the market closes (or any later day, for any date in the last ~60 days):

    venv\\Scripts\\python scripts\\daily_report.py              # today
    venv\\Scripts\\python scripts\\daily_report.py 2026-10-19   # a given day
    venv\\Scripts\\python scripts\\daily_report.py 2026-10-07 --no-save   # look only, don't log

For every round trip (flat -> flat on one symbol) it reports:
  * net P&L, gross P&L and charges (recorded vs the cost model in core/costs.py)
  * NIFTY's move over the holding window and the P&L left after removing it
    (alpha = net - notional x NIFTY return, i.e. beta 1)
  * slippage per fill vs the decision price (ref_price) and vs the open of the 5-min bar it
    filled in (what the backtest harness assumes)
and checks the pass/fail rules P2-P5 for the day:
  P2 slippage (median <= 2bp, worst <= 5bp vs ref), P3 charges within 10% of the model,
  P4 risk limits (entries/day, Rs per name, gross exposure, flat by 15:20, nothing overnight),
  P5 turnover (entries/day, charges <= Rs 500/day).
P1 (replay fidelity) needs the harness replay - see research/strat_lab/forward_test.py.

One row per day is written to data/validation/track_a_daily.csv (re-running a day replaces it).
"""
import csv
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd  # noqa: E402
import pytz  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()

from alpha_prime.core.config import settings  # noqa: E402
from alpha_prime.core.costs import intraday_round_trip_cost  # noqa: E402

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.utc
OUT = os.path.join(ROOT, "data", "validation", "track_a_daily.csv")
FLAT_BY = "15:20"
MAX_CHARGES_PER_DAY = 500.0


def load_trades(day: str):
    c = sqlite3.connect(str(settings.db_path))
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT * FROM trades WHERE trading_mode = 'paper' AND date(timestamp) <= ? ORDER BY timestamp, id",
        (day,))]
    for r in rows:
        r["ts"] = UTC.localize(datetime.strptime(r["timestamp"][:19], "%Y-%m-%d %H:%M:%S")).astimezone(IST)
        for k in ("ref_price", "nifty_price", "charges", "gross_pnl"):
            r.setdefault(k, None)  # older DBs have no instrumentation columns
    ok = [r for r in rows if str(r.get("status") or "").upper() not in ("REJECTED", "CANCELLED", "FAILED")]
    return [r for r in ok if r["ts"].strftime("%Y-%m-%d") == day], ok


def bars_5m(symbols, day):
    """5-min bars for `day` per symbol (yfinance keeps ~60 days). Missing -> empty frame."""
    import yfinance as yf
    out = {}
    d0 = datetime.strptime(day, "%Y-%m-%d")
    tick = {s: ("^NSEI" if s == "NIFTY" else f"{s}.NS") for s in symbols}
    try:
        df = yf.download(list(tick.values()), start=d0.strftime("%Y-%m-%d"),
                         end=(d0 + timedelta(days=1)).strftime("%Y-%m-%d"),
                         interval="5m", progress=False, group_by="ticker", auto_adjust=False)
    except Exception:
        df = pd.DataFrame()
    for s, t in tick.items():
        try:
            sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            sub = sub.dropna(subset=["Open"])
            sub.index = sub.index.tz_convert(IST)
            out[s] = sub
        except Exception:
            out[s] = pd.DataFrame()
    return out


def bar_open_at(bars, ts):
    """Open of the 5-min bar that contains ts."""
    if bars is None or bars.empty:
        return None
    start = ts.replace(second=0, microsecond=0) - timedelta(minutes=ts.minute % 5)
    if start in bars.index:
        return float(bars.loc[start, "Open"])
    return None


def nifty_at(nifty_bars, ts, recorded):
    if recorded:
        return float(recorded)
    if nifty_bars is None or nifty_bars.empty:
        return None
    upto = nifty_bars[nifty_bars.index <= ts]
    return float(upto["Close"].iloc[-1]) if len(upto) else None


def slip_bp(fill, ref, action):
    if not fill or not ref:
        return None
    sign = 1 if action == "BUY" else -1  # positive = adverse
    return (fill - ref) / ref * 1e4 * sign


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = "--no-save" not in sys.argv
    day = args[0] if args else datetime.now(IST).strftime("%Y-%m-%d")
    t = settings.trading
    cap = t.capital
    today, history = load_trades(day)
    print(f"\nTrack A daily report - {day} (paper)\n")

    # anything still open at the end of the day (from this day or carried in)?
    net_qty = defaultdict(int)
    for r in history:
        net_qty[r["symbol"]] += r["quantity"] if r["action"] == "BUY" else -r["quantity"]
    overnight = {s: q for s, q in net_qty.items() if q > 0}

    if not today:
        print("  No paper trades on this day.")
    syms = sorted({r["symbol"] for r in today})
    bars = bars_5m(syms + ["NIFTY"], day) if today else {}
    nb = bars.get("NIFTY")

    # ---- round trips (flat -> flat per symbol), running gross exposure, per-fill slippage
    trips, open_trip, pos = [], {}, defaultdict(lambda: [0, 0.0])  # qty, cost
    gross, max_gross, max_name = 0.0, 0.0, 0.0
    slips_ref, slips_bar = [], []
    entries = sum(1 for r in today if r["action"] == "BUY")
    # positions carried in from earlier days count toward exposure
    carried = defaultdict(lambda: [0, 0.0])
    for r in history:
        if r["ts"].strftime("%Y-%m-%d") < day:
            q, c = carried[r["symbol"]]
            if r["action"] == "BUY":
                carried[r["symbol"]] = [q + r["quantity"], c + r["quantity"] * r["price"]]
            elif q:
                carried[r["symbol"]] = [q - r["quantity"], c * (q - r["quantity"]) / q]
    for s, (q, c) in carried.items():
        if q > 0:
            pos[s] = [q, c]
            gross += c
    max_gross = gross

    for r in today:
        s, q, px, act = r["symbol"], r["quantity"], r["price"], r["action"]
        sb = slip_bp(px, r["ref_price"], act)
        if sb is not None:
            slips_ref.append(sb)
        bo = bar_open_at(bars.get(s), r["ts"])
        bb = slip_bp(px, bo, act)
        if bb is not None:
            slips_bar.append(bb)
        held_q, held_c = pos[s]
        if act == "BUY":
            if held_q == 0:
                open_trip[s] = dict(sym=s, entry_ts=r["ts"], buy_value=0.0, sell_value=0.0, net=0.0, gross=0.0,
                                    charges=0.0, n_fills=0,
                                    nifty_in=nifty_at(nb, r["ts"], r["nifty_price"]))
            pos[s] = [held_q + q, held_c + q * px]
            gross += q * px
            open_trip[s]["buy_value"] += q * px
            max_name = max(max_name, pos[s][1])
        else:
            avg = held_c / held_q if held_q else px
            pos[s] = [held_q - q, held_c - avg * q]
            gross -= avg * q
            tr = open_trip.get(s)
            if tr is None:  # closing a position carried in from an earlier day
                tr = open_trip[s] = dict(sym=s, entry_ts=None, buy_value=avg * q, sell_value=0.0, net=0.0,
                                         gross=0.0, charges=0.0, n_fills=0, nifty_in=None, carried=True)
            tr["sell_value"] += q * px
            fill_gross = r["gross_pnl"] if r["gross_pnl"] is not None else (px - avg) * q
            net = r["pnl"] or 0.0
            tr["net"] += net
            tr["gross"] += fill_gross
            # rows from before the instrumentation columns: charges = gross - net
            tr["charges"] += r["charges"] if r["charges"] is not None else max(fill_gross - net, 0.0)
            if pos[s][0] == 0:
                tr["exit_ts"] = r["ts"]
                tr["nifty_out"] = nifty_at(nb, r["ts"], r["nifty_price"])
                trips.append(open_trip.pop(s))
        max_gross = max(max_gross, gross)

    # ---- per-trip table
    tot = defaultdict(float)
    charge_gaps = []
    if trips:
        print(f"  {'symbol':<11}{'in':>6}{'out':>7}{'notional':>11}{'gross':>9}{'charges':>9}{'net':>9}"
              f"{'NIFTY%':>8}{'alpha':>9}")
    for tr in trips:
        model = intraday_round_trip_cost(tr["buy_value"], tr["sell_value"])
        if tr["charges"] and model:
            charge_gaps.append(abs(tr["charges"] - model) / model * 100)
        n_ret = ((tr["nifty_out"] / tr["nifty_in"] - 1) if tr.get("nifty_in") and tr.get("nifty_out") else None)
        market = tr["buy_value"] * n_ret if n_ret is not None else 0.0
        alpha = tr["net"] - market
        for k, v in (("net", tr["net"]), ("gross", tr["gross"]), ("charges", tr["charges"]),
                     ("market", market), ("alpha", alpha), ("model", model)):
            tot[k] += v
        print(f"  {tr['sym']:<11}{tr['entry_ts'].strftime('%H:%M') if tr['entry_ts'] else 'prev':>6}"
              f"{tr['exit_ts'].strftime('%H:%M'):>7}{tr['buy_value']:>11,.0f}{tr['gross']:>9,.0f}"
              f"{tr['charges']:>9,.0f}{tr['net']:>9,.0f}"
              f"{(n_ret * 100 if n_ret is not None else float('nan')):>8.2f}{alpha:>9,.0f}")
    still_open = {s: v for s, v in open_trip.items()}

    print()
    print(f"  Net P&L        Rs {tot['net']:>10,.0f}   ({tot['net'] / cap * 100:+.2f}% of capital)")
    print(f"  Gross P&L      Rs {tot['gross']:>10,.0f}")
    print(f"  Charges        Rs {tot['charges']:>10,.0f}   (cost model: Rs {tot['model']:,.0f})")
    print(f"  Market part    Rs {tot['market']:>10,.0f}   (notional x NIFTY move over each hold)")
    print(f"  Alpha          Rs {tot['alpha']:>10,.0f}   (net minus market part)")
    med_ref = statistics.median(slips_ref) if slips_ref else None
    worst_ref = max(slips_ref) if slips_ref else None
    med_bar = statistics.median(slips_bar) if slips_bar else None
    print(f"  Slippage vs decision price : "
          + (f"median {med_ref:+.1f}bp, worst {worst_ref:+.1f}bp over {len(slips_ref)} fills"
             if slips_ref else "n/a (no ref_price recorded)"))
    print(f"  Fill vs 5-min bar open     : "
          + (f"median {med_bar:+.1f}bp over {len(slips_bar)} fills" if slips_bar else "n/a"))
    if settings.trading.mode == "paper":
        print("    (paper fills are yfinance quotes, so these measure quote drift, not real slippage)")

    # ---- checks
    last_exit = max((tr["exit_ts"] for tr in trips), default=None)
    gross_cap = cap * t.max_gross_exposure_pct / 100
    name_cap = cap * t.max_position_size_pct / 100
    checks = [
        ("P2 slippage", (med_ref is None) or (med_ref <= 2 and worst_ref <= 5),
         "no data" if med_ref is None else f"median {med_ref:+.1f}bp / worst {worst_ref:+.1f}bp (<= 2 / 5)"),
        ("P3 charges vs model", all(g <= 10 for g in charge_gaps),
         f"max gap {max(charge_gaps):.1f}%" if charge_gaps else "no closed trips"),
        ("P4 entries/day", entries <= t.max_trades_per_day, f"{entries} (max {t.max_trades_per_day})"),
        ("P4 Rs per name", max_name <= name_cap * 1.001, f"max Rs {max_name:,.0f} (cap {name_cap:,.0f})"),
        ("P4 gross exposure", max_gross <= gross_cap * 1.001, f"max Rs {max_gross:,.0f} (cap {gross_cap:,.0f})"),
        ("P4 flat by " + FLAT_BY, last_exit is None or last_exit.strftime("%H:%M") <= FLAT_BY,
         last_exit.strftime("%H:%M") if last_exit else "no exits"),
        ("P4 nothing overnight", not overnight and not still_open,
         "flat" if not overnight else f"OPEN: {overnight}"),
        ("P5 charges/day", tot["charges"] <= MAX_CHARGES_PER_DAY,
         f"Rs {tot['charges']:,.0f} (<= {MAX_CHARGES_PER_DAY:,.0f})"),
        ("Kill switch: day > -2.2%", tot["net"] / cap * 100 > -2.2, f"{tot['net'] / cap * 100:+.2f}%"),
    ]
    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    if not save:
        print("\n  (--no-save: not written to the experiment log)\n")
        return

    # ---- persist one row per day
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    row = dict(date=day, trips=len(trips), entries=entries, net=round(tot["net"], 2), gross=round(tot["gross"], 2),
               charges=round(tot["charges"], 2), charges_model=round(tot["model"], 2),
               market=round(tot["market"], 2), alpha=round(tot["alpha"], 2),
               slip_ref_median_bp=None if med_ref is None else round(med_ref, 2),
               slip_ref_worst_bp=None if worst_ref is None else round(worst_ref, 2),
               slip_bar_median_bp=None if med_bar is None else round(med_bar, 2),
               max_gross=round(max_gross, 0), max_name=round(max_name, 0),
               last_exit=last_exit.strftime("%H:%M") if last_exit else "",
               overnight=";".join(f"{s}:{q}" for s, q in overnight.items()),
               all_pass=all(ok for _, ok, _ in checks))
    rows = []
    if os.path.exists(OUT):
        with open(OUT, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["date"] != day]
    rows.append(row)
    rows.sort(key=lambda r: r["date"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerows(rows)

    cum = sum(float(r["net"]) for r in rows)
    print(f"\n  Experiment so far: {len(rows)} day(s), cumulative net Rs {cum:,.0f} "
          f"({cum / cap * 100:+.2f}%). Kill switch at -4% (Rs {-0.04 * cap:,.0f}).")
    print(f"  Saved to {os.path.relpath(OUT, ROOT)}\n")


if __name__ == "__main__":
    main()
