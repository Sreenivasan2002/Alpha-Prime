"""
Week-by-week backtest of the app's rules, with a 15:15 square-off and a 12:00 cut-off side by side.

    venv\\Scripts\\python research\\strat_lab\\weekly_report.py 2026-08-17 2026-10-16

Books (all net of Indian intraday charges and 1bp/side slippage, harness rules):
  current      the live app's rules today: baseline signal + 3 entries/day + 3 open positions
  current_12   same, but nothing new after 11:55 and everything closed at 12:00
  old          the rules before 18 Oct: up to 10 open positions, no daily entry cap
  old_12       old rules with the 12:00 cut-off
The LLM debate layer is not modelled (it was never backtested).
Writes results/weekly_<start>_<end>.csv.
"""
import os
import sys
from datetime import date

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)

import pandas as pd  # noqa: E402

import harness  # noqa: E402
from strategies import app_current  # noqa: E402


class Current(app_current.Strategy):
    name = "current"
    max_positions = 3
    max_trades_per_day = 3


class Current12(Current):
    name = "current_12"
    entry_end = "11:55"
    squareoff = "12:00"


class Old(app_current.Strategy):
    name = "old"


class Old12(Old):
    name = "old_12"
    entry_end = "11:55"
    squareoff = "12:00"


def weekly_table(res):
    t = res.trades.copy()
    d = res.daily.copy()
    d.index = pd.to_datetime(d.index)
    wk = d.groupby(d.index - pd.to_timedelta(d.index.weekday, unit="D")).sum()
    if len(t):
        t["wk"] = pd.to_datetime(t.entry_time.map(lambda x: x.date()))
        t["wk"] = t.wk - pd.to_timedelta(t.wk.dt.weekday, unit="D")
    rows = []
    for w, net in wk.items():
        g = t[t.wk == w] if len(t) else t
        n = len(g)
        rows.append(dict(week=w.date(), trades=n, wins=int((g.pnl > 0).sum()) if n else 0,
                         win_rate=round((g.pnl > 0).mean() * 100, 1) if n else None, net=round(net, 0)))
    return pd.DataFrame(rows)


def main():
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 8, 17)
    end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 10, 16)
    res = {cls.name: harness.run(cls(), (start, end)) for cls in (Current, Current12, Old, Old12)}
    tabs = {k: weekly_table(r) for k, r in res.items()}

    out = tabs["current"][["week", "trades", "wins", "win_rate", "net"]].rename(
        columns={"trades": "trades_1515", "wins": "wins_1515", "win_rate": "win%_1515", "net": "net_1515"})
    c12 = tabs["current_12"].rename(columns={"trades": "trades_12", "wins": "wins_12",
                                             "win_rate": "win%_12", "net": "net_12"})
    out = out.merge(c12, on="week", how="outer")
    for k in ("old", "old_12"):
        out = out.merge(tabs[k][["week", "net"]].rename(columns={"net": f"net_{k}"}), on="week", how="outer")
    out = out.sort_values("week")
    path = os.path.join(LAB, "results", f"weekly_{start}_{end}.csv")
    out.to_csv(path, index=False)

    print(f"\nWeek-wise backtest {start} .. {end} (net of charges + 1bp slippage)\n")
    print(out.to_string(index=False))
    print()
    for k, r in res.items():
        m = r.metrics
        print(f"  {k:<11} days {m['days']:>3}  trades {m['trades']:>4}  win {m.get('win_rate', 0):>5.1f}%  "
              f"net {m['net']:>11,.0f}  ({m['ret_pct']:+.2f}%)  PF {m.get('profit_factor')}  "
              f"charges {m.get('costs', 0):>9,.0f}  maxDD {m['max_dd_pct']:.2f}%  "
              f"up/down days {m['up_days']}/{m['down_days']}")
    print(f"\n  saved {path}")


if __name__ == "__main__":
    main()
