r"""
Backtest the app's current rules over a date range, with day-wise P&L snapshots.

    venv\Scripts\python scripts\backtest_week.py 2026-10-05 2026-10-09

Mirrors the live app as closely as the mechanical parts allow: NIFTY 100 universe, only
stocks currently up 1-5% vs yesterday's close (the scanner's top-gainer rule), 5-min
technical BUY with strength >= MIN_SIGNAL_STRENGTH, trend not bearish, volume confirmed,
risk-based sizing, stop / target / breakeven / trailing, 15:15 square-off, daily limits,
Indian intraday charges. It does NOT replay the LLM agents (scanner reasoning, Bull/Bear
debate, risk manager), which filter trades further in live use.
"""
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

import pandas as pd  # noqa: E402
from alpha_prime.backtest import Backtester, BacktestConfig  # noqa: E402
from alpha_prime.data.market_data import market_data  # noqa: E402

start = date.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2026-10-05")
end = date.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else "2026-10-09")
SNAPS = ["11:00", "12:00", "13:00", "14:00", "15:00"]

cfg = BacktestConfig()
cfg.gain_filter = (1.0, 5.0)
cfg.require_volume_confirmed = True
symbols = market_data.get_nifty100_stocks()
print(f"Rules: strength >= {cfg.min_signal_strength}, stop {cfg.default_stop_loss_pct}%, target {cfg.default_target_pct}%, "
      f"risk/trade {cfg.risk_per_trade_pct}%, max {cfg.max_position_size_pct}% per stock, max {cfg.max_open_positions} positions")
print(f"Universe: NIFTY 100 ({len(symbols)}), gainers +1..5%, volume confirmed | {start} to {end}\n")

bt = Backtester(cfg)
data = bt.load_data(symbols, period="30d", interval="5m")
print(f"Loaded {len(data)} symbols\n")
res = bt.simulate(data, start_date=start, end_date=end, snapshot_times=SNAPS)

trades = pd.DataFrame([t.__dict__ for t in res.trades])
if not trades.empty:
    trades["day"] = pd.to_datetime(trades["exit_time"]).dt.date.astype(str)

print("DAY-WISE P&L (cumulative for that day, net of charges on closed trades; open trades marked to market)")
print(f"{'Day':<16}" + "".join(f"{s:>11}" for s in SNAPS) + f"{'Day final':>12}{'Trades':>8}")
week = 0.0
for d in pd.bdate_range(start, end).date:
    ds = str(d)
    snap = res.snapshots.get(ds)
    if snap is None:
        print(f"{d.strftime('%a %d %b'):<16}  no market data (holiday?)")
        continue
    day_tr = trades[trades["day"] == ds] if not trades.empty else trades
    final = float(day_tr["pnl"].sum()) if not day_tr.empty else 0.0
    week += final
    print(f"{d.strftime('%a %d %b'):<16}" + "".join(f"{snap.get(s, 0):>+11,.0f}" for s in SNAPS)
          + f"{final:>+12,.0f}{len(day_tr):>8}")
print(f"\nWEEK TOTAL (net of charges): Rs {week:+,.2f}   ({week / cfg.capital * 100:+.2f}% of Rs {cfg.capital:,.0f})")
m = res.metrics
if m.get("num_trades"):
    print(f"Trades {m['num_trades']} | win rate {m['win_rate_pct']}% | gross Rs {m['total_gross_pnl']:+,.0f} | "
          f"charges Rs {m['total_cost']:,.0f} | profit factor {m['profit_factor']} | exits {m['exit_reasons']}")
    os.makedirs("backtest_results", exist_ok=True)
    out = f"backtest_results/week_{start}_{end}_trades.csv"
    trades.to_csv(out, index=False)
    print(f"Trade list saved: {out}")
