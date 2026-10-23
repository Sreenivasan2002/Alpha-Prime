r"""
Backtest the app's current rules over a date range and report results WEEK BY WEEK.

    venv\Scripts\python scripts\backtest_range.py 2026-09-01 2026-09-30

Same rules as scripts/backtest_week.py (NIFTY 100, top gainers +1..5%, volume confirmed,
5-min technical BUY >= MIN_SIGNAL_STRENGTH, risk-based sizing, stop/target/trailing,
15:15 square-off, daily limits, Indian intraday charges). LLM agents are not replayed.
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

start = date.fromisoformat(sys.argv[1])
end = date.fromisoformat(sys.argv[2])
period = "60d"
# optional: square-off time, e.g. 12:00 (default = live setting 15:15)
sq = sys.argv[3] if len(sys.argv) > 3 else None

cfg = BacktestConfig()
cfg.gain_filter = (1.0, 5.0)
cfg.require_volume_confirmed = True
if sq:
    cfg.squareoff = (int(sq[:2]), int(sq[3:5]))
symbols = market_data.get_nifty100_stocks()
print(f"Rules: strength >= {cfg.min_signal_strength}, stop {cfg.default_stop_loss_pct}%, target {cfg.default_target_pct}%, "
      f"risk/trade {cfg.risk_per_trade_pct}%, max {cfg.max_position_size_pct}%/stock, max {cfg.max_open_positions} positions")
print(f"Universe: NIFTY 100, gainers +1..5%, volume confirmed | {start} to {end}\n")

bt = Backtester(cfg)
data = bt.load_data(symbols, period=period, interval="5m")
res = bt.simulate(data, start_date=start, end_date=end)
t = pd.DataFrame([x.__dict__ for x in res.trades])
if t.empty:
    sys.exit("No trades in this range.")
t["day"] = pd.to_datetime(t["exit_time"]).dt.tz_localize(None).dt.normalize()
t["week"] = t["day"] - pd.to_timedelta(t["day"].dt.weekday, unit="D")

trading_days = sorted({d for df in data.values() for d in df["_date"].unique() if start <= d <= end})
daily = t.groupby("day")["pnl"].sum()

print(f"{'Week (Mon)':<12}{'Days':>5}{'Trades':>8}{'Win %':>8}{'Gross':>11}{'Charges':>10}{'NET':>11}"
      f"{'Best day':>11}{'Worst day':>11}{'Up/Down days':>14}")
total = 0.0
for wk, g in t.groupby("week"):
    wk_days = [d for d in trading_days if pd.Timestamp(d) - pd.Timedelta(days=pd.Timestamp(d).weekday()) == wk]
    dd = daily[(daily.index >= wk) & (daily.index < wk + pd.Timedelta(days=7))]
    net = g["pnl"].sum()
    total += net
    print(f"{wk.strftime('%d %b'):<12}{len(wk_days):>5}{len(g):>8}{(g.pnl > 0).mean() * 100:>7.0f}%"
          f"{g.gross_pnl.sum():>+11,.0f}{g.cost.sum():>10,.0f}{net:>+11,.0f}{dd.max():>+11,.0f}{dd.min():>+11,.0f}"
          f"{f'{(dd > 0).sum()}/{(dd < 0).sum()}':>14}")
m = res.metrics
print(f"\nTOTAL {start} to {end}: NET Rs {total:+,.2f} ({total / cfg.capital * 100:+.2f}% of Rs {cfg.capital:,.0f})")
print(f"{len(trading_days)} trading days | {m['num_trades']} trades | win rate {m['win_rate_pct']}% | gross Rs {m['total_gross_pnl']:+,.0f} | "
      f"charges Rs {m['total_cost']:,.0f} | profit factor {m['profit_factor']} | max drawdown {m['max_drawdown_pct']}%")
print(f"Profitable days: {(daily > 0).sum()} of {len(trading_days)} | exits {m['exit_reasons']}")
print("\nDAILY:")
for d, v in daily.items():
    print(f"  {d.strftime('%a %d %b')}: {v:+,.0f}")
os.makedirs("backtest_results", exist_ok=True)
out = f"backtest_results/range_{start}_{end}_sq{cfg.squareoff[0]:02d}{cfg.squareoff[1]:02d}_trades.csv"
t.to_csv(out, index=False)
print(f"\nTrade list saved: {out}")
