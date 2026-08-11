"""
Backtest runner for Alpha-Prime.

Replays historical intraday data through the live signal engine + risk/exit rules
to measure the strategy on history. NEVER touches the live trading DB or broker.

Examples:
  # Default: NIFTY 50 (first 15 symbols), last 60 days of 5m bars
  python run_backtest.py

  # Specific symbols
  python run_backtest.py --symbols RELIANCE,TCS,INFY,SBIN --period 60d

  # Whole NIFTY 100, save trade log + equity to CSV
  python run_backtest.py --universe "NIFTY 100" --max-symbols 40 --out backtest_results

  # A/B the trailing-stop fix (new breakeven+wide-trail vs old tight 0.5%/0.7% trail)
  python run_backtest.py --symbols RELIANCE,TCS,INFY,SBIN,ICICIBANK --compare

Note: yfinance caps 5m history at ~60 days. Wider universes take longer (the TA
engine recomputes per in-window bar while flat).
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from alpha_prime.backtest import Backtester, BacktestConfig
from alpha_prime.data.market_data import market_data


def _fmt_metrics(title: str, m: dict) -> str:
    if m.get("num_trades", 0) == 0:
        return f"\n=== {title} ===\n  {m.get('note', 'No trades.')}"
    lines = [f"\n=== {title} ==="]
    # (key, label, unit, signed?)
    order = [
        ("num_trades", "Trades", "", False),
        ("total_gross_pnl", "Gross P&L", "Rs", True),
        ("total_cost", "Transaction costs", "Rs", False),
        ("total_pnl", "Net P&L", "Rs", True),
        ("return_on_capital_pct", "Net return on capital", "%", True),
        ("win_rate_pct", "Win rate", "%", False),
        ("avg_win", "Avg win", "Rs", False),
        ("avg_loss", "Avg loss", "Rs", False),
        ("win_loss_ratio", "Win/Loss size ratio", "x", False),
        ("avg_win_pct", "Avg win", "%", True),
        ("avg_loss_pct", "Avg loss", "%", True),
        ("profit_factor", "Profit factor", "", False),
        ("expectancy_per_trade", "Expectancy/trade", "Rs", True),
        ("max_drawdown_pct", "Max drawdown", "%", False),
        ("avg_bars_held", "Avg bars held", "", False),
    ]
    for key, label, unit, signed in order:
        if key in m and m[key] is not None:
            val = m[key]
            if unit == "Rs":
                lines.append(f"  {label:22s}: Rs {val:+,.2f}" if signed else f"  {label:22s}: Rs {val:,.2f}")
            elif unit == "%":
                lines.append(f"  {label:22s}: {val:+.2f}%" if signed else f"  {label:22s}: {val:.2f}%")
            elif unit == "x":
                lines.append(f"  {label:22s}: {val:.2f}x")
            else:
                lines.append(f"  {label:22s}: {val}")
    if "exit_reasons" in m:
        lines.append(f"  {'Exit reasons':22s}: {m['exit_reasons']}")
    return "\n".join(lines)


def _save(result, outdir: str, tag: str):
    import pandas as pd
    Path(outdir).mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
    eq_df = pd.DataFrame(result.equity_curve)
    tp = Path(outdir) / f"trades_{tag}.csv"
    ep = Path(outdir) / f"equity_{tag}.csv"
    trades_df.to_csv(tp, index=False)
    eq_df.to_csv(ep, index=False)
    print(f"  saved: {tp}  ({len(trades_df)} trades)")
    print(f"  saved: {ep}")


def main():
    ap = argparse.ArgumentParser(description="Alpha-Prime backtester")
    ap.add_argument("--symbols", help="Comma-separated symbols (overrides --universe)")
    ap.add_argument("--universe", default="NIFTY 50",
                    help='Index name, e.g. "NIFTY 50", "NIFTY 100" (default: NIFTY 50)')
    ap.add_argument("--max-symbols", type=int, default=15, help="Cap symbols for runtime (default 15)")
    ap.add_argument("--period", default="60d", help="History window (yfinance: 5m maxes ~60d)")
    ap.add_argument("--interval", default="5m", help="Bar interval (default 5m)")
    ap.add_argument("--capital", type=float, default=None, help="Override starting capital")
    ap.add_argument("--out", default=None, help="Directory to save trades/equity CSVs")
    ap.add_argument("--compare", action="store_true",
                    help="Also run the OLD trailing-stop (arm 0.5%%, trail 0.7%%, no breakeven) to A/B the exit fix")
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = market_data.get_index_stocks(args.universe)[: args.max_symbols]

    print(f"Backtest universe: {len(symbols)} symbols | period={args.period} interval={args.interval}")
    print(f"Symbols: {', '.join(symbols)}")

    cfg = BacktestConfig()
    if args.capital:
        cfg.capital = args.capital
    print(f"Capital Rs {cfg.capital:,.0f} | risk/trade {cfg.risk_per_trade_pct}% | "
          f"max pos {cfg.max_position_size_pct}% of capital | SL {cfg.default_stop_loss_pct}% "
          f"target {cfg.default_target_pct}% | trail arm {cfg.trail_arm_pct}%/{cfg.trailing_stop_pct}%")

    bt = Backtester(cfg)
    try:
        result = bt.run(symbols, period=args.period, interval=args.interval)
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(_fmt_metrics(f"CURRENT settings ({result.bars_scanned:,} bars, {result.duration_sec}s)", result.metrics))
    if args.out:
        _save(result, args.out, "current")

    if args.compare:
        legacy = BacktestConfig()
        if args.capital:
            legacy.capital = args.capital
        # Reproduce the OLD exit behaviour: trail tightly from +0.5%, 0.7% below peak,
        # and effectively no breakeven lock (arm above any realistic gain).
        legacy.trail_arm_pct = 0.5
        legacy.trailing_stop_pct = 0.7
        legacy.breakeven_arm_pct = 100.0
        res2 = Backtester(legacy).run(symbols, period=args.period, interval=args.interval)
        print(_fmt_metrics("OLD trailing stop (arm 0.5%, trail 0.7%, no breakeven)", res2.metrics))
        if args.out:
            _save(res2, args.out, "old_trailing")
        # Headline delta
        a, b = result.metrics, res2.metrics
        if a.get("num_trades") and b.get("num_trades"):
            print("\n=== DELTA (current - old trailing) ===")
            print(f"  Total P&L:     Rs {a['total_pnl'] - b['total_pnl']:+,.2f}")
            print(f"  Win rate:      {a['win_rate_pct'] - b['win_rate_pct']:+.1f} pts")
            print(f"  Expectancy:    Rs {a['expectancy_per_trade'] - b['expectancy_per_trade']:+,.2f}/trade")

    print("\n(Backtest only — no live DB or broker was touched.)")


if __name__ == "__main__":
    main()
