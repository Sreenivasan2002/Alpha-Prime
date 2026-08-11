"""
Parameter sweep for Alpha-Prime — grid-searches exit/stop settings to find what
maximises NET (after-cost) expectancy, instead of assuming a config is best.

Signals are computed ONCE (they don't depend on stop/target/trail), then every
parameter combo is simulated against the same cached signals — so a 36-combo sweep
costs roughly one backtest's worth of TA, not 36.

Examples:
  python run_sweep.py --symbols RELIANCE,TCS,INFY,SBIN,ICICIBANK --period 60d
  python run_sweep.py --universe "NIFTY 50" --max-symbols 10 --period 60d --out backtest_results
"""

import argparse
import itertools
import sys
from pathlib import Path

from loguru import logger

from alpha_prime.backtest import Backtester, BacktestConfig
from alpha_prime.data.market_data import market_data

# Grid (kept modest; edit to taste)
GRID = {
    "default_stop_loss_pct": [0.8, 1.0, 1.2],
    "default_target_pct": [1.2, 1.8, 2.5],
    "trail_arm_pct": [0.5, 1.0],
    "trailing_stop_pct": [0.7, 1.0],
}


def main():
    ap = argparse.ArgumentParser(description="Alpha-Prime parameter sweep")
    ap.add_argument("--symbols", help="Comma-separated symbols (overrides --universe)")
    ap.add_argument("--universe", default="NIFTY 50", help='Index name (default: NIFTY 50)')
    ap.add_argument("--max-symbols", type=int, default=8, help="Cap symbols for runtime (default 8)")
    ap.add_argument("--period", default="60d")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--no-costs", action="store_true", help="Disable transaction costs (gross sweep)")
    ap.add_argument("--out", default=None, help="Directory to save the ranked CSV")
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = market_data.get_index_stocks(args.universe)[: args.max_symbols]

    base = BacktestConfig()
    if args.capital:
        base.capital = args.capital
    if args.no_costs:
        base.include_costs = False

    print(f"Sweep universe: {len(symbols)} symbols | period={args.period} | costs={'OFF' if args.no_costs else 'ON'}")
    print(f"Symbols: {', '.join(symbols)}")

    bt = Backtester(base)
    data = bt.load_data(symbols, period=args.period, interval=args.interval)
    if not data:
        print("ERROR: no usable data", file=sys.stderr)
        sys.exit(1)

    print("Precomputing entry signals once (reused across all combos)...")
    bt.precompute_signals(data)

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"Running {len(combos)} parameter combinations...\n")

    rows = []
    for combo in combos:
        cfg = BacktestConfig()
        if args.capital:
            cfg.capital = args.capital
        cfg.include_costs = base.include_costs
        for k, v in zip(keys, combo):
            setattr(cfg, k, v)
        res = bt.simulate(data, cfg=cfg)
        m = res.metrics
        if m.get("num_trades", 0) == 0:
            continue
        rows.append({
            "SL%": cfg.default_stop_loss_pct,
            "Tgt%": cfg.default_target_pct,
            "trailArm%": cfg.trail_arm_pct,
            "trail%": cfg.trailing_stop_pct,
            "trades": m["num_trades"],
            "net_pnl": m["total_pnl"],
            "ret%": m["return_on_capital_pct"],
            "win%": m["win_rate_pct"],
            "PF": m["profit_factor"],
            "exp/trade": m["expectancy_per_trade"],
            "maxDD%": m["max_drawdown_pct"],
        })

    rows.sort(key=lambda r: r["net_pnl"], reverse=True)

    hdr = f"{'SL%':>4} {'Tgt%':>5} {'tArm%':>6} {'trl%':>5} {'trades':>6} {'net_pnl':>11} {'ret%':>6} {'win%':>6} {'PF':>5} {'exp/tr':>8} {'maxDD%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pf = f"{r['PF']:.2f}" if r["PF"] is not None else "  inf"
        print(f"{r['SL%']:>4} {r['Tgt%']:>5} {r['trailArm%']:>6} {r['trail%']:>5} "
              f"{r['trades']:>6} {r['net_pnl']:>11,.0f} {r['ret%']:>+6.2f} {r['win%']:>6.1f} {pf:>5} "
              f"{r['exp/trade']:>+8.1f} {r['maxDD%']:>7.2f}")

    if rows:
        best = rows[0]
        print(f"\nBest by net P&L: SL={best['SL%']}% target={best['Tgt%']}% "
              f"trail_arm={best['trailArm%']}% trail={best['trail%']}% "
              f"-> net Rs {best['net_pnl']:,.0f} ({best['ret%']:+.2f}%), exp Rs {best['exp/trade']:+.1f}/trade")
    else:
        print("\nNo combo produced trades.")

    if args.out and rows:
        import pandas as pd
        Path(args.out).mkdir(parents=True, exist_ok=True)
        p = Path(args.out) / "sweep_results.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"saved: {p}")

    print("\n(Sweep only — no live DB or broker was touched.)")


if __name__ == "__main__":
    main()
