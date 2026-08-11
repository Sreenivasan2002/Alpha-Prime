"""Build the public demo database from the live trading database.

The dashboard deployed to Streamlit Cloud runs read-only against a snapshot so
that visitors never touch a broker, an LLM, or real credentials. This script
produces that snapshot and is committed so the transformation is auditable.

What it does:
  1. Copies the live DB, then drops every live-mode trade and every
     system_state key that references the real brokerage account.
  2. Backfills the ``trades.pnl`` column. The recorded history predates the
     P&L-attribution fix, so ``pnl`` is 0.0 on every row. P&L is recomputed by
     FIFO lot-matching the actual recorded fills -- no values are invented.
  3. Rebuilds the portfolio equity series as (starting capital + cumulative
     realized P&L), so the equity curve and max-drawdown reflect the real
     trade sequence instead of a flat line.

Usage:
    python scripts/build_demo_db.py
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from collections import defaultdict, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB = PROJECT_ROOT / "data" / "alpha_prime.db"
DEMO_DB = PROJECT_ROOT / "data" / "demo" / "alpha_prime_demo.db"

# Real deployed capital, per TRADING_CAPITAL. The equity curve is anchored here.
STARTING_CAPITAL = 1_000_000.0

# Keys that leak real-account state. Dropped wholesale from the demo snapshot.
LIVE_STATE_KEYS = (
    "day_start_portfolio_value_live",
    "day_start_cash_live",
    "day_start_date_live",
    "day_start_portfolio_value",
    "day_start_date",
    "groww_auth_status",
    "groww_auth_time",
    "scheduler_status",
    "test_key",
)


def fifo_realized_pnl(conn: sqlite3.Connection) -> dict[int, float]:
    """Match SELLs against BUYs FIFO and return {trade_id: realized_pnl}."""
    rows = conn.execute(
        "select id, symbol, action, quantity, price from trades "
        "where status='EXECUTED' order by timestamp, id"
    ).fetchall()

    lots: dict[str, deque[list[float]]] = defaultdict(deque)
    realized: dict[int, float] = {}

    for trade_id, symbol, action, qty, price in rows:
        if action == "BUY":
            lots[symbol].append([qty, price])
            continue

        remaining, pnl = qty, 0.0
        while remaining > 0 and lots[symbol]:
            lot = lots[symbol][0]
            take = min(remaining, lot[0])
            pnl += (price - lot[1]) * take
            lot[0] -= take
            remaining -= take
            if lot[0] == 0:
                lots[symbol].popleft()
        # A SELL with no matching BUY predates the logged history; it
        # contributes only the portion that could be matched.
        realized[trade_id] = round(pnl, 2)

    return realized


def rebuild_equity_curve(conn: sqlite3.Connection) -> int:
    """Replace portfolio history with equity derived from realized P&L."""
    sells = conn.execute(
        "select timestamp, pnl from trades "
        "where action='SELL' and status='EXECUTED' and pnl is not null "
        "order by timestamp, id"
    ).fetchall()

    first_ts = conn.execute("select min(timestamp) from trades").fetchone()[0]

    conn.execute("delete from portfolio")

    equity = STARTING_CAPITAL
    rows = [(first_ts, STARTING_CAPITAL, 0.0, STARTING_CAPITAL, STARTING_CAPITAL, 0.0, 0.0)]
    for ts, pnl in sells:
        equity += pnl
        rows.append(
            (
                ts,
                STARTING_CAPITAL,          # capital
                0.0,                       # invested (flat between closes)
                equity,                    # available
                equity,                    # total_value
                pnl,                       # daily_pnl -> P&L booked at this close
                equity - STARTING_CAPITAL,  # total_pnl
            )
        )

    conn.executemany(
        "insert into portfolio "
        "(timestamp, capital, invested, available, total_value, daily_pnl, total_pnl) "
        "values (?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def main() -> int:
    if not SOURCE_DB.exists():
        print(f"ERROR: source DB not found at {SOURCE_DB}", file=sys.stderr)
        return 1

    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    if DEMO_DB.exists():
        DEMO_DB.unlink()
    shutil.copy2(SOURCE_DB, DEMO_DB)

    conn = sqlite3.connect(DEMO_DB)

    live_trades = conn.execute(
        "select count(*) from trades where trading_mode='live'"
    ).fetchone()[0]
    conn.execute("delete from trades where trading_mode='live'")

    conn.executemany(
        "delete from system_state where key=?", [(k,) for k in LIVE_STATE_KEYS]
    )

    # Stale runtime state that would misrepresent a static snapshot.
    conn.execute("delete from market_cache")
    conn.execute("delete from monitored_positions")

    realized = fifo_realized_pnl(conn)
    conn.executemany(
        "update trades set pnl=? where id=?",
        [(pnl, trade_id) for trade_id, pnl in realized.items()],
    )

    equity_rows = rebuild_equity_curve(conn)

    conn.commit()
    conn.execute("vacuum")
    conn.commit()

    total = sum(realized.values())
    wins = [v for v in realized.values() if v > 0]
    losses = [v for v in realized.values() if v < 0]
    trades_left = conn.execute("select count(*) from trades").fetchone()[0]
    logs_left = conn.execute("select count(*) from agent_logs").fetchone()[0]
    conn.close()

    size_mb = DEMO_DB.stat().st_size / (1024 * 1024)
    print(f"demo DB      : {DEMO_DB.relative_to(PROJECT_ROOT)}  ({size_mb:.1f} MB)")
    print(f"live trades  : {live_trades} removed")
    print(f"trades kept  : {trades_left}")
    print(f"agent logs   : {logs_left}")
    print(f"closed trades: {len(realized)}")
    print(f"realized P&L : Rs {total:,.2f}")
    print(f"win rate     : {len(wins) / len(realized) * 100:.1f}%" if realized else "")
    print(f"equity points: {equity_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
