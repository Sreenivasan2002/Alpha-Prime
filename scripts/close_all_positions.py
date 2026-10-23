"""
Close every open PAPER position at the current market price.

    venv\\Scripts\\python scripts\\close_all_positions.py

The paper book is rebuilt from the database on every order, so this is safe while
run_trading.py runs: its monitor drops positions that are no longer held.
Refuses when the market is closed or has no live session (it would sell at a stale price).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from loguru import logger  # noqa: E402

logger.remove()

from alpha_prime.core.config import settings  # noqa: E402
from alpha_prime.core.broker import PaperTradingBroker  # noqa: E402
from alpha_prime.core.database import remove_monitored_position  # noqa: E402
from alpha_prime.data.market_data import market_data, is_market_open  # noqa: E402

if settings.trading.mode != "paper":
    sys.exit("Refusing: TRADING_MODE is not 'paper'. This script only closes paper positions.")

if not is_market_open() or not market_data.session_is_live():
    sys.exit("Refusing: the market is closed or has no live session - a sell now would use a stale price.")

broker = PaperTradingBroker()          # loads open positions from the database
positions = broker.get_positions()
if not positions:
    print("No open paper positions.")
    sys.exit(0)

total = 0.0
for symbol, pos in positions.items():
    r = broker.place_order(symbol=symbol, action="SELL", quantity=pos["quantity"],
                           rationale="Manual close-all (scripts/close_all_positions.py)",
                           agent_name="manual_close_all")
    if r.get("status") == "EXECUTED":
        remove_monitored_position(symbol)
        total += r["realized_pnl"]
        print(f"SOLD {pos['quantity']:>5} {symbol:<11} @ Rs {r['price']:,.2f}   gross {r['gross_pnl']:+,.2f}   "
              f"charges {r['charges']:,.2f}   NET {r['realized_pnl']:+,.2f}")
    else:
        print(f"FAILED {symbol}: {r.get('error')}")

print(f"\nTotal net P&L on positions closed now: Rs {total:+,.2f}")
