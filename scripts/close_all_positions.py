"""
Close every open PAPER position at the current market price.

    venv\\Scripts\\python scripts\\close_all_positions.py

STOP run_trading.py and the Streamlit dashboard FIRST. A running scheduler keeps its own
in-memory copy of positions and would try to sell the same shares again.
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
from alpha_prime.core.database import clear_monitored_positions  # noqa: E402

if settings.trading.mode != "paper":
    sys.exit("Refusing: TRADING_MODE is not 'paper'. This script only closes paper positions.")

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
        total += r["realized_pnl"]
        print(f"SOLD {pos['quantity']:>5} {symbol:<11} @ Rs {r['price']:,.2f}   gross {r['gross_pnl']:+,.2f}   "
              f"charges {r['charges']:,.2f}   NET {r['realized_pnl']:+,.2f}")
    else:
        print(f"FAILED {symbol}: {r.get('error')}")

clear_monitored_positions(trading_mode="paper")
print(f"\nTotal net P&L on positions closed now: Rs {total:+,.2f}")
