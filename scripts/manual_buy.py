"""
Manually BUY one stock in PAPER mode through the normal place_trade() path, so every
hard-coded gate still applies (trading window, cooldown, live 5-min BUY check,
daily-loss limit, risk-based position sizing, per-stock cap).

    venv\\Scripts\\python scripts\\manual_buy.py SIEMENS

Run it ONLY while run_trading.py is stopped, then restart run_trading.py: the new
process loads the position from the database and manages its stop/target/15:15 exit.
"""
import json
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from loguru import logger  # noqa: E402

logger.remove()

from alpha_prime.core.config import settings  # noqa: E402
from alpha_prime.core.scheduler import SCHEDULER_LOCK_PORT  # noqa: E402

if len(sys.argv) != 2:
    sys.exit("Usage: venv\\Scripts\\python scripts\\manual_buy.py SYMBOL")
symbol = sys.argv[1].strip().upper()

if settings.trading.mode != "paper":
    sys.exit("Refusing: TRADING_MODE is not 'paper'.")

# Refuse while a scheduler is running - it would not know about this position.
probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    probe.bind(("127.0.0.1", SCHEDULER_LOCK_PORT))
except OSError:
    sys.exit("Refusing: the trading scheduler is running. Stop it with Ctrl+C first, "
             "then run this, then restart start_experiment.bat.")
finally:
    probe.close()

from alpha_prime.agents.tools import place_trade  # noqa: E402
from alpha_prime.data.market_data import market_data  # noqa: E402

price = market_data.get_live_price(symbol)
if not price:
    sys.exit(f"Could not get a price for {symbol}.")
t = settings.trading
stop = round(price * (1 - t.default_stop_loss_pct / 100), 2)
target = round(price * (1 + t.default_target_pct / 100), 2)
print(f"{symbol} @ Rs {price}: stop {stop} (-{t.default_stop_loss_pct}%), target {target} (+{t.default_target_pct}%)")

# Ask for a large quantity on purpose: the risk gate clamps it to the allowed size.
res = json.loads(place_trade.invoke({"symbol": symbol, "action": "BUY", "quantity": 1_000_000,
                                     "rationale": "Manual paper buy (scripts/manual_buy.py)",
                                     "stop_loss": stop, "target": target}))
if res.get("status") == "EXECUTED":
    print(f"BOUGHT {res['quantity']} {symbol} @ Rs {res['price']} = Rs {res['total_value']:,.0f}")
    print("Now restart:  .\\start_experiment.bat")
else:
    print("NOT BOUGHT ->", res.get("error", res))
