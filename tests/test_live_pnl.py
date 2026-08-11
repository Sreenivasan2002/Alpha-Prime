"""
End-to-end tests for LIVE-mode realized P&L recording.

Verifies the fix in GrowwBroker.place_order(): a live SELL now records an
accurate realized `pnl` (exit - avg_entry) * qty in the trades table, so that:
  1. The 3-day stock cooldown (pnl < 0 path) fires for live trades.
  2. get_today_pnl_pct() counts realized P&L even after a position goes flat,
     and does NOT double-count partial closes (Groww position realised_pnl was
     dropped in favor of the trades-table as the single source of truth).

Groww API keys rotate daily, so this runs against a stateful MOCK client that
mimics the recorded Groww response shapes — never a live session.

Run:  E:/Algo Trading/venv/Scripts/python.exe tests/test_live_pnl.py
"""

import os
import sys
import time
import tempfile

# Make the package importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import alpha_prime.core.database as db
from alpha_prime.core.config import settings
from alpha_prime.data.market_data import market_data, clean_symbol
import alpha_prime.core.broker as broker_mod
from alpha_prime.core.broker import GrowwBroker, get_today_pnl_pct
from alpha_prime.agents.tools import _check_stock_cooldown

IST = broker_mod.IST


# --------------------------------------------------------------------------- #
# Mock Groww client — returns the recorded Groww response shapes.
# --------------------------------------------------------------------------- #
class MockGrowwClient:
    """Stateful stand-in for growwapi.GrowwAPI.

    The test drives `positions`, `fill_price`, and `fill_status` to model the
    broker's view before/after a fill.
    """

    def __init__(self):
        self.positions = []          # list of Groww position dicts
        self.fill_price = 0.0        # average_price reported by order status
        self.fill_status = "EXECUTED"
        self.last_order = None

    def place_order(self, **params):
        self.last_order = params
        return {"groww_order_id": "GW-MOCK-1"}

    def get_order_status(self, segment=None, groww_order_id=None):
        qty = (self.last_order or {}).get("quantity", 0)
        return {
            "order_status": self.fill_status,
            "filled_quantity": qty if self.fill_status == "EXECUTED" else 0,
            "average_price": self.fill_price if self.fill_status == "EXECUTED" else 0,
        }

    def get_positions_for_user(self, segment=None):
        return {"positions": list(self.positions)}

    def get_available_margin_details(self):
        return {"clear_cash": 500000.0,
                "equity_margin_details": {"cnc_balance_available": 500000.0}}

    def get_holdings_for_user(self):
        return {"holdings": []}


def _pos(symbol, qty, net_price, ltp, realised_pnl=0.0):
    """Build a Groww-shaped position dict (subset of the recorded fields)."""
    return {
        "trading_symbol": symbol,
        "segment": "CASH",
        "exchange": "NSE",
        "product": "MIS",
        "quantity": qty,
        "net_price": net_price,
        "ltp": ltp,                       # so get_positions skips the network
        "realised_pnl": realised_pnl,
    }


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #
CAPITAL = 1_000_000.0
_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def reset_trades():
    with db.db_session() as conn:
        conn.execute("DELETE FROM trades")


def stamp_last_sell_today():
    """Force the most recent SELL's timestamp to IST-now so get_today_pnl_pct's
    date match is deterministic regardless of the UTC/IST boundary at run time."""
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    with db.db_session() as conn:
        conn.execute(
            "UPDATE trades SET timestamp = ? WHERE id = "
            "(SELECT id FROM trades WHERE action='SELL' ORDER BY id DESC LIMIT 1)",
            (now_ist,))


def make_broker(mock):
    """Return a GrowwBroker wired to the mock client and registered as the
    live singleton so get_today_pnl_pct()/cooldown see the same instance."""
    b = GrowwBroker()
    b._client = mock
    b._initialized = True
    broker_mod.reset_broker()
    broker_mod._groww_broker = b
    return b


def setup_environment():
    # Isolated temp DB
    fd, path = tempfile.mkstemp(suffix=".db", prefix="alpha_test_")
    os.close(fd)
    db.DB_PATH = path
    db.init_database()

    # Live mode + known capital for deterministic percentages
    settings.trading.mode = "live"
    settings.trading.capital = CAPITAL

    # No network, no real sleeps
    market_data.get_live_price = lambda symbol: 1450.0
    time.sleep = lambda *a, **k: None
    return path


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
def scenario_full_close_loss():
    print("\n[A] Full-close losing SELL -> realized pnl, cooldown, flat-day P&L")
    reset_trades()
    mock = MockGrowwClient()
    mock.positions = [_pos("INFY", 100, 1500.0, 1500.0)]
    mock.fill_price = 1450.0
    mock.fill_status = "EXECUTED"
    b = make_broker(mock)

    res = b.place_order("INFY", "SELL", 100, agent_name="test")

    # (1) realized P&L computed and returned: (1450 - 1500) * 100 = -5000
    check("SELL returns realized_pnl=-5000",
          res.get("realized_pnl") == -5000.0, f"got {res.get('realized_pnl')}")

    # (2) trades table stored the loss
    trades = db.get_trades(symbol="INFY", trading_mode="live")
    sell = next((t for t in trades if t["action"] == "SELL"), None)
    check("trades table SELL pnl=-5000",
          sell is not None and sell["pnl"] == -5000.0,
          f"got {sell['pnl'] if sell else None}")

    # (3) cooldown now fires for a live realized loss
    check("cooldown active after live loss",
          _check_stock_cooldown("INFY") is True)

    # (4) position has gone flat (dropped from Groww positions); realized P&L
    #     must still be in today's total via the trades table.
    mock.positions = []
    stamp_last_sell_today()
    pct = get_today_pnl_pct()
    # -5000 / 1,000,000 * 100 = -0.5
    check("get_today_pnl_pct counts flat-position realized loss (-0.5%)",
          abs(pct - (-0.5)) < 1e-6, f"got {pct}")


def scenario_no_double_count():
    print("\n[B] Partial close still open -> no double-count of realized P&L")
    reset_trades()
    mock = MockGrowwClient()
    # A prior SELL recorded a -5000 realized loss in the trades table...
    mock.positions = [_pos("INFY", 100, 1500.0, 1500.0)]
    mock.fill_price = 1450.0
    b = make_broker(mock)
    b.place_order("INFY", "SELL", 50, agent_name="test")  # partial close
    stamp_last_sell_today()

    # ...and Groww still reports the remaining open 50 with its OWN realised_pnl.
    # Old code added both -> double count. New code uses only the trades table
    # for realized and the position for unrealised.
    mock.positions = [_pos("INFY", 50, 1500.0, 1480.0, realised_pnl=-2500.0)]
    pct = get_today_pnl_pct()

    # unrealised on remaining 50: (1480 - 1500) * 50 = -1000
    # realized from trades table: (1450 - 1500) * 50 = -2500
    # expected total = -3500 -> -0.35% ; double-counting would give -0.60%
    check("get_today_pnl_pct = unrealised + trades-realized only (-0.35%)",
          abs(pct - (-0.35)) < 1e-6, f"got {pct}")


def scenario_winning_sell():
    print("\n[C] Winning SELL -> positive pnl, no cooldown")
    reset_trades()
    mock = MockGrowwClient()
    mock.positions = [_pos("TCS", 10, 3000.0, 3000.0)]
    mock.fill_price = 3100.0
    b = make_broker(mock)

    res = b.place_order("TCS", "SELL", 10, agent_name="test")
    check("SELL returns realized_pnl=+1000",
          res.get("realized_pnl") == 1000.0, f"got {res.get('realized_pnl')}")
    check("no cooldown after a win",
          _check_stock_cooldown("TCS") is False)


def scenario_rejected_sell():
    print("\n[D] Rejected SELL -> no phantom pnl recorded")
    reset_trades()
    mock = MockGrowwClient()
    mock.positions = [_pos("WIPRO", 20, 400.0, 380.0)]
    mock.fill_price = 0.0
    mock.fill_status = "REJECTED"
    b = make_broker(mock)

    res = b.place_order("WIPRO", "SELL", 20, agent_name="test")
    check("rejected SELL records realized_pnl=0",
          res.get("realized_pnl") == 0.0, f"got {res.get('realized_pnl')}")
    trades = db.get_trades(symbol="WIPRO", trading_mode="live")
    sell = next((t for t in trades if t["action"] == "SELL"), None)
    check("rejected SELL stored pnl=0 (no false loss/cooldown)",
          sell is not None and sell["pnl"] == 0.0,
          f"got {sell['pnl'] if sell else None}")
    check("no cooldown from a rejected SELL",
          _check_stock_cooldown("WIPRO") is False)


def main():
    path = setup_environment()
    try:
        scenario_full_close_loss()
        scenario_no_double_count()
        scenario_winning_sell()
        scenario_rejected_sell()
    finally:
        # best-effort cleanup of the temp DB (+ WAL/SHM sidecars)
        for p in (path, path + "-wal", path + "-shm"):
            try:
                os.remove(p)
            except OSError:
                pass

    print(f"\n{'='*52}\n  {_passed} passed, {_failed} failed\n{'='*52}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
