"""
Tests for the turnover / exposure caps and per-trade instrumentation (added 2026-10-18).

  A. MAX_TRADES_PER_DAY      - the 4th BUY entry of the day is refused (counted from the DB)
  B. MAX_ENTRIES_PER_CYCLE   - a 2nd entry in one scan cycle is refused; the next cycle resets it
  C. MAX_GROSS_EXPOSURE_PCT  - quantity is clamped to the room left under the gross cap, and
                               refused once the cap is used up
  D. Fail-closed             - if open positions can't be read, the BUY is refused
  E. Instrumentation         - paper BUY/SELL rows carry ref_price, nifty_price, charges, gross_pnl

No network: prices, NIFTY level and the broker are stubbed; the DB is a temp file.

Run:  E:/Algo Trading/venv/Scripts/python.exe tests/test_turnover_caps.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpha_prime.core.database as db
from alpha_prime.core.config import settings
from alpha_prime.data.market_data import market_data
import alpha_prime.core.broker as broker_mod
import alpha_prime.agents.tools as tools

CAPITAL = 1_000_000.0
PRICE = 1000.0

_passed = 0
_failed = 0


def check(label, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}  {detail}")


class FakeBroker:
    def __init__(self):
        self.positions = {}
        self.fail = False

    def get_positions(self):
        if self.fail:
            raise RuntimeError("broker down")
        return self.positions


def setup_environment():
    fd, path = tempfile.mkstemp(suffix=".db", prefix="alpha_caps_test_")
    os.close(fd)
    db.DB_PATH = path
    db.init_database()

    t = settings.trading
    t.mode = "paper"
    t.capital = CAPITAL
    t.margin_multiplier = 5.0
    t.max_open_positions = 3
    t.max_trades_per_day = 3
    t.max_gross_exposure_pct = 100.0
    t.max_entries_per_cycle = 1
    t.risk_per_trade_pct = 0.5
    t.max_position_size_pct = 20.0
    t.paper_include_charges = True

    market_data.get_live_price = lambda symbol: PRICE
    broker_mod.get_today_pnl_pct = lambda: 0.0
    broker_mod._nifty_level = lambda: 25000.0
    return path


def clear_trades():
    with db.db_session() as conn:
        conn.execute("DELETE FROM trades")


def buy_row(symbol):
    db.record_trade(symbol=symbol, action="BUY", quantity=10, price=PRICE, trading_mode="paper")


def scenario_daily_cap(fake):
    print("\n[A] Daily trade cap")
    clear_trades()
    fake.positions = {}
    for s in ("AAA", "BBB", "CCC"):
        buy_row(s)
    tools.begin_trading_cycle("A")
    r = tools._enforce_risk_limits("DDD", stop_loss=990.0, requested_qty=10)
    check("4th entry of the day refused", not r["ok"] and "Daily trade cap" in r["reason"], r)
    # a REJECTED order does not use up the cap
    clear_trades()
    buy_row("AAA")
    buy_row("BBB")
    db.record_trade(symbol="CCC", action="BUY", quantity=10, price=PRICE,
                    status="REJECTED", trading_mode="paper")
    r = tools._enforce_risk_limits("DDD", stop_loss=990.0, requested_qty=10)
    check("rejected orders are not counted", r["ok"], r)
    # live rows don't count against the paper cap
    clear_trades()
    for s in ("AAA", "BBB", "CCC"):
        db.record_trade(symbol=s, action="BUY", quantity=10, price=PRICE, trading_mode="live")
    r = tools._enforce_risk_limits("DDD", stop_loss=990.0, requested_qty=10)
    check("other mode's entries are not counted", r["ok"], r)


def scenario_cycle_cap(fake):
    print("\n[B] One entry per scan cycle")
    clear_trades()
    fake.positions = {}
    tools.begin_trading_cycle("B1")
    r1 = tools._enforce_risk_limits("AAA", stop_loss=990.0, requested_qty=10)
    tools._record_cycle_entry()
    r2 = tools._enforce_risk_limits("BBB", stop_loss=990.0, requested_qty=10)
    check("1st entry in cycle allowed", r1["ok"], r1)
    check("2nd entry in same cycle refused", not r2["ok"] and "per scan cycle" in r2["reason"], r2)
    tools.begin_trading_cycle("B2")
    r3 = tools._enforce_risk_limits("BBB", stop_loss=990.0, requested_qty=10)
    check("next cycle allows an entry again", r3["ok"], r3)


def scenario_gross_cap(fake):
    print("\n[C] Gross exposure cap (Rs 10L on Rs 10L capital)")
    clear_trades()
    tools.begin_trading_cycle("C")
    # Rs 9.5L already open -> room for Rs 50k = 50 shares at Rs 1000
    fake.positions = {"AAA": {"quantity": 500, "avg_price": 1000.0, "current_value": 500_000.0},
                      "BBB": {"quantity": 450, "avg_price": 1000.0, "current_value": 450_000.0}}
    # stop 1% away -> risk sizing alone would allow 500 shares, notional cap 200
    r = tools._enforce_risk_limits("CCC", stop_loss=990.0, requested_qty=10_000)
    check("quantity clamped to remaining gross room (50)", r["ok"] and r["quantity"] == 50, r)
    check("binding constraint reported as gross_exposure", "gross_exposure" in r.get("note", ""), r)
    fake.positions = {"AAA": {"quantity": 500, "avg_price": 1000.0, "current_value": 500_000.0},
                      "BBB": {"quantity": 500, "avg_price": 1000.0, "current_value": 500_000.0}}
    r = tools._enforce_risk_limits("CCC", stop_loss=990.0, requested_qty=10)
    check("refused once gross cap is used up", not r["ok"] and "Gross exposure" in r["reason"], r)
    # falls back to qty * avg_price when current_value is missing
    fake.positions = {"AAA": {"quantity": 900, "avg_price": 1000.0}}
    r = tools._enforce_risk_limits("CCC", stop_loss=990.0, requested_qty=10_000)
    check("uses qty*avg_price when current_value missing (room 100)", r["ok"] and r["quantity"] == 100, r)
    # with no positions the old caps still bind (notional 20% = 200 shares)
    fake.positions = {}
    r = tools._enforce_risk_limits("CCC", stop_loss=990.0, requested_qty=10_000)
    check("empty book: notional cap (200) still binds", r["ok"] and r["quantity"] == 200, r)


def scenario_fail_closed(fake):
    print("\n[D] Fail closed")
    clear_trades()
    tools.begin_trading_cycle("D")
    fake.fail = True
    r = tools._enforce_risk_limits("AAA", stop_loss=990.0, requested_qty=10)
    fake.fail = False
    check("positions unreadable -> BUY refused", not r["ok"] and "open positions" in r["reason"], r)


def scenario_instrumentation():
    print("\n[E] Paper trade instrumentation")
    clear_trades()
    pb = broker_mod.PaperTradingBroker()
    pb.place_order("AAA", "BUY", 100, price=1000.0, ref_price=999.0, agent_name="test")
    broker_mod._nifty_level = lambda: 25100.0
    res = pb.place_order("AAA", "SELL", 100, price=1010.0, ref_price=1011.0, agent_name="test")
    with db.db_session() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM trades ORDER BY id")]
    b, s = rows[0], rows[1]
    check("BUY ref_price stored", b["ref_price"] == 999.0, b)
    check("BUY nifty_price stored", b["nifty_price"] == 25000.0, b)
    check("BUY has no charges (charged on the round trip)", b["charges"] is None, b)
    check("SELL ref_price / nifty_price stored", s["ref_price"] == 1011.0 and s["nifty_price"] == 25100.0, s)
    check("SELL gross_pnl = 1000", abs(s["gross_pnl"] - 1000.0) < 1e-6, s)
    check("SELL charges match result and net = gross - charges",
          abs(s["charges"] - res["charges"]) < 0.01 and abs(s["pnl"] - (s["gross_pnl"] - s["charges"])) < 0.01, s)


def main():
    path = setup_environment()
    fake = FakeBroker()
    tools.get_broker = lambda: fake
    try:
        scenario_daily_cap(fake)
        scenario_cycle_cap(fake)
        scenario_gross_cap(fake)
        scenario_fail_closed(fake)
        scenario_instrumentation()
    finally:
        for p in (path, path + "-wal", path + "-shm"):
            try:
                os.remove(p)
            except OSError:
                pass
    print(f"\n{'=' * 52}\n  {_passed} passed, {_failed} failed\n{'=' * 52}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
