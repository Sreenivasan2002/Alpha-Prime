"""
Tests for the 21 Oct 2026 accuracy repairs (audit: 45 confirmed findings).

  A. Average-cost replay of open positions (top-up, partial sell, reopen after flat)
  B. Paper broker: cash includes realised P&L; two broker objects always agree (DB is truth)
  C. Summary: today / total P&L split; snapshots only on request
  D. Monitored positions keep registered_at; a top-up keeps a raised stop and the peak
  E. Exit labels + trigger price (STOP_LOSS vs TRAILING_STOP vs TARGET_HIT)
  F. Stale square-off decides by the position's OPEN date (trades table)
  G. Holiday calendar + intraday-staleness guard
  H. Dashboard performance stats in rupees (no Sharpe before 10 days)
  I. VOID rows (holiday phantom fills) are ignored everywhere

No network: prices and session checks are stubbed; the DB is a temp file.

Run:  E:/Algo Trading/venv/Scripts/python.exe tests/test_accuracy_repairs.py
"""
import os
import sys
import tempfile
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import pytz  # noqa: E402

import alpha_prime.core.database as db  # noqa: E402
from alpha_prime.core.config import settings  # noqa: E402
from alpha_prime.data.market_data import market_data  # noqa: E402
import alpha_prime.core.broker as broker_mod  # noqa: E402
import alpha_prime.core.scheduler as sched  # noqa: E402
import alpha_prime.core.market_hours as mh  # noqa: E402

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.utc
PRICES = {}
_passed = _failed = 0


def check(label, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}  {detail}")


def setup():
    fd, path = tempfile.mkstemp(suffix=".db", prefix="alpha_repair_test_")
    os.close(fd)
    db.DB_PATH = path
    db.init_database()
    t = settings.trading
    t.mode, t.capital, t.margin_multiplier = "paper", 1_000_000.0, 5.0
    t.paper_include_charges = True
    market_data.get_live_price = lambda s: PRICES.get(s)
    market_data.session_is_live = lambda *a, **k: True
    broker_mod._nifty_level = lambda: 25000.0
    sched.is_market_open = lambda: True
    return path


def clear():
    with db.db_session() as conn:
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM monitored_positions")
        conn.execute("DELETE FROM portfolio")


def fill(sym, action, qty, price, ts=None, pnl=0.0, status="EXECUTED"):
    tid = db.record_trade(symbol=sym, action=action, quantity=qty, price=price, pnl=pnl,
                          status=status, trading_mode="paper")
    if ts:
        with db.db_session() as conn:
            conn.execute("UPDATE trades SET timestamp = ? WHERE id = ?", (ts, tid))
    return tid


def utc_str(dt_ist):
    return dt_ist.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def scenario_replay():
    print("\n[A] Average-cost replay")
    clear()
    fill("CANBK", "BUY", 100, 100.0)
    fill("CANBK", "SELL", 100, 101.0)          # flat
    fill("CANBK", "BUY", 50, 120.0)            # reopen: cost basis must be 120, not mean(100,120)
    pos = {p["symbol"]: p for p in db.get_open_positions("paper")}
    check("reopen after flat starts a new cost basis (120)", abs(pos["CANBK"]["avg_buy_price"] - 120.0) < 1e-9,
          pos.get("CANBK"))
    fill("PNB", "BUY", 1000, 100.0)
    fill("PNB", "BUY", 10, 200.0)              # small top-up: weighted, not unweighted mean (150)
    pos = {p["symbol"]: p for p in db.get_open_positions("paper")}
    want = (1000 * 100 + 10 * 200) / 1010
    check("top-up is quantity-weighted", abs(pos["PNB"]["avg_buy_price"] - want) < 1e-9, pos["PNB"])
    fill("PNB", "SELL", 510, 105.0)            # partial sell keeps the average
    pos = {p["symbol"]: p for p in db.get_open_positions("paper")}
    check("partial sell keeps avg and reduces qty", pos["PNB"]["net_qty"] == 500
          and abs(pos["PNB"]["avg_buy_price"] - want) < 1e-9, pos["PNB"])


def scenario_broker():
    print("\n[B] Paper broker: realised P&L in cash, copies agree")
    clear()
    PRICES.update({"AAA": 1000.0, "BBB": 500.0})
    a, b = broker_mod.PaperTradingBroker(), broker_mod.PaperTradingBroker()
    a.place_order("AAA", "BUY", 100, price=1000.0)
    check("second broker object sees the first one's BUY", "AAA" in b.get_positions())
    r = b.place_order("AAA", "SELL", 100, price=990.0)     # loss, sold through the OTHER object
    check("SELL through another object succeeds", r["status"] == "EXECUTED", r)
    check("first object no longer holds AAA", "AAA" not in a.get_positions())
    fresh = broker_mod.PaperTradingBroker()
    want_cash = 5_000_000 + r["realized_pnl"]
    check("restart keeps realised P&L in cash", abs(fresh.available_cash - want_cash) < 0.01,
          (fresh.available_cash, want_cash))
    dup = a.place_order("AAA", "SELL", 100, price=990.0)
    check("a stale copy cannot sell the same shares twice", dup["status"] == "FAILED", dup)


def scenario_summary():
    print("\n[C] Summary and snapshots")
    clear()
    PRICES.update({"AAA": 1010.0})
    yday = utc_str(IST.localize(datetime.now() - timedelta(days=1)).replace(hour=11))
    fill("ZZZ", "BUY", 10, 100.0, ts=yday)
    fill("ZZZ", "SELL", 10, 90.0, ts=yday, pnl=-150.0)       # yesterday: -150
    fill("AAA", "BUY", 100, 1000.0)
    fill("AAA", "SELL", 50, 1020.0, pnl=900.0)               # today: +900 realised, 50 left
    s = broker_mod.PaperTradingBroker().get_portfolio_summary()
    check("realised total = -150 + 900", abs(s["realised_pnl"] - 750) < 1e-6, s["realised_pnl"])
    check("realised today = +900", abs(s["realised_today"] - 900) < 1e-6, s["realised_today"])
    check("unrealised = 50 x (1010 - 1000)", abs(s["unrealised_pnl"] - 500) < 1e-6, s["unrealised_pnl"])
    check("today P&L = 900 + 500, % of REAL capital", abs(s["today_pnl"] - 1400) < 1e-6
          and abs(s["today_pnl_pct"] - 0.14) < 1e-9, (s["today_pnl"], s["today_pnl_pct"]))
    n0 = len(db.get_portfolio_history(limit=100))
    broker_mod.PaperTradingBroker().get_portfolio_summary()
    check("plain summary writes no snapshot", len(db.get_portfolio_history(limit=100)) == n0)
    broker_mod.PaperTradingBroker().get_portfolio_summary(save_snapshot=True)
    h = db.get_portfolio_history(limit=1)[0]
    check("snapshot stores today/total P&L, not unrealised only",
          abs(h["daily_pnl"] - 1400) < 1e-6 and abs(h["total_pnl"] - 1250) < 1e-6, h)


def scenario_monitor():
    print("\n[D] Monitor persistence and top-ups")
    clear()
    m = sched.PositionMonitor()
    m.register_position("AAA", entry_price=1000.0, stop_loss=988.0, target=1025.0)
    with db.db_session() as conn:
        conn.execute("UPDATE monitored_positions SET registered_at = '2026-01-01 04:00:00' WHERE symbol='AAA'")
    m._stop_losses["AAA"] = 1000.0                      # stop already raised to breakeven
    m._peak_prices["AAA"] = 1012.0
    m._persist_position("AAA")
    reg = db.get_monitored_positions()[0]["registered_at"]
    check("an update keeps the original registered_at", str(reg).startswith("2026-01-01"), reg)
    m.register_position("AAA", entry_price=1005.0, stop_loss=990.0, target=1030.0)   # top-up
    st = m.get_status()["AAA"]
    check("top-up keeps the raised stop (1000), the peak and the target",
          st["stop_loss"] == 1000.0 and st["peak"] == 1012.0 and st["target"] == 1025.0, st)
    check("top-up sets entry to the new average", st["entry"] == 1005.0, st)


def scenario_labels():
    print("\n[E] Exit labels")
    clear()
    m = sched.PositionMonitor()
    m.register_position("LOS", entry_price=100.0, stop_loss=98.8, target=102.5)
    m.register_position("TRL", entry_price=100.0, stop_loss=98.8, target=110.0)
    m._stop_losses["TRL"] = 100.5                       # trailed above entry
    m._peak_prices["TRL"] = 101.6
    m.register_position("TGT", entry_price=100.0, stop_loss=98.8, target=102.5)
    PRICES.update({"LOS": 98.5, "TRL": 100.4, "TGT": 102.6})
    sig = {s["symbol"]: s for s in m.check_positions()}
    check("stop below entry -> STOP_LOSS, trigger = stop", sig["LOS"]["reason"] == "STOP_LOSS"
          and sig["LOS"]["trigger_price"] == 98.8, sig.get("LOS"))
    check("stop at/above entry -> TRAILING_STOP", sig["TRL"]["reason"] == "TRAILING_STOP", sig.get("TRL"))
    check("target -> TARGET_HIT, trigger = target", sig["TGT"]["reason"] == "TARGET_HIT"
          and sig["TGT"]["trigger_price"] == 102.5, sig.get("TGT"))


def scenario_stale():
    print("\n[F] Stale square-off by open date")
    clear()
    broker_mod.reset_broker()
    PRICES.update({"OLD": 200.0, "NEW": 300.0})
    yday = utc_str(IST.localize(datetime.now() - timedelta(days=1)).replace(hour=10, minute=30))
    fill("OLD", "BUY", 10, 200.0, ts=yday)
    fill("NEW", "BUY", 10, 300.0)
    sched.position_monitor.register_position("OLD", 200.0)   # registered "today" (e.g. on restart)
    sched.position_monitor.register_position("NEW", 300.0)
    sched.TradingScheduler()._square_off_stale_positions()
    held = broker_mod.PaperTradingBroker().get_positions()
    check("position opened yesterday is closed even though registered today", "OLD" not in held, held)
    check("today's position is kept", "NEW" in held, held)
    check("closed position left the monitor", not sched.position_monitor.is_monitored("OLD"))
    sched.position_monitor.remove_position("NEW")


def scenario_calendar():
    print("\n[G] Holidays and stale intraday bars")
    check("20 Oct 2026 (Dussehra) is not a trading day", not mh.is_trading_day(date(2026, 10, 20)))
    check("21 Oct 2026 is a trading day", mh.is_trading_day(date(2026, 10, 21)))
    check("weekend is not a trading day", not mh.is_trading_day(date(2026, 10, 24)))
    from alpha_prime.agents.tools import _intraday_staleness
    now = datetime.now(IST)
    old = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex([now - timedelta(days=1)]))
    fresh = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex([now - timedelta(minutes=6)]))
    check("yesterday's last bar is stale", _intraday_staleness(old) is not None)
    check("a bar from 6 min ago is fresh", _intraday_staleness(fresh) is None, _intraday_staleness(fresh))


def scenario_performance_and_void():
    print("\n[H/I] Performance stats and VOID rows")
    clear()
    from alpha_prime.ui.dashboard import performance_stats
    fill("AAA", "BUY", 100, 1000.0)
    fill("AAA", "SELL", 100, 1010.0, pnl=900.0)
    fill("BBB", "BUY", 100, 1000.0)
    fill("BBB", "SELL", 100, 990.0, pnl=-1100.0)
    fill("CCC", "BUY", 100, 1000.0, status="VOID")
    fill("CCC", "SELL", 100, 1100.0, pnl=9900.0, status="VOID")
    st = performance_stats("paper", 1_000_000.0)
    check("VOID SELL excluded from stats (2 trades)", st["trades"] == 2, st["trades"])
    check("profit factor in rupees = 900 / 1100", abs(st["profit_factor"] - 900 / 1100) < 1e-9, st["profit_factor"])
    check("no Sharpe before 10 trading days", st["sharpe"] is None, st["sharpe"])
    check("realised P&L ignores VOID rows", abs(db.get_realised_pnl("paper") + 200) < 1e-9, db.get_realised_pnl("paper"))
    check("VOID BUY leaves no open position", "CCC" not in {p["symbol"] for p in db.get_open_positions("paper")})
    check("VOID BUY is not a daily entry", db.count_today_entries("paper") == 2, db.count_today_entries("paper"))


def main():
    path = setup()
    try:
        scenario_replay()
        scenario_broker()
        scenario_summary()
        scenario_monitor()
        scenario_labels()
        scenario_stale()
        scenario_calendar()
        scenario_performance_and_void()
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
