"""
Pre-market GO / NO-GO check. Run it before starting the scheduler:

    venv\\Scripts\\python scripts\\preflight_check.py

Exits 0 when everything needed for a paper-trading day works, 1 otherwise.
The OpenAI check makes a real (1-token) chat call: listing models still succeeds
on a revoked key, so it cannot be used to prove the key works.
"""
import os
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pytz  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()  # keep the output readable

from alpha_prime.core.config import settings  # noqa: E402

IST = pytz.timezone("Asia/Kolkata")
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: {detail}")


print("\nAlpha-Prime pre-flight check -", datetime.now(IST).strftime("%a %d %b %Y, %H:%M IST"), "\n")
t, m = settings.trading, settings.market

check("Trading mode", t.mode == "paper", t.mode)
check("Capital", t.capital == 1_000_000, f"Rs {t.capital:,.0f} (x{t.margin_multiplier:g} margin)")
check("Paper charges", t.paper_include_charges, "deducted from P&L" if t.paper_include_charges else "OFF - P&L will be gross")
check("Turnover / exposure caps",
      t.max_trades_per_day <= 3 and t.max_open_positions <= 3 and t.max_entries_per_cycle <= 1
      and t.max_gross_exposure_pct <= 100,
      f"{t.max_trades_per_day} entries/day, {t.max_entries_per_cycle}/cycle, {t.max_open_positions} open, "
      f"gross <= Rs {t.capital * t.max_gross_exposure_pct / 100:,.0f} (frozen for the 4-week validation)")
check("Cycle interval", 60 % m.analysis_interval_minutes == 0,
      f"every {m.analysis_interval_minutes} min, entries {m.trading_start_hour:02d}:{m.trading_start_minute:02d}-"
      f"{m.trading_end_hour:02d}:{m.trading_end_minute:02d}")

try:
    from openai import OpenAI
    r = OpenAI(api_key=settings.openai.api_key, timeout=20).chat.completions.create(
        model=settings.openai.model, messages=[{"role": "user", "content": "Reply with OK"}], max_tokens=2)
    check("OpenAI key", True, f"{settings.openai.model} answered")
except Exception as e:
    msg = str(e)
    hint = " -> create a new key at platform.openai.com/api-keys and put it in .env" if "401" in msg else ""
    check("OpenAI key", False, msg[:110] + hint)

try:
    from alpha_prime.data.market_data import market_data
    p = market_data.get_live_price("RELIANCE")
    check("Market data (yfinance)", bool(p), f"RELIANCE Rs {p}" if p else "no price returned")
except Exception as e:
    check("Market data (yfinance)", False, str(e)[:110])

try:
    c = sqlite3.connect(str(settings.db_path))
    open_pos = c.execute("""select count(*) from (select symbol, sum(case when action='BUY' then quantity else -quantity end) q
                            from trades where status='EXECUTED' and trading_mode='paper' group by symbol having q > 0)""").fetchone()[0]
    n = c.execute("select count(*) from trades").fetchone()[0]
    check("Database", open_pos == 0, f"{n} trades recorded, {open_pos} open paper positions")
except Exception as e:
    check("Database", False, str(e)[:110])

now = datetime.now(IST)
from alpha_prime.core.market_hours import is_trading_day  # noqa: E402
check("Trading day", is_trading_day(now.date()),
      now.strftime("%A %d %b") + ("" if is_trading_day(now.date()) else " - weekend or NSE holiday, market closed"))

print()
if all(results):
    print("  GO - start trading with:  venv\\Scripts\\python run_trading.py\n")
    sys.exit(0)
print("  NO-GO - fix the FAIL lines above first.\n")
sys.exit(1)
