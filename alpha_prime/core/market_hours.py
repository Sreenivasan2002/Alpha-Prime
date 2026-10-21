"""NSE session-time helpers.

These are pure calendar/clock functions with no market-data dependency. They
live in `core` rather than in `data.market_data` so that callers which only
need to know whether the session is open -- the dashboard header, for one --
do not have to import yfinance and the rest of the data stack.

`data.market_data` re-exports these names, so existing imports keep working.
"""

from datetime import date, datetime, timedelta
from typing import Optional

import pytz

from alpha_prime.core.config import settings

IST = pytz.timezone("Asia/Kolkata")

# NSE equity trading holidays (weekdays only). Without this the app "traded" on
# 20 Oct 2026 (Dussehra) at the previous session's prices.
#   verified = no 5-min bars for NIFTY on that day in yfinance data
#   others   = from the NSE 2026 holiday circular; re-check against nseindia.com and add
#              next year's list each December.
# The data check in market_data.session_is_live() is the backstop for anything missing here.
NSE_HOLIDAYS = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi (verified)
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti (verified)
    date(2026, 10, 20),  # Dussehra (verified)
    date(2026, 11, 10),  # Diwali Balipratipada
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}


def is_trading_day(d: Optional[date] = None) -> bool:
    """Weekday that is not an NSE holiday (IST)."""
    d = d or datetime.now(IST).date()
    return d.weekday() < 5 and d not in NSE_HOLIDAYS


def clean_symbol(symbol: str) -> str:
    """Remove exchange suffix from symbol"""
    return symbol.replace(".NS", "").replace(".BO", "").upper().strip()


def is_market_open() -> bool:
    """Check if Indian market is currently open"""
    now = datetime.now(IST)

    # Weekend / NSE holiday check
    if not is_trading_day(now.date()):
        return False

    market_open = now.replace(
        hour=settings.market.market_open_hour,
        minute=settings.market.market_open_minute,
        second=0, microsecond=0
    )
    market_close = now.replace(
        hour=settings.market.market_close_hour,
        minute=settings.market.market_close_minute,
        second=0, microsecond=0
    )

    return market_open <= now <= market_close


def is_trading_window() -> bool:
    """Check if we are within the configured trading window for placing NEW BUY orders.
    This is a subset of market hours - we skip early morning volatility and late-day.
    Market monitor (SL/target) still runs during all market hours.
    """
    now = datetime.now(IST)

    if not is_trading_day(now.date()):
        return False

    if not is_market_open():
        return False

    trading_start = now.replace(
        hour=settings.market.trading_start_hour,
        minute=settings.market.trading_start_minute,
        second=0, microsecond=0
    )
    trading_end = now.replace(
        hour=settings.market.trading_end_hour,
        minute=settings.market.trading_end_minute,
        second=0, microsecond=0
    )

    return trading_start <= now <= trading_end


def is_pre_market() -> bool:
    """Check if we're in pre-market session"""
    now = datetime.now(IST)
    if not is_trading_day(now.date()):
        return False

    pre_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)

    return pre_open <= now < market_open


def time_to_market_open() -> Optional[timedelta]:
    """Get time until market opens"""
    now = datetime.now(IST)
    if is_market_open():
        return timedelta(0)

    # Next market open
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now >= market_open or not is_trading_day(now.date()):
        # Next trading day (skips weekends and NSE holidays)
        days_ahead = 1
        while not is_trading_day((now + timedelta(days=days_ahead)).date()):
            days_ahead += 1
        market_open = (now + timedelta(days=days_ahead)).replace(
            hour=9, minute=15, second=0, microsecond=0
        )
    return market_open - now
