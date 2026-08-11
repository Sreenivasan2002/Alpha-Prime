"""NSE session-time helpers.

These are pure calendar/clock functions with no market-data dependency. They
live in `core` rather than in `data.market_data` so that callers which only
need to know whether the session is open -- the dashboard header, for one --
do not have to import yfinance and the rest of the data stack.

`data.market_data` re-exports these names, so existing imports keep working.
"""

from datetime import datetime, timedelta
from typing import Optional

import pytz

from alpha_prime.core.config import settings

IST = pytz.timezone("Asia/Kolkata")


def clean_symbol(symbol: str) -> str:
    """Remove exchange suffix from symbol"""
    return symbol.replace(".NS", "").replace(".BO", "").upper().strip()


def is_market_open() -> bool:
    """Check if Indian market is currently open"""
    now = datetime.now(IST)

    # Weekend check
    if now.weekday() >= 5:
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

    if now.weekday() >= 5:
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
    if now.weekday() >= 5:
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
    if now >= market_open:
        # Next business day
        days_ahead = 1
        while (now + timedelta(days=days_ahead)).weekday() >= 5:
            days_ahead += 1
        market_open = (now + timedelta(days=days_ahead)).replace(
            hour=9, minute=15, second=0, microsecond=0
        )
    return market_open - now
