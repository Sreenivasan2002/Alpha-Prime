"""
LangChain Tools for Alpha-Prime Trading Agents
These tools are callable by the AI agents to interact with market data,
technical analysis, and broker systems.
"""

import json
from typing import Optional
from langchain_core.tools import tool
from loguru import logger

from alpha_prime.data.market_data import market_data, is_market_open, clean_symbol
from alpha_prime.data.technical_analysis import technical_analyzer
from alpha_prime.core.broker import get_broker
from alpha_prime.core.database import (
    get_trades, get_signals, get_open_positions,
    add_to_watchlist, get_watchlist, record_signal, log_agent_activity
)


@tool
def get_stock_price(symbol: str) -> str:
    """Get the current/latest price for an Indian stock (NSE).
    Args:
        symbol: Stock symbol like RELIANCE, TCS, INFY, HDFCBANK
    """
    price = market_data.get_live_price(symbol)
    if price:
        return json.dumps({"symbol": clean_symbol(symbol), "price": price, "exchange": "NSE"})
    return json.dumps({"error": f"Could not fetch price for {symbol}"})


@tool
def get_multiple_stock_prices(symbols: str) -> str:
    """Get prices for multiple stocks at once. Provide comma-separated symbols.
    Args:
        symbols: Comma-separated stock symbols like 'RELIANCE,TCS,INFY'
    """
    symbol_list = [s.strip() for s in symbols.split(",")]
    prices = market_data.get_multiple_prices(symbol_list)
    return json.dumps(prices)


@tool
def get_stock_info(symbol: str) -> str:
    """Get detailed fundamental information about a stock including PE ratio, market cap, sector, etc.
    Args:
        symbol: Stock symbol like RELIANCE, TCS, INFY
    """
    info = market_data.get_stock_info(symbol)
    if info:
        return json.dumps(info)
    return json.dumps({"error": f"Could not fetch info for {symbol}"})


@tool
def run_technical_analysis(symbol: str, period: str = "3mo") -> str:
    """Run comprehensive technical analysis on a stock. Returns trend, signals from RSI, MACD, Bollinger Bands, moving averages, candlestick patterns, support/resistance levels, and an overall BUY/SELL/HOLD recommendation.
    Args:
        symbol: Stock symbol like RELIANCE, TCS
        period: Data period - 1mo, 3mo, 6mo, 1y (default: 3mo)
    """
    df = market_data.get_historical_data(symbol, period=period, interval="1d")
    if df is None:
        return json.dumps({"error": f"No data available for {symbol}"})

    report = technical_analyzer.analyze(df, symbol=clean_symbol(symbol))
    if report:
        return json.dumps(report.to_dict())
    return json.dumps({"error": f"Analysis failed for {symbol}"})


@tool
def run_intraday_analysis(symbol: str) -> str:
    """Run technical analysis on intraday (5-minute) data for a stock. Best for same-day trading decisions. Uses 5 days of data to ensure enough candles for indicator computation.
    Args:
        symbol: Stock symbol like RELIANCE, TCS
    """
    df = market_data.get_intraday_data(symbol, interval="5m", period="5d")
    if df is None:
        return json.dumps({"error": f"No intraday data for {symbol}"})

    report = technical_analyzer.analyze(df, symbol=clean_symbol(symbol))
    if report:
        return json.dumps(report.to_dict())
    return json.dumps({"error": f"Intraday analysis failed for {symbol}"})


@tool
def get_historical_data(symbol: str, period: str = "6mo") -> str:
    """Get historical OHLCV data for a stock. Returns last 10 data points with full details.
    Args:
        symbol: Stock symbol
        period: 1mo, 3mo, 6mo, 1y, 2y
    """
    df = market_data.get_historical_data(symbol, period=period)
    if df is None:
        return json.dumps({"error": f"No data for {symbol}"})

    # Return last 10 rows as JSON
    recent = df.tail(10).reset_index()
    recent["Date"] = recent["Date"].astype(str)
    return recent.to_json(orient="records")
        

@tool
def place_trade(symbol: str, action: str, quantity: int,
                rationale: str, stop_loss: float = 0,
                target: float = 0) -> str:
    """Place a BUY or SELL trade order.
    Args:
        symbol: Stock symbol to trade
        action: BUY or SELL
        quantity: Number of shares
        rationale: Reason for this trade
        stop_loss: Stop loss price (0 = no stop loss)
        target: Target price (0 = no target)
    """
    broker = get_broker()
    result = broker.place_order(
        symbol=symbol,
        action=action,
        quantity=quantity,
        stop_loss=stop_loss if stop_loss > 0 else None,
        target=target if target > 0 else None,
        rationale=rationale,
        agent_name="trading_agent"
    )

    # Auto-register BUY positions with the position monitor for trailing SL
    if action.upper() == "BUY" and result.get("status") in ("EXECUTED", "PLACED"):
        try:
            from alpha_prime.core.scheduler import position_monitor
            entry_price = result.get("price", 0)
            if entry_price > 0:
                position_monitor.register_position(
                    symbol=clean_symbol(symbol),
                    entry_price=entry_price,
                    stop_loss=stop_loss if stop_loss > 0 else 0,
                    target=target if target > 0 else 0
                )
        except Exception as e:
            logger.warning(f"Could not register position for monitoring: {e}")

    return json.dumps(result)


@tool
def get_portfolio() -> str:
    """Get the current portfolio summary including all positions, P&L, and available cash."""
    broker = get_broker()
    summary = broker.get_portfolio_summary()
    return json.dumps(summary)


@tool
def get_current_positions() -> str:
    """Get all currently open positions with their current values and P&L."""
    broker = get_broker()
    positions = broker.get_positions()
    if positions:
        return json.dumps(positions)
    return json.dumps({"message": "No open positions"})


@tool
def get_recent_trades(limit: int = 20) -> str:
    """Get recent trade history.
    Args:
        limit: Number of recent trades to fetch (default: 20)
    """
    trades = get_trades(limit=limit)
    return json.dumps(trades, default=str)


@tool
def get_recent_signals(limit: int = 20) -> str:
    """Get recent trading signals generated by agents.
    Args:
        limit: Number of signals to fetch (default: 20)
    """
    signals = get_signals(limit=limit)
    return json.dumps(signals, default=str)


@tool
def check_market_status() -> str:
    """Check if the Indian stock market (NSE) is currently open for trading."""
    from alpha_prime.data.market_data import is_market_open, is_pre_market, time_to_market_open
    from datetime import datetime
    import pytz

    IST = pytz.timezone("Asia/Kolkata")
    now = datetime.now(IST)

    return json.dumps({
        "is_open": is_market_open(),
        "is_pre_market": is_pre_market(),
        "current_time_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "market_hours": "09:15 - 15:30 IST",
        "time_to_open": str(time_to_market_open()) if not is_market_open() else "0"
    })


@tool
def get_nifty50_stocks() -> str:
    """Get the list of NIFTY 50 constituent stocks."""
    stocks = market_data.get_nifty50_stocks()
    return json.dumps(stocks)


@tool
def save_signal(symbol: str, signal_type: str, strength: float,
                reasoning: str, agent_name: str = "analyst") -> str:
    """Save a trading signal for later reference.
    Args:
        symbol: Stock symbol
        signal_type: BUY, SELL, or HOLD
        strength: Signal strength from 0.0 to 1.0
        reasoning: Explanation for this signal
        agent_name: Name of the agent generating this signal
    """
    record_signal(
        symbol=clean_symbol(symbol),
        signal_type=signal_type,
        strength=strength,
        agent_name=agent_name,
        reasoning=reasoning
    )
    return json.dumps({"status": "Signal saved", "symbol": symbol, "signal": signal_type})


# Group tools by agent role
MARKET_ANALYSIS_TOOLS = [
    get_stock_price,
    get_multiple_stock_prices,
    get_stock_info,
    get_historical_data,
    run_technical_analysis,
    run_intraday_analysis,
    get_nifty50_stocks,
    check_market_status,
]

TRADING_TOOLS = [
    place_trade,
    get_portfolio,
    get_current_positions,
    get_recent_trades,
]

SIGNAL_TOOLS = [
    save_signal,
    get_recent_signals,
]

ALL_TOOLS = MARKET_ANALYSIS_TOOLS + TRADING_TOOLS + SIGNAL_TOOLS
