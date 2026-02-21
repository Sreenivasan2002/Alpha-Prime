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
from alpha_prime.core.config import settings
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
    # === HARD-CODED TRADE VALIDATION (last line of defense) ===
    if action.upper() == "BUY":
        # 1. Must have stop-loss defined
        if stop_loss <= 0:
            return json.dumps({
                "status": "REJECTED",
                "error": "BUY rejected: stop_loss is required. Cannot buy without a stop-loss."
            })

        # 2. Must have target defined
        if target <= 0:
            return json.dumps({
                "status": "REJECTED",
                "error": "BUY rejected: target is required. Cannot buy without a profit target."
            })

        # 3. Validate current price to check target gives at least 0.5% profit
        try:
            current_price = market_data.get_live_price(symbol)
            if current_price and current_price > 0:
                target_pct = (target - current_price) / current_price * 100
                sl_pct = (current_price - stop_loss) / current_price * 100

                # Target must be at least 0.5% above current price
                if target_pct < 0.5:
                    return json.dumps({
                        "status": "REJECTED",
                        "error": f"BUY rejected: target ({target:.2f}) is only {target_pct:.2f}% above current price ({current_price:.2f}). Need at least 0.5%."
                    })

                # Stop-loss must not be more than 3% below (too risky)
                if sl_pct > 3.0:
                    return json.dumps({
                        "status": "REJECTED",
                        "error": f"BUY rejected: stop-loss ({stop_loss:.2f}) is {sl_pct:.2f}% below current price. Max allowed: 3%."
                    })

                # Risk:Reward must be at least 1:1 (target distance >= SL distance)
                if target_pct < sl_pct:
                    return json.dumps({
                        "status": "REJECTED",
                        "error": f"BUY rejected: Risk:Reward is bad. Target={target_pct:.2f}% vs SL={sl_pct:.2f}%. Target must be >= SL distance."
                    })

                # Run quick intraday check - verify trend is not BEARISH
                try:
                    df = market_data.get_intraday_data(symbol, interval="5m", period="5d")
                    if df is not None:
                        from alpha_prime.data.technical_analysis import technical_analyzer
                        report = technical_analyzer.analyze(df, symbol=clean_symbol(symbol))
                        if report:
                            if report.overall_signal == "SELL":
                                return json.dumps({
                                    "status": "REJECTED",
                                    "error": f"BUY rejected: Real-time analysis shows SELL signal for {symbol}. Cannot buy against sell signal."
                                })
                            if report.trend and report.trend.upper() == "BEARISH":
                                return json.dumps({
                                    "status": "REJECTED",
                                    "error": f"BUY rejected: {symbol} is in BEARISH trend. Cannot buy in downtrend."
                                })
                            if report.overall_strength < 0.5:
                                return json.dumps({
                                    "status": "REJECTED",
                                    "error": f"BUY rejected: {symbol} signal strength is only {report.overall_strength:.2f}. Need >= 0.50."
                                })
                except Exception as e:
                    logger.warning(f"Pre-trade validation analysis failed for {symbol}: {e}")
                    # Don't block trade if validation itself fails

        except Exception as e:
            logger.warning(f"Pre-trade price validation failed for {symbol}: {e}")

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
    """Get recent trade history for the current trading mode (paper or live).
    Args:
        limit: Number of recent trades to fetch (default: 20)
    """
    mode = getattr(settings.trading, "mode", "paper")
    trades = get_trades(limit=limit, trading_mode=mode)
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


@tool
def check_news_sentiment(symbol: str) -> str:
    """Check recent news sentiment for a stock using Google News RSS. Returns sentiment summary (POSITIVE/NEGATIVE/NEUTRAL) based on recent headlines.
    Args:
        symbol: Stock symbol like RELIANCE, TCS, INFY
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET
    from datetime import datetime, timedelta

    symbol_clean = clean_symbol(symbol)

    # Map common symbols to company names for better search
    company_names = {
        "RELIANCE": "Reliance Industries",
        "TCS": "TCS Tata Consultancy",
        "HDFCBANK": "HDFC Bank",
        "INFY": "Infosys",
        "ICICIBANK": "ICICI Bank",
        "SBIN": "State Bank India SBI",
        "BHARTIARTL": "Bharti Airtel",
        "ITC": "ITC Limited",
        "KOTAKBANK": "Kotak Mahindra Bank",
        "LT": "Larsen Toubro",
        "WIPRO": "Wipro",
        "HCLTECH": "HCL Technologies",
        "MARUTI": "Maruti Suzuki",
        "TATASTEEL": "Tata Steel",
        "TMPV": "Tata Motors Passenger Vehicles",
        "SUNPHARMA": "Sun Pharmaceutical",
        "BAJFINANCE": "Bajaj Finance",
        "AXISBANK": "Axis Bank",
        "ADANIENT": "Adani Enterprises",
        "NTPC": "NTPC Limited",
        "POWERGRID": "Power Grid Corporation",
    }

    search_term = company_names.get(symbol_clean, symbol_clean)
    query = f"{search_term} stock NSE"

    try:
        # Google News RSS feed
        encoded_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        response = urllib.request.urlopen(req, timeout=10)
        xml_data = response.read()
        root = ET.fromstring(xml_data)

        headlines = []
        channel = root.find("channel")
        if channel is not None:
            for item in channel.findall("item")[:10]:  # Last 10 headlines
                title = item.find("title")
                pub_date = item.find("pubDate")
                if title is not None and title.text:
                    headlines.append({
                        "title": title.text,
                        "date": pub_date.text if pub_date is not None else ""
                    })

        if not headlines:
            return json.dumps({
                "symbol": symbol_clean,
                "sentiment": "NEUTRAL",
                "confidence": 0.3,
                "headline_count": 0,
                "summary": f"No recent news found for {symbol_clean}",
                "headlines": []
            })

        # Simple keyword-based sentiment scoring
        positive_words = [
            "surge", "rally", "jump", "gain", "rise", "high", "profit", "growth",
            "buy", "upgrade", "bullish", "strong", "record", "soar", "boom",
            "positive", "outperform", "beat", "above", "up", "success",
            "dividend", "bonus", "revenue", "earnings", "expansion"
        ]
        negative_words = [
            "fall", "drop", "crash", "loss", "decline", "low", "sell", "downgrade",
            "bearish", "weak", "concern", "risk", "fear", "warning", "cut",
            "negative", "underperform", "miss", "below", "down", "trouble",
            "debt", "fraud", "penalty", "fine", "recession", "layoff",
            "probe", "investigation", "lawsuit"
        ]

        pos_count = 0
        neg_count = 0
        total_headlines = len(headlines)

        for h in headlines:
            title_lower = h["title"].lower()
            pos_hits = sum(1 for w in positive_words if w in title_lower)
            neg_hits = sum(1 for w in negative_words if w in title_lower)
            if pos_hits > neg_hits:
                pos_count += 1
            elif neg_hits > pos_hits:
                neg_count += 1

        # Determine overall sentiment
        if total_headlines == 0:
            sentiment = "NEUTRAL"
            confidence = 0.3
        elif pos_count > neg_count and pos_count >= total_headlines * 0.4:
            sentiment = "POSITIVE"
            confidence = min(pos_count / total_headlines, 0.9)
        elif neg_count > pos_count and neg_count >= total_headlines * 0.4:
            sentiment = "NEGATIVE"
            confidence = min(neg_count / total_headlines, 0.9)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5

        # Build summary
        headline_texts = [h["title"] for h in headlines[:5]]
        summary = f"{symbol_clean} news sentiment: {sentiment} ({confidence:.0%} confidence). "
        summary += f"{pos_count} positive, {neg_count} negative out of {total_headlines} headlines."

        return json.dumps({
            "symbol": symbol_clean,
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": total_headlines - pos_count - neg_count,
            "headline_count": total_headlines,
            "summary": summary,
            "recent_headlines": headline_texts
        })

    except Exception as e:
        logger.warning(f"News sentiment check failed for {symbol_clean}: {e}")
        return json.dumps({
            "symbol": symbol_clean,
            "sentiment": "NEUTRAL",
            "confidence": 0.0,
            "headline_count": 0,
            "summary": f"Could not fetch news for {symbol_clean}: {str(e)}",
            "headlines": []
        })


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
    check_news_sentiment,
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
