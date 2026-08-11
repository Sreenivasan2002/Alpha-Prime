"""
LangChain Tools for Alpha-Prime Trading Agents
These tools are callable by the AI agents to interact with market data,
technical analysis, and broker systems.
"""

import json
import pandas as pd
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
    # Reject non-positive quantities up front (covers BUY and SELL). A negative
    # quantity would otherwise create a short and invert cash flow in the broker.
    try:
        quantity = int(quantity)
    except (TypeError, ValueError):
        return json.dumps({"status": "REJECTED",
                           "error": f"Invalid quantity '{quantity}': must be a positive integer."})
    if quantity < 1:
        return json.dumps({"status": "REJECTED",
                           "error": f"Quantity must be >= 1 (got {quantity})."})

    # === HARD-CODED TRADE VALIDATION (last line of defense) ===
    if action.upper() == "BUY":
        # -1. Trading window check — no new BUYs outside 10:00-14:30
        from alpha_prime.data.market_data import is_trading_window
        if not is_trading_window():
            from datetime import datetime
            import pytz
            now_ist = datetime.now(pytz.timezone("Asia/Kolkata"))
            start_h = settings.market.trading_start_hour
            start_m = settings.market.trading_start_minute
            end_h = settings.market.trading_end_hour
            end_m = settings.market.trading_end_minute
            return json.dumps({
                "status": "REJECTED",
                "error": (
                    f"BUY rejected: Outside trading window "
                    f"({start_h}:{start_m:02d}-{end_h}:{end_m:02d} IST). "
                    f"Current time: {now_ist.strftime('%H:%M')} IST. "
                    f"Wait for trading window to open."
                )
            })

        # 0. Cooldown check - prevent buying stocks that recently lost money
        symbol_clean = clean_symbol(symbol)
        if _check_stock_cooldown(symbol_clean):
            return json.dumps({
                "status": "REJECTED",
                "error": f"BUY rejected: {symbol_clean} has recent trading losses. 3-day cooldown active."
            })

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
                            # For intraday data, strength 0.10-0.30 is normal if signal=BUY
                            # Only block if engine says signal is not BUY
                            if report.overall_signal != "BUY":
                                return json.dumps({
                                    "status": "REJECTED",
                                    "error": f"BUY rejected: {symbol} real-time engine says {report.overall_signal}, not BUY."
                                })
                except Exception as e:
                    logger.warning(f"Pre-trade validation analysis failed for {symbol}: {e}")
                    # Don't block trade if validation itself fails

        except Exception as e:
            logger.warning(f"Pre-trade price validation failed for {symbol}: {e}")

        # === HARD-CODED RISK LIMITS (code-enforced; never trust LLM math) ===
        # Daily-loss circuit breaker + max-open-positions + risk-based sizing.
        # Quantity is CLAMPED here so an LLM over-sizing a trade (the cause of
        # the DIVISLAB/ZYDUSLIFE Rs.17k losses) can no longer blow up the book.
        risk_check = _enforce_risk_limits(
            symbol_clean=clean_symbol(symbol),
            stop_loss=stop_loss,
            requested_qty=quantity,
        )
        if not risk_check["ok"]:
            return json.dumps({"status": "REJECTED", "error": risk_check["reason"]})
        if risk_check["quantity"] != quantity:
            logger.warning(
                f"[place_trade] {clean_symbol(symbol)} quantity clamped "
                f"{quantity} -> {risk_check['quantity']} ({risk_check['note']})"
            )
        quantity = risk_check["quantity"]

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

    from alpha_prime.data.market_data import is_trading_window

    start_h = settings.market.trading_start_hour
    start_m = settings.market.trading_start_minute
    end_h = settings.market.trading_end_hour
    end_m = settings.market.trading_end_minute

    return json.dumps({
        "is_open": is_market_open(),
        "is_pre_market": is_pre_market(),
        "is_trading_window": is_trading_window(),
        "current_time_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "market_hours": "09:15 - 15:30 IST",
        "trading_window": f"{start_h}:{start_m:02d} - {end_h}:{end_m:02d} IST",
        "trading_window_note": f"Only place BUY orders between {start_h}:{start_m:02d} and {end_h}:{end_m:02d} IST. Before {start_h}:{start_m:02d} is too volatile.",
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
    symbol_clean = clean_symbol(symbol)

    # === HARD-CODED SIGNAL VALIDATION ===
    # The LLM CANNOT be trusted to report accurate strength values.
    # We re-run the technical analysis engine and use ITS strength.
    if signal_type.upper() == "BUY":
        try:
            df = market_data.get_intraday_data(symbol_clean, interval="5m", period="5d")
            if df is not None:
                from alpha_prime.data.technical_analysis import technical_analyzer
                report = technical_analyzer.analyze(df, symbol=symbol_clean)
                if report:
                    engine_strength = report.overall_strength
                    engine_signal = report.overall_signal

                    # REJECT if engine says HOLD or SELL
                    if engine_signal != "BUY":
                        logger.warning(
                            f"[save_signal] BLOCKED: LLM said BUY {symbol_clean} "
                            f"but engine says {engine_signal}. Saving as HOLD."
                        )
                        record_signal(
                            symbol=symbol_clean,
                            signal_type="HOLD",
                            strength=engine_strength,
                            agent_name=agent_name,
                            reasoning=f"BLOCKED: LLM recommended BUY but engine signal is {engine_signal} (strength={engine_strength:.3f}). {reasoning}"
                        )
                        return json.dumps({
                            "status": "BLOCKED",
                            "symbol": symbol_clean,
                            "signal": "HOLD",
                            "reason": f"Engine says {engine_signal} (strength={engine_strength:.3f}), not BUY. Signal overridden to HOLD."
                        })

                    # REJECT if engine strength is below minimum threshold
                    # For intraday 5-min data, strengths are naturally lower (0.15-0.30 typical)
                    # The key is: engine says BUY + trend BULLISH + volume confirmed = GO
                    # We use a lower threshold because top gainers already proved momentum
                    min_strength = getattr(settings.trading, 'min_signal_strength', 0.10)
                    if engine_strength < min_strength:
                        logger.warning(
                            f"[save_signal] BLOCKED: {symbol_clean} engine strength "
                            f"{engine_strength:.3f} < minimum {min_strength}. LLM claimed {strength:.3f}."
                        )
                        record_signal(
                            symbol=symbol_clean,
                            signal_type="HOLD",
                            strength=engine_strength,
                            agent_name=agent_name,
                            reasoning=f"BLOCKED: Engine strength {engine_strength:.3f} < minimum {min_strength}. LLM claimed {strength:.3f}. {reasoning}"
                        )
                        return json.dumps({
                            "status": "BLOCKED",
                            "symbol": symbol_clean,
                            "signal": "HOLD",
                            "reason": f"Engine strength {engine_strength:.3f} is below minimum {min_strength}. LLM claimed {strength:.3f}."
                        })

                    # REJECT if trend is BEARISH
                    if report.trend and report.trend.upper() == "BEARISH":
                        logger.warning(
                            f"[save_signal] BLOCKED: {symbol_clean} trend is BEARISH."
                        )
                        record_signal(
                            symbol=symbol_clean,
                            signal_type="HOLD",
                            strength=engine_strength,
                            agent_name=agent_name,
                            reasoning=f"BLOCKED: Trend is BEARISH. Cannot save BUY signal. {reasoning}"
                        )
                        return json.dumps({
                            "status": "BLOCKED",
                            "symbol": symbol_clean,
                            "signal": "HOLD",
                            "reason": f"Trend is BEARISH. Cannot buy in a downtrend."
                        })

                    # Volume confirmation: warn but don't hard-block
                    # The engine already applies a 20% strength penalty for unconfirmed volume
                    # And the scanner checked daily volume via get_top_gainers
                    if not report.volume_confirmed:
                        logger.info(
                            f"[save_signal] WARNING: {symbol_clean} 5-min volume not confirmed. "
                            f"Engine already applied 20% strength penalty. Proceeding with caution."
                        )

                    # OVERRIDE the LLM's strength with the engine's strength
                    if abs(strength - engine_strength) > 0.05:
                        logger.warning(
                            f"[save_signal] CORRECTED: {symbol_clean} LLM strength "
                            f"{strength:.3f} -> engine strength {engine_strength:.3f}"
                        )
                    strength = engine_strength

                    # Check for recent losses on this stock (cooldown)
                    recent_losses = _check_stock_cooldown(symbol_clean)
                    if recent_losses:
                        logger.warning(
                            f"[save_signal] BLOCKED: {symbol_clean} lost money recently. Cooldown active."
                        )
                        record_signal(
                            symbol=symbol_clean,
                            signal_type="HOLD",
                            strength=engine_strength,
                            agent_name=agent_name,
                            reasoning=f"BLOCKED: Stock has recent losses. Cooldown period active. {reasoning}"
                        )
                        return json.dumps({
                            "status": "BLOCKED",
                            "symbol": symbol_clean,
                            "signal": "HOLD",
                            "reason": f"{symbol_clean} has recent trading losses. Cooldown period - avoid this stock."
                        })

        except Exception as e:
            logger.warning(f"[save_signal] Validation failed for {symbol_clean}: {e}")
            # If validation itself fails, block the signal to be safe
            return json.dumps({
                "status": "BLOCKED",
                "symbol": symbol_clean,
                "signal": "HOLD",
                "reason": f"Could not validate signal: {e}"
            })

    record_signal(
        symbol=symbol_clean,
        signal_type=signal_type,
        strength=strength,
        agent_name=agent_name,
        reasoning=reasoning
    )
    return json.dumps({"status": "Signal saved", "symbol": symbol_clean, "signal": signal_type, "verified_strength": round(strength, 3)})


def _check_stock_cooldown(symbol_clean: str) -> bool:
    """Check if a stock has recent losses (within last 3 trading days).
    Returns True if the stock should be avoided (cooldown active).
    """
    try:
        from alpha_prime.core.database import get_trades
        mode = getattr(settings.trading, "mode", "paper")
        recent_trades = get_trades(limit=50, symbol=symbol_clean, trading_mode=mode)

        if not recent_trades:
            return False

        # Check last 3 days of trades for this stock
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=3)

        for trade in recent_trades:
            try:
                trade_time = datetime.fromisoformat(str(trade.get("timestamp", "")))
            except (ValueError, TypeError):
                continue

            if trade_time < cutoff:
                break  # Older than 3 days, stop checking

            # If there was a SELL with a loss (pnl < 0), cooldown is active
            if trade.get("action") == "SELL" and trade.get("pnl", 0) < 0:
                logger.info(
                    f"[cooldown] {symbol_clean} has recent loss "
                    f"(pnl={trade.get('pnl')}) on {trade.get('timestamp')}. Cooldown active."
                )
                return True

            # Also check if we bought recently and the stock dropped
            if trade.get("action") == "BUY":
                buy_price = trade.get("price", 0)
                if buy_price > 0:
                    from alpha_prime.data.market_data import market_data as md
                    current_price = md.get_live_price(symbol_clean)
                    if current_price and current_price < buy_price * 0.99:
                        # Stock is down >1% from our recent buy — avoid
                        logger.info(
                            f"[cooldown] {symbol_clean} bought at {buy_price}, "
                            f"now at {current_price} ({((current_price/buy_price)-1)*100:.1f}%). Cooldown."
                        )
                        return True

    except Exception as e:
        logger.warning(f"[cooldown] Error checking cooldown for {symbol_clean}: {e}")

    return False


def _enforce_risk_limits(symbol_clean: str, stop_loss: float, requested_qty: int) -> dict:
    """Code-enforced risk gate for BUY orders — does NOT trust LLM arithmetic.

    Enforces, in order:
      0. Quantity sanity: must be a positive integer (a negative qty would create a
         SHORT and CREDIT cash in the broker).
      1. Daily-loss circuit breaker: block ALL new BUYs if today's P&L <= -max_daily_loss_pct.
      2. Max open positions: block a NEW symbol once we hold max_open_positions.
      3. Position sizing on the TOTAL (existing + new) position, then subtract what
         we already hold so repeated top-ups can't stack past the cap:
           (entry - stop_loss) * total_qty <= capital * risk_per_trade_pct/100   (primary)
           total_qty * entry            <= capital * max_position_size_pct/100    (notional)
         Both caps use REAL capital (not margin buying power) so a single position's
         loss-at-stop AND overnight-gap exposure stay bounded regardless of leverage.

    Returns {"ok": bool, "quantity": int, "reason": str, "note": str}.
    """
    t = settings.trading

    # 0. Quantity must be a positive whole number
    try:
        requested_qty = int(requested_qty)
    except (TypeError, ValueError):
        return {"ok": False, "quantity": 0, "note": "",
                "reason": f"Invalid quantity '{requested_qty}': must be a positive integer."}
    if requested_qty < 1:
        return {"ok": False, "quantity": 0, "note": "",
                "reason": f"Quantity must be >= 1 (got {requested_qty})."}

    # 1. Daily-loss circuit breaker
    try:
        from alpha_prime.core.broker import get_today_pnl_pct
        today_pnl = get_today_pnl_pct()
        if today_pnl <= -abs(t.max_daily_loss_pct):
            return {
                "ok": False, "quantity": 0, "note": "",
                "reason": (
                    f"Daily loss limit hit: today's P&L {today_pnl:.2f}% "
                    f"<= -{t.max_daily_loss_pct}%. No new BUY trades today."
                ),
            }
    except Exception as e:
        logger.warning(f"[risk] Daily P&L check failed (continuing): {e}")

    # Need a reliable current price to size the position
    current_price = market_data.get_live_price(symbol_clean)
    if not current_price or current_price <= 0:
        return {"ok": False, "quantity": 0, "note": "",
                "reason": f"Could not fetch price for {symbol_clean} to size position."}

    # A long's stop-loss must sit below the entry price
    if stop_loss and stop_loss >= current_price:
        return {"ok": False, "quantity": 0, "note": "",
                "reason": (f"Stop-loss ({stop_loss:.2f}) must be below entry "
                           f"({current_price:.2f}) for a BUY.")}

    # 2. Max open positions (and capture any existing holding for top-up math)
    held_qty = 0
    try:
        positions = get_broker().get_positions() or {}
        held_qty = int(positions.get(symbol_clean, {}).get("quantity", 0) or 0)
        if symbol_clean not in positions and len(positions) >= t.max_open_positions:
            return {"ok": False, "quantity": 0, "note": "",
                    "reason": (f"Max open positions ({t.max_open_positions}) reached. "
                               f"Cannot open a new position in {symbol_clean}.")}
    except Exception as e:
        logger.warning(f"[risk] Open-positions check failed (continuing): {e}")

    # 3. Position sizing — caps are on the TOTAL position; subtract what we hold.
    #    Uses the shared pure sizing function (same logic the backtester runs).
    from alpha_prime.core.risk import compute_max_quantity
    max_total_qty, binding = compute_max_quantity(
        capital=t.capital, price=current_price, stop_loss=stop_loss,
        risk_per_trade_pct=t.risk_per_trade_pct,
        max_position_size_pct=t.max_position_size_pct,
    )
    max_new_qty = max_total_qty - held_qty
    if max_new_qty < 1:
        return {"ok": False, "quantity": 0, "note": "",
                "reason": (f"{symbol_clean}: already hold {held_qty} share(s); total cap "
                           f"{max_total_qty} ({binding}) reached — no room to add.")}

    final_qty = min(requested_qty, max_new_qty)
    note = ("capped by {b} (total cap {tc}, held {h}, max new {mn} @ {p:.2f})".format(
        b=binding, tc=max_total_qty, h=held_qty, mn=max_new_qty, p=current_price)
        if final_qty < requested_qty else "")
    return {"ok": True, "quantity": final_qty, "reason": "", "note": note}


@tool
def get_top_gainers(min_gain_pct: float = 1.0, max_gain_pct: float = 5.0) -> str:
    """Scan NIFTY 100 stocks and return today's top gainers - stocks already moving up today.
    These are momentum candidates: stocks with confirmed upward movement.
    Volume is time-adjusted so early-morning scans work correctly.
    Args:
        min_gain_pct: Minimum daily gain percentage to qualify (default: 1.0%)
        max_gain_pct: Maximum daily gain percentage (default: 5.0%, avoid stocks that already moved too much)
    """
    import yfinance as yf
    from datetime import datetime
    import pytz

    IST = pytz.timezone("Asia/Kolkata")
    now = datetime.now(IST)

    # Calculate what fraction of the trading day has elapsed (9:15 to 15:30 = 375 min)
    market_open_min = 9 * 60 + 15   # 555 min from midnight
    market_close_min = 15 * 60 + 30  # 930 min from midnight
    current_min = now.hour * 60 + now.minute
    total_market_minutes = market_close_min - market_open_min  # 375

    elapsed_minutes = max(current_min - market_open_min, 1)
    day_fraction = min(elapsed_minutes / total_market_minutes, 1.0)

    logger.info(f"[top_gainers] Market time elapsed: {elapsed_minutes} min ({day_fraction:.0%} of day)")

    # Use NIFTY 100 for speed
    all_stocks = market_data.get_nifty100_stocks()
    yf_symbols = [f"{s}.NS" for s in all_stocks]

    gainers = []

    try:
        logger.info(f"[top_gainers] Scanning {len(yf_symbols)} stocks for today's gainers...")

        # Download 2 days of daily data to compare
        data = yf.download(yf_symbols, period="2d", interval="1d", progress=False, group_by="ticker")

        if data.empty:
            return json.dumps({"error": "Could not fetch market data", "gainers": []})

        for sym_nse in yf_symbols:
            sym_clean = sym_nse.replace(".NS", "")
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    stock_data = data[sym_nse] if sym_nse in data.columns.get_level_values(0) else None
                    if stock_data is None or stock_data.empty or len(stock_data) < 2:
                        continue
                    prev_close = float(stock_data["Close"].iloc[-2])
                    current_price = float(stock_data["Close"].iloc[-1])
                    current_volume = float(stock_data["Volume"].iloc[-1])
                    prev_volume = float(stock_data["Volume"].iloc[-2])
                else:
                    continue

                if prev_close <= 0:
                    continue

                gain_pct = ((current_price - prev_close) / prev_close) * 100

                if min_gain_pct <= gain_pct <= max_gain_pct:
                    # TIME-ADJUSTED volume ratio:
                    # If only 10% of the day has passed, today's volume should be ~10% of yesterday's
                    # So we normalize: (today_vol / day_fraction) / yesterday_vol
                    # This gives us the PROJECTED full-day volume pace
                    if prev_volume > 0 and day_fraction > 0:
                        projected_full_day_vol = current_volume / day_fraction
                        vol_ratio = round(projected_full_day_vol / prev_volume, 2)
                    else:
                        vol_ratio = 1.0

                    # Volume confirmed if projected pace >= 80% of yesterday
                    # (lower threshold than 1.2x because we're projecting)
                    volume_confirmed = vol_ratio >= 0.8

                    gainers.append({
                        "symbol": sym_clean,
                        "current_price": round(current_price, 2),
                        "prev_close": round(prev_close, 2),
                        "gain_pct": round(gain_pct, 2),
                        "volume_ratio": vol_ratio,
                        "volume_confirmed": volume_confirmed,
                        "day_fraction": round(day_fraction, 2),
                    })

            except Exception:
                continue

        # Sort by: volume_confirmed first, then gain % descending
        gainers.sort(key=lambda x: (x["volume_confirmed"], x["gain_pct"]), reverse=True)

        top = gainers[:10]

        logger.info(f"[top_gainers] Found {len(gainers)} gainers, returning top {len(top)}")

        return json.dumps({
            "total_gainers": len(gainers),
            "top_gainers": top,
            "scan_universe": "NIFTY 100",
            "filter": f"{min_gain_pct}% to {max_gain_pct}% daily gain",
            "market_elapsed": f"{elapsed_minutes} min ({day_fraction:.0%} of day)"
        })

    except Exception as e:
        logger.error(f"[top_gainers] Error: {e}")
        return json.dumps({"error": str(e), "gainers": []})


def _fetch_news_headlines(symbol_clean: str) -> list:
    """Fetch recent news headlines from Google News RSS for a given stock symbol.
    Returns list of dicts with 'title' and 'date' keys.
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    # Map common NSE symbols to company names for better search
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
        "HINDUNILVR": "Hindustan Unilever",
        "ASIANPAINT": "Asian Paints",
        "TITAN": "Titan Company",
        "ULTRACEMCO": "UltraTech Cement",
        "NESTLEIND": "Nestle India",
        "JSWSTEEL": "JSW Steel",
        "TATACONSUM": "Tata Consumer Products",
        "ONGC": "ONGC",
        "COALINDIA": "Coal India",
        "BPCL": "BPCL",
        "TECHM": "Tech Mahindra",
        "DRREDDY": "Dr Reddys Laboratories",
        "CIPLA": "Cipla",
        "DIVISLAB": "Divis Laboratories",
        "EICHERMOT": "Eicher Motors",
        "HEROMOTOCO": "Hero MotoCorp",
        "BAJAJFINSV": "Bajaj Finserv",
        "M&M": "Mahindra and Mahindra",
        "GRASIM": "Grasim Industries",
        "APOLLOHOSP": "Apollo Hospitals",
        "HINDALCO": "Hindalco Industries",
    }

    search_term = company_names.get(symbol_clean, symbol_clean)
    query = f"{search_term} stock NSE"

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
        for item in channel.findall("item")[:10]:
            title = item.find("title")
            pub_date = item.find("pubDate")
            if title is not None and title.text:
                headlines.append({
                    "title": title.text,
                    "date": pub_date.text if pub_date is not None else ""
                })
    return headlines


def _ai_sentiment_analysis(symbol_clean: str, headlines: list) -> dict:
    """Use GPT-4o-mini to analyze financial sentiment of headlines.
    Returns dict with sentiment, confidence, reasoning, per-headline analysis.
    Much more accurate than keyword matching — understands financial context,
    negation, sarcasm, and nuance.
    """
    from openai import OpenAI

    headline_text = "\n".join(
        [f"{i+1}. {h['title']}" for i, h in enumerate(headlines)]
    )

    prompt = f"""You are a financial sentiment analyst specializing in the Indian stock market (NSE).
Analyze these recent news headlines for {symbol_clean} and determine the overall sentiment for INTRADAY TRADING.

HEADLINES:
{headline_text}

INSTRUCTIONS:
1. Classify EACH headline as POSITIVE, NEGATIVE, or NEUTRAL for the stock price
2. Consider financial context: "profit booking" is NEGATIVE (selling pressure), "earnings beat" is POSITIVE
3. Watch for negation: "not expected to rise" is NEGATIVE
4. Consider recency and impact: recent, high-impact news matters more
5. Give an OVERALL sentiment and confidence score

RESPOND IN EXACTLY THIS JSON FORMAT (no other text):
{{
  "overall_sentiment": "POSITIVE" or "NEGATIVE" or "NEUTRAL",
  "confidence": 0.0 to 1.0,
  "positive_count": number,
  "negative_count": number,
  "neutral_count": number,
  "reasoning": "1-2 sentence explanation",
  "impact_on_trading": "BULLISH_BOOST" or "BEARISH_WARNING" or "NO_IMPACT",
  "key_headline": "the most impactful headline text"
}}"""

    try:
        client = OpenAI(api_key=settings.openai.api_key)
        response = client.chat.completions.create(
            model=settings.openai.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            timeout=15,
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON from response — handle markdown code blocks
        if result_text.startswith("```"):
            # Remove ```json and ``` markers
            lines = result_text.split("\n")
            result_text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        return json.loads(result_text)

    except Exception as e:
        logger.warning(f"AI sentiment analysis failed for {symbol_clean}: {e}")
        return None


@tool
def check_news_sentiment(symbol: str) -> str:
    """Check recent news sentiment for a stock using AI-powered analysis (GPT-4o-mini) on Google News headlines. Returns sentiment (POSITIVE/NEGATIVE/NEUTRAL), confidence score, reasoning, and trading impact.
    Args:
        symbol: Stock symbol like RELIANCE, TCS, INFY
    """
    symbol_clean = clean_symbol(symbol)

    try:
        # Step 1: Fetch headlines from Google News RSS
        headlines = _fetch_news_headlines(symbol_clean)

        if not headlines:
            return json.dumps({
                "symbol": symbol_clean,
                "sentiment": "NEUTRAL",
                "confidence": 0.3,
                "headline_count": 0,
                "summary": f"No recent news found for {symbol_clean}",
                "impact_on_trading": "NO_IMPACT",
                "headlines": []
            })

        headline_texts = [h["title"] for h in headlines[:5]]

        # Step 2: AI-powered sentiment analysis using GPT-4o-mini
        ai_result = _ai_sentiment_analysis(symbol_clean, headlines)

        if ai_result:
            # AI analysis succeeded — use its results
            sentiment = ai_result.get("overall_sentiment", "NEUTRAL")
            confidence = float(ai_result.get("confidence", 0.5))
            pos_count = int(ai_result.get("positive_count", 0))
            neg_count = int(ai_result.get("negative_count", 0))
            reasoning = ai_result.get("reasoning", "")
            impact = ai_result.get("impact_on_trading", "NO_IMPACT")
            key_headline = ai_result.get("key_headline", "")

            summary = (
                f"{symbol_clean} news sentiment: {sentiment} "
                f"({confidence:.0%} confidence). "
                f"{pos_count} positive, {neg_count} negative out of "
                f"{len(headlines)} headlines. {reasoning}"
            )

            return json.dumps({
                "symbol": symbol_clean,
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "positive_count": pos_count,
                "negative_count": neg_count,
                "neutral_count": len(headlines) - pos_count - neg_count,
                "headline_count": len(headlines),
                "summary": summary,
                "reasoning": reasoning,
                "impact_on_trading": impact,
                "key_headline": key_headline,
                "recent_headlines": headline_texts,
                "method": "ai_gpt4omini"
            })
        else:
            # AI failed — fall back to simple keyword matching
            logger.info(f"Falling back to keyword sentiment for {symbol_clean}")
            return _keyword_sentiment_fallback(symbol_clean, headlines)

    except Exception as e:
        logger.warning(f"News sentiment check failed for {symbol_clean}: {e}")
        return json.dumps({
            "symbol": symbol_clean,
            "sentiment": "NEUTRAL",
            "confidence": 0.0,
            "headline_count": 0,
            "summary": f"Could not fetch news for {symbol_clean}: {str(e)}",
            "impact_on_trading": "NO_IMPACT",
            "headlines": [],
            "method": "error"
        })


def _keyword_sentiment_fallback(symbol_clean: str, headlines: list) -> str:
    """Fallback keyword-based sentiment when AI analysis is unavailable.
    Less accurate but works without API calls.
    """
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

    headline_texts = [h["title"] for h in headlines[:5]]
    summary = (
        f"{symbol_clean} news sentiment: {sentiment} "
        f"({confidence:.0%} confidence). "
        f"{pos_count} positive, {neg_count} negative out of "
        f"{total_headlines} headlines."
    )

    return json.dumps({
        "symbol": symbol_clean,
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": total_headlines - pos_count - neg_count,
        "headline_count": total_headlines,
        "summary": summary,
        "impact_on_trading": "NO_IMPACT",
        "recent_headlines": headline_texts,
        "method": "keyword_fallback"
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
    get_top_gainers,
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
