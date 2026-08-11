"""
Groww API Broker Integration for Alpha-Prime
Handles order placement, portfolio management, and account operations.
Supports both paper trading and live trading modes.

Groww API Auth Flow:
  1. Call GrowwAPI.get_access_token(api_key, secret=secret_key) -> returns {"accessToken": "..."}
  2. Create client: GrowwAPI(token=access_token)
  3. Use client methods for trading
"""

import json
from datetime import datetime
from typing import Optional, Dict, List
from loguru import logger
import pytz

from alpha_prime.core.config import settings
from alpha_prime.core.database import (
    record_trade, get_open_positions, save_portfolio_snapshot,
    log_agent_activity, set_system_state, get_system_state
)
from alpha_prime.data.market_data import market_data, clean_symbol

IST = pytz.timezone("Asia/Kolkata")


class PaperTradingBroker:
    """Paper trading broker for simulation/testing"""

    def __init__(self):
        self.capital = settings.trading.effective_capital  # Use margin-adjusted capital
        self.actual_capital = settings.trading.capital       # Real cash
        self.positions: Dict[str, dict] = {}
        self.available_cash = self.capital
        self._load_state()

    def _load_state(self):
        """Load existing positions from database (paper mode only)"""
        try:
            open_pos = get_open_positions(trading_mode="paper")
            for pos in open_pos:
                symbol = pos["symbol"]
                qty = pos["net_qty"]
                avg_price = pos["avg_buy_price"] or 0
                if qty > 0:
                    self.positions[symbol] = {
                        "quantity": qty,
                        "avg_price": avg_price,
                        "invested": qty * avg_price
                    }
                    self.available_cash -= qty * avg_price
        except Exception as e:
            logger.warning(f"Could not load paper trading state: {e}")

    def place_order(self, symbol: str, action: str, quantity: int,
                    order_type: str = "MARKET", price: float = None,
                    stop_loss: float = None, target: float = None,
                    rationale: str = "", agent_name: str = "") -> Dict:
        """Place a paper trade order"""
        symbol = clean_symbol(symbol)
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            quantity = 0
        if quantity <= 0:
            return {"status": "FAILED", "error": f"Invalid quantity {quantity}: must be positive."}
        current_price = price or market_data.get_live_price(symbol)

        if current_price is None:
            return {"status": "FAILED", "error": f"Could not get price for {symbol}"}

        realized_pnl = 0.0

        if action.upper() == "BUY":
            total_cost = current_price * quantity
            if total_cost > self.available_cash:
                return {
                    "status": "FAILED",
                    "error": f"Insufficient funds. Need Rs.{total_cost:.2f}, have Rs.{self.available_cash:.2f}"
                }

            if symbol in self.positions:
                existing = self.positions[symbol]
                total_qty = existing["quantity"] + quantity
                total_invested = existing["invested"] + total_cost
                self.positions[symbol] = {
                    "quantity": total_qty,
                    "avg_price": total_invested / total_qty,
                    "invested": total_invested
                }
            else:
                self.positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": current_price,
                    "invested": total_cost
                }

            self.available_cash -= total_cost

        elif action.upper() == "SELL":
            if symbol not in self.positions:
                return {"status": "FAILED", "error": f"No position in {symbol}"}
            if self.positions[symbol]["quantity"] < quantity:
                return {
                    "status": "FAILED",
                    "error": f"Insufficient shares. Have {self.positions[symbol]['quantity']}, trying to sell {quantity}"
                }

            sell_value = current_price * quantity
            avg_buy = self.positions[symbol]["avg_price"]
            # Realized P&L for this sell (FIFO/average-cost): (exit - avg_entry) * qty
            realized_pnl = (current_price - avg_buy) * quantity

            self.positions[symbol]["quantity"] -= quantity
            self.positions[symbol]["invested"] -= avg_buy * quantity

            if self.positions[symbol]["quantity"] == 0:
                del self.positions[symbol]

            self.available_cash += sell_value
        else:
            return {"status": "FAILED", "error": f"Unknown action: {action}"}

        trade_id = record_trade(
            symbol=symbol, action=action.upper(), quantity=quantity,
            price=current_price, order_type=order_type,
            stop_loss=stop_loss, target=target,
            rationale=rationale, agent_name=agent_name,
            order_id=f"PAPER-{datetime.now(IST).strftime('%Y%m%d%H%M%S')}",
            status="EXECUTED",
            trading_mode="paper",
            pnl=realized_pnl,
        )

        result = {
            "status": "EXECUTED",
            "trade_id": trade_id,
            "symbol": symbol,
            "action": action.upper(),
            "quantity": quantity,
            "price": current_price,
            "total_value": current_price * quantity,
            "realized_pnl": round(realized_pnl, 2),
            "available_cash": round(self.available_cash, 2),
            "mode": "PAPER"
        }

        logger.info(f"Paper trade: {action} {quantity} {symbol} @ Rs.{current_price}")
        log_agent_activity(agent_name or "broker", "trade",
                          f"{action} {quantity} {symbol} @ Rs.{current_price}", result)
        return result

    def get_positions(self) -> Dict[str, dict]:
        """Get current positions with live values"""
        positions_with_value = {}
        for symbol, pos in self.positions.items():
            current_price = market_data.get_live_price(symbol) or pos["avg_price"]
            current_value = current_price * pos["quantity"]
            pnl = current_value - pos["invested"]
            pnl_pct = (pnl / pos["invested"] * 100) if pos["invested"] > 0 else 0

            positions_with_value[symbol] = {
                "quantity": pos["quantity"],
                "avg_price": round(pos["avg_price"], 2),
                "current_price": round(current_price, 2),
                "invested": round(pos["invested"], 2),
                "current_value": round(current_value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2)
            }
        return positions_with_value

    def get_portfolio_summary(self) -> Dict:
        """Get complete portfolio summary"""
        positions = self.get_positions()
        total_invested = sum(p["invested"] for p in positions.values())
        total_current = sum(p["current_value"] for p in positions.values())
        total_pnl = total_current - total_invested
        total_value = self.available_cash + total_current

        summary = {
            "capital": self.capital,
            "available_cash": round(self.available_cash, 2),
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current, 2),
            "total_portfolio_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / self.capital * 100), 2) if self.capital > 0 else 0,
            "num_positions": len(positions),
            "positions": positions,
            "mode": "PAPER"
        }

        save_portfolio_snapshot(
            capital=self.capital, invested=total_invested,
            available=self.available_cash, total_value=total_value,
            daily_pnl=total_pnl, total_pnl=total_pnl, positions=positions
        )
        return summary

    def get_user_profile(self) -> Dict:
        return {"name": "Paper Trader", "mode": "PAPER", "capital": self.capital}


class GrowwBroker:
    """Live trading broker using Groww API"""

    def __init__(self):
        self._client = None
        self._initialized = False
        self._access_token = None

    def _ensure_client(self):
        """Initialize Groww API client with proper auth flow.

        Groww auth flow:
          1. GrowwAPI.get_access_token(api_key, secret=secret_key)
             -> returns the JWT token as a STRING (not a dict)
             (source: growwapi returns response.json()["token"])
          2. GrowwAPI(token=jwt_string)
        """
        if self._initialized and self._client:
            return

        api_key = settings.groww.api_key
        secret_key = settings.groww.secret_key

        if not api_key or api_key == "your_groww_api_key_here":
            raise ValueError("Groww API Key not configured. Please update in Settings.")
        if not secret_key or secret_key == "your_groww_secret_key_here":
            raise ValueError("Groww Secret Key not configured. Please update in Settings.")

        try:
            from growwapi import GrowwAPI

            # Step 1: Get access token using API key + secret
            # NOTE: get_access_token() returns the JWT token STRING directly,
            # NOT a dict. The SDK does: return response.json()["token"]
            logger.info("Authenticating with Groww API...")
            token_response = GrowwAPI.get_access_token(
                api_key=api_key,
                secret=secret_key
            )

            # Handle both possible return types for safety
            if isinstance(token_response, str):
                # Direct JWT token string (this is the actual behavior)
                self._access_token = token_response
            elif isinstance(token_response, dict):
                # Dict response (fallback if SDK changes)
                self._access_token = (
                    token_response.get("accessToken")
                    or token_response.get("access_token")
                    or token_response.get("token")
                )
            else:
                raise ValueError(f"Unexpected token response type: {type(token_response)}")

            if not self._access_token:
                raise ValueError("Could not extract access token from response")

            # Step 2: Create client with access token
            self._client = GrowwAPI(token=self._access_token)
            self._initialized = True

            set_system_state("groww_auth_status", "authenticated")
            set_system_state("groww_auth_time", datetime.now(IST).isoformat())
            logger.info("Groww API authenticated successfully")

        except Exception as e:
            self._initialized = False
            self._client = None
            set_system_state("groww_auth_status", f"failed: {str(e)}")
            logger.error(f"Groww API auth failed: {e}")
            raise

    def reinitialize(self, api_key: str = None, secret_key: str = None):
        """Reinitialize with new API keys (for daily key rotation)"""
        self._initialized = False
        self._client = None
        self._access_token = None
        if api_key and secret_key:
            settings.update_groww_keys(api_key, secret_key)
        self._ensure_client()

    def get_user_profile(self) -> Dict:
        """Get user profile from Groww"""
        self._ensure_client()
        try:
            profile = self._client.get_user_profile()
            return profile if isinstance(profile, dict) else {"raw": str(profile)}
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return {"error": str(e)}

    def get_funds(self) -> Dict:
        """Get available funds/margin from Groww"""
        self._ensure_client()
        try:
            margin = self._client.get_available_margin_details()
            return margin if isinstance(margin, dict) else {"raw": str(margin)}
        except Exception as e:
            logger.error(f"Error fetching funds: {e}")
            return {"error": str(e)}

    def get_holdings(self) -> Dict:
        """Get holdings (delivery positions) from Groww"""
        self._ensure_client()
        try:
            holdings = self._client.get_holdings_for_user()
            return holdings if isinstance(holdings, dict) else {"raw": str(holdings)}
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            return {"error": str(e)}

    def place_order(self, symbol: str, action: str, quantity: int,
                    order_type: str = "MARKET", price: float = None,
                    stop_loss: float = None, target: float = None,
                    rationale: str = "", agent_name: str = "") -> Dict:
        """Place a live order via Groww API"""
        self._ensure_client()
        symbol = clean_symbol(symbol)
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            quantity = 0
        if quantity <= 0:
            return {"status": "FAILED", "error": f"Invalid quantity {quantity}: must be positive."}

        try:
            transaction_type = "BUY" if action.upper() == "BUY" else "SELL"

            # For a SELL we must know the average entry price BEFORE the order
            # fills, because once it executes the position shrinks (or vanishes)
            # and the cost basis is no longer recoverable from get_positions().
            # This entry price is used below to compute realized P&L so that the
            # trades table, the 3-day cooldown, and the daily-loss breaker all
            # see accurate live P&L (paper mode already does this natively).
            avg_buy_price = None
            if transaction_type == "SELL":
                try:
                    pre_positions = self.get_positions()
                    pos = pre_positions.get(symbol)
                    if pos:
                        avg_buy_price = pos.get("avg_price")
                except Exception as pe:
                    logger.warning(
                        f"Could not read entry price for {symbol} before SELL "
                        f"(realized P&L will be recorded as 0): {pe}"
                    )

            # Build order params matching Groww API signature
            # IMPORTANT: product must be "MIS" for intraday, "CNC" for delivery
            # Groww constants: PRODUCT_MIS="MIS", PRODUCT_CNC="CNC"
            order_params = {
                "validity": "DAY",
                "exchange": "NSE",
                "order_type": order_type,          # MARKET or LIMIT
                "product": "MIS",                  # MIS=intraday, CNC=delivery
                "quantity": quantity,
                "segment": "CASH",
                "trading_symbol": symbol,
                "transaction_type": transaction_type,
            }

            if order_type == "LIMIT" and price:
                order_params["price"] = price
            if stop_loss and order_type in ("SL", "SL_M"):
                order_params["trigger_price"] = stop_loss

            # Generate a unique reference ID
            ref_id = f"AP-{datetime.now(IST).strftime('%Y%m%d%H%M%S')}"
            order_params["order_reference_id"] = ref_id

            logger.info(f"Placing Groww order: {transaction_type} {quantity} {symbol} (product=MIS)")
            response = self._client.place_order(**order_params)

            order_id = ""
            if isinstance(response, dict):
                # Groww returns groww_order_id (snake_case)
                order_id = response.get("groww_order_id",
                            response.get("growwOrderId",
                            response.get("orderId",
                            response.get("order_id", ""))))

            # Get execution price - check order status for fill price
            exec_price = price or (market_data.get_live_price(symbol) or 0)
            order_status = "PLACED"

            # Try to get actual fill price from order status
            if order_id:
                try:
                    import time
                    time.sleep(1)  # Brief wait for execution
                    status_resp = self._client.get_order_status(
                        segment="CASH", groww_order_id=str(order_id)
                    )
                    if isinstance(status_resp, dict):
                        order_status = status_resp.get("order_status", "PLACED")
                        filled_qty = status_resp.get("filled_quantity", 0)
                        avg_price = status_resp.get("average_price", 0)
                        if avg_price and avg_price > 0:
                            exec_price = avg_price
                        if order_status == "EXECUTED":
                            logger.info(f"Order EXECUTED: {quantity} {symbol} @ Rs.{exec_price}")
                except Exception as se:
                    logger.debug(f"Could not check order status: {se}")

            # Compute realized P&L for a SELL: (exit - avg_entry) * qty.
            # Only when we have a valid entry price and the order was not
            # rejected/cancelled — a non-filled sell must not record a phantom
            # gain/loss that would wrongly trip the cooldown or daily breaker.
            realized_pnl = 0.0
            if transaction_type == "SELL":
                rejected = str(order_status).upper() in ("REJECTED", "CANCELLED", "FAILED")
                if avg_buy_price and exec_price and not rejected:
                    realized_pnl = (exec_price - avg_buy_price) * quantity
                    logger.info(
                        f"Realized P&L for SELL {quantity} {symbol}: "
                        f"(exit {exec_price} - entry {avg_buy_price}) x {quantity} "
                        f"= Rs.{realized_pnl:.2f}"
                    )
                elif not rejected:
                    logger.warning(
                        f"No entry price available for {symbol}; live SELL P&L "
                        f"recorded as 0 (realized-loss cooldown disabled for this trade)."
                    )

            trade_id = record_trade(
                symbol=symbol, action=action.upper(), quantity=quantity,
                price=exec_price, order_type=order_type,
                stop_loss=stop_loss, target=target,
                rationale=rationale, agent_name=agent_name,
                order_id=str(order_id), status=order_status,
                trading_mode="live",
                pnl=realized_pnl,
            )

            result = {
                "status": order_status,
                "trade_id": trade_id,
                "order_id": str(order_id),
                "symbol": symbol,
                "action": action.upper(),
                "quantity": quantity,
                "price": exec_price,
                "realized_pnl": round(realized_pnl, 2),
                "mode": "LIVE",
                "groww_response": response if isinstance(response, dict) else str(response)
            }

            logger.info(f"Order {order_status}: {action} {quantity} {symbol} @ Rs.{exec_price}, order_id={order_id}")
            log_agent_activity(agent_name or "broker", "trade",
                             f"LIVE {action} {quantity} {symbol} @ Rs.{exec_price} [{order_status}]", result)
            return result

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    def get_positions(self) -> Dict[str, dict]:
        """Get positions from Groww.

        Groww actual response format per position:
          trading_symbol, segment, exchange, product,
          credit_quantity, credit_price, debit_quantity, debit_price,
          quantity (net), net_price, realised_pnl,
          carry_forward_credit_quantity, carry_forward_credit_price,
          carry_forward_debit_quantity, carry_forward_debit_price,
          net_carry_forward_quantity, net_carry_forward_price
        """
        self._ensure_client()
        try:
            positions_data = self._client.get_positions_for_user(segment="CASH")
            result = {}

            if isinstance(positions_data, dict):
                positions_list = positions_data.get("positions", positions_data.get("data", []))
                if isinstance(positions_list, list):
                    for pos in positions_list:
                        symbol = pos.get("trading_symbol", pos.get("tradingSymbol", ""))
                        qty = int(pos.get("quantity", 0))

                        # Groww uses net_price / credit_price for avg buy price
                        avg = float(pos.get("net_price", 0)
                                   or pos.get("credit_price", 0)
                                   or pos.get("averagePrice", 0)
                                   or pos.get("average_price", 0))

                        # LTP may not be in position data - fetch live
                        ltp = float(pos.get("lastTradedPrice", 0)
                                   or pos.get("ltp", 0))
                        if ltp == 0 and symbol:
                            ltp = market_data.get_live_price(symbol) or avg

                        realised_pnl = float(pos.get("realised_pnl", 0)
                                            or pos.get("realisedPnl", 0))

                        if qty != 0:
                            invested = abs(qty) * avg
                            current_value = abs(qty) * ltp
                            unrealised_pnl = current_value - invested

                            result[symbol] = {
                                "quantity": abs(qty),
                                "avg_price": round(avg, 2),
                                "current_price": round(ltp, 2),
                                "invested": round(invested, 2),
                                "current_value": round(current_value, 2),
                                "pnl": round(unrealised_pnl, 2),
                                "pnl_pct": round((unrealised_pnl / invested * 100), 2) if invested > 0 else 0,
                                "realised_pnl": round(realised_pnl, 2),
                                "product": pos.get("product", ""),
                                "side": "LONG" if qty > 0 else "SHORT"
                            }
            return result
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {}

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary from Groww (includes funds, positions, and holdings)"""
        self._ensure_client()
        try:
            funds_data = self.get_funds()
            positions = self.get_positions()
            holdings_data = self.get_holdings()

            # Parse available cash from margin response
            # Actual Groww response: clear_cash, equity_margin_details, fno_margin_details
            available = 0
            total_margin = 0
            if isinstance(funds_data, dict) and "error" not in funds_data:
                available = float(funds_data.get("clear_cash", 0))
                if available == 0:
                    available = float(funds_data.get("availableBalance",
                                   funds_data.get("available_balance", 0)))
                equity = funds_data.get("equity_margin_details", {})
                if isinstance(equity, dict):
                    total_margin = float(equity.get("cnc_balance_available", available))
                else:
                    total_margin = available

            # Intraday positions
            pos_invested = sum(p.get("invested", 0) for p in positions.values())
            pos_current = sum(p.get("current_value", 0) for p in positions.values())
            pos_pnl = pos_current - pos_invested

            # Holdings (delivery) - parse from Groww response
            holdings_list = []
            holdings_invested = 0
            holdings_current = 0
            if isinstance(holdings_data, dict) and "error" not in holdings_data:
                raw_holdings = holdings_data.get("holdings", [])
                if isinstance(raw_holdings, list):
                    for h in raw_holdings:
                        symbol = h.get("trading_symbol", "")
                        qty = float(h.get("quantity", 0))
                        avg_price = float(h.get("average_price", 0))
                        invested = qty * avg_price
                        # Fetch current price for each holding
                        current_price = market_data.get_live_price(symbol) or avg_price
                        current_value = qty * current_price
                        pnl = current_value - invested
                        pnl_pct = (pnl / invested * 100) if invested > 0 else 0

                        holdings_list.append({
                            "symbol": symbol,
                            "quantity": int(qty),
                            "avg_price": round(avg_price, 2),
                            "current_price": round(current_price, 2),
                            "invested": round(invested, 2),
                            "current_value": round(current_value, 2),
                            "pnl": round(pnl, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "type": "DELIVERY"
                        })
                        holdings_invested += invested
                        holdings_current += current_value

            total_invested = pos_invested + holdings_invested
            total_current = pos_current + holdings_current
            total_pnl = total_current - total_invested

            summary = {
                "capital": total_margin or (available + total_invested),
                "available_cash": round(available, 2),
                "total_invested": round(total_invested, 2),
                "total_current_value": round(total_current, 2),
                "total_portfolio_value": round(available + total_current, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round((total_pnl / total_invested * 100), 2) if total_invested > 0 else 0,
                "num_positions": len(positions),
                "num_holdings": len(holdings_list),
                "positions": positions,
                "holdings": holdings_list,
                "mode": "LIVE",
                "funds_raw": funds_data,
            }

            save_portfolio_snapshot(
                capital=summary["capital"],
                invested=total_invested,
                available=available,
                total_value=summary["total_portfolio_value"],
                daily_pnl=total_pnl,
                total_pnl=total_pnl,
                positions=positions
            )
            return summary

        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            return {
                "error": str(e),
                "mode": "LIVE",
                "capital": 0, "available_cash": 0,
                "total_invested": 0, "total_current_value": 0,
                "total_portfolio_value": 0, "total_pnl": 0,
                "total_pnl_pct": 0, "num_positions": 0,
                "num_holdings": 0, "positions": {}, "holdings": []
            }

    def get_order_book(self) -> list:
        """Get today's order book"""
        self._ensure_client()
        try:
            orders = self._client.get_order_list(segment="CASH")
            return orders if isinstance(orders, (list, dict)) else []
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []


# Singleton instances to avoid re-creating on every call
_paper_broker = None
_groww_broker = None


def get_broker():
    """Factory function to get the appropriate broker based on trading mode"""
    global _paper_broker, _groww_broker

    if settings.trading.mode == "live":
        if _groww_broker is None:
            _groww_broker = GrowwBroker()
        return _groww_broker
    else:
        if _paper_broker is None:
            _paper_broker = PaperTradingBroker()
        return _paper_broker


def reset_broker():
    """Reset broker instances (needed after API key changes)"""
    global _paper_broker, _groww_broker
    _paper_broker = None
    _groww_broker = None


def set_day_start_if_needed():
    """Set day-start CASH value if not set for today (IST). Separate per paper/live mode.
    We only track available_cash at day start so that delivery holdings price fluctuations
    don't affect our intraday P&L calculation."""
    mode = getattr(settings.trading, "mode", "paper")
    key_date = f"day_start_date_{mode}"
    key_value = f"day_start_cash_{mode}"
    today_ist = datetime.now(IST).strftime("%Y-%m-%d")
    day_start_date = get_system_state(key_date)
    if day_start_date == today_ist:
        return
    try:
        broker = get_broker()
        summary = broker.get_portfolio_summary()
        # Track only available cash at day start (not delivery holdings)
        cash = summary.get("available_cash", 0) or summary.get("capital", 0)
        set_system_state(key_value, str(round(cash, 2)))
        set_system_state(key_date, today_ist)
        logger.info(f"Day start cash [{mode}]: Rs.{cash:,.0f} for {today_ist}")
    except Exception as e:
        logger.warning(f"Could not set day start value: {e}")


def get_today_pnl_pct() -> float:
    """Return today's INTRADAY P&L as percent of trading capital.
    Only counts: unrealised P&L from open MIS positions + realised intraday P&L today.
    Does NOT include delivery holdings value changes (which falsely inflated P&L to 31%)."""
    try:
        broker = get_broker()
        summary = broker.get_portfolio_summary()

        # Unrealised P&L from currently-open intraday (MIS) positions only.
        # We deliberately do NOT add each position's Groww `realised_pnl` here:
        # since live SELLs now record an accurate `pnl` in the trades table
        # (see GrowwBroker.place_order), the realised portion is summed from the
        # trades table below. Counting both would double-count partial closes,
        # and the trades-table path also captures positions that have gone flat
        # (qty 0, dropped from get_positions).
        intraday_pnl = 0.0
        positions = summary.get("positions", {})
        for sym, pos in positions.items():
            intraday_pnl += pos.get("pnl", 0)

        # Get realised P&L from today's completed trades (single source of truth
        # for realised intraday P&L, accurate in both paper and live modes).
        from alpha_prime.core.database import get_trades
        today_trades = get_trades(limit=100, trading_mode=settings.trading.mode)
        today_ist = datetime.now(IST).strftime("%Y-%m-%d")
        realised_pnl = 0.0
        for t in (today_trades or []):
            ts = t.get("timestamp", "")
            if today_ist in ts and t.get("action") == "SELL":
                realised_pnl += t.get("pnl", 0) or 0

        total_intraday_pnl = intraday_pnl + realised_pnl
        capital = settings.trading.capital  # Use actual capital, not margin-inflated
        if capital <= 0:
            return 0.0

        pnl_pct = round(total_intraday_pnl / capital * 100, 2)
        return pnl_pct
    except Exception as e:
        logger.warning(f"Could not compute today intraday P&L %: {e}")
        return 0.0
