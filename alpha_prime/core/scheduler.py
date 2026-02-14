"""
Trading Scheduler for Alpha-Prime
Manages automated execution of the trading pipeline during market hours.
Includes: auto square-off, trailing stop-loss, position monitoring.
"""

import threading
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz
from loguru import logger

from alpha_prime.core.config import settings
from alpha_prime.core.database import log_agent_activity, set_system_state, get_system_state
from alpha_prime.data.market_data import is_market_open, is_pre_market, market_data
from alpha_prime.agents.pipeline import run_full_pipeline, run_portfolio_review

IST = pytz.timezone("Asia/Kolkata")


class PositionMonitor:
    """
    Monitors open positions for trailing stop-loss and auto square-off.
    Tracks the highest price reached for each position and sells if price
    drops below a trailing threshold.
    """

    def __init__(self):
        self._peak_prices = {}     # symbol -> highest price since buy
        self._entry_prices = {}    # symbol -> entry price
        self._stop_losses = {}     # symbol -> absolute stop-loss price
        self._targets = {}         # symbol -> target price
        self._lock = threading.Lock()

    def register_position(self, symbol: str, entry_price: float,
                          stop_loss: float = 0, target: float = 0):
        """Register a new position for monitoring"""
        with self._lock:
            self._peak_prices[symbol] = entry_price
            self._entry_prices[symbol] = entry_price
            if stop_loss > 0:
                self._stop_losses[symbol] = stop_loss
            else:
                # Default 1.5% trailing stop from entry
                self._stop_losses[symbol] = entry_price * 0.985
            if target > 0:
                self._targets[symbol] = target
            else:
                self._targets[symbol] = entry_price * 1.02  # 2% default target

            logger.info(
                f"[PositionMonitor] Registered {symbol}: entry={entry_price:.2f}, "
                f"SL={self._stop_losses[symbol]:.2f}, target={self._targets[symbol]:.2f}"
            )

    def remove_position(self, symbol: str):
        """Remove a position from monitoring (after sell)"""
        with self._lock:
            self._peak_prices.pop(symbol, None)
            self._entry_prices.pop(symbol, None)
            self._stop_losses.pop(symbol, None)
            self._targets.pop(symbol, None)
            logger.info(f"[PositionMonitor] Removed {symbol} from monitoring")

    def check_positions(self) -> list:
        """
        Check all monitored positions against current prices.
        Returns list of symbols that should be SOLD (hit trailing SL or target).
        """
        sell_signals = []

        with self._lock:
            symbols = list(self._peak_prices.keys())

        for symbol in symbols:
            try:
                current_price = market_data.get_live_price(symbol)
                if current_price is None or current_price <= 0:
                    continue

                with self._lock:
                    if symbol not in self._peak_prices:
                        continue

                    peak = self._peak_prices[symbol]
                    entry = self._entry_prices.get(symbol, 0)
                    sl = self._stop_losses.get(symbol, 0)
                    target = self._targets.get(symbol, 0)

                    # Update peak price (trailing)
                    if current_price > peak:
                        self._peak_prices[symbol] = current_price
                        # Move stop-loss up: trail at 1.5% below new peak
                        new_sl = current_price * 0.985
                        if new_sl > sl:
                            self._stop_losses[symbol] = new_sl
                            logger.info(
                                f"[TrailingSL] {symbol} new peak {current_price:.2f}, "
                                f"SL moved up to {new_sl:.2f}"
                            )

                    # Check if stop-loss hit
                    if current_price <= sl:
                        pnl = current_price - entry
                        pnl_pct = (pnl / entry * 100) if entry > 0 else 0
                        sell_signals.append({
                            "symbol": symbol,
                            "reason": "TRAILING_STOP_LOSS",
                            "current_price": current_price,
                            "stop_loss": sl,
                            "entry": entry,
                            "pnl": round(pnl, 2),
                            "pnl_pct": round(pnl_pct, 2),
                        })
                        logger.warning(
                            f"[TrailingSL] {symbol} HIT SL! Price {current_price:.2f} "
                            f"<= SL {sl:.2f} (P&L: {pnl_pct:+.2f}%)"
                        )

                    # Check if target hit
                    elif target > 0 and current_price >= target:
                        pnl = current_price - entry
                        pnl_pct = (pnl / entry * 100) if entry > 0 else 0
                        sell_signals.append({
                            "symbol": symbol,
                            "reason": "TARGET_HIT",
                            "current_price": current_price,
                            "target": target,
                            "entry": entry,
                            "pnl": round(pnl, 2),
                            "pnl_pct": round(pnl_pct, 2),
                        })
                        logger.info(
                            f"[Target] {symbol} HIT TARGET! Price {current_price:.2f} "
                            f">= Target {target:.2f} (P&L: {pnl_pct:+.2f}%)"
                        )

            except Exception as e:
                logger.error(f"[PositionMonitor] Error checking {symbol}: {e}")

        return sell_signals

    def get_status(self) -> dict:
        """Get current monitoring status"""
        with self._lock:
            status = {}
            for symbol in self._peak_prices:
                status[symbol] = {
                    "entry": self._entry_prices.get(symbol, 0),
                    "peak": self._peak_prices.get(symbol, 0),
                    "stop_loss": self._stop_losses.get(symbol, 0),
                    "target": self._targets.get(symbol, 0),
                }
            return status


# Global position monitor
position_monitor = PositionMonitor()


class TradingScheduler:
    """Manages automated trading schedule with position monitoring"""

    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone=IST)
        self._running = False
        self._pipeline_lock = threading.Lock()
        self._last_run = None

    def start(self):
        """Start the scheduler with all jobs"""
        if self._running:
            logger.warning("Scheduler already running")
            return

        interval = settings.market.analysis_interval_minutes

        # Main trading job - runs every N minutes during market hours
        self.scheduler.add_job(
            self._run_trading_cycle,
            IntervalTrigger(minutes=interval, timezone=IST),
            id="trading_cycle",
            name="Trading Cycle",
            replace_existing=True,
            max_instances=1,
        )

        # Position monitor - runs every 2 minutes to check SL/target
        self.scheduler.add_job(
            self._monitor_positions,
            IntervalTrigger(minutes=2, timezone=IST),
            id="position_monitor",
            name="Position Monitor",
            replace_existing=True,
            max_instances=1,
        )

        # Pre-market scan - runs at 9:00 AM IST
        self.scheduler.add_job(
            self._pre_market_scan,
            CronTrigger(hour=9, minute=0, timezone=IST),
            id="pre_market_scan",
            name="Pre-Market Scan",
            replace_existing=True,
        )

        # Auto square-off at 3:15 PM IST (before 3:30 market close)
        self.scheduler.add_job(
            self._auto_square_off,
            CronTrigger(hour=15, minute=15, timezone=IST),
            id="auto_square_off",
            name="Auto Square-Off",
            replace_existing=True,
        )

        # End-of-day review - runs at 3:25 PM IST
        self.scheduler.add_job(
            self._end_of_day_review,
            CronTrigger(hour=15, minute=25, timezone=IST),
            id="eod_review",
            name="End of Day Review",
            replace_existing=True,
        )

        self.scheduler.start()
        self._running = True
        set_system_state("scheduler_status", "running")
        logger.info(
            f"Trading scheduler started (interval: {interval} min, "
            f"position monitor: 2 min, auto square-off: 3:15 PM)"
        )

        # Sync existing positions from broker to position monitor
        self._sync_positions_from_broker()

    def stop(self):
        """Stop the scheduler"""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            set_system_state("scheduler_status", "stopped")
            logger.info("Trading scheduler stopped")

    def is_running(self) -> bool:
        return self._running

    def _sync_positions_from_broker(self):
        """Load existing positions into position monitor on startup"""
        try:
            from alpha_prime.core.broker import get_broker
            broker = get_broker()
            positions = broker.get_positions()
            for symbol, pos in positions.items():
                avg_price = pos.get("avg_price", 0)
                if avg_price > 0:
                    # Register with default SL/target
                    position_monitor.register_position(
                        symbol=symbol,
                        entry_price=avg_price,
                        stop_loss=avg_price * (1 - settings.trading.default_stop_loss_pct / 100),
                        target=avg_price * (1 + settings.trading.default_target_pct / 100)
                    )
            if positions:
                logger.info(f"Synced {len(positions)} positions to monitor")
        except Exception as e:
            logger.warning(f"Could not sync positions: {e}")

    def _monitor_positions(self):
        """Check positions against SL/target and auto-sell if triggered"""
        now = datetime.now(IST)

        # Only monitor during market hours
        if not is_market_open():
            return

        try:
            sell_signals = position_monitor.check_positions()

            if not sell_signals:
                return

            from alpha_prime.core.broker import get_broker
            broker = get_broker()

            for signal in sell_signals:
                symbol = signal["symbol"]
                reason = signal["reason"]
                price = signal["current_price"]

                logger.info(
                    f"[AutoSell] Selling {symbol} - {reason} at Rs.{price:.2f} "
                    f"(P&L: {signal['pnl_pct']:+.2f}%)"
                )

                # Get actual quantity from broker
                positions = broker.get_positions()
                qty = positions.get(symbol, {}).get("quantity", 0)

                if qty > 0:
                    result = broker.place_order(
                        symbol=symbol,
                        action="SELL",
                        quantity=qty,
                        rationale=f"Auto-{reason.lower().replace('_', ' ')}: "
                                  f"price={price:.2f}, P&L={signal['pnl_pct']:+.2f}%",
                        agent_name="position_monitor"
                    )
                    logger.info(f"[AutoSell] {symbol} result: {result.get('status')}")
                    log_agent_activity(
                        "position_monitor", "auto_sell",
                        f"{reason}: Sold {qty} {symbol} @ {price:.2f}",
                        {"signal": signal, "result": result}
                    )

                    # Remove from monitor
                    position_monitor.remove_position(symbol)
                else:
                    logger.warning(f"[AutoSell] {symbol} has 0 quantity, removing from monitor")
                    position_monitor.remove_position(symbol)

        except Exception as e:
            logger.error(f"Position monitor error: {e}")

    def _auto_square_off(self):
        """Square off ALL intraday positions at 3:15 PM"""
        now = datetime.now(IST)

        # Only on weekdays
        if now.weekday() >= 5:
            return

        logger.info("=" * 60)
        logger.info("AUTO SQUARE-OFF: Closing all intraday positions at 3:15 PM")
        logger.info("=" * 60)

        try:
            from alpha_prime.core.broker import get_broker
            broker = get_broker()
            positions = broker.get_positions()

            if not positions:
                logger.info("[SquareOff] No open positions to close")
                log_agent_activity("scheduler", "square_off", "No positions to close")
                return

            total_pnl = 0
            for symbol, pos in positions.items():
                qty = pos.get("quantity", 0)
                if qty <= 0:
                    continue

                current_price = pos.get("current_price", 0)
                entry_price = pos.get("avg_price", 0)
                pnl = (current_price - entry_price) * qty

                logger.info(
                    f"[SquareOff] Selling {qty} {symbol} @ Rs.{current_price:.2f} "
                    f"(Entry: {entry_price:.2f}, P&L: Rs.{pnl:+.2f})"
                )

                result = broker.place_order(
                    symbol=symbol,
                    action="SELL",
                    quantity=qty,
                    rationale=f"EOD auto square-off at 3:15 PM. "
                              f"Entry={entry_price:.2f}, Exit={current_price:.2f}, "
                              f"P&L=Rs.{pnl:+.2f}",
                    agent_name="auto_square_off"
                )

                total_pnl += pnl
                position_monitor.remove_position(symbol)

                log_agent_activity(
                    "auto_square_off", "sell",
                    f"Squared off {qty} {symbol} @ {current_price:.2f}, P&L: Rs.{pnl:+.2f}",
                    {"result": result}
                )

            logger.info(f"[SquareOff] Total day P&L: Rs.{total_pnl:+.2f}")
            set_system_state("last_square_off", now.isoformat())
            set_system_state("last_square_off_pnl", f"{total_pnl:+.2f}")
            log_agent_activity(
                "auto_square_off", "complete",
                f"All positions squared off. Total P&L: Rs.{total_pnl:+.2f}"
            )

        except Exception as e:
            logger.error(f"Square-off error: {e}")
            log_agent_activity("auto_square_off", "error", str(e))

    def _run_trading_cycle(self):
        """Execute one trading cycle"""
        now = datetime.now(IST)

        # Only trade during market hours
        if not is_market_open():
            logger.debug(f"Market closed at {now.strftime('%H:%M')} IST, skipping cycle")
            return

        # Don't trade in the last 15 minutes (closing auction)
        if now.hour == 15 and now.minute >= 15:
            logger.info("Last 15 minutes of market - skipping new trades")
            return

        # Prevent concurrent runs
        if not self._pipeline_lock.acquire(blocking=False):
            logger.warning("Previous trading cycle still running, skipping")
            return

        try:
            logger.info(f"Starting trading cycle at {now.strftime('%H:%M:%S')} IST")
            set_system_state("last_cycle_start", now.isoformat())
            log_agent_activity("scheduler", "cycle_start",
                             f"Trading cycle started at {now.strftime('%H:%M')}")

            # Run the full pipeline
            result = run_full_pipeline()

            # After pipeline, register any new positions for monitoring
            self._sync_new_positions(result)

            self._last_run = now
            set_system_state("last_cycle_end", datetime.now(IST).isoformat())
            set_system_state("last_cycle_status", "success")

            logger.info("Trading cycle completed successfully")

        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
            set_system_state("last_cycle_status", f"error: {str(e)}")
            log_agent_activity("scheduler", "error", str(e))

        finally:
            self._pipeline_lock.release()

    def _sync_new_positions(self, pipeline_result):
        """After a pipeline run, sync any new positions to the monitor"""
        try:
            from alpha_prime.core.broker import get_broker
            broker = get_broker()
            positions = broker.get_positions()

            for symbol, pos in positions.items():
                if symbol not in position_monitor._peak_prices:
                    avg_price = pos.get("avg_price", 0)
                    if avg_price > 0:
                        position_monitor.register_position(
                            symbol=symbol,
                            entry_price=avg_price,
                            stop_loss=avg_price * (1 - settings.trading.default_stop_loss_pct / 100),
                            target=avg_price * (1 + settings.trading.default_target_pct / 100)
                        )
        except Exception as e:
            logger.warning(f"Could not sync new positions: {e}")

    def _pre_market_scan(self):
        """Pre-market analysis at 9:00 AM"""
        logger.info("Running pre-market scan...")
        log_agent_activity("scheduler", "pre_market", "Pre-market scan started")

        try:
            from alpha_prime.agents.pipeline import run_scanner_only
            result = run_scanner_only()
            set_system_state("pre_market_scan", result[:2000])
            logger.info("Pre-market scan completed")
        except Exception as e:
            logger.error(f"Pre-market scan error: {e}")

    def _end_of_day_review(self):
        """End of day portfolio review at 3:25 PM"""
        logger.info("Running end-of-day review...")
        log_agent_activity("scheduler", "eod_review", "End-of-day review started")

        try:
            result = run_portfolio_review()
            set_system_state("eod_review", result[:2000])
            logger.info("End-of-day review completed")
        except Exception as e:
            logger.error(f"EOD review error: {e}")

    def run_now(self):
        """Manually trigger a trading cycle"""
        logger.info("Manual trading cycle triggered")
        thread = threading.Thread(target=self._run_trading_cycle, daemon=True)
        thread.start()
        return "Trading cycle triggered"

    def get_status(self) -> dict:
        """Get scheduler status"""
        return {
            "running": self._running,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_cycle_status": get_system_state("last_cycle_status", "N/A"),
            "position_monitor": position_monitor.get_status(),
            "last_square_off": get_system_state("last_square_off", "N/A"),
            "next_jobs": [
                {
                    "name": job.name,
                    "next_run": str(job.next_run_time) if job.next_run_time else "N/A"
                }
                for job in self.scheduler.get_jobs()
            ] if self._running else [],
            "interval_minutes": settings.market.analysis_interval_minutes,
            "market_open": is_market_open(),
            "trading_mode": settings.trading.mode,
        }


# Singleton
trading_scheduler = TradingScheduler()
