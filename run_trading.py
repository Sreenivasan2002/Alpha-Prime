"""
Alpha-Prime Trading System - Headless Trading Runner
Run this to start the automated trading scheduler without the UI.

Usage:
    venv\Scripts\python run_trading.py
"""

import sys
import os
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from alpha_prime.core.config import settings
from alpha_prime.core.database import init_database
from alpha_prime.core.scheduler import trading_scheduler
from alpha_prime.data.market_data import is_market_open

# Configure logging
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_path, exist_ok=True)
logger.add(
    os.path.join(log_path, "trading_{time}.log"),
    rotation="1 day",
    retention="30 days",
    level="INFO"
)


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received")
    trading_scheduler.stop()
    sys.exit(0)


def main():
    """Main entry point for headless trading"""
    print("=" * 60)
    print("  ALPHA-PRIME - Autonomous Intraday Trading System")
    print("  Indian Markets (NSE) | Multi-Agent AI Pipeline")
    print("=" * 60)
    print()

    # Check configuration
    if not settings.openai.api_key:
        print("ERROR: OPENAI_API_KEY not set in .env file")
        print("Please update your .env file with the required API keys.")
        return

    print(f"  Trading Mode:    {settings.trading.mode.upper()}")
    print(f"  Market Open:     {'YES' if is_market_open() else 'NO'}")
    print(f"  Analysis Interval: {settings.market.analysis_interval_minutes} minutes")
    print(f"  Max Daily Loss:  {settings.trading.max_daily_loss_pct}%")
    print(f"  Max Position:    {settings.trading.max_position_size_pct}%")
    print()

    # Initialize
    init_database()
    logger.info("Database initialized")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start scheduler
    trading_scheduler.start()
    print("Scheduler started. Press Ctrl+C to stop.")
    print()

    # Keep alive
    try:
        while True:
            time.sleep(60)
            status = trading_scheduler.get_status()
            market = "OPEN" if status["market_open"] else "CLOSED"
            logger.debug(f"Heartbeat - Market: {market}, Scheduler: {'Running' if status['running'] else 'Stopped'}")
    except KeyboardInterrupt:
        print("\nShutting down...")
        trading_scheduler.stop()
        print("Goodbye!")


if __name__ == "__main__":
    main()
