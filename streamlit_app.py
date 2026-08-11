"""Entrypoint for the public Streamlit Cloud deployment.

Streamlit Cloud auto-detects this filename, so this module is the only way
the hosted app starts. It forces DEMO_MODE on *before* any project module is
imported, which means the hosted deployment cannot come up in a live-trading
configuration even if environment variables or secrets are misconfigured.

Run the real system locally with `streamlit run run_dashboard.py` instead.
"""

import os

# Must be set before importing alpha_prime: core.config reads the environment
# at import time to pick the database path and the execution guards.
os.environ["DEMO_MODE"] = "true"
os.environ["TRADING_MODE"] = "paper"

# Ensure no credential is picked up even if one is present in the environment.
os.environ.pop("GROWW_API_KEY", None)
os.environ.pop("GROWW_SECRET_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)

# The host has no .env file, so without these the app would silently fall back
# to the library defaults in core/config.py -- including a Rs 10,000 capital,
# which would misreport every percentage on a snapshot built against Rs 10L.
# These are the settings the committed snapshot and backtest were produced with.
os.environ.setdefault("TRADING_CAPITAL", "1000000")
os.environ.setdefault("MARGIN_MULTIPLIER", "5.0")
os.environ.setdefault("MAX_DAILY_LOSS_PCT", "2.0")
os.environ.setdefault("MAX_POSITION_SIZE_PCT", "20.0")
os.environ.setdefault("MAX_OPEN_POSITIONS", "10")
os.environ.setdefault("DEFAULT_STOP_LOSS_PCT", "1.2")
os.environ.setdefault("DEFAULT_TARGET_PCT", "2.5")
os.environ.setdefault("RISK_PER_TRADE_PCT", "0.5")
os.environ.setdefault("BREAKEVEN_ARM_PCT", "0.6")
os.environ.setdefault("TRAIL_ARM_PCT", "1.0")
os.environ.setdefault("TRAILING_STOP_PCT", "1.0")

from alpha_prime.ui.dashboard import main  # noqa: E402

# Streamlit re-executes this script top-to-bottom on every interaction, so the
# app is started by calling main() at module level rather than under a
# __main__ guard.
main()
