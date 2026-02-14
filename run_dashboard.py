"""
Alpha-Prime Trading System - Dashboard Launcher
Run this to start the Streamlit UI.

Usage:
    From project root with venv activated:
        streamlit run run_dashboard.py

    Or directly:
        venv/Scripts/streamlit run run_dashboard.py
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_prime.ui.dashboard import main

if __name__ == "__main__":
    main()
