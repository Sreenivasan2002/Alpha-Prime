"""
Alpha-Prime Trading System - Quick Analysis Tool
Run this to get a quick AI-powered analysis of specific stocks.

Usage:
    venv\Scripts\python run_analysis.py RELIANCE TCS INFY
    venv\Scripts\python run_analysis.py  # (scans market for opportunities)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from alpha_prime.core.config import settings
from alpha_prime.core.database import init_database


def main():
    init_database()

    symbols = sys.argv[1:] if len(sys.argv) > 1 else []

    if not settings.openai.api_key:
        print("ERROR: OPENAI_API_KEY not set in .env file")
        return

    if symbols:
        print(f"\nAnalyzing: {', '.join(symbols)}")
        print("=" * 60)

        from alpha_prime.agents.pipeline import run_analysis
        result = run_analysis(symbols)
        print("\n" + result)
    else:
        print("\nScanning market for opportunities...")
        print("=" * 60)

        from alpha_prime.agents.pipeline import run_scanner_only
        result = run_scanner_only()
        print("\n" + result)


if __name__ == "__main__":
    main()
