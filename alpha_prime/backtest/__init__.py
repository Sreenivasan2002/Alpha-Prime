"""Backtesting harness for Alpha-Prime.

Replays historical OHLCV through the SAME technical-analysis engine and the SAME
risk/exit rules used in live trading, so we can measure the impact of the risk
changes on history. No LLM agents, no network at decision time, no lookahead.
"""

from alpha_prime.backtest.backtester import (
    Backtester, BacktestConfig, BacktestResult, Trade,
)

__all__ = ["Backtester", "BacktestConfig", "BacktestResult", "Trade"]
