"""
Indian intraday (MIS) equity transaction costs.

Used by the paper broker so paper P&L is reported NET of charges, like a real
account would be. Rates are typical 2026 discount-broker values; they mirror the
defaults in backtest.BacktestConfig so paper and backtest numbers are comparable.
"""

BROKERAGE_PCT = 0.001        # 0.1% per order ...
BROKERAGE_CAP = 20.0         # ... capped at Rs 20 per order
STT_PCT_SELL = 0.00025       # 0.025% on the sell side only
EXCHANGE_TXN_PCT = 0.0000297  # NSE transaction charge, both sides
SEBI_PCT = 0.000001          # Rs 10 per crore, both sides
STAMP_PCT_BUY = 0.00003      # 0.003% on the buy side only
GST_PCT = 0.18               # on brokerage + exchange + SEBI fee


def intraday_round_trip_cost(buy_value: float, sell_value: float) -> float:
    """Total charges in rupees for buying `buy_value` and selling `sell_value` of shares."""
    brokerage = min(BROKERAGE_PCT * buy_value, BROKERAGE_CAP) + min(BROKERAGE_PCT * sell_value, BROKERAGE_CAP)
    stt = STT_PCT_SELL * sell_value
    exchange = EXCHANGE_TXN_PCT * (buy_value + sell_value)
    sebi = SEBI_PCT * (buy_value + sell_value)
    stamp = STAMP_PCT_BUY * buy_value
    gst = GST_PCT * (brokerage + exchange + sebi)
    return brokerage + stt + exchange + sebi + stamp + gst
