"""
Pure risk / position-sizing functions for Alpha-Prime.

These are deliberately IO-free (no broker, no network, no DB) so that the EXACT
same logic drives both live trading (tools._enforce_risk_limits,
scheduler.PositionMonitor) and the backtester. Keeping one source of truth means
a backtest faithfully reflects how the system behaves with real money.
"""

import math


def compute_max_quantity(capital: float, price: float, stop_loss: float,
                         risk_per_trade_pct: float, max_position_size_pct: float) -> tuple:
    """Maximum TOTAL position size (existing + new) allowed for one symbol.

    Two caps, both on REAL capital (not margin buying power):
      - risk_per_trade : (price - stop_loss) * qty <= capital * risk_per_trade_pct/100
      - max_position_size : qty * price          <= capital * max_position_size_pct/100

    The caller subtracts any quantity already held to get the max NEW quantity.

    Returns (max_total_qty: int, binding: str). binding names the tighter cap.
    """
    caps = []  # (label, max_total_qty)

    if stop_loss and price and 0 < stop_loss < price:
        sl_distance = price - stop_loss
        max_risk_amount = capital * (risk_per_trade_pct / 100.0)
        caps.append(("risk_per_trade", int(math.floor(max_risk_amount / sl_distance))))

    if price and price > 0:
        max_notional = capital * (max_position_size_pct / 100.0)
        caps.append(("max_position_size", int(math.floor(max_notional / price))))

    if not caps:
        return 0, "invalid_price"

    binding, max_total = min(caps, key=lambda c: c[1])
    return max_total, binding


def compute_trailed_stop(entry: float, peak: float, current_price: float, current_sl: float,
                         breakeven_arm_pct: float, trail_arm_pct: float,
                         trailing_stop_pct: float) -> float:
    """New stop-loss for an open long. The stop is only ever RAISED, never lowered.

    Two stages (gain measured from entry using the latest price; `peak` is the
    highest price seen so far and must already be updated by the caller):
      1. Breakeven lock: once gain >= breakeven_arm_pct, move SL up to entry.
      2. Trail: once gain >= trail_arm_pct, trail trailing_stop_pct below the peak.

    Trailing intentionally does NOT start before trail_arm_pct, so a small early
    wiggle can't knock a winner out at ~+0.5% before it reaches its target.
    """
    if entry <= 0:
        return current_sl

    gain = (current_price - entry) / entry
    new_sl = current_sl

    if gain >= breakeven_arm_pct / 100.0:
        new_sl = max(new_sl, entry)  # breakeven lock
    if gain >= trail_arm_pct / 100.0:
        new_sl = max(new_sl, peak * (1 - trailing_stop_pct / 100.0))  # trail below peak

    return new_sl
