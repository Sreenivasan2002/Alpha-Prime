"""Baseline: the live app's mechanical rules (scanner + TA engine + risk), no LLM layer.
Top gainers +1..5% vs previous close -> real alpha_prime TechnicalAnalyzer on the last 150
completed 5-min bars -> BUY with strength >= 0.10, trend not BEARISH, volume confirmed.
Stop 1.2%, target 2.5%, breakeven 0.6% / trail 1.0% after +1.0%, square-off 15:15."""
from loguru import logger

from alpha_prime.data.technical_analysis import technical_analyzer

logger.remove()


class Strategy:
    name = "app_current"
    entry_start = "10:00"
    entry_end = "14:30"
    squareoff = "15:15"
    max_positions = 10
    profit_lock_pct = 1.5
    min_strength = 0.10

    def __init__(self):
        self._cache = {}

    def on_bar(self, ctx):
        orders = []
        for sym in ctx.symbols:
            if sym in ctx.positions or not ctx.has_bar(sym):
                continue
            pc, px = ctx.prev_close(sym), ctx.last(sym, "c")
            if not pc or not px or not (1.0 <= (px / pc - 1) * 100 <= 5.0):
                continue
            key = (sym, ctx.ts)
            if key not in self._cache:
                bars = ctx.bars(sym, 150)
                r = technical_analyzer.analyze(bars, symbol=sym) if len(bars) >= 20 else None
                self._cache[key] = bool(r and r.overall_signal == "BUY" and r.overall_strength >= self.min_strength
                                        and str(r.trend).upper() != "BEARISH" and r.volume_confirmed)
            if self._cache[key]:
                orders.append(dict(sym=sym, side="long", stop_pct=1.2, target_pct=2.5,
                                   trail=(0.6, 1.0, 1.0), tag="app"))
        return orders
