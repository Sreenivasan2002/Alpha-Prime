"""
Event-driven intraday backtester for Alpha-Prime.

Design goals:
  * FIDELITY: entries use the real TechnicalAnalyzer engine; sizing uses the shared
    compute_max_quantity(); the trailing stop uses the shared compute_trailed_stop().
    So a backtest reflects how the live system actually behaves.
  * NO LOOKAHEAD: the BUY decision at a bar is computed from data up to the PREVIOUS
    bar's close; the trade is filled at the current bar's OPEN. Intrabar exits use the
    bar's High/Low; if both stop and target are touched in one bar we assume the STOP
    filled first (pessimistic). Trailing stop is updated only at bar close.
  * NET OF COSTS: realistic Indian intraday costs (brokerage, STT, exchange, SEBI,
    stamp, GST) + optional slippage are subtracted from every trade. With ~0.5% edges
    these matter a lot, so metrics are NET unless costs are disabled.
  * INTRADAY ONLY: every position is squared off at the configured square-off time
    (or the day's last bar) — no overnight holds, mirroring the live square-off.
  * ISOLATION: never touches the live alpha_prime.db. Output is in-memory + optional CSV.

Performance: entry signals (TA) are memoised per (symbol, bar), so a parameter sweep
over exit/sizing settings reuses the same signals instead of recomputing them.

yfinance limits intraday history (5m ~= last 60 days).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from alpha_prime.core.config import settings
from alpha_prime.core.risk import compute_max_quantity, compute_trailed_stop
from alpha_prime.data.market_data import market_data, clean_symbol
from alpha_prime.data.technical_analysis import technical_analyzer

IST = "Asia/Kolkata"


@dataclass
class BacktestConfig:
    """All knobs for a run. Defaults pulled from live settings so the backtest matches
    production unless deliberately overridden (e.g. to A/B a setting or sweep)."""
    capital: float = field(default_factory=lambda: settings.trading.capital)
    margin_multiplier: float = field(default_factory=lambda: settings.trading.margin_multiplier)
    risk_per_trade_pct: float = field(default_factory=lambda: settings.trading.risk_per_trade_pct)
    max_position_size_pct: float = field(default_factory=lambda: settings.trading.max_position_size_pct)
    max_open_positions: int = field(default_factory=lambda: settings.trading.max_open_positions)
    default_stop_loss_pct: float = field(default_factory=lambda: settings.trading.default_stop_loss_pct)
    default_target_pct: float = field(default_factory=lambda: settings.trading.default_target_pct)
    breakeven_arm_pct: float = field(default_factory=lambda: settings.trading.breakeven_arm_pct)
    trail_arm_pct: float = field(default_factory=lambda: settings.trading.trail_arm_pct)
    trailing_stop_pct: float = field(default_factory=lambda: settings.trading.trailing_stop_pct)
    max_daily_loss_pct: float = field(default_factory=lambda: settings.trading.max_daily_loss_pct)
    daily_profit_target_pct: float = field(default_factory=lambda: settings.trading.daily_profit_target_pct)
    min_signal_strength: float = field(default_factory=lambda: settings.trading.min_signal_strength)
    cooldown_days: int = 3

    # Entry filters (mirror the scanner/analyst gates)
    require_bullish_trend: bool = True
    require_volume_confirmed: bool = False  # live warns but doesn't hard-block

    # Trading window (IST)
    trading_start: tuple = field(default_factory=lambda: (settings.market.trading_start_hour, settings.market.trading_start_minute))
    trading_end: tuple = field(default_factory=lambda: (settings.market.trading_end_hour, settings.market.trading_end_minute))
    squareoff: tuple = (15, 15)

    lookback_bars: int = 150  # bars of history fed to the TA engine per decision

    # --- Transaction costs (Indian intraday equity / MIS), all as fractions ---
    include_costs: bool = True
    brokerage_pct: float = 0.001        # 0.1% per side ...
    brokerage_cap: float = 20.0         # ... capped at Rs 20 per order (Groww-style)
    stt_pct_sell: float = 0.00025       # 0.025% on the SELL side only
    exchange_txn_pct: float = 0.0000297  # NSE ~0.00297% each side
    sebi_pct: float = 0.000001          # Rs 10 / crore each side
    stamp_pct_buy: float = 0.00003      # 0.003% on the BUY side only
    gst_pct: float = 0.18               # 18% GST on (brokerage + exchange + sebi)
    slippage_pct: float = 0.0           # optional per-side slippage as fraction of price

    @property
    def effective_capital(self) -> float:
        return self.capital * self.margin_multiplier


@dataclass
class Trade:
    symbol: str
    entry_time: str
    entry_price: float
    qty: int
    stop: float
    target: float
    exit_time: str = ""
    exit_price: float = 0.0
    gross_pnl: float = 0.0
    cost: float = 0.0
    pnl: float = 0.0          # NET of costs
    pnl_pct: float = 0.0      # gross %, on price
    exit_reason: str = ""
    bars_held: int = 0


@dataclass
class _OpenPos:
    symbol: str
    entry_time: pd.Timestamp
    entry_iloc: int
    entry_price: float
    qty: int
    stop: float
    target: float
    peak: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: List[dict]
    metrics: dict
    symbols: List[str]
    bars_scanned: int
    duration_sec: float


class Backtester:
    def __init__(self, cfg: Optional[BacktestConfig] = None):
        self.cfg = cfg or BacktestConfig()
        self._signal_cache: Dict[tuple, bool] = {}  # (symbol, iloc) -> bool

    # ---- data ----
    def load_data(self, symbols: List[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
        data = {}
        for sym in symbols:
            df = market_data.get_historical_data(sym, period=period, interval=interval)
            if df is None or df.empty or len(df) < max(30, self.cfg.lookback_bars // 4):
                logger.warning(f"[backtest] skipping {sym}: insufficient data")
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            idx = pd.to_datetime(df.index)
            try:
                idx = idx.tz_convert(IST) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(IST)
            except (TypeError, AttributeError):
                pass
            df.index = idx
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df["_date"] = df.index.date
            df["_minutes"] = df.index.hour * 60 + df.index.minute
            data[clean_symbol(sym)] = df
        return data

    # ---- time helpers ----
    def _in_window(self, minutes: int) -> bool:
        start = self.cfg.trading_start[0] * 60 + self.cfg.trading_start[1]
        end = self.cfg.trading_end[0] * 60 + self.cfg.trading_end[1]
        return start <= minutes <= end

    def _squareoff_minutes(self) -> int:
        return self.cfg.squareoff[0] * 60 + self.cfg.squareoff[1]

    # ---- signal (memoised) ----
    def _entry_signal(self, df: pd.DataFrame, iloc: int, symbol: str) -> bool:
        """BUY decision from data strictly BEFORE bar `iloc` (no lookahead). Memoised so
        a parameter sweep over exit settings reuses the same signals."""
        key = (symbol, iloc)
        cached = self._signal_cache.get(key)
        if cached is not None:
            return cached
        start = max(0, iloc - self.cfg.lookback_bars)
        window = df.iloc[start:iloc]
        result = False
        if len(window) >= 20:
            report = technical_analyzer.analyze(window, symbol=symbol)
            if report is not None and report.overall_signal == "BUY" \
                    and report.overall_strength >= self.cfg.min_signal_strength \
                    and not (self.cfg.require_bullish_trend and str(report.trend).upper() == "BEARISH") \
                    and not (self.cfg.require_volume_confirmed and not report.volume_confirmed):
                result = True
        self._signal_cache[key] = result
        return result

    def precompute_signals(self, data: Dict[str, pd.DataFrame]):
        """Fill the signal cache for every in-window bar. Call once before a sweep."""
        sq = self._squareoff_minutes()
        n = 0
        for sym, df in data.items():
            minutes = df["_minutes"].values
            for iloc in range(1, len(df)):
                if self._in_window(minutes[iloc]) and minutes[iloc] < sq:
                    self._entry_signal(df, iloc, sym)
                    n += 1
        logger.info(f"[backtest] precomputed signals for {n} in-window bars")

    # ---- costs ----
    def _round_trip_cost(self, entry_price: float, exit_price: float, qty: int) -> float:
        c = self.cfg
        if not c.include_costs:
            return 0.0
        buy_turn = entry_price * qty
        sell_turn = exit_price * qty
        brokerage = min(c.brokerage_pct * buy_turn, c.brokerage_cap) + min(c.brokerage_pct * sell_turn, c.brokerage_cap)
        stt = c.stt_pct_sell * sell_turn
        exch = c.exchange_txn_pct * (buy_turn + sell_turn)
        sebi = c.sebi_pct * (buy_turn + sell_turn)
        stamp = c.stamp_pct_buy * buy_turn
        gst = c.gst_pct * (brokerage + exch + sebi)
        slippage = c.slippage_pct * (buy_turn + sell_turn)
        return brokerage + stt + exch + sebi + stamp + gst + slippage

    # ---- run / simulate ----
    def run(self, symbols: List[str], period: str = "60d", interval: str = "5m") -> BacktestResult:
        data = self.load_data(symbols, period, interval)
        if not data:
            raise ValueError("No usable data loaded for any symbol.")
        return self.simulate(data)

    def simulate(self, data: Dict[str, pd.DataFrame], cfg: Optional[BacktestConfig] = None) -> BacktestResult:
        """Run the simulation. `cfg` overrides exit/sizing/cost settings (for sweeps);
        entry SIGNALS always come from self.cfg via the memoised cache."""
        t0 = time.time()
        run_cfg = cfg or self.cfg
        # cost model uses run_cfg too
        prev_cfg = self.cfg
        self.cfg = run_cfg
        try:
            result = self._simulate_impl(data, run_cfg)
        finally:
            self.cfg = prev_cfg
        result.duration_sec = round(time.time() - t0, 1)
        return result

    def _simulate_impl(self, data: Dict[str, pd.DataFrame], cfg: BacktestConfig) -> BacktestResult:
        iloc_map = {s: {ts: i for i, ts in enumerate(df.index)} for s, df in data.items()}
        last_iloc_of_day = {}
        for s, df in data.items():
            for i, d in enumerate(df["_date"].values):
                last_iloc_of_day[(s, d)] = i
        timeline = sorted(set().union(*[set(df.index) for df in data.values()]))

        cash_used = 0.0
        realized_total = 0.0
        open_positions: Dict[str, _OpenPos] = {}
        last_loss_date: Dict[str, date] = {}
        trades: List[Trade] = []
        equity_curve: List[dict] = []

        cur_day = None
        day_realized = 0.0
        day_blocked = False
        squareoff_min = self._squareoff_minutes()

        for ts in timeline:
            d = ts.date()
            if d != cur_day:
                if cur_day is not None:
                    equity_curve.append({"date": str(cur_day), "equity": round(cfg.capital + realized_total, 2)})
                cur_day = d
                day_realized = 0.0
                day_blocked = False

            minutes = ts.hour * 60 + ts.minute

            # ---------- Phase A: manage / exit open positions ----------
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                df = data[sym]
                iloc = iloc_map[sym].get(ts)
                if iloc is None:
                    continue
                bar = df.iloc[iloc]
                o, h, l, c = float(bar.Open), float(bar.High), float(bar.Low), float(bar.Close)

                exit_price = None
                reason = ""
                is_last_of_day = last_iloc_of_day.get((sym, d)) == iloc

                if minutes >= squareoff_min or is_last_of_day:
                    exit_price, reason = c, "EOD_SQUAREOFF"
                elif o <= pos.stop:
                    exit_price, reason = o, "STOP_GAP"
                elif l <= pos.stop:
                    exit_price, reason = pos.stop, "STOP"
                elif o >= pos.target:
                    exit_price, reason = o, "TARGET_GAP"
                elif h >= pos.target:
                    exit_price, reason = pos.target, "TARGET"

                if exit_price is not None:
                    gross = (exit_price - pos.entry_price) * pos.qty
                    cost = self._round_trip_cost(pos.entry_price, exit_price, pos.qty)
                    net = gross - cost
                    realized_total += net
                    day_realized += net
                    cash_used -= pos.entry_price * pos.qty
                    if net < 0:
                        last_loss_date[sym] = d
                    trades.append(Trade(
                        symbol=sym, entry_time=str(pos.entry_time), entry_price=round(pos.entry_price, 2),
                        qty=pos.qty, stop=round(pos.stop, 2), target=round(pos.target, 2),
                        exit_time=str(ts), exit_price=round(exit_price, 2),
                        gross_pnl=round(gross, 2), cost=round(cost, 2), pnl=round(net, 2),
                        pnl_pct=round((exit_price - pos.entry_price) / pos.entry_price * 100, 3),
                        exit_reason=reason, bars_held=iloc - pos.entry_iloc,
                    ))
                    del open_positions[sym]
                else:
                    pos.peak = max(pos.peak, h)
                    pos.stop = compute_trailed_stop(
                        entry=pos.entry_price, peak=pos.peak, current_price=c, current_sl=pos.stop,
                        breakeven_arm_pct=cfg.breakeven_arm_pct, trail_arm_pct=cfg.trail_arm_pct,
                        trailing_stop_pct=cfg.trailing_stop_pct,
                    )

            # ---------- daily circuit breaker (realized + open unrealized) ----------
            if not day_blocked:
                unrealized = 0.0
                for sym, pos in open_positions.items():
                    iloc = iloc_map[sym].get(ts)
                    if iloc is not None:
                        unrealized += (float(data[sym].iloc[iloc].Close) - pos.entry_price) * pos.qty
                day_pnl_pct = (day_realized + unrealized) / cfg.capital * 100 if cfg.capital else 0
                if day_pnl_pct <= -abs(cfg.max_daily_loss_pct) or day_pnl_pct >= cfg.daily_profit_target_pct:
                    day_blocked = True

            # ---------- Phase B: entries ----------
            if not day_blocked and self._in_window(minutes) and minutes < squareoff_min:
                for sym in data.keys():
                    if sym in open_positions:
                        continue
                    if len(open_positions) >= cfg.max_open_positions:
                        break
                    iloc = iloc_map[sym].get(ts)
                    if iloc is None or iloc == 0:
                        continue
                    ll = last_loss_date.get(sym)
                    if ll is not None and (d - ll).days < cfg.cooldown_days:
                        continue

                    entry_price = float(data[sym].iloc[iloc].Open)
                    if entry_price <= 0:
                        continue
                    stop = entry_price * (1 - cfg.default_stop_loss_pct / 100.0)
                    target = entry_price * (1 + cfg.default_target_pct / 100.0)

                    max_total_qty, _ = compute_max_quantity(
                        capital=cfg.capital, price=entry_price, stop_loss=stop,
                        risk_per_trade_pct=cfg.risk_per_trade_pct,
                        max_position_size_pct=cfg.max_position_size_pct,
                    )
                    avail_bp = cfg.effective_capital - cash_used
                    bp_qty = int(avail_bp // entry_price)
                    qty = max(0, min(max_total_qty, bp_qty))
                    if qty < 1:
                        continue

                    if not self._entry_signal(data[sym], iloc, sym):
                        continue

                    open_positions[sym] = _OpenPos(
                        symbol=sym, entry_time=ts, entry_iloc=iloc, entry_price=entry_price,
                        qty=qty, stop=stop, target=target, peak=entry_price,
                    )
                    cash_used += entry_price * qty

        if cur_day is not None:
            equity_curve.append({"date": str(cur_day), "equity": round(cfg.capital + realized_total, 2)})

        bars_scanned = sum(len(df) for df in data.values())
        metrics = self._metrics(trades, equity_curve, cfg)
        return BacktestResult(
            config=cfg, trades=trades, equity_curve=equity_curve, metrics=metrics,
            symbols=list(data.keys()), bars_scanned=bars_scanned, duration_sec=0.0,
        )

    # ---- metrics ----
    @staticmethod
    def _metrics(trades: List[Trade], equity_curve: List[dict], cfg: BacktestConfig) -> dict:
        n = len(trades)
        if n == 0:
            return {"num_trades": 0, "note": "No trades taken in this window."}
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = sum(t.pnl for t in losses)
        total_pnl = sum(t.pnl for t in trades)
        total_gross = sum(t.gross_pnl for t in trades)
        total_cost = sum(t.cost for t in trades)

        peak = -float("inf")
        max_dd = 0.0
        for pt in equity_curve:
            peak = max(peak, pt["equity"])
            if peak > 0:
                max_dd = max(max_dd, (peak - pt["equity"]) / peak * 100)

        avg_win = (gross_win / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0
        profit_factor = (gross_win / abs(gross_loss)) if gross_loss != 0 else float("inf")

        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        return {
            "num_trades": n,
            "total_pnl": round(total_pnl, 2),
            "total_gross_pnl": round(total_gross, 2),
            "total_cost": round(total_cost, 2),
            "return_on_capital_pct": round(total_pnl / cfg.capital * 100, 2),
            "win_rate_pct": round(len(wins) / n * 100, 1),
            "wins": len(wins),
            "losses": len(losses),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_win_pct": round(sum(t.pnl_pct for t in wins) / len(wins), 3) if wins else 0,
            "avg_loss_pct": round(sum(t.pnl_pct for t in losses) / len(losses), 3) if losses else 0,
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
            "expectancy_per_trade": round(total_pnl / n, 2),
            "win_loss_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss else None,
            "max_drawdown_pct": round(max_dd, 2),
            "exit_reasons": reasons,
            "avg_bars_held": round(sum(t.bars_held for t in trades) / n, 1),
        }
