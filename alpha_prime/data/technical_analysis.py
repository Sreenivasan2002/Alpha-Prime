"""
Technical Analysis Engine for Alpha-Prime
Computes indicators, detects candlestick patterns, and identifies chart patterns.
Uses pandas-ta for comprehensive technical indicator computation.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class TechnicalSignal:
    """Represents a technical analysis signal"""
    indicator: str
    signal: str  # BUY, SELL, NEUTRAL
    strength: float  # 0.0 to 1.0
    value: float
    description: str


@dataclass
class PatternSignal:
    """Represents a candlestick or chart pattern signal"""
    pattern_name: str
    signal: str  # BULLISH, BEARISH, NEUTRAL
    strength: float
    description: str


@dataclass
class TechnicalReport:
    """Complete technical analysis report for a symbol"""
    symbol: str
    current_price: float
    signals: List[TechnicalSignal] = field(default_factory=list)
    patterns: List[PatternSignal] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    overall_signal: str = "NEUTRAL"  # BUY, SELL, HOLD
    overall_strength: float = 0.0
    summary: str = ""

    volume_confirmed: bool = False  # Whether volume supports the move

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "trend": self.trend,
            "overall_signal": self.overall_signal,
            "overall_strength": round(self.overall_strength, 3),
            "volume_confirmed": self.volume_confirmed,
            "support_levels": [round(s, 2) for s in self.support_levels],
            "resistance_levels": [round(r, 2) for r in self.resistance_levels],
            "signals": [
                {"indicator": s.indicator, "signal": s.signal,
                 "strength": round(s.strength, 3), "value": round(s.value, 4),
                 "description": s.description}
                for s in self.signals
            ],
            "patterns": [
                {"pattern": p.pattern_name, "signal": p.signal,
                 "strength": round(p.strength, 3), "description": p.description}
                for p in self.patterns
            ],
            "summary": self.summary
        }


class TechnicalAnalyzer:
    """Comprehensive technical analysis engine"""

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Optional[TechnicalReport]:
        """
        Run full technical analysis on OHLCV DataFrame.
        Returns a TechnicalReport with all signals.
        """
        if df is None or df.empty or len(df) < 10:
            logger.warning(f"Insufficient data for analysis ({len(df) if df is not None else 0} rows): {symbol}")
            return None

        try:
            current_price = float(df["Close"].iloc[-1])
            report = TechnicalReport(symbol=symbol, current_price=current_price)

            # Compute indicators
            self._compute_moving_averages(df, report)
            self._compute_rsi(df, report)
            self._compute_macd(df, report)
            self._compute_bollinger_bands(df, report)
            self._compute_stochastic(df, report)
            self._compute_atr(df, report)
            self._compute_vwap(df, report)
            self._compute_adx(df, report)
            self._compute_volume_confirmation(df, report)
            self._detect_candlestick_patterns(df, report)
            self._compute_support_resistance(df, report)
            self._determine_trend(df, report)
            self._compute_overall_signal(report)
            self._generate_summary(report)

            return report
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return None

    def _compute_moving_averages(self, df: pd.DataFrame, report: TechnicalReport):
        """Compute SMA and EMA signals"""
        close = df["Close"]
        current_price = report.current_price

        # SMA 20 / 50 / 200
        for period in [20, 50, 200]:
            if len(df) >= period:
                sma = ta.sma(close, length=period)
                if sma is not None and not sma.empty:
                    sma_val = float(sma.iloc[-1])
                    if current_price > sma_val:
                        signal = "BUY"
                        strength = min((current_price - sma_val) / sma_val * 10, 1.0)
                    else:
                        signal = "SELL"
                        strength = min((sma_val - current_price) / sma_val * 10, 1.0)

                    report.signals.append(TechnicalSignal(
                        indicator=f"SMA_{period}",
                        signal=signal,
                        strength=abs(strength),
                        value=sma_val,
                        description=f"Price {'above' if signal == 'BUY' else 'below'} SMA{period} ({sma_val:.2f})"
                    ))

        # EMA crossover (9/21)
        if len(df) >= 21:
            ema9 = ta.ema(close, length=9)
            ema21 = ta.ema(close, length=21)
            if ema9 is not None and ema21 is not None:
                ema9_val = float(ema9.iloc[-1])
                ema21_val = float(ema21.iloc[-1])
                ema9_prev = float(ema9.iloc[-2])
                ema21_prev = float(ema21.iloc[-2])

                if ema9_val > ema21_val and ema9_prev <= ema21_prev:
                    report.signals.append(TechnicalSignal(
                        indicator="EMA_CROSSOVER",
                        signal="BUY",
                        strength=0.8,
                        value=ema9_val,
                        description="Bullish EMA crossover (EMA9 crossed above EMA21)"
                    ))
                elif ema9_val < ema21_val and ema9_prev >= ema21_prev:
                    report.signals.append(TechnicalSignal(
                        indicator="EMA_CROSSOVER",
                        signal="SELL",
                        strength=0.8,
                        value=ema9_val,
                        description="Bearish EMA crossover (EMA9 crossed below EMA21)"
                    ))

    def _compute_rsi(self, df: pd.DataFrame, report: TechnicalReport):
        """Compute RSI signal"""
        if len(df) < 14:
            return
        rsi = ta.rsi(df["Close"], length=14)
        if rsi is None or rsi.empty:
            return
        rsi_val = float(rsi.iloc[-1])

        if rsi_val < 30:
            signal = "BUY"
            strength = (30 - rsi_val) / 30
            desc = f"RSI oversold at {rsi_val:.1f} - strong reversal up"
        elif rsi_val > 70:
            signal = "SELL"
            strength = (rsi_val - 70) / 30
            desc = f"RSI overbought at {rsi_val:.1f} - strong reversal down"
        elif rsi_val < 45:
            signal = "BUY"
            strength = 0.3
            desc = f"RSI in lower range at {rsi_val:.1f} - mild bullish bias"
        elif rsi_val > 55:
            signal = "SELL"
            strength = 0.3
            desc = f"RSI in upper range at {rsi_val:.1f} - mild bearish bias"
        else:
            signal = "NEUTRAL"
            strength = 0.0
            desc = f"RSI neutral at {rsi_val:.1f}"

        report.signals.append(TechnicalSignal(
            indicator="RSI",
            signal=signal,
            strength=min(strength, 1.0),
            value=rsi_val,
            description=desc
        ))

    def _compute_macd(self, df: pd.DataFrame, report: TechnicalReport):
        """Compute MACD signal"""
        if len(df) < 26:
            return
        macd_df = ta.macd(df["Close"])
        if macd_df is None or macd_df.empty:
            return

        macd_cols = macd_df.columns
        macd_line = float(macd_df[macd_cols[0]].iloc[-1])
        signal_line = float(macd_df[macd_cols[2]].iloc[-1])
        histogram = float(macd_df[macd_cols[1]].iloc[-1])

        prev_macd = float(macd_df[macd_cols[0]].iloc[-2])
        prev_signal = float(macd_df[macd_cols[2]].iloc[-2])

        if macd_line > signal_line and prev_macd <= prev_signal:
            signal = "BUY"
            strength = 0.85
            desc = "MACD bullish crossover"
        elif macd_line < signal_line and prev_macd >= prev_signal:
            signal = "SELL"
            strength = 0.85
            desc = "MACD bearish crossover"
        elif histogram > 0:
            signal = "BUY"
            strength = 0.4
            desc = f"MACD histogram positive ({histogram:.4f})"
        else:
            signal = "SELL"
            strength = 0.4
            desc = f"MACD histogram negative ({histogram:.4f})"

        report.signals.append(TechnicalSignal(
            indicator="MACD",
            signal=signal,
            strength=strength,
            value=macd_line,
            description=desc
        ))

    def _compute_bollinger_bands(self, df: pd.DataFrame, report: TechnicalReport):
        """Compute Bollinger Bands signal"""
        if len(df) < 20:
            return
        bb = ta.bbands(df["Close"], length=20, std=2)
        if bb is None or bb.empty:
            return

        bb_cols = bb.columns
        upper = float(bb[bb_cols[2]].iloc[-1])  # Upper band
        lower = float(bb[bb_cols[0]].iloc[-1])  # Lower band
        middle = float(bb[bb_cols[1]].iloc[-1])  # Middle band
        price = report.current_price

        bandwidth = (upper - lower) / middle if middle != 0 else 0
        pct_b = (price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5

        if price <= lower:
            signal = "BUY"
            strength = 0.8
            desc = f"Price at lower Bollinger Band ({lower:.2f}) - oversold"
        elif price >= upper:
            signal = "SELL"
            strength = 0.8
            desc = f"Price at upper Bollinger Band ({upper:.2f}) - overbought"
        elif pct_b < 0.3:
            signal = "BUY"
            strength = 0.35
            desc = f"Price in lower Bollinger zone (%B: {pct_b:.2f}) - bullish bias"
        elif pct_b > 0.7:
            signal = "SELL"
            strength = 0.35
            desc = f"Price in upper Bollinger zone (%B: {pct_b:.2f}) - bearish bias"
        else:
            signal = "NEUTRAL"
            strength = 0.0
            desc = f"Price within Bollinger Bands (%B: {pct_b:.2f})"

        report.signals.append(TechnicalSignal(
            indicator="BOLLINGER",
            signal=signal,
            strength=strength,
            value=pct_b,
            description=desc
        ))

    def _compute_stochastic(self, df: pd.DataFrame, report: TechnicalReport):
        """Compute Stochastic Oscillator"""
        if len(df) < 14:
            return
        stoch = ta.stoch(df["High"], df["Low"], df["Close"])
        if stoch is None or stoch.empty:
            return

        stoch_cols = stoch.columns
        k = float(stoch[stoch_cols[0]].iloc[-1])
        d = float(stoch[stoch_cols[1]].iloc[-1])

        if k < 20 and d < 20:
            signal = "BUY"
            strength = 0.75
            desc = f"Stochastic oversold (K:{k:.1f}, D:{d:.1f})"
        elif k > 80 and d > 80:
            signal = "SELL"
            strength = 0.75
            desc = f"Stochastic overbought (K:{k:.1f}, D:{d:.1f})"
        elif k < 35:
            signal = "BUY"
            strength = 0.3
            desc = f"Stochastic in lower zone (K:{k:.1f}, D:{d:.1f})"
        elif k > 65:
            signal = "SELL"
            strength = 0.3
            desc = f"Stochastic in upper zone (K:{k:.1f}, D:{d:.1f})"
        else:
            signal = "NEUTRAL"
            strength = 0.0
            desc = f"Stochastic neutral (K:{k:.1f}, D:{d:.1f})"

        report.signals.append(TechnicalSignal(
            indicator="STOCHASTIC",
            signal=signal,
            strength=strength,
            value=k,
            description=desc
        ))

    def _compute_atr(self, df: pd.DataFrame, report: TechnicalReport):
        """Compute Average True Range for volatility assessment"""
        if len(df) < 14:
            return
        atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        if atr is None or atr.empty:
            return
        atr_val = float(atr.iloc[-1])
        atr_pct = atr_val / report.current_price * 100

        report.signals.append(TechnicalSignal(
            indicator="ATR",
            signal="NEUTRAL",
            strength=0.0,
            value=atr_val,
            description=f"ATR: {atr_val:.2f} ({atr_pct:.2f}% of price) - {'High' if atr_pct > 3 else 'Normal'} volatility"
        ))

    def _compute_vwap(self, df: pd.DataFrame, report: TechnicalReport):
        """Compute VWAP if volume data available"""
        if "Volume" not in df.columns or len(df) < 2:
            return
        try:
            vwap = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
            if vwap is not None and not vwap.empty:
                vwap_val = float(vwap.iloc[-1])
                price = report.current_price

                if price > vwap_val:
                    signal = "BUY"
                    strength = min((price - vwap_val) / vwap_val * 20, 1.0)
                    desc = f"Price above VWAP ({vwap_val:.2f}) - bullish intraday"
                else:
                    signal = "SELL"
                    strength = min((vwap_val - price) / vwap_val * 20, 1.0)
                    desc = f"Price below VWAP ({vwap_val:.2f}) - bearish intraday"

                report.signals.append(TechnicalSignal(
                    indicator="VWAP",
                    signal=signal,
                    strength=strength,
                    value=vwap_val,
                    description=desc
                ))
        except Exception:
            pass

    def _compute_adx(self, df: pd.DataFrame, report: TechnicalReport):
        """Compute ADX for trend strength"""
        if len(df) < 14:
            return
        adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        if adx_df is None or adx_df.empty:
            return

        adx_cols = adx_df.columns
        adx_val = float(adx_df[adx_cols[0]].iloc[-1])
        plus_di = float(adx_df[adx_cols[1]].iloc[-1])
        minus_di = float(adx_df[adx_cols[2]].iloc[-1])

        if adx_val > 25:
            if plus_di > minus_di:
                signal = "BUY"
                desc = f"Strong uptrend (ADX:{adx_val:.1f}, +DI:{plus_di:.1f} > -DI:{minus_di:.1f})"
            else:
                signal = "SELL"
                desc = f"Strong downtrend (ADX:{adx_val:.1f}, -DI:{minus_di:.1f} > +DI:{plus_di:.1f})"
            strength = min(adx_val / 50, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 0.0
            desc = f"Weak trend (ADX:{adx_val:.1f})"

        report.signals.append(TechnicalSignal(
            indicator="ADX",
            signal=signal,
            strength=strength,
            value=adx_val,
            description=desc
        ))

    def _compute_volume_confirmation(self, df: pd.DataFrame, report: TechnicalReport):
        """Check if volume confirms the current price move.
        High volume on up-moves = bullish confirmation.
        Low volume on up-moves = weak rally, likely to reverse.
        Volume increasing trend = conviction behind the move.
        """
        if "Volume" not in df.columns or len(df) < 20:
            return

        try:
            vol = df["Volume"]
            close = df["Close"]

            # Skip if volume is all zeros (some data feeds don't have volume)
            if vol.iloc[-20:].sum() == 0:
                return

            # Current volume vs 20-period average
            avg_volume_20 = float(vol.iloc[-20:].mean())
            current_volume = float(vol.iloc[-1])

            if avg_volume_20 <= 0:
                return

            volume_ratio = current_volume / avg_volume_20

            # Check if price is moving up or down
            price_change = float(close.iloc[-1]) - float(close.iloc[-2])
            price_change_pct = (price_change / float(close.iloc[-2])) * 100 if float(close.iloc[-2]) > 0 else 0

            # Volume trend: compare last 5 bars avg vs previous 5 bars avg
            if len(vol) >= 10:
                recent_vol_avg = float(vol.iloc[-5:].mean())
                prior_vol_avg = float(vol.iloc[-10:-5].mean())
                vol_trend_ratio = recent_vol_avg / prior_vol_avg if prior_vol_avg > 0 else 1.0
            else:
                vol_trend_ratio = 1.0

            # Determine volume signal
            if price_change > 0 and volume_ratio >= 1.2:
                # Price up + above-average volume = strong bullish
                signal = "BUY"
                strength = min(volume_ratio / 3.0, 0.9)  # Cap at 0.9
                desc = f"Volume confirmation: {volume_ratio:.1f}x avg volume on up-move (+{price_change_pct:.2f}%)"
                report.volume_confirmed = True
            elif price_change > 0 and volume_ratio < 0.7:
                # Price up but low volume = weak rally
                signal = "SELL"
                strength = 0.4
                desc = f"Volume divergence: price up but volume only {volume_ratio:.1f}x avg - weak rally"
                report.volume_confirmed = False
            elif price_change < 0 and volume_ratio >= 1.5:
                # Price down + high volume = strong selling
                signal = "SELL"
                strength = min(volume_ratio / 3.0, 0.9)
                desc = f"Heavy selling: {volume_ratio:.1f}x avg volume on down-move ({price_change_pct:.2f}%)"
                report.volume_confirmed = False
            elif price_change < 0 and volume_ratio < 0.7:
                # Price down but low volume = not much conviction in selling
                signal = "BUY"
                strength = 0.3
                desc = f"Low-volume dip: volume {volume_ratio:.1f}x avg on down-move - sellers lack conviction"
                report.volume_confirmed = True
            else:
                signal = "NEUTRAL"
                strength = 0.0
                desc = f"Volume neutral: {volume_ratio:.1f}x avg volume"
                report.volume_confirmed = volume_ratio >= 0.9

            # Add volume trend bonus
            if vol_trend_ratio > 1.3 and signal == "BUY":
                strength = min(strength + 0.1, 0.9)
                desc += f" | Volume trending up ({vol_trend_ratio:.1f}x)"

            report.signals.append(TechnicalSignal(
                indicator="VOLUME",
                signal=signal,
                strength=strength,
                value=volume_ratio,
                description=desc
            ))

        except Exception as e:
            logger.debug(f"Volume confirmation error: {e}")

    def _detect_candlestick_patterns(self, df: pd.DataFrame, report: TechnicalReport):
        """Detect candlestick patterns using custom logic (no TA-Lib dependency)"""
        if len(df) < 5:
            return

        try:
            o = df["Open"].values
            h = df["High"].values
            l = df["Low"].values
            c = df["Close"].values

            # Use the last few candles
            idx = -1
            body = c[idx] - o[idx]
            body_abs = abs(body)
            upper_shadow = h[idx] - max(c[idx], o[idx])
            lower_shadow = min(c[idx], o[idx]) - l[idx]
            total_range = h[idx] - l[idx]

            if total_range == 0:
                return

            body_pct = body_abs / total_range

            # Doji - small body relative to range
            if body_pct < 0.1 and total_range > 0:
                report.patterns.append(PatternSignal(
                    pattern_name="Doji",
                    signal="NEUTRAL",
                    strength=0.6,
                    description="Doji pattern - market indecision, potential reversal"
                ))

            # Hammer - small body at top, long lower shadow
            if (lower_shadow > 2 * body_abs and upper_shadow < body_abs * 0.5 and
                body_pct < 0.4 and c[idx] > o[idx]):
                report.patterns.append(PatternSignal(
                    pattern_name="Hammer",
                    signal="BULLISH",
                    strength=0.75,
                    description="Hammer pattern - bullish reversal signal"
                ))

            # Inverted Hammer
            if (upper_shadow > 2 * body_abs and lower_shadow < body_abs * 0.5 and
                body_pct < 0.4 and c[idx] > o[idx]):
                report.patterns.append(PatternSignal(
                    pattern_name="Inverted Hammer",
                    signal="BULLISH",
                    strength=0.65,
                    description="Inverted Hammer - potential bullish reversal"
                ))

            # Shooting Star
            if (upper_shadow > 2 * body_abs and lower_shadow < body_abs * 0.5 and
                body_pct < 0.4 and c[idx] < o[idx]):
                report.patterns.append(PatternSignal(
                    pattern_name="Shooting Star",
                    signal="BEARISH",
                    strength=0.75,
                    description="Shooting Star - bearish reversal signal"
                ))

            # Marubozu (strong body, no/tiny shadows)
            if body_pct > 0.85:
                if body > 0:
                    report.patterns.append(PatternSignal(
                        pattern_name="Bullish Marubozu",
                        signal="BULLISH",
                        strength=0.8,
                        description="Bullish Marubozu - strong buying pressure"
                    ))
                else:
                    report.patterns.append(PatternSignal(
                        pattern_name="Bearish Marubozu",
                        signal="BEARISH",
                        strength=0.8,
                        description="Bearish Marubozu - strong selling pressure"
                    ))

            # Engulfing patterns (need 2 candles)
            if len(df) >= 2:
                prev_body = c[-2] - o[-2]
                curr_body = c[-1] - o[-1]

                # Bullish Engulfing
                if (prev_body < 0 and curr_body > 0 and
                    o[-1] <= c[-2] and c[-1] >= o[-2]):
                    report.patterns.append(PatternSignal(
                        pattern_name="Bullish Engulfing",
                        signal="BULLISH",
                        strength=0.85,
                        description="Bullish Engulfing - strong reversal signal"
                    ))

                # Bearish Engulfing
                if (prev_body > 0 and curr_body < 0 and
                    o[-1] >= c[-2] and c[-1] <= o[-2]):
                    report.patterns.append(PatternSignal(
                        pattern_name="Bearish Engulfing",
                        signal="BEARISH",
                        strength=0.85,
                        description="Bearish Engulfing - strong reversal signal"
                    ))

            # Morning Star (3 candles)
            if len(df) >= 3:
                body_3 = c[-3] - o[-3]
                body_2 = abs(c[-2] - o[-2])
                body_1 = c[-1] - o[-1]
                range_2 = h[-2] - l[-2]

                # Morning Star
                if (body_3 < 0 and body_1 > 0 and
                    range_2 > 0 and body_2 / range_2 < 0.3 and
                    c[-1] > (o[-3] + c[-3]) / 2):
                    report.patterns.append(PatternSignal(
                        pattern_name="Morning Star",
                        signal="BULLISH",
                        strength=0.9,
                        description="Morning Star - strong bullish reversal pattern"
                    ))

                # Evening Star
                if (body_3 > 0 and body_1 < 0 and
                    range_2 > 0 and body_2 / range_2 < 0.3 and
                    c[-1] < (o[-3] + c[-3]) / 2):
                    report.patterns.append(PatternSignal(
                        pattern_name="Evening Star",
                        signal="BEARISH",
                        strength=0.9,
                        description="Evening Star - strong bearish reversal pattern"
                    ))

        except Exception as e:
            logger.debug(f"Candlestick pattern detection error: {e}")

    def _compute_support_resistance(self, df: pd.DataFrame, report: TechnicalReport):
        """Compute support and resistance levels using pivot points"""
        if len(df) < 5:
            return

        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values

        # Recent pivot points
        pivot = (high[-2] + low[-2] + close[-2]) / 3
        r1 = 2 * pivot - low[-2]
        r2 = pivot + (high[-2] - low[-2])
        s1 = 2 * pivot - high[-2]
        s2 = pivot - (high[-2] - low[-2])

        report.resistance_levels = sorted([float(r1), float(r2)], reverse=True)
        report.support_levels = sorted([float(s1), float(s2)])

        # Also find key levels from recent price action
        recent_highs = pd.Series(high[-20:])
        recent_lows = pd.Series(low[-20:])

        # Simple resistance/support from recent peaks/troughs
        price = report.current_price
        near_highs = recent_highs[recent_highs > price].unique()
        near_lows = recent_lows[recent_lows < price].unique()

        if len(near_highs) > 0:
            report.resistance_levels.extend([float(x) for x in sorted(near_highs)[:3]])
        if len(near_lows) > 0:
            report.support_levels.extend([float(x) for x in sorted(near_lows, reverse=True)[:3]])

        report.resistance_levels = sorted(list(set([round(x, 2) for x in report.resistance_levels])))
        report.support_levels = sorted(list(set([round(x, 2) for x in report.support_levels])), reverse=True)

    def _determine_trend(self, df: pd.DataFrame, report: TechnicalReport):
        """Determine overall trend direction"""
        close = df["Close"]

        # Use multiple timeframe analysis
        bullish_count = 0
        bearish_count = 0

        # Short-term trend (5 days)
        if len(close) >= 5:
            if float(close.iloc[-1]) > float(close.iloc[-5]):
                bullish_count += 1
            else:
                bearish_count += 1

        # Medium-term trend (20 days)
        if len(close) >= 20:
            sma20 = ta.sma(close, length=20)
            if sma20 is not None and float(close.iloc[-1]) > float(sma20.iloc[-1]):
                bullish_count += 1
            else:
                bearish_count += 1

        # Long-term trend (50 days)
        if len(close) >= 50:
            sma50 = ta.sma(close, length=50)
            if sma50 is not None and float(close.iloc[-1]) > float(sma50.iloc[-1]):
                bullish_count += 1
            else:
                bearish_count += 1

        if bullish_count > bearish_count:
            report.trend = "BULLISH"
        elif bearish_count > bullish_count:
            report.trend = "BEARISH"
        else:
            report.trend = "NEUTRAL"

    def _compute_overall_signal(self, report: TechnicalReport):
        """Compute overall buy/sell/hold signal from all indicators.
        Stricter scoring: requires strong majority + trend alignment + RSI in safe zone.
        Designed for intraday with 0.5%+ profit target.
        """
        buy_score = 0.0
        sell_score = 0.0
        buy_count = 0
        sell_count = 0
        neutral_count = 0

        for signal in report.signals:
            if signal.signal == "BUY":
                buy_score += signal.strength
                buy_count += 1
            elif signal.signal == "SELL":
                sell_score += signal.strength
                sell_count += 1
            else:
                neutral_count += 1

        # Add pattern signals (reduced weight - patterns alone shouldn't drive trades)
        for pattern in report.patterns:
            if pattern.signal == "BULLISH":
                buy_score += pattern.strength * 0.3
                buy_count += 1
            elif pattern.signal == "BEARISH":
                sell_score += pattern.strength * 0.3
                sell_count += 1

        total_signals = buy_count + sell_count + neutral_count
        total_directional = buy_count + sell_count
        if total_directional == 0:
            report.overall_signal = "HOLD"
            report.overall_strength = 0.0
            return

        # --- STRICT VALIDATION GATES ---
        # Gate 1: Require at least 3 BUY indicators to consider a BUY
        min_indicators_required = 3
        if buy_count < min_indicators_required and sell_count < min_indicators_required:
            report.overall_signal = "HOLD"
            report.overall_strength = 0.0
            return

        # Gate 2: Require clear majority (>60% of directional signals must agree)
        majority_threshold = 0.60
        buy_ratio = buy_count / total_directional
        sell_ratio = sell_count / total_directional

        # Gate 3: Trend must align (no buying in BEARISH trend, no selling in BULLISH)
        trend = report.trend.upper() if report.trend else "NEUTRAL"

        if buy_ratio >= majority_threshold and buy_count >= min_indicators_required:
            # Penalize if trend is against the signal
            if trend == "BEARISH":
                report.overall_signal = "HOLD"
                report.overall_strength = 0.0
                return

            report.overall_signal = "BUY"
            avg_strength = buy_score / buy_count
            # Conservative strength: majority_ratio * avg_strength (no 1.5x multiplier)
            # Cap at 0.85 to prevent overconfidence
            raw_strength = min(buy_ratio * avg_strength, 0.85)

            # Volume confirmation penalty: reduce strength if volume doesn't confirm
            if not report.volume_confirmed:
                raw_strength *= 0.8  # 20% penalty for unconfirmed volume
            report.overall_strength = raw_strength

        elif sell_ratio >= majority_threshold and sell_count >= min_indicators_required:
            if trend == "BULLISH":
                report.overall_signal = "HOLD"
                report.overall_strength = 0.0
                return

            report.overall_signal = "SELL"
            avg_strength = sell_score / sell_count
            report.overall_strength = min(sell_ratio * avg_strength, 0.85)

        else:
            # No clear majority - HOLD
            report.overall_signal = "HOLD"
            report.overall_strength = 0.0

    def _generate_summary(self, report: TechnicalReport):
        """Generate human-readable summary"""
        parts = []
        parts.append(f"{report.symbol} @ {report.current_price:.2f}")
        parts.append(f"Trend: {report.trend}")
        parts.append(f"Signal: {report.overall_signal} (strength: {report.overall_strength:.2f})")

        buy_signals = [s for s in report.signals if s.signal == "BUY"]
        sell_signals = [s for s in report.signals if s.signal == "SELL"]

        if buy_signals:
            parts.append(f"Bullish: {', '.join([s.indicator for s in buy_signals])}")
        if sell_signals:
            parts.append(f"Bearish: {', '.join([s.indicator for s in sell_signals])}")
        if report.patterns:
            parts.append(f"Patterns: {', '.join([p.pattern_name for p in report.patterns])}")
        if report.support_levels:
            parts.append(f"Support: {report.support_levels[0]:.2f}")
        if report.resistance_levels:
            parts.append(f"Resistance: {report.resistance_levels[0]:.2f}")

        report.summary = " | ".join(parts)


# Singleton
technical_analyzer = TechnicalAnalyzer()
