"""
Momentum Strategy
==================
Identifies and follows strong price trends using RSI, MACD, and volume.

Logic:
- RSI for overbought/oversold conditions
- MACD for trend direction and crossovers
- Volume confirmation to filter weak signals
- Combined score determines signal strength

Entry conditions (BUY):
- RSI < 35 (oversold, rebounding)
- MACD line crosses above signal line
- Volume above 20-period average

Entry conditions (SELL):
- RSI > 65 (overbought, reversing)
- MACD line crosses below signal line
- Volume above 20-period average
"""

import logging
import numpy as np
from typing import List

from core.strategy.base import BaseStrategy, Signal, SignalDirection
from data.ingestion.market_data import Candle

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__("momentum", config)
        self._min_candles = 50  # Need enough for MACD (26) + signal (9) + buffer

        # RSI parameters
        self._rsi_period = 14
        self._rsi_oversold = 35
        self._rsi_overbought = 65

        # MACD parameters
        self._macd_fast = 12
        self._macd_slow = 26
        self._macd_signal = 9

        # Volume
        self._volume_period = 20
        self._volume_multiplier = 1.0  # Volume must be >= 1x average

        # Risk
        self._default_stop_loss = config.risk.stop_loss_pct
        self._default_take_profit = config.risk.take_profit_pct

    def analyze(self, candles: List[Candle], symbol: str) -> Signal:
        if not self._has_enough_data(candles, symbol):
            return self._hold_signal(symbol, "insufficient_data")

        closes = np.array(self._closes(candles))
        volumes = np.array(self._volumes(candles))

        # Calculate indicators
        rsi = self._calc_rsi(closes, self._rsi_period)
        macd_line, signal_line, histogram = self._calc_macd(closes)
        vol_ratio = volumes[-1] / np.mean(volumes[-self._volume_period:]) if np.mean(volumes[-self._volume_period:]) > 0 else 0

        current_rsi = rsi[-1]
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        prev_macd = macd_line[-2]
        prev_signal = signal_line[-2]
        current_hist = histogram[-1]
        prev_hist = histogram[-2]

        # Score components (each 0-1)
        rsi_score = 0.0
        macd_score = 0.0
        volume_score = min(vol_ratio / 2.0, 1.0)  # Normalize: 2x avg = 1.0

        direction = SignalDirection.HOLD

        # BUY signals
        if current_rsi < self._rsi_oversold:
            rsi_score = (self._rsi_oversold - current_rsi) / self._rsi_oversold
            if prev_macd <= prev_signal and current_macd > current_signal:
                macd_score = 1.0  # Fresh crossover
            elif current_hist > prev_hist and current_hist > 0:
                macd_score = 0.6  # Accelerating bullish momentum
            elif current_macd > current_signal:
                macd_score = 0.4  # Already above signal
            if rsi_score > 0.2 and macd_score > 0.3:
                direction = SignalDirection.BUY

        # SELL signals
        elif current_rsi > self._rsi_overbought:
            rsi_score = (current_rsi - self._rsi_overbought) / (100 - self._rsi_overbought)
            if prev_macd >= prev_signal and current_macd < current_signal:
                macd_score = 1.0  # Fresh crossover
            elif current_hist < prev_hist and current_hist < 0:
                macd_score = 0.6  # Accelerating bearish momentum
            elif current_macd < current_signal:
                macd_score = 0.4  # Already below signal
            if rsi_score > 0.2 and macd_score > 0.3:
                direction = SignalDirection.SELL

        if direction == SignalDirection.HOLD:
            return self._hold_signal(symbol, "no_signal")

        # Volume confirmation
        has_volume = vol_ratio >= self._volume_multiplier
        if not has_volume:
            volume_score *= 0.5  # Penalize but don't discard

        # Combined strength: RSI 35%, MACD 45%, Volume 20%
        strength = (rsi_score * 0.35) + (macd_score * 0.45) + (volume_score * 0.20)
        strength = min(max(strength, 0.0), 1.0)

        # Expected profit scales with signal strength
        expected_profit = self._default_take_profit * strength

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=round(strength, 4),
            strategy=self.name,
            expected_profit_pct=round(expected_profit, 4),
            stop_loss_pct=self._default_stop_loss,
            take_profit_pct=self._default_take_profit,
            metadata={
                "rsi": round(current_rsi, 2),
                "rsi_score": round(rsi_score, 4),
                "macd": round(current_macd, 6),
                "macd_signal": round(current_signal, 6),
                "macd_histogram": round(current_hist, 6),
                "macd_score": round(macd_score, 4),
                "volume_ratio": round(vol_ratio, 4),
                "volume_score": round(volume_score, 4),
                "volume_confirmed": has_volume,
            },
        )

    # =========================================================================
    # INDICATOR CALCULATIONS
    # =========================================================================

    @staticmethod
    def _calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI using exponential moving average."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.zeros_like(closes)
        avg_loss = np.zeros_like(closes)

        # SMA for first value
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        # EMA for subsequent
        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _calc_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.zeros_like(data)
        multiplier = 2.0 / (period + 1)
        ema[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        return ema

    def _calc_macd(self, closes: np.ndarray):
        """Calculate MACD line, signal line, and histogram."""
        ema_fast = self._calc_ema(closes, self._macd_fast)
        ema_slow = self._calc_ema(closes, self._macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calc_ema(macd_line, self._macd_signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
