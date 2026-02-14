"""
Breakout Strategy
==================
Identifies price breakouts from consolidation ranges.

Logic:
- Dynamic support/resistance from recent highs/lows
- ATR for volatility-adjusted breakout thresholds
- Volume surge confirms genuine breakout vs fakeout
- Wider take-profit (breakouts run further)

Entry conditions (BUY):
- Price breaks above resistance level
- Breakout exceeds ATR-based threshold
- Volume >= 2x average (strong conviction)

Entry conditions (SELL):
- Price breaks below support level
- Breakdown exceeds ATR-based threshold
- Volume >= 2x average
"""

import logging
import numpy as np
from typing import List

from core.strategy.base import BaseStrategy, Signal, SignalDirection
from data.ingestion.market_data import Candle

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__("breakout", config)
        self._min_candles = 50

        # Support/Resistance lookback
        self._sr_period = 20  # Candles to find S/R levels
        self._consolidation_period = 10  # Recent candles for range detection

        # ATR parameters
        self._atr_period = 14
        self._atr_multiplier = 0.5  # Breakout must exceed 0.5x ATR

        # Volume
        self._volume_period = 20
        self._volume_breakout = 2.0  # 2x average for breakout confirmation

        # Risk - wider for breakouts (they run further)
        self._default_stop_loss = config.risk.stop_loss_pct
        self._default_take_profit = config.risk.take_profit_pct * 1.5  # Wider TP

    def analyze(self, candles: List[Candle], symbol: str) -> Signal:
        if not self._has_enough_data(candles, symbol):
            return self._hold_signal(symbol, "insufficient_data")

        closes = np.array(self._closes(candles))
        highs = np.array(self._highs(candles))
        lows = np.array(self._lows(candles))
        volumes = np.array(self._volumes(candles))

        current_price = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]

        # Calculate levels
        resistance = np.max(highs[-self._sr_period - 1:-1])  # Exclude current candle
        support = np.min(lows[-self._sr_period - 1:-1])
        atr = self._calc_atr(highs, lows, closes, self._atr_period)
        current_atr = atr[-1]

        # Volume analysis
        avg_volume = np.mean(volumes[-self._volume_period:])
        vol_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 0

        # Range width (consolidation detection)
        recent_highs = highs[-self._consolidation_period:]
        recent_lows = lows[-self._consolidation_period:]
        range_width = (np.max(recent_highs) - np.min(recent_lows))
        range_pct = range_width / current_price * 100 if current_price > 0 else 0

        # Breakout threshold
        breakout_threshold = current_atr * self._atr_multiplier

        direction = SignalDirection.HOLD
        breakout_score = 0.0
        volume_score = 0.0
        range_score = 0.0

        # BULLISH BREAKOUT: price breaks above resistance
        breakout_distance = current_high - resistance
        if breakout_distance > breakout_threshold:
            direction = SignalDirection.BUY

            # How far above resistance (normalized by ATR)
            breakout_score = min(breakout_distance / (current_atr * 2), 1.0)

            # Volume confirmation
            if vol_ratio >= self._volume_breakout:
                volume_score = min((vol_ratio - 1.0) / 3.0, 1.0)
            else:
                volume_score = 0.1  # Weak without volume

            # Tighter consolidation before breakout = stronger signal
            if range_pct < 3.0:  # Tight range
                range_score = 0.8
            elif range_pct < 5.0:
                range_score = 0.5
            else:
                range_score = 0.2

        # BEARISH BREAKDOWN: price breaks below support
        breakdown_distance = support - current_low
        if breakdown_distance > breakout_threshold and direction == SignalDirection.HOLD:
            direction = SignalDirection.SELL

            breakout_score = min(breakdown_distance / (current_atr * 2), 1.0)

            if vol_ratio >= self._volume_breakout:
                volume_score = min((vol_ratio - 1.0) / 3.0, 1.0)
            else:
                volume_score = 0.1

            if range_pct < 3.0:
                range_score = 0.8
            elif range_pct < 5.0:
                range_score = 0.5
            else:
                range_score = 0.2

        if direction == SignalDirection.HOLD:
            return self._hold_signal(symbol, "no_breakout")

        # Combined strength: Breakout 35%, Volume 40%, Range 25%
        # Volume weighted heavily — breakouts without volume are fakeouts
        strength = (breakout_score * 0.35) + (volume_score * 0.40) + (range_score * 0.25)
        strength = min(max(strength, 0.0), 1.0)

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
                "price": current_price,
                "resistance": round(resistance, 6),
                "support": round(support, 6),
                "atr": round(current_atr, 6),
                "breakout_threshold": round(breakout_threshold, 6),
                "breakout_distance": round(max(breakout_distance, breakdown_distance), 6),
                "breakout_score": round(breakout_score, 4),
                "volume_ratio": round(vol_ratio, 4),
                "volume_confirmed": vol_ratio >= self._volume_breakout,
                "range_pct": round(range_pct, 4),
                "range_score": round(range_score, 4),
            },
        )

    # =========================================================================
    # INDICATOR CALCULATIONS
    # =========================================================================

    @staticmethod
    def _calc_atr(highs: np.ndarray, lows: np.ndarray,
                  closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        atr = np.zeros_like(closes)
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(closes)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr
