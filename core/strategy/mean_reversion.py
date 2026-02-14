"""
Mean Reversion Strategy
========================
Identifies price extremes that are likely to revert to the mean.

Logic:
- Bollinger Bands for dynamic support/resistance
- Z-score for measuring deviation from mean
- Volume divergence as confirmation
- Tighter risk parameters (quick in, quick out)

Entry conditions (BUY):
- Price below lower Bollinger Band
- Z-score < -1.5 (1.5 std devs below mean)
- Volume spike (suggests capitulation/reversal)

Entry conditions (SELL):
- Price above upper Bollinger Band
- Z-score > 1.5 (1.5 std devs above mean)
- Volume spike (suggests exhaustion)
"""

import logging
import numpy as np
from typing import List

from core.strategy.base import BaseStrategy, Signal, SignalDirection
from data.ingestion.market_data import Candle

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__("mean_reversion", config)
        self._min_candles = 40

        # Bollinger Band parameters
        self._bb_period = 20
        self._bb_std = 2.0

        # Z-score
        self._zscore_period = 20
        self._zscore_entry = 1.5  # Entry threshold
        self._zscore_extreme = 2.5  # Strong signal threshold

        # Volume
        self._volume_period = 20
        self._volume_spike = 1.5  # 1.5x average = spike

        # Risk - tighter for mean reversion (quick trades)
        self._default_stop_loss = config.risk.stop_loss_pct * 0.8  # Tighter stops
        self._default_take_profit = config.risk.take_profit_pct * 0.7  # Quicker exits

    def analyze(self, candles: List[Candle], symbol: str) -> Signal:
        if not self._has_enough_data(candles, symbol):
            return self._hold_signal(symbol, "insufficient_data")

        closes = np.array(self._closes(candles))
        volumes = np.array(self._volumes(candles))
        current_price = closes[-1]

        # Calculate indicators
        sma, upper_band, lower_band = self._calc_bollinger(
            closes, self._bb_period, self._bb_std
        )
        zscore = self._calc_zscore(closes, self._zscore_period)
        vol_ratio = volumes[-1] / np.mean(volumes[-self._volume_period:]) if np.mean(volumes[-self._volume_period:]) > 0 else 0

        current_zscore = zscore[-1]
        current_sma = sma[-1]
        current_upper = upper_band[-1]
        current_lower = lower_band[-1]

        # Band width (volatility measure)
        band_width = (current_upper - current_lower) / current_sma if current_sma > 0 else 0

        # Score components
        bb_score = 0.0
        zscore_score = 0.0
        volume_score = 0.0
        direction = SignalDirection.HOLD

        # BUY: Price below lower band, negative z-score
        if current_price <= current_lower and current_zscore < -self._zscore_entry:
            direction = SignalDirection.BUY

            # BB score: how far below the band
            bb_distance = (current_lower - current_price) / current_lower if current_lower > 0 else 0
            bb_score = min(bb_distance / 0.02, 1.0)  # 2% below band = 1.0

            # Z-score intensity
            z_intensity = abs(current_zscore) - self._zscore_entry
            zscore_score = min(z_intensity / (self._zscore_extreme - self._zscore_entry), 1.0)

            # Volume spike = capitulation = good for mean reversion
            if vol_ratio >= self._volume_spike:
                volume_score = min((vol_ratio - 1.0) / 2.0, 1.0)
            else:
                volume_score = 0.2  # Low volume mean reversion is weaker

        # SELL: Price above upper band, positive z-score
        elif current_price >= current_upper and current_zscore > self._zscore_entry:
            direction = SignalDirection.SELL

            bb_distance = (current_price - current_upper) / current_upper if current_upper > 0 else 0
            bb_score = min(bb_distance / 0.02, 1.0)

            z_intensity = current_zscore - self._zscore_entry
            zscore_score = min(z_intensity / (self._zscore_extreme - self._zscore_entry), 1.0)

            if vol_ratio >= self._volume_spike:
                volume_score = min((vol_ratio - 1.0) / 2.0, 1.0)
            else:
                volume_score = 0.2

        if direction == SignalDirection.HOLD:
            return self._hold_signal(symbol, "within_bands")

        # Combined strength: BB 30%, Z-score 45%, Volume 25%
        strength = (bb_score * 0.30) + (zscore_score * 0.45) + (volume_score * 0.25)
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
                "sma": round(current_sma, 6),
                "upper_band": round(current_upper, 6),
                "lower_band": round(current_lower, 6),
                "band_width": round(band_width, 6),
                "zscore": round(current_zscore, 4),
                "bb_score": round(bb_score, 4),
                "zscore_score": round(zscore_score, 4),
                "volume_ratio": round(vol_ratio, 4),
                "volume_spike": vol_ratio >= self._volume_spike,
            },
        )

    # =========================================================================
    # INDICATOR CALCULATIONS
    # =========================================================================

    @staticmethod
    def _calc_bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0):
        """Calculate Bollinger Bands."""
        sma = np.zeros_like(closes)
        upper = np.zeros_like(closes)
        lower = np.zeros_like(closes)

        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1: i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            sma[i] = mean
            upper[i] = mean + (num_std * std)
            lower[i] = mean - (num_std * std)

        return sma, upper, lower

    @staticmethod
    def _calc_zscore(closes: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate rolling z-score."""
        zscore = np.zeros_like(closes)
        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1: i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            if std > 0:
                zscore[i] = (closes[i] - mean) / std
        return zscore
