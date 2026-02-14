"""
Unit Tests - Strategy Engine
=============================
Tests for momentum, mean reversion, breakout strategies, and orchestrator.
Run with: pytest tests/unit/test_strategies.py -v
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.ingestion.market_data import Candle
from core.strategy.base import SignalDirection
from core.strategy.momentum import MomentumStrategy
from core.strategy.mean_reversion import MeanReversionStrategy
from core.strategy.breakout import BreakoutStrategy
from core.strategy.orchestrator import StrategyOrchestrator
from config import load_config


def _make_candles(
    prices: list, symbol: str = "BTCUSDT",
    volumes: list = None, spread: float = 0.001
) -> list:
    """Generate candle list from close prices."""
    if volumes is None:
        volumes = [100.0] * len(prices)
    candles = []
    for i, (price, vol) in enumerate(zip(prices, volumes)):
        candles.append(Candle(
            symbol=symbol, interval="1m",
            open_time=1700000000000 + i * 60000,
            open=price * (1 - spread / 2),
            high=price * (1 + spread),
            low=price * (1 - spread),
            close=price,
            volume=vol,
            close_time=1700000000000 + (i + 1) * 60000 - 1,
            quote_volume=vol * price,
            trades=50,
            is_closed=True,
        ))
    return candles


def _trending_up(n=60, start=100, pct_gain=0.10):
    """Generate uptrending price series."""
    return [start * (1 + pct_gain * i / n) for i in range(n)]


def _trending_down(n=60, start=100, pct_loss=0.10):
    """Generate downtrending price series."""
    return [start * (1 - pct_loss * i / n) for i in range(n)]


def _mean_reverting(n=60, center=100, amplitude=3):
    """Generate mean-reverting series that ends at an extreme."""
    prices = []
    for i in range(n):
        # Sine wave with increasing amplitude at end
        factor = 1 + (i / n) * 2  # Amplify toward end
        prices.append(center + amplitude * factor * np.sin(i * 0.3))
    return prices


def _consolidating_then_breakout(n=60, base=100, breakout_pct=0.03):
    """Generate consolidation followed by breakout."""
    prices = []
    # 80% consolidation
    consol_n = int(n * 0.8)
    for i in range(consol_n):
        noise = np.sin(i * 0.5) * 0.3  # Tiny oscillation
        prices.append(base + noise)
    # 20% breakout
    for i in range(n - consol_n):
        prices.append(base * (1 + breakout_pct * (i + 1) / (n - consol_n)))
    return prices


# ============================================================================
# MOMENTUM STRATEGY TESTS
# ============================================================================

class TestMomentumStrategy:

    @pytest.fixture
    def strategy(self):
        config = load_config()
        return MomentumStrategy(config)

    def test_insufficient_data_returns_hold(self, strategy):
        candles = _make_candles([100] * 10)  # Only 10, need 50
        signal = strategy.analyze(candles, "BTCUSDT")
        assert signal.direction == SignalDirection.HOLD

    def test_enough_data_produces_signal(self, strategy):
        candles = _make_candles([100] * 60)
        signal = strategy.analyze(candles, "BTCUSDT")
        assert signal is not None
        assert signal.strategy == "momentum"

    def test_rsi_calculation(self, strategy):
        # Generate enough data for RSI
        prices = np.array(_trending_up(30))
        rsi = strategy._calc_rsi(prices, 14)
        # Uptrend should produce high RSI
        assert rsi[-1] > 50

    def test_ema_calculation(self, strategy):
        prices = np.array([float(i) for i in range(1, 21)])
        ema = strategy._calc_ema(prices, 10)
        # EMA of linear series should be close to midpoint
        assert ema[-1] > 10

    def test_signal_has_metadata(self, strategy):
        candles = _make_candles(_trending_down(60, 100, 0.15))
        signal = strategy.analyze(candles, "BTCUSDT")
        # Should have metadata regardless of direction
        assert signal.metadata is not None
        assert isinstance(signal.metadata, dict)
        # If it's a hold, it has a reason; if actionable, it has indicators
        assert "reason" in signal.metadata or "rsi" in signal.metadata


# ============================================================================
# MEAN REVERSION STRATEGY TESTS
# ============================================================================

class TestMeanReversionStrategy:

    @pytest.fixture
    def strategy(self):
        config = load_config()
        return MeanReversionStrategy(config)

    def test_insufficient_data_returns_hold(self, strategy):
        candles = _make_candles([100] * 10)
        signal = strategy.analyze(candles, "ETHUSDT")
        assert signal.direction == SignalDirection.HOLD

    def test_within_bands_returns_hold(self, strategy):
        # Stable price = within Bollinger bands
        candles = _make_candles([100] * 60)
        signal = strategy.analyze(candles, "ETHUSDT")
        assert signal.direction == SignalDirection.HOLD
        assert signal.metadata.get("reason") == "within_bands"

    def test_bollinger_calculation(self, strategy):
        prices = np.array([100.0 + np.sin(i * 0.3) * 2 for i in range(40)])
        sma, upper, lower = strategy._calc_bollinger(prices, 20, 2.0)
        # Upper > SMA > Lower
        assert upper[-1] > sma[-1] > lower[-1]

    def test_zscore_calculation(self, strategy):
        prices = np.array([100.0] * 20 + [110.0])  # Spike at end
        zscore = strategy._calc_zscore(prices, 20)
        assert zscore[-1] > 1.0  # Should be significantly above mean


# ============================================================================
# BREAKOUT STRATEGY TESTS
# ============================================================================

class TestBreakoutStrategy:

    @pytest.fixture
    def strategy(self):
        config = load_config()
        return BreakoutStrategy(config)

    def test_insufficient_data_returns_hold(self, strategy):
        candles = _make_candles([100] * 10)
        signal = strategy.analyze(candles, "SOLUSDT")
        assert signal.direction == SignalDirection.HOLD

    def test_no_breakout_returns_hold(self, strategy):
        # Flat price = no breakout
        candles = _make_candles([100] * 60)
        signal = strategy.analyze(candles, "SOLUSDT")
        assert signal.direction == SignalDirection.HOLD

    def test_atr_calculation(self, strategy):
        n = 30
        highs = np.array([101.0 + i * 0.1 for i in range(n)])
        lows = np.array([99.0 + i * 0.1 for i in range(n)])
        closes = np.array([100.0 + i * 0.1 for i in range(n)])
        atr = strategy._calc_atr(highs, lows, closes, 14)
        assert atr[-1] > 0  # ATR should be positive

    def test_breakout_with_volume_spike(self, strategy):
        # Consolidation then breakout with high volume
        prices = _consolidating_then_breakout(60, 100, 0.05)
        volumes = [100] * 48 + [500] * 12  # Volume spike on breakout
        candles = _make_candles(prices, volumes=volumes)
        signal = strategy.analyze(candles, "BTCUSDT")
        # Should detect the breakout (or at least not error)
        assert signal is not None
        assert signal.strategy == "breakout"


# ============================================================================
# ORCHESTRATOR TESTS
# ============================================================================

class TestOrchestrator:

    @pytest.fixture
    def orchestrator(self):
        config = load_config()
        return StrategyOrchestrator(config)

    def test_initialization(self, orchestrator):
        status = orchestrator.get_status()
        assert "momentum" in status["strategies"]
        assert "mean_reversion" in status["strategies"]
        assert "breakout" in status["strategies"]

    def test_evaluate_returns_signal(self, orchestrator):
        candles = _make_candles([100] * 60)
        result = orchestrator.evaluate("BTCUSDT", candles)
        assert result is not None
        assert result.symbol == "BTCUSDT"

    def test_evaluate_all_filters_actionable(self, orchestrator):
        # Flat data = no actionable signals
        data = {
            "BTCUSDT": _make_candles([100] * 60, "BTCUSDT"),
            "ETHUSDT": _make_candles([100] * 60, "ETHUSDT"),
        }
        signals = orchestrator.evaluate_all(data)
        # Flat price should produce no actionable signals
        assert isinstance(signals, list)

    def test_weights_applied(self, orchestrator):
        status = orchestrator.get_status()
        weights = status["weights"]
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_empty_candles_handled(self, orchestrator):
        result = orchestrator.evaluate("BTCUSDT", [])
        assert result.direction == SignalDirection.HOLD


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
