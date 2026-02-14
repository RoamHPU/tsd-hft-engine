"""
Strategy Base Class
====================
Abstract interface that all trading strategies must implement.
Defines the contract: receive candles, produce signals.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from data.ingestion.market_data import Candle

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal produced by a strategy."""
    symbol: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    strategy: str
    expected_profit_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_actionable(self) -> bool:
        """Signal is worth acting on (not HOLD and strong enough)."""
        return self.direction != SignalDirection.HOLD and self.strength >= 0.3


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.
    
    Every strategy must:
    1. Accept a list of candles
    2. Produce a Signal (BUY, SELL, or HOLD)
    3. Report its own confidence/strength
    """

    def __init__(self, name: str, config):
        self.name = name
        self._config = config
        self._min_candles = 30  # Minimum candles needed to generate signal
        logger.info(f"Strategy initialized: {name}")

    @abstractmethod
    def analyze(self, candles: List[Candle], symbol: str) -> Signal:
        """
        Analyze candle data and produce a trading signal.
        
        Args:
            candles: List of OHLCV candles, oldest first
            symbol: Trading pair symbol
            
        Returns:
            Signal with direction, strength, and risk parameters
        """
        pass

    def _hold_signal(self, symbol: str, reason: str = "") -> Signal:
        """Convenience: return a HOLD signal."""
        return Signal(
            symbol=symbol,
            direction=SignalDirection.HOLD,
            strength=0.0,
            strategy=self.name,
            expected_profit_pct=0.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            metadata={"reason": reason},
        )

    def _has_enough_data(self, candles: List[Candle], symbol: str) -> bool:
        """Check if we have enough candles to analyze."""
        if len(candles) < self._min_candles:
            logger.debug(
                f"{self.name}: Not enough data for {symbol} "
                f"({len(candles)}/{self._min_candles})"
            )
            return False
        return True

    @staticmethod
    def _closes(candles: List[Candle]) -> List[float]:
        """Extract close prices from candles."""
        return [c.close for c in candles]

    @staticmethod
    def _volumes(candles: List[Candle]) -> List[float]:
        """Extract volumes from candles."""
        return [c.volume for c in candles]

    @staticmethod
    def _highs(candles: List[Candle]) -> List[float]:
        """Extract high prices from candles."""
        return [c.high for c in candles]

    @staticmethod
    def _lows(candles: List[Candle]) -> List[float]:
        """Extract low prices from candles."""
        return [c.low for c in candles]
