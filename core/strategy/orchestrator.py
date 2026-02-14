"""
Strategy Orchestrator
======================
Aggregates signals from all strategies using weighted scoring.
Produces a single actionable decision per symbol per evaluation cycle.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.strategy.base import BaseStrategy, Signal, SignalDirection
from core.strategy.momentum import MomentumStrategy
from core.strategy.mean_reversion import MeanReversionStrategy
from core.strategy.breakout import BreakoutStrategy
from data.ingestion.market_data import Candle

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSignal:
    """Combined signal from all strategies."""
    symbol: str
    direction: SignalDirection
    combined_strength: float
    expected_profit_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    contributing_signals: List[Signal]
    strategy_label: str  # Dominant strategy name
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_actionable(self) -> bool:
        return (
            self.direction != SignalDirection.HOLD
            and self.combined_strength >= 0.35
        )


class StrategyOrchestrator:
    """
    Manages all strategies and produces weighted aggregate signals.
    
    Flow:
    1. Feed candles to each strategy
    2. Collect individual signals
    3. Weight by strategy allocation (from config)
    4. Resolve conflicts (opposing signals cancel out)
    5. Output single decision per symbol
    """

    def __init__(self, config):
        self._config = config
        self._weights = config.strategies.weights  # {"momentum": 0.4, ...}

        # Initialize strategies
        self._strategies: Dict[str, BaseStrategy] = {}
        for name in config.strategies.enabled:
            if name == "momentum":
                self._strategies[name] = MomentumStrategy(config)
            elif name == "mean_reversion":
                self._strategies[name] = MeanReversionStrategy(config)
            elif name == "breakout":
                self._strategies[name] = BreakoutStrategy(config)
            else:
                logger.warning(f"Unknown strategy: {name}")

        logger.info(
            f"Orchestrator initialized | Strategies: {list(self._strategies.keys())} | "
            f"Weights: {self._weights}"
        )

    def evaluate(self, symbol: str, candles: List[Candle]) -> AggregatedSignal:
        """
        Evaluate all strategies for a symbol and produce an aggregate signal.
        """
        signals: List[Signal] = []

        # Collect signals from each strategy
        for name, strategy in self._strategies.items():
            try:
                signal = strategy.analyze(candles, symbol)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Strategy {name} error on {symbol}: {e}")

        if not signals:
            return self._neutral_signal(symbol, signals)

        # Separate buy/sell/hold signals
        buy_signals = [s for s in signals if s.direction == SignalDirection.BUY]
        sell_signals = [s for s in signals if s.direction == SignalDirection.SELL]

        # Calculate weighted scores
        buy_score = sum(
            s.strength * self._weights.get(s.strategy, 0)
            for s in buy_signals
        )
        sell_score = sum(
            s.strength * self._weights.get(s.strategy, 0)
            for s in sell_signals
        )

        # Conflict resolution: opposing signals reduce confidence
        if buy_signals and sell_signals:
            # Net score
            net_score = buy_score - sell_score
            if abs(net_score) < 0.15:
                # Too close — conflicting signals, stay out
                return self._neutral_signal(symbol, signals, reason="conflicting_signals")
            direction = SignalDirection.BUY if net_score > 0 else SignalDirection.SELL
            combined_strength = abs(net_score)
        elif buy_signals:
            direction = SignalDirection.BUY
            combined_strength = buy_score
        elif sell_signals:
            direction = SignalDirection.SELL
            combined_strength = sell_score
        else:
            return self._neutral_signal(symbol, signals)

        # Cap at 1.0
        combined_strength = min(combined_strength, 1.0)

        # Use risk params from the strongest contributing signal
        active_signals = buy_signals if direction == SignalDirection.BUY else sell_signals
        dominant = max(active_signals, key=lambda s: s.strength * self._weights.get(s.strategy, 0))

        # Expected profit: weighted average of contributing signals
        expected_profit = sum(
            s.expected_profit_pct * self._weights.get(s.strategy, 0)
            for s in active_signals
        ) / sum(self._weights.get(s.strategy, 0) for s in active_signals) if active_signals else 0

        return AggregatedSignal(
            symbol=symbol,
            direction=direction,
            combined_strength=round(combined_strength, 4),
            expected_profit_pct=round(expected_profit, 4),
            stop_loss_pct=dominant.stop_loss_pct,
            take_profit_pct=dominant.take_profit_pct,
            contributing_signals=signals,
            strategy_label=dominant.strategy,
        )

    def evaluate_all(self, candle_data: Dict[str, List[Candle]]) -> List[AggregatedSignal]:
        """
        Evaluate all symbols and return actionable signals only.
        
        Args:
            candle_data: {symbol: [candles]} for all symbols
            
        Returns:
            List of actionable aggregated signals, sorted by strength
        """
        actionable = []
        for symbol, candles in candle_data.items():
            if not candles:
                continue
            signal = self.evaluate(symbol, candles)
            if signal.is_actionable:
                actionable.append(signal)

        # Sort by strength (strongest first)
        actionable.sort(key=lambda s: s.combined_strength, reverse=True)

        if actionable:
            logger.info(
                f"Evaluation complete | {len(actionable)} actionable signals "
                f"from {len(candle_data)} symbols"
            )
            for sig in actionable[:3]:  # Log top 3
                logger.info(
                    f"  {sig.symbol} {sig.direction.value.upper()} "
                    f"strength={sig.combined_strength:.4f} "
                    f"strategy={sig.strategy_label}"
                )

        return actionable

    def _neutral_signal(
        self, symbol: str, signals: List[Signal], reason: str = "no_signal"
    ) -> AggregatedSignal:
        """Return a neutral (HOLD) aggregate signal."""
        return AggregatedSignal(
            symbol=symbol,
            direction=SignalDirection.HOLD,
            combined_strength=0.0,
            expected_profit_pct=0.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            contributing_signals=signals,
            strategy_label=reason,
        )

    def get_status(self) -> dict:
        """Get orchestrator status."""
        return {
            "strategies": list(self._strategies.keys()),
            "weights": self._weights,
        }
