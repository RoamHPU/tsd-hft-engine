"""
Trading Loop
==============
The main engine loop that:
1. Fetches/updates market data
2. Evaluates strategies
3. Executes approved trades
4. Monitors open positions for exits
5. Sleeps until next cycle

Runs as a background asyncio task.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from data.ingestion.market_data import MarketDataManager
from core.strategy.orchestrator import StrategyOrchestrator
from core.execution.engine import ExecutionEngine

logger = logging.getLogger(__name__)


class TradingLoop:
    """
    Main trading loop coordinator.
    
    Cycle:
    1. Update market data (REST fetch per symbol)
    2. Run strategy evaluation
    3. Execute top signals (respecting position limits)
    4. Check exits on open positions
    5. Log cycle stats
    6. Sleep until next interval
    """

    def __init__(
        self,
        config,
        market_data: MarketDataManager,
        orchestrator: StrategyOrchestrator,
        execution: ExecutionEngine,
    ):
        self._config = config
        self._market = market_data
        self._orchestrator = orchestrator
        self._execution = execution

        self._running = False
        self._cycle_count = 0
        self._cycle_interval = config.evaluation.cycle_interval_seconds
        self._signals_generated = 0
        self._trades_executed = 0

        logger.info(
            f"TradingLoop initialized | "
            f"Interval: {self._cycle_interval}s | "
            f"Symbols: {len(config.all_trading_pairs)}"
        )

    async def start(self):
        """Start the trading loop."""
        self._running = True
        logger.info("Trading loop STARTED")

        # Preload historical candles
        logger.info("Preloading historical candles...")
        self._market.preload_candles(
            interval="1m",
            limit=100,
        )

        # Start WebSocket streams
        self._market.start_streams(intervals=["1m"])

        # Main loop
        while self._running:
            try:
                cycle_start = time.time()
                await self._run_cycle()
                elapsed = time.time() - cycle_start

                # Sleep for remainder of interval
                sleep_time = max(0, self._cycle_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Trading loop error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error

    async def stop(self):
        """Stop the trading loop gracefully."""
        self._running = False
        self._market.stop_streams()
        logger.info(
            f"Trading loop STOPPED | "
            f"Cycles: {self._cycle_count} | "
            f"Signals: {self._signals_generated} | "
            f"Trades: {self._trades_executed}"
        )

    async def _run_cycle(self):
        """Execute one trading cycle."""
        self._cycle_count += 1

        # 1. Gather candle data for all symbols
        candle_data = {}
        for symbol in self._config.all_trading_pairs:
            candles = self._market.get_candles(symbol, "1m")
            if candles:
                candle_data[symbol] = candles
            else:
                # Fallback: REST fetch if WebSocket hasn't delivered yet
                candles = self._market.fetch_candles(symbol, "1m", 100)
                if candles:
                    candle_data[symbol] = candles

        if not candle_data:
            if self._cycle_count % 10 == 0:  # Log every 10th cycle
                logger.warning("No candle data available")
            return

        # 2. Check exits on existing positions first
        self._execution.check_exits()

        # 3. Evaluate strategies
        actionable_signals = self._orchestrator.evaluate_all(candle_data)
        self._signals_generated += len(actionable_signals)

        # 4. Execute top signals (limited by available position slots)
        from core.risk.manager import RiskManager
        risk_status = self._execution._risk.get_status()
        available_slots = risk_status.get("available_slots", 0)

        executed = 0
        for signal in actionable_signals:
            if executed >= available_slots:
                break

            # Don't open duplicate positions in same symbol
            open_positions = self._execution._risk.get_open_positions()
            already_in = any(
                pos.symbol == signal.symbol
                for pos in open_positions.values()
            )
            if already_in:
                continue

            order_id = self._execution.execute_signal(signal)
            if order_id:
                executed += 1
                self._trades_executed += 1

        # 5. Periodic status log
        if self._cycle_count % 60 == 0:  # Every ~60 cycles
            self._log_status()

    def _log_status(self):
        """Log periodic status update."""
        risk_status = self._execution._risk.get_status()
        data_status = self._market.get_data_status()

        logger.info(
            f"CYCLE {self._cycle_count} | "
            f"Capital: ${risk_status['capital']:.2f} | "
            f"PnL: ${risk_status['daily_pnl']:.2f} | "
            f"Positions: {risk_status['open_positions']}/{self._config.positions.max_concurrent} | "
            f"Trades today: {risk_status['daily_trades']} | "
            f"Data: {data_status['symbols_with_candles']} symbols"
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "cycle_interval_seconds": self._cycle_interval,
            "signals_generated": self._signals_generated,
            "trades_executed": self._trades_executed,
        }
