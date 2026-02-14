"""
Execution Engine
=================
Routes trading signals through risk management to order execution.

Flow:
1. Receive aggregated signal from orchestrator
2. Calculate position size
3. Build trade proposal
4. Submit to risk manager
5. If approved, execute on exchange
6. Record everything in trade store
7. Register position for tracking
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from core.strategy.orchestrator import AggregatedSignal
from core.strategy.base import SignalDirection
from core.risk.manager import RiskManager, TradeProposal, Position, RiskVerdict
from core.exchange.binance_adapter import BinanceAdapter
from data.storage.trade_store import TradeStore

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Converts signals into executed trades.
    
    Responsibilities:
    - Position sizing based on signal strength and config
    - Trade proposal construction
    - Risk gate (every trade passes through RiskManager)
    - Order execution (paper or live)
    - Full audit trail in TradeStore
    """

    def __init__(
        self,
        config,
        risk_manager: RiskManager,
        exchange: BinanceAdapter,
        trade_store: TradeStore,
    ):
        self._config = config
        self._risk = risk_manager
        self._exchange = exchange
        self._store = trade_store
        self._is_paper = config.is_paper

        logger.info(
            f"ExecutionEngine initialized | Paper: {self._is_paper} | "
            f"Min position: ${config.positions.min_position_usdt} | "
            f"Max position: ${config.positions.max_position_usdt}"
        )

    def execute_signal(self, signal: AggregatedSignal) -> Optional[str]:
        """
        Attempt to execute a trading signal.
        
        Returns:
            order_id if executed, None if rejected or failed
        """
        if not signal.is_actionable:
            return None

        symbol = signal.symbol
        side = signal.direction.value  # "buy" or "sell"

        # 1. Get current price
        ticker = self._exchange.get_price(symbol)
        if not ticker:
            logger.warning(f"Cannot execute {symbol}: no price available")
            return None
        price = ticker.price

        # 2. Calculate position size
        quantity = self._calculate_position_size(signal, price)
        if quantity is None:
            return None

        # 3. Build trade proposal
        proposal = TradeProposal(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            strategy=signal.strategy_label,
            expected_profit_pct=signal.expected_profit_pct,
            signal_strength=signal.combined_strength,
        )

        # 4. Risk gate
        risk_result = self._risk.evaluate_trade(proposal)

        # Record the risk decision
        self._store.record_risk_event(
            "trade_approved" if risk_result.approved else "trade_rejected",
            risk_result,
            self._risk.get_status(),
        )

        if not risk_result.approved:
            logger.info(
                f"REJECTED {symbol} {side} | {risk_result.verdict.value} | "
                f"{risk_result.reason}"
            )
            return None

        # Use adjusted quantity if risk manager modified it
        final_quantity = risk_result.adjusted_quantity or quantity

        # 5. Execute order
        logger.info(
            f"EXECUTING {symbol} {side.upper()} | "
            f"qty={final_quantity} @ ${price:,.2f} | "
            f"value=${final_quantity * price:.2f} | "
            f"strategy={signal.strategy_label} | "
            f"strength={signal.combined_strength:.4f}"
        )

        order = self._exchange.place_market_order(symbol, side, final_quantity)

        if not order:
            logger.error(f"Order execution failed: {symbol} {side}")
            self._store.record_system_event(
                "order_failed",
                f"Market order failed: {symbol} {side} qty={final_quantity}",
                severity="error",
            )
            return None

        # 6. Record trade
        trade_id = self._store.record_trade(proposal, order, is_paper=self._is_paper)

        # 7. Register position with risk manager
        stop_loss_price = self._calc_stop_price(
            price, side, signal.stop_loss_pct
        )
        take_profit_price = self._calc_tp_price(
            price, side, signal.take_profit_pct
        )

        position = Position(
            symbol=symbol,
            side=side,
            quantity=order.filled_quantity,
            entry_price=order.avg_fill_price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            strategy=signal.strategy_label,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )
        self._risk.register_position(position)

        logger.info(
            f"FILLED {symbol} {side.upper()} | "
            f"qty={order.filled_quantity} @ ${order.avg_fill_price:,.2f} | "
            f"SL=${stop_loss_price:,.2f} TP=${take_profit_price:,.2f} | "
            f"order_id={order.order_id}"
        )

        return order.order_id

    def check_exits(self):
        """
        Check all open positions for stop-loss or take-profit hits.
        Called on every price update or evaluation cycle.
        """
        positions = self._risk.get_open_positions()
        if not positions:
            return

        for key, pos in list(positions.items()):
            ticker = self._exchange.get_price(pos.symbol)
            if not ticker:
                continue

            current_price = ticker.price
            should_exit = False
            exit_reason = ""

            if pos.side == "buy":
                if current_price <= pos.stop_loss_price:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif current_price >= pos.take_profit_price:
                    should_exit = True
                    exit_reason = "take_profit"
            elif pos.side == "sell":
                if current_price >= pos.stop_loss_price:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif current_price <= pos.take_profit_price:
                    should_exit = True
                    exit_reason = "take_profit"

            if should_exit:
                self._exit_position(key, pos, current_price, exit_reason)

    def _exit_position(self, key: str, pos: Position, exit_price: float, reason: str):
        """Close a position."""
        exit_side = "sell" if pos.side == "buy" else "buy"

        logger.info(
            f"EXITING {pos.symbol} ({reason}) | "
            f"Entry=${pos.entry_price:,.2f} Exit=${exit_price:,.2f}"
        )

        order = self._exchange.place_market_order(
            pos.symbol, exit_side, pos.quantity
        )

        if order:
            # Calculate P&L
            if pos.side == "buy":
                pnl = (exit_price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - exit_price) * pos.quantity

            # Subtract commissions
            pnl -= order.commission
            pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100

            # Update risk manager
            self._risk.close_position(key, exit_price, pnl)

            # Update trade store
            # Find the original order_id (stored in position key pattern)
            self._store.record_system_event(
                "position_closed",
                f"{pos.symbol} {reason} | PnL=${pnl:.4f} ({pnl_pct:.2f}%)",
                severity="info",
                details={
                    "symbol": pos.symbol,
                    "reason": reason,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                },
            )

            emoji = "+" if pnl >= 0 else ""
            logger.info(
                f"CLOSED {pos.symbol} | {reason} | "
                f"PnL: {emoji}${pnl:.4f} ({emoji}{pnl_pct:.2f}%)"
            )
        else:
            logger.error(f"Exit order failed: {pos.symbol} {exit_side}")

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def _calculate_position_size(
        self, signal: AggregatedSignal, price: float
    ) -> Optional[float]:
        """
        Calculate position size based on signal strength and config.
        
        Sizing logic:
        - Base: default_position_pct of deployable capital
        - Scale: +/- based on signal strength
        - Floor: min_position_usdt
        - Ceiling: max_position_usdt and max_position_pct
        """
        status = self._risk.get_status()
        deployable = status["deployable"]

        # Base position in USDT
        base_pct = self._config.positions.default_position_pct / 100
        base_usdt = deployable * base_pct

        # Scale by signal strength (0.5x to 1.5x base)
        scale = 0.5 + signal.combined_strength  # 0.5 at strength=0, 1.5 at strength=1
        target_usdt = base_usdt * scale

        # Apply bounds
        min_pos = self._config.positions.min_position_usdt
        max_pos = min(
            self._config.positions.max_position_usdt,
            deployable * (self._config.positions.max_position_pct / 100),
        )

        target_usdt = max(min_pos, min(target_usdt, max_pos))

        # Convert to quantity
        if price <= 0:
            return None

        quantity = target_usdt / price

        # Round to reasonable precision (8 decimal places for crypto)
        quantity = round(quantity, 8)

        if quantity * price < min_pos:
            logger.debug(
                f"Position too small for {signal.symbol}: "
                f"${quantity * price:.2f} < ${min_pos}"
            )
            return None

        return quantity

    @staticmethod
    def _calc_stop_price(entry: float, side: str, stop_pct: float) -> float:
        """Calculate stop-loss price."""
        if side == "buy":
            return entry * (1 - stop_pct / 100)
        else:
            return entry * (1 + stop_pct / 100)

    @staticmethod
    def _calc_tp_price(entry: float, side: str, tp_pct: float) -> float:
        """Calculate take-profit price."""
        if side == "buy":
            return entry * (1 + tp_pct / 100)
        else:
            return entry * (1 - tp_pct / 100)

    def get_status(self) -> dict:
        """Get execution engine status."""
        return {
            "paper_mode": self._is_paper,
            "min_position": self._config.positions.min_position_usdt,
            "max_position": self._config.positions.max_position_usdt,
        }
