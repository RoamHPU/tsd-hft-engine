"""
Risk Manager
============
The non-negotiable safety layer. Every trade MUST pass through here.
No module bypasses risk checks. No exceptions.

This is the kill switch, position sizer, exposure tracker, and loss limiter.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class RiskVerdict(Enum):
    """Risk check result."""
    APPROVED = "approved"
    REJECTED_SIZE = "rejected_position_too_large"
    REJECTED_EXPOSURE = "rejected_exposure_limit"
    REJECTED_DAILY_LOSS = "rejected_daily_loss_limit"
    REJECTED_CONSECUTIVE = "rejected_consecutive_losses"
    REJECTED_TRADE_LIMIT = "rejected_daily_trade_limit"
    REJECTED_KILL_SWITCH = "rejected_kill_switch_active"
    REJECTED_INSUFFICIENT_CAPITAL = "rejected_insufficient_capital"
    REJECTED_BELOW_MINIMUM = "rejected_below_minimum_size"
    REJECTED_BELOW_PROFIT_THRESHOLD = "rejected_below_profit_threshold"


@dataclass
class TradeProposal:
    """A proposed trade that must pass risk checks."""
    symbol: str
    side: str                    # "buy" or "sell"
    quantity: float
    price: float
    strategy: str
    expected_profit_pct: float   # Expected return percentage
    signal_strength: float       # 0.0 to 1.0

    @property
    def notional_value(self) -> float:
        return self.quantity * self.price


@dataclass
class RiskCheckResult:
    """Result of a risk evaluation."""
    verdict: RiskVerdict
    proposal: TradeProposal
    reason: str
    adjusted_quantity: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def approved(self) -> bool:
        return self.verdict == RiskVerdict.APPROVED


@dataclass
class Position:
    """An open position."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: str
    strategy: str
    stop_loss_price: float
    take_profit_price: float

    @property
    def notional_value(self) -> float:
        return self.quantity * self.entry_price


class RiskManager:
    """
    Central risk management engine.

    Every trade proposal flows through evaluate_trade() before execution.
    The risk manager tracks:
    - Open positions and total exposure
    - Daily P&L and loss limits
    - Consecutive losses
    - Kill switch state
    - Per-asset concentration
    """

    def __init__(self, config):
        self.config = config
        self.risk = config.risk
        self.capital = config.capital
        self.positions_config = config.positions
        self.fees = config.fees

        # State tracking
        self._open_positions: Dict[str, Position] = {}
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._consecutive_losses: int = 0
        self._kill_switch_active: bool = False
        self._day_start: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._trade_log: List[dict] = []
        self._current_capital: float = config.capital.total_usdt

        logger.info(
            f"RiskManager initialized | Capital: ${self._current_capital:.2f} | "
            f"Kill switch: {'ON' if self.risk.kill_switch_enabled else 'OFF'} | "
            f"Daily loss limit: ${self.risk.daily_loss_limit_usdt:.2f}"
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def evaluate_trade(self, proposal: TradeProposal) -> RiskCheckResult:
        """
        Evaluate a trade proposal against all risk rules.
        Returns APPROVED or a specific rejection reason.
        This is the ONLY way a trade gets permission to execute.
        """
        self._maybe_reset_daily()

        # Check kill switch first
        if self._kill_switch_active:
            return self._reject(proposal, RiskVerdict.REJECTED_KILL_SWITCH,
                                "Kill switch is active - all trading halted")

        # Check daily trade limit
        if self._daily_trades >= self.risk.max_trades_per_day:
            return self._reject(proposal, RiskVerdict.REJECTED_TRADE_LIMIT,
                                f"Daily trade limit reached ({self.risk.max_trades_per_day})")

        # Check consecutive losses
        if self._consecutive_losses >= self.risk.consecutive_loss_halt:
            return self._reject(proposal, RiskVerdict.REJECTED_CONSECUTIVE,
                                f"Consecutive loss limit ({self.risk.consecutive_loss_halt}) reached")

        # Check daily loss limit
        if self._daily_pnl <= -self.risk.daily_loss_limit_usdt:
            return self._reject(proposal, RiskVerdict.REJECTED_DAILY_LOSS,
                                f"Daily loss limit reached: ${self._daily_pnl:.2f}")

        # Check profit threshold exceeds fees
        if proposal.expected_profit_pct < self.fees.min_profit_threshold_pct:
            return self._reject(proposal, RiskVerdict.REJECTED_BELOW_PROFIT_THRESHOLD,
                                f"Expected profit {proposal.expected_profit_pct:.2f}% < "
                                f"minimum threshold {self.fees.min_profit_threshold_pct}%")

        # Check position size
        result = self._check_position_size(proposal)
        if not result.approved:
            return result

        # Check exposure limits
        result = self._check_exposure(proposal)
        if not result.approved:
            return result

        # All checks passed
        self._daily_trades += 1
        logger.info(
            f"TRADE APPROVED: {proposal.side} {proposal.quantity} {proposal.symbol} "
            f"@ ${proposal.price:.4f} (${proposal.notional_value:.2f})"
        )
        return RiskCheckResult(
            verdict=RiskVerdict.APPROVED,
            proposal=proposal,
            reason="All risk checks passed",
        )

    def register_position(self, position: Position):
        """Register a newly opened position."""
        key = f"{position.symbol}_{position.side}_{position.entry_time}"
        self._open_positions[key] = position
        logger.info(f"Position registered: {position.symbol} {position.side} "
                     f"${position.notional_value:.2f}")

    def close_position(self, key: str, exit_price: float, pnl: float):
        """Record a position closure and update tracking."""
        if key in self._open_positions:
            pos = self._open_positions.pop(key)
            self._daily_pnl += pnl
            self._current_capital += pnl

            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

            # Check kill switch after every close
            self._check_kill_switch()

            self._trade_log.append({
                "symbol": pos.symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            logger.info(
                f"Position closed: {pos.symbol} PnL: ${pnl:.2f} | "
                f"Daily PnL: ${self._daily_pnl:.2f} | "
                f"Capital: ${self._current_capital:.2f}"
            )

    def get_status(self) -> dict:
        """Get current risk manager status."""
        return {
            "capital": self._current_capital,
            "deployable": self._current_capital * (self.capital.max_allocation_pct / 100),
            "open_positions": len(self._open_positions),
            "total_exposure": sum(p.notional_value for p in self._open_positions.values()),
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "consecutive_losses": self._consecutive_losses,
            "kill_switch_active": self._kill_switch_active,
            "available_slots": self.positions_config.max_concurrent - len(self._open_positions),
        }

    def activate_kill_switch(self, reason: str):
        """Manually activate the kill switch."""
        self._kill_switch_active = True
        logger.critical(f"🛑 KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self):
        """Manually deactivate the kill switch. Requires explicit action."""
        self._kill_switch_active = False
        logger.warning("Kill switch deactivated manually")

    def update_capital(self, new_capital: float):
        """Update current capital (e.g., after deposit/withdrawal)."""
        old = self._current_capital
        self._current_capital = new_capital
        logger.info(f"Capital updated: ${old:.2f} → ${new_capital:.2f}")

    # =========================================================================
    # INTERNAL CHECKS
    # =========================================================================

    def _check_position_size(self, proposal: TradeProposal) -> RiskCheckResult:
        """Validate position size against limits."""
        value = proposal.notional_value

        # Below minimum
        if value < self.positions_config.min_position_usdt:
            return self._reject(proposal, RiskVerdict.REJECTED_BELOW_MINIMUM,
                                f"Position ${value:.2f} < minimum ${self.positions_config.min_position_usdt}")

        # Above maximum
        max_by_pct = self._current_capital * (self.positions_config.max_position_pct / 100)
        max_allowed = min(self.positions_config.max_position_usdt, max_by_pct)
        if value > max_allowed:
            return self._reject(proposal, RiskVerdict.REJECTED_SIZE,
                                f"Position ${value:.2f} > max ${max_allowed:.2f}")

        # Check sufficient free capital
        current_exposure = sum(p.notional_value for p in self._open_positions.values())
        deployable = self._current_capital * (self.capital.max_allocation_pct / 100)
        available = deployable - current_exposure

        if value > available:
            return self._reject(proposal, RiskVerdict.REJECTED_INSUFFICIENT_CAPITAL,
                                f"Position ${value:.2f} > available capital ${available:.2f}")

        return RiskCheckResult(
            verdict=RiskVerdict.APPROVED, proposal=proposal, reason="Size OK"
        )

    def _check_exposure(self, proposal: TradeProposal) -> RiskCheckResult:
        """Check exposure concentration limits."""
        # Check max concurrent positions
        if len(self._open_positions) >= self.positions_config.max_concurrent:
            return self._reject(proposal, RiskVerdict.REJECTED_EXPOSURE,
                                f"Max concurrent positions ({self.positions_config.max_concurrent}) reached")

        # Check single-asset concentration
        asset_exposure = sum(
            p.notional_value for p in self._open_positions.values()
            if p.symbol == proposal.symbol
        )
        max_asset = self._current_capital * (self.risk.max_single_asset_pct / 100)
        if asset_exposure + proposal.notional_value > max_asset:
            return self._reject(proposal, RiskVerdict.REJECTED_EXPOSURE,
                                f"{proposal.symbol} exposure would exceed "
                                f"${max_asset:.2f} ({self.risk.max_single_asset_pct}%)")

        return RiskCheckResult(
            verdict=RiskVerdict.APPROVED, proposal=proposal, reason="Exposure OK"
        )

    def _check_kill_switch(self):
        """Check if kill switch should trigger."""
        if not self.risk.kill_switch_enabled:
            return

        drawdown = self.capital.total_usdt - self._current_capital
        if drawdown >= self.risk.kill_switch_drawdown_usdt:
            self._kill_switch_active = True
            logger.critical(
                f"🛑 KILL SWITCH AUTO-TRIGGERED | "
                f"Drawdown: ${drawdown:.2f} >= limit ${self.risk.kill_switch_drawdown_usdt:.2f}"
            )

    def _maybe_reset_daily(self):
        """Reset daily counters at midnight UTC."""
        now = datetime.now(timezone.utc)
        if now.date() > self._day_start.date():
            logger.info(
                f"Daily reset | Previous day PnL: ${self._daily_pnl:.2f} | "
                f"Trades: {self._daily_trades}"
            )
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._consecutive_losses = 0
            self._day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _reject(self, proposal: TradeProposal, verdict: RiskVerdict, reason: str) -> RiskCheckResult:
        """Create a rejection result."""
        logger.warning(f"TRADE REJECTED: {proposal.symbol} - {reason}")
        return RiskCheckResult(verdict=verdict, proposal=proposal, reason=reason)
