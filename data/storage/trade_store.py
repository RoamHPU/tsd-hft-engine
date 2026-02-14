"""
Trade Store
===========
Persistence layer for trades, risk events, and performance data.
Every trade and risk decision gets recorded here for audit and analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from data.storage.models import (
    Trade, RiskEvent, DailyPerformance, SystemEvent,
    TradeDirection, TradeStatus, OrderType, init_db,
)
from core.risk.manager import RiskCheckResult, TradeProposal, RiskVerdict
from core.exchange.binance_adapter import OrderResult

logger = logging.getLogger(__name__)


class TradeStore:
    """
    Central persistence for all trading data.
    
    Responsibilities:
    - Record every trade (paper and live)
    - Record every risk decision (approved and rejected)
    - Track daily performance summaries
    - Log system events
    """

    def __init__(self, db_url: str = "sqlite:///data/tsd_hft.db"):
        self._Session = init_db(db_url)
        logger.info(f"TradeStore initialized | DB: {db_url}")

    def record_trade(
        self,
        proposal: TradeProposal,
        order_result: OrderResult,
        is_paper: bool = True,
    ) -> int:
        """Record a completed trade."""
        session = self._Session()
        try:
            trade = Trade(
                order_id=order_result.order_id,
                symbol=order_result.symbol,
                side=TradeDirection(order_result.side),
                order_type=OrderType.MARKET,  # TODO: detect from order
                status=TradeStatus(order_result.status.lower()) if order_result.status.lower() in [s.value for s in TradeStatus] else TradeStatus.FILLED,
                strategy=proposal.strategy,
                requested_quantity=proposal.quantity,
                filled_quantity=order_result.filled_quantity,
                requested_price=proposal.price,
                avg_fill_price=order_result.avg_fill_price,
                notional_value=order_result.filled_quantity * order_result.avg_fill_price,
                commission=order_result.commission,
                commission_asset=order_result.commission_asset,
                signal_strength=proposal.signal_strength,
                expected_profit_pct=proposal.expected_profit_pct,
                is_paper=is_paper,
                raw_response=order_result.raw_response,
            )
            session.add(trade)
            session.commit()
            trade_id = trade.id
            logger.info(f"Trade recorded: #{trade_id} {order_result.symbol} {order_result.side}")
            return trade_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record trade: {e}")
            raise
        finally:
            session.close()

    def update_trade_exit(
        self, order_id: str, exit_price: float, pnl: float, pnl_pct: float
    ):
        """Update a trade with exit information."""
        session = self._Session()
        try:
            trade = session.query(Trade).filter_by(order_id=order_id).first()
            if trade:
                trade.exit_price = exit_price
                trade.exit_time = datetime.now(timezone.utc)
                trade.pnl = pnl
                trade.pnl_pct = pnl_pct
                session.commit()
                logger.info(f"Trade exit recorded: {order_id} PnL=${pnl:.2f} ({pnl_pct:.2f}%)")
            else:
                logger.warning(f"Trade not found for exit update: {order_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update trade exit: {e}")
        finally:
            session.close()

    def record_risk_event(
        self,
        event_type: str,
        result: RiskCheckResult,
        risk_status: dict,
    ):
        """Record a risk decision (approved or rejected)."""
        session = self._Session()
        try:
            event = RiskEvent(
                event_type=event_type,
                symbol=result.proposal.symbol,
                verdict=result.verdict.value,
                reason=result.reason,
                capital_at_event=risk_status.get("capital", 0),
                daily_pnl_at_event=risk_status.get("daily_pnl", 0),
                open_positions_count=risk_status.get("open_positions", 0),
                total_exposure=risk_status.get("total_exposure", 0),
                consecutive_losses=risk_status.get("consecutive_losses", 0),
                proposed_value=result.proposal.notional_value,
                proposed_side=result.proposal.side,
                proposed_strategy=result.proposal.strategy,
            )
            session.add(event)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record risk event: {e}")
        finally:
            session.close()

    def record_system_event(
        self, event_type: str, message: str,
        severity: str = "info", details: dict = None
    ):
        """Record a system event."""
        session = self._Session()
        try:
            event = SystemEvent(
                event_type=event_type,
                message=message,
                severity=severity,
                details=details,
            )
            session.add(event)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record system event: {e}")
        finally:
            session.close()

    def save_daily_performance(
        self,
        date_str: str,
        starting_capital: float,
        ending_capital: float,
        trades: List[Trade] = None,
        total_fees: float = 0.0,
        strategy_pnl: Dict[str, float] = None,
    ):
        """Save or update daily performance summary."""
        session = self._Session()
        try:
            daily_pnl = ending_capital - starting_capital
            daily_pnl_pct = (daily_pnl / starting_capital * 100) if starting_capital > 0 else 0

            total_trades = len(trades) if trades else 0
            winning = sum(1 for t in (trades or []) if t.pnl and t.pnl > 0)
            losing = sum(1 for t in (trades or []) if t.pnl and t.pnl < 0)
            win_rate = (winning / total_trades * 100) if total_trades > 0 else None

            existing = session.query(DailyPerformance).filter_by(date=date_str).first()
            if existing:
                existing.ending_capital = ending_capital
                existing.daily_pnl = daily_pnl
                existing.daily_pnl_pct = daily_pnl_pct
                existing.total_trades = total_trades
                existing.winning_trades = winning
                existing.losing_trades = losing
                existing.win_rate = win_rate
                existing.total_fees = total_fees
                existing.strategy_pnl = strategy_pnl
            else:
                perf = DailyPerformance(
                    date=date_str,
                    starting_capital=starting_capital,
                    ending_capital=ending_capital,
                    daily_pnl=daily_pnl,
                    daily_pnl_pct=daily_pnl_pct,
                    total_trades=total_trades,
                    winning_trades=winning,
                    losing_trades=losing,
                    win_rate=win_rate,
                    total_fees=total_fees,
                    strategy_pnl=strategy_pnl,
                )
                session.add(perf)

            session.commit()
            logger.info(f"Daily performance saved: {date_str} PnL=${daily_pnl:.2f}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save daily performance: {e}")
        finally:
            session.close()

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_recent_trades(self, limit: int = 50) -> List[dict]:
        """Get recent trades as dictionaries."""
        session = self._Session()
        try:
            trades = (
                session.query(Trade)
                .order_by(Trade.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "order_id": t.order_id,
                    "symbol": t.symbol,
                    "side": t.side.value,
                    "status": t.status.value,
                    "strategy": t.strategy,
                    "filled_quantity": t.filled_quantity,
                    "avg_fill_price": t.avg_fill_price,
                    "notional_value": t.notional_value,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "commission": t.commission,
                    "is_paper": t.is_paper,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in trades
            ]
        finally:
            session.close()

    def get_daily_performance(self, days: int = 30) -> List[dict]:
        """Get recent daily performance summaries."""
        session = self._Session()
        try:
            perfs = (
                session.query(DailyPerformance)
                .order_by(DailyPerformance.date.desc())
                .limit(days)
                .all()
            )
            return [
                {
                    "date": p.date,
                    "starting_capital": p.starting_capital,
                    "ending_capital": p.ending_capital,
                    "daily_pnl": p.daily_pnl,
                    "daily_pnl_pct": p.daily_pnl_pct,
                    "total_trades": p.total_trades,
                    "win_rate": p.win_rate,
                    "total_fees": p.total_fees,
                }
                for p in perfs
            ]
        finally:
            session.close()

    def get_trade_count(self) -> int:
        """Total number of trades."""
        session = self._Session()
        try:
            return session.query(Trade).count()
        finally:
            session.close()

    def get_total_pnl(self) -> float:
        """Sum of all closed trade P&L."""
        session = self._Session()
        try:
            from sqlalchemy import func
            result = session.query(func.sum(Trade.pnl)).filter(Trade.pnl.isnot(None)).scalar()
            return result or 0.0
        finally:
            session.close()
