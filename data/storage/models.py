"""
Data Models
===========
SQLAlchemy models for trade logging, audit trail, and performance tracking.
Every trade, risk decision, and system event gets recorded here.
"""

import enum
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Enum, Text, JSON,
    create_engine, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class TradeDirection(enum.Enum):
    BUY = "buy"
    SELL = "sell"


class TradeStatus(enum.Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"


class Trade(Base):
    """Record of every trade executed or attempted."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Order identification
    order_id = Column(String(64), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Trade details
    side = Column(Enum(TradeDirection), nullable=False)
    order_type = Column(Enum(OrderType), nullable=False)
    status = Column(Enum(TradeStatus), nullable=False)
    strategy = Column(String(50), nullable=False, index=True)
    
    # Quantities and prices
    requested_quantity = Column(Float, nullable=False)
    filled_quantity = Column(Float, nullable=False, default=0.0)
    requested_price = Column(Float, nullable=True)  # Null for market orders
    avg_fill_price = Column(Float, nullable=True)
    
    # Financial
    notional_value = Column(Float, nullable=False)  # quantity * price
    commission = Column(Float, nullable=False, default=0.0)
    commission_asset = Column(String(10), default="USDT")
    
    # Risk context at time of trade
    signal_strength = Column(Float, nullable=True)
    expected_profit_pct = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    
    # Result (filled after position close)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    is_paper = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
    
    # Raw exchange response
    raw_response = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_trades_symbol_created", "symbol", "created_at"),
        Index("idx_trades_strategy_created", "strategy", "created_at"),
    )

    def __repr__(self):
        return (
            f"<Trade {self.order_id} {self.side.value} {self.filled_quantity} "
            f"{self.symbol} @ {self.avg_fill_price} PnL={self.pnl}>"
        )


class RiskEvent(Base):
    """Audit log of every risk decision."""
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # What happened
    event_type = Column(String(50), nullable=False, index=True)
    # e.g., "trade_approved", "trade_rejected", "kill_switch_activated",
    #        "daily_limit_reached", "position_closed"
    
    # Context
    symbol = Column(String(20), nullable=True)
    verdict = Column(String(50), nullable=False)
    reason = Column(Text, nullable=False)
    
    # State snapshot at time of event
    capital_at_event = Column(Float, nullable=False)
    daily_pnl_at_event = Column(Float, nullable=False, default=0.0)
    open_positions_count = Column(Integer, nullable=False, default=0)
    total_exposure = Column(Float, nullable=False, default=0.0)
    consecutive_losses = Column(Integer, nullable=False, default=0)
    
    # Proposed trade details (if applicable)
    proposed_value = Column(Float, nullable=True)
    proposed_side = Column(String(10), nullable=True)
    proposed_strategy = Column(String(50), nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_risk_events_type_created", "event_type", "created_at"),
    )

    def __repr__(self):
        return f"<RiskEvent {self.event_type} {self.verdict} {self.symbol}>"


class DailyPerformance(Base):
    """Daily performance summary — one row per day."""
    __tablename__ = "daily_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), unique=True, nullable=False, index=True)  # YYYY-MM-DD
    
    # P&L
    starting_capital = Column(Float, nullable=False)
    ending_capital = Column(Float, nullable=False)
    daily_pnl = Column(Float, nullable=False, default=0.0)
    daily_pnl_pct = Column(Float, nullable=False, default=0.0)
    
    # Trade stats
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=True)
    
    # Risk metrics
    max_drawdown = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    total_fees = Column(Float, nullable=False, default=0.0)
    
    # Strategy breakdown
    strategy_pnl = Column(JSON, nullable=True)  # {"momentum": 12.5, "mean_reversion": -3.2}
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<DailyPerformance {self.date} PnL=${self.daily_pnl:.2f}>"


class SystemEvent(Base):
    """System-level events: startup, shutdown, config changes, errors."""
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False, index=True)
    # e.g., "startup", "shutdown", "config_change", "error", "kill_switch",
    #        "paper_to_live", "capital_update"
    message = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False, default="info")  # info, warning, error, critical
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<SystemEvent {self.event_type} [{self.severity}]>"


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

def init_db(db_url: str = "sqlite:///data/tsd_hft.db") -> sessionmaker:
    """
    Initialize database and return a session factory.
    Uses SQLite for development, PostgreSQL for production.
    """
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session
