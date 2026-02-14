"""
API Routes - Health & Status
=============================
Health checks, system status, and risk dashboard endpoints.
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# These get set during app initialization
_risk_manager = None
_exchange = None
_config = None
_trade_store = None
_orchestrator = None
_start_time = None


def init_routes(config, risk_manager, exchange, trade_store, orchestrator=None):
    """Initialize route dependencies."""
    global _config, _risk_manager, _exchange, _trade_store, _orchestrator, _start_time
    _config = config
    _risk_manager = risk_manager
    _exchange = exchange
    _trade_store = trade_store
    _orchestrator = orchestrator
    _start_time = datetime.now(timezone.utc)


@router.get("/health")
async def health():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": (datetime.now(timezone.utc) - _start_time).total_seconds() if _start_time else 0,
    }


@router.get("/status")
async def system_status():
    """Comprehensive system status."""
    risk_status = _risk_manager.get_status() if _risk_manager else {}

    # Test exchange connectivity
    exchange_ok = False
    btc_price = None
    if _exchange:
        ticker = _exchange.get_price("BTCUSDT")
        if ticker:
            exchange_ok = True
            btc_price = ticker.price

    return {
        "system": {
            "status": "running",
            "environment": _config.environment if _config else "unknown",
            "paper_mode": _config.is_paper if _config else True,
            "uptime_seconds": (datetime.now(timezone.utc) - _start_time).total_seconds() if _start_time else 0,
        },
        "capital": {
            "total": risk_status.get("capital", 0),
            "deployable": risk_status.get("deployable", 0),
            "reserve": (_config.capital.total_usdt * _config.capital.min_reserve_pct / 100) if _config else 0,
        },
        "risk": {
            "open_positions": risk_status.get("open_positions", 0),
            "total_exposure": risk_status.get("total_exposure", 0),
            "daily_pnl": risk_status.get("daily_pnl", 0),
            "daily_trades": risk_status.get("daily_trades", 0),
            "consecutive_losses": risk_status.get("consecutive_losses", 0),
            "kill_switch_active": risk_status.get("kill_switch_active", False),
            "available_slots": risk_status.get("available_slots", 0),
        },
        "exchange": {
            "connected": exchange_ok,
            "name": _config.exchange.name if _config else "unknown",
            "btc_price": btc_price,
        },
        "performance": {
            "total_trades": _trade_store.get_trade_count() if _trade_store else 0,
            "total_pnl": _trade_store.get_total_pnl() if _trade_store else 0,
        },
    }


@router.get("/risk")
async def risk_status():
    """Detailed risk manager status."""
    if not _risk_manager:
        return {"error": "Risk manager not initialized"}
    return _risk_manager.get_status()


@router.get("/trades/recent")
async def recent_trades(limit: int = 50):
    """Get recent trades."""
    if not _trade_store:
        return {"trades": [], "error": "Trade store not initialized"}
    return {"trades": _trade_store.get_recent_trades(limit=limit)}


@router.get("/performance/daily")
async def daily_performance(days: int = 30):
    """Get daily performance history."""
    if not _trade_store:
        return {"performance": [], "error": "Trade store not initialized"}
    return {"performance": _trade_store.get_daily_performance(days=days)}


@router.post("/kill-switch/activate")
async def activate_kill_switch():
    """Manually activate the kill switch."""
    if _risk_manager:
        _risk_manager.activate_kill_switch("Manual activation via API")
        return {"status": "kill_switch_activated"}
    return {"error": "Risk manager not initialized"}


@router.post("/kill-switch/deactivate")
async def deactivate_kill_switch():
    """Manually deactivate the kill switch."""
    if _risk_manager:
        _risk_manager.deactivate_kill_switch()
        return {"status": "kill_switch_deactivated"}
    return {"error": "Risk manager not initialized"}


@router.get("/strategies")
async def strategies_status():
    """Get strategy orchestrator status including AI consensus."""
    if not _orchestrator:
        return {"error": "Orchestrator not initialized"}
    return _orchestrator.get_status()


@router.get("/ai/status")
async def ai_consensus_status():
    """Get AI consensus provider status and diagnostics."""
    if not _orchestrator:
        return {"error": "Orchestrator not initialized"}
    status = _orchestrator.get_status()
    ai = status.get("ai_consensus", {})
    if not ai:
        return {"error": "AI consensus strategy not enabled", "strategies": status.get("strategies", [])}
    return ai
