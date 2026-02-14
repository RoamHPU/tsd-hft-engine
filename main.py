"""
TSD HFT Engine - Main Entry Point
==================================
Trading system only. No philosophy. No mythology.
Clean startup -> config -> risk -> exchange -> strategies -> trade.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure directories exist
Path("logs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/hft_engine.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("tsd-hft")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


# Global references for API access
_trading_loop = None


def initialize():
    """Initialize all engine components. Returns (config, components dict)."""
    global _trading_loop

    logger.info("=" * 60)
    logger.info("TSD HFT ENGINE v2.0 - Starting")
    logger.info("=" * 60)

    # 1. Load configuration
    from config import load_config
    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)

    logger.info(f"Environment: {config.environment}")
    logger.info(f"Capital: ${config.capital.total_usdt:.2f} USDT")
    logger.info(f"Paper trading: {config.is_paper}")

    # 2. Initialize risk manager
    from core.risk.manager import RiskManager
    risk_manager = RiskManager(config)

    # 3. Initialize exchange adapter
    from core.exchange.binance_adapter import BinanceAdapter
    exchange = BinanceAdapter(config, paper_mode=config.is_paper)

    # 4. Initialize trade store
    from data.storage.trade_store import TradeStore
    trade_store = TradeStore()

    # 5. Initialize market data manager
    from data.ingestion.market_data import MarketDataManager
    market_data = MarketDataManager(config)

    # 6. Initialize strategy orchestrator
    from core.strategy.orchestrator import StrategyOrchestrator
    orchestrator = StrategyOrchestrator(config)

    # 7. Initialize execution engine
    from core.execution.engine import ExecutionEngine
    execution = ExecutionEngine(config, risk_manager, exchange, trade_store)

    # 8. Initialize trading loop
    from core.execution.trading_loop import TradingLoop
    _trading_loop = TradingLoop(config, market_data, orchestrator, execution)

    # 9. Verify exchange connectivity
    test_price = exchange.get_price("BTCUSDT")
    if test_price:
        logger.info(f"Exchange connected | BTC: ${test_price.price:,.2f}")
    else:
        logger.warning("Exchange connectivity check failed")

    # 10. Record startup
    trade_store.record_system_event(
        "startup",
        f"Engine started | Capital: ${config.capital.total_usdt:.2f} | Paper: {config.is_paper}",
        severity="info",
        details={"capital": config.capital.total_usdt, "paper_mode": config.is_paper},
    )

    # 11. Wire up API routes
    from api.routes.health import router as health_router, init_routes
    init_routes(config, risk_manager, exchange, trade_store)
    app.include_router(health_router, prefix="/api")

    # 12. Startup summary
    status = risk_manager.get_status()
    logger.info("-" * 60)
    logger.info("STARTUP SUMMARY")
    logger.info(f"  Capital:          ${status['capital']:.2f}")
    logger.info(f"  Deployable:       ${status['deployable']:.2f}")
    logger.info(f"  Kill switch:      {'ACTIVE' if status['kill_switch_active'] else 'ARMED'}")
    logger.info(f"  Max positions:    {config.positions.max_concurrent}")
    logger.info(f"  Daily loss limit: ${config.risk.daily_loss_limit_usdt:.2f}")
    logger.info(f"  Strategies:       {', '.join(config.strategies.enabled)}")
    logger.info(f"  Cycle interval:   {config.evaluation.cycle_interval_seconds}s")
    logger.info(f"  API:              http://localhost:{config.port}")
    logger.info("-" * 60)
    logger.info("TSD HFT Engine ready")

    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    config = initialize()

    # Start trading loop as background task
    loop_task = asyncio.create_task(_trading_loop.start())
    logger.info("Trading loop started as background task")

    yield  # App is running

    # Shutdown
    logger.info("Shutting down trading loop...")
    await _trading_loop.stop()
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutdown complete")


app = FastAPI(
    title="TSD HFT Engine",
    description="High-Frequency Trading Engine for Cryptocurrency Markets",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def root():
    loop_status = _trading_loop.get_status() if _trading_loop else {}
    running = loop_status.get("running", False)
    cycles = loop_status.get("cycle_count", 0)
    trades = loop_status.get("trades_executed", 0)
    signals = loop_status.get("signals_generated", 0)

    return f"""<!DOCTYPE html>
<html><head><title>TSD HFT Engine</title>
<meta http-equiv="refresh" content="30">
<style>
    body {{ font-family: -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0;
           display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }}
    .c {{ text-align: center; max-width: 700px; }}
    h1 {{ color: #00ff88; font-size: 2em; margin-bottom: 5px; }}
    .sub {{ color: #888; margin-bottom: 20px; }}
    .s {{ background: #1a1a2e; padding: 20px; border-radius: 8px; margin: 15px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; text-align: left; }}
    .label {{ color: #888; }}
    .val {{ color: #00ff88; font-weight: bold; }}
    .val.warn {{ color: #ffaa00; }}
    a {{ color: #00aaff; text-decoration: none; margin: 0 10px; }}
    a:hover {{ text-decoration: underline; }}
    .l {{ display: flex; gap: 15px; justify-content: center; margin-top: 20px; flex-wrap: wrap; }}
    .status {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }}
    .status.on {{ background: #0a3d1a; color: #00ff88; }}
    .status.off {{ background: #3d0a0a; color: #ff4444; }}
</style></head>
<body><div class="c">
    <h1>TSD HFT Engine v2.0</h1>
    <p class="sub">Disciplined Capital Growth
        <span class="status {'on' if running else 'off'}">{'TRADING' if running else 'OFFLINE'}</span>
    </p>
    <div class="s">
        <div class="grid">
            <span class="label">Cycles</span><span class="val">{cycles:,}</span>
            <span class="label">Signals</span><span class="val">{signals:,}</span>
            <span class="label">Trades</span><span class="val">{trades:,}</span>
            <span class="label">Mode</span><span class="val warn">PAPER</span>
        </div>
    </div>
    <div class="l">
        <a href="/api/health">Health</a>
        <a href="/api/status">Status</a>
        <a href="/api/risk">Risk</a>
        <a href="/api/trades/recent">Trades</a>
        <a href="/api/performance/daily">Performance</a>
        <a href="/docs">API Docs</a>
    </div>
</div></body></html>"""


if __name__ == "__main__":
    import uvicorn
    from config import load_config
    config = load_config()
    uvicorn.run("main:app", host="0.0.0.0", port=config.port, reload=False)
