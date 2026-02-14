# TSD HFT Engine

**High-Frequency Trading Engine for Cryptocurrency Markets**

A clean, modular, risk-first trading system built for Binance.
Designed for disciplined capital growth starting from ~$473 USDT.

---

## Architecture

```
tsd-hft-engine/
├── config/
│   ├── __init__.py              ← Config loader + validation
│   └── trading_config.yaml      ← SINGLE SOURCE OF TRUTH
├── core/
│   ├── exchange/
│   │   └── binance_adapter.py   ← Exchange connectivity + orders
│   ├── strategy/
│   │   ├── base.py              ← Strategy interface
│   │   ├── momentum.py          ← Momentum signals
│   │   ├── mean_reversion.py    ← Mean reversion signals
│   │   └── breakout.py          ← Breakout signals
│   ├── risk/
│   │   └── manager.py           ← Risk checks, kill switch, exposure
│   ├── execution/
│   │   └── engine.py            ← Order routing, fill tracking, slippage
│   └── evaluation/
│       └── metrics.py           ← Sharpe, Sortino, drawdown, PnL
├── data/
│   ├── ingestion/
│   │   └── market_feed.py       ← WebSocket + REST price feeds
│   └── storage/
│       ├── models.py            ← SQLAlchemy models
│       └── trade_store.py       ← Trade log + audit trail
├── api/
│   └── routes/
│       ├── trading.py           ← Trade endpoints
│       ├── risk.py              ← Risk status endpoints
│       └── health.py            ← Health checks
├── dashboard/
│   └── templates/
│       └── trading.html         ← Clean trading dashboard
├── tests/
│   ├── unit/                    ← Unit tests per module
│   ├── integration/             ← End-to-end flow tests
│   └── backtest/                ← Historical backtesting
├── main.py                      ← Entry point
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Design Principles

1. **Single source of truth**: One config file. Every module reads from it.
2. **Risk is non-negotiable**: Every trade passes through the risk manager. No bypass.
3. **Kill switch always armed**: Auto-triggers at 10% drawdown.
4. **Paper first**: System starts in paper mode. Live only after validation.
5. **No cross-contamination**: Trading logic only. No mythology in the engine.
6. **Testable**: Every component has a clear interface and is independently testable.
7. **Auditable**: Full trade log and audit trail.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/RoamHPU/tsd-hft-engine.git
cd tsd-hft-engine
cp .env.example .env
# Edit .env with your Binance API keys

# Run with Docker
docker build -t tsd-hft .
docker run -p 8000:8000 --env-file .env tsd-hft

# Or run locally
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

## Configuration

All configuration lives in `config/trading_config.yaml`.
Key settings are validated on startup — the engine won't start with invalid config.

Update capital by editing `capital.total_usdt` in the config file.
The system auto-calculates deployable capital, reserve, and risk limits.

---

## 30-Day Build Roadmap

### Week 1: Foundation (Days 1-7)
- [x] Project structure and architecture
- [x] Config system (single source of truth + validation)
- [x] Risk manager (kill switch, position limits, daily loss limits)
- [x] Exchange adapter (Binance connectivity, orders, paper mode)
- [ ] Data storage layer (SQLAlchemy models for trades + audit)
- [ ] Market data ingestion (REST polling + WebSocket feeds)
- [ ] FastAPI server with health + status endpoints
- [ ] Unit tests for config, risk manager, exchange adapter

### Week 2: Strategy Engine (Days 8-14)
- [ ] Strategy base class (signal interface)
- [ ] Momentum strategy (RSI, MACD, volume-weighted)
- [ ] Mean reversion strategy (Bollinger Bands, z-score)
- [ ] Breakout strategy (support/resistance, volume confirmation)
- [ ] Strategy orchestrator (weighted signal aggregation)
- [ ] Execution engine (order routing through risk → exchange)
- [ ] Paper trading integration tests
- [ ] Backtest framework scaffold

### Week 3: Dashboard + Evaluation (Days 15-21)
- [ ] FastAPI trading routes (start/stop, manual override)
- [ ] Risk status API endpoints
- [ ] Trading dashboard (clean HTML/JS, live updates)
- [ ] Evaluation metrics engine (Sharpe, Sortino, win rate, drawdown)
- [ ] Performance reporting (daily/weekly summaries)
- [ ] Historical backtesting against 30 days of data
- [ ] Validate Sharpe > 1.5 in paper mode

### Week 4: Hardening + Deployment (Days 22-30)
- [ ] Comprehensive error handling and recovery
- [ ] Reconnection logic for WebSocket drops
- [ ] Docker optimization (multi-stage build)
- [ ] AWS EC2 deployment scripts
- [ ] Monitoring and alerting (health checks, PnL alerts)
- [ ] Security audit (API key handling, rate limits)
- [ ] Documentation
- [ ] Decision gate: paper → live (only if metrics pass)

### Go-Live Criteria (ALL must be met)
- [ ] 100+ paper trades completed
- [ ] Sharpe ratio > 1.5 over 30-day evaluation
- [ ] Max drawdown < 10% in paper mode
- [ ] Win rate > 55%
- [ ] All unit + integration tests passing
- [ ] Kill switch tested and verified
- [ ] Manual review of trade log

---

## Current Status

**Phase**: Week 1 - Foundation
**Capital**: $473.45 USDT
**Mode**: Paper trading
**Exchange**: Binance (spot)
