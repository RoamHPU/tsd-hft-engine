"""
Configuration Loader
====================
Single entry point for all trading configuration.
Every module imports config from here. No exceptions.
"""

import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "trading_config.yaml"


@dataclass(frozen=True)
class ExchangeConfig:
    name: str
    base_url: str
    use_testnet: bool
    requests_per_minute: int
    orders_per_second: int
    orders_per_day: int


@dataclass(frozen=True)
class CapitalConfig:
    total_usdt: float
    max_allocation_pct: float
    deployable_usdt: float
    reserve_usdt: float
    min_reserve_pct: float


@dataclass(frozen=True)
class PositionConfig:
    max_concurrent: int
    min_position_usdt: float
    max_position_usdt: float
    max_position_pct: float
    default_position_pct: float


@dataclass(frozen=True)
class RiskConfig:
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_enabled: bool
    trailing_stop_pct: float
    daily_loss_limit_pct: float
    daily_loss_limit_usdt: float
    max_trades_per_day: int
    consecutive_loss_halt: int
    kill_switch_enabled: bool
    kill_switch_drawdown_pct: float
    kill_switch_drawdown_usdt: float
    max_single_asset_pct: float
    max_correlated_exposure_pct: float


@dataclass(frozen=True)
class FeeConfig:
    spot_maker_pct: float
    spot_taker_pct: float
    futures_maker_pct: float
    futures_taker_pct: float
    bnb_discount_pct: float
    use_bnb_for_fees: bool
    min_profit_threshold_pct: float


@dataclass(frozen=True)
class StrategyConfig:
    enabled: List[str]
    weights: Dict[str, float]


@dataclass(frozen=True)
class PaperTradingConfig:
    enabled: bool
    initial_capital: float
    track_slippage: bool
    simulated_slippage_bps: int
    simulated_latency_ms: int


@dataclass(frozen=True)
class EvaluationConfig:
    cycle_interval_seconds: int
    track_metrics: List[str]
    min_sharpe_for_live: float
    evaluation_window_days: int
    min_trades_for_evaluation: int


@dataclass(frozen=True)
class TradingConfig:
    """Master configuration object. Immutable after creation."""
    exchange: ExchangeConfig
    capital: CapitalConfig
    positions: PositionConfig
    risk: RiskConfig
    fees: FeeConfig
    trading_pairs_tier_1: List[str]
    trading_pairs_tier_2: List[str]
    trading_pairs_tier_3: List[str]
    strategies: StrategyConfig
    paper_trading: PaperTradingConfig
    evaluation: EvaluationConfig
    environment: str
    host: str
    port: int

    @property
    def all_trading_pairs(self) -> List[str]:
        return self.trading_pairs_tier_1 + self.trading_pairs_tier_2 + self.trading_pairs_tier_3

    @property
    def is_paper(self) -> bool:
        return self.paper_trading.enabled

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


def _load_yaml() -> dict:
    """Load raw YAML config."""
    config_path = Path(os.environ.get("TSD_CONFIG_PATH", CONFIG_PATH))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_config() -> TradingConfig:
    """
    Load and validate the master trading configuration.
    Returns an immutable TradingConfig object.
    """
    raw = _load_yaml()

    # Build typed config objects
    exchange = ExchangeConfig(
        name=raw["exchange"]["name"],
        base_url=raw["exchange"]["base_url"],
        use_testnet=raw["exchange"]["use_testnet"],
        requests_per_minute=raw["exchange"]["rate_limits"]["requests_per_minute"],
        orders_per_second=raw["exchange"]["rate_limits"]["orders_per_second"],
        orders_per_day=raw["exchange"]["rate_limits"]["orders_per_day"],
    )

    cap = raw["capital"]
    capital = CapitalConfig(
        total_usdt=cap["total_usdt"],
        max_allocation_pct=cap["max_allocation_pct"],
        deployable_usdt=cap["total_usdt"] * (cap["max_allocation_pct"] / 100),
        reserve_usdt=cap["total_usdt"] * (cap["min_reserve_pct"] / 100),
        min_reserve_pct=cap["min_reserve_pct"],
    )

    pos = raw["positions"]
    positions = PositionConfig(
        max_concurrent=pos["max_concurrent"],
        min_position_usdt=pos["min_position_usdt"],
        max_position_usdt=pos["max_position_usdt"],
        max_position_pct=pos["max_position_pct"],
        default_position_pct=pos["default_position_pct"],
    )

    r = raw["risk"]
    risk = RiskConfig(
        stop_loss_pct=r["stop_loss_pct"],
        take_profit_pct=r["take_profit_pct"],
        trailing_stop_enabled=r["trailing_stop_enabled"],
        trailing_stop_pct=r["trailing_stop_pct"],
        daily_loss_limit_pct=r["daily_loss_limit_pct"],
        daily_loss_limit_usdt=capital.total_usdt * (r["daily_loss_limit_pct"] / 100),
        max_trades_per_day=r["max_trades_per_day"],
        consecutive_loss_halt=r["consecutive_loss_halt"],
        kill_switch_enabled=r["kill_switch_enabled"],
        kill_switch_drawdown_pct=r["kill_switch_drawdown_pct"],
        kill_switch_drawdown_usdt=capital.total_usdt * (r["kill_switch_drawdown_pct"] / 100),
        max_single_asset_pct=r["max_single_asset_pct"],
        max_correlated_exposure_pct=r["max_correlated_exposure_pct"],
    )

    f = raw["fees"]
    fees = FeeConfig(
        spot_maker_pct=f["spot"]["maker_pct"],
        spot_taker_pct=f["spot"]["taker_pct"],
        futures_maker_pct=f["futures"]["maker_pct"],
        futures_taker_pct=f["futures"]["taker_pct"],
        bnb_discount_pct=f["bnb_discount_pct"],
        use_bnb_for_fees=f["use_bnb_for_fees"],
        min_profit_threshold_pct=f["min_profit_threshold_pct"],
    )

    s = raw["strategies"]
    strategies = StrategyConfig(
        enabled=s["enabled"],
        weights=s["weights"],
    )

    pt = raw["paper_trading"]
    paper_trading = PaperTradingConfig(
        enabled=pt["enabled"],
        initial_capital=pt["initial_capital"],
        track_slippage=pt["track_slippage"],
        simulated_slippage_bps=pt["simulated_slippage_bps"],
        simulated_latency_ms=pt["simulated_latency_ms"],
    )

    ev = raw["evaluation"]
    evaluation = EvaluationConfig(
        cycle_interval_seconds=ev.get("cycle_interval_seconds", 60),
        track_metrics=ev["track_metrics"],
        min_sharpe_for_live=ev["min_sharpe_for_live"],
        evaluation_window_days=ev["evaluation_window_days"],
        min_trades_for_evaluation=ev["min_trades_for_evaluation"],
    )

    pairs = raw["trading_pairs"]
    dep = raw["deployment"]

    config = TradingConfig(
        exchange=exchange,
        capital=capital,
        positions=positions,
        risk=risk,
        fees=fees,
        trading_pairs_tier_1=pairs["tier_1"],
        trading_pairs_tier_2=pairs["tier_2"],
        trading_pairs_tier_3=pairs["tier_3"],
        strategies=strategies,
        paper_trading=paper_trading,
        evaluation=evaluation,
        environment=dep["environment"],
        host=dep["host"],
        port=dep["port"],
    )

    _validate_config(config)
    logger.info(f"Config loaded: {config.environment} | Capital: ${config.capital.total_usdt:.2f} | Paper: {config.is_paper}")
    return config


def _validate_config(config: TradingConfig):
    """Validate configuration for internal consistency."""
    errors = []

    # Capital sanity checks
    if config.capital.total_usdt <= 0:
        errors.append("Capital must be positive")

    if config.positions.min_position_usdt > config.capital.deployable_usdt:
        errors.append(
            f"Min position (${config.positions.min_position_usdt}) > "
            f"deployable capital (${config.capital.deployable_usdt:.2f})"
        )

    # Max concurrent positions must be achievable
    max_possible = int(config.capital.deployable_usdt / config.positions.min_position_usdt)
    if config.positions.max_concurrent > max_possible:
        errors.append(
            f"max_concurrent ({config.positions.max_concurrent}) impossible with "
            f"${config.capital.deployable_usdt:.2f} deployable at "
            f"${config.positions.min_position_usdt} minimum"
        )

    # Risk limits must be set
    if not config.risk.kill_switch_enabled:
        errors.append("Kill switch MUST be enabled")

    # Strategy weights must sum to ~1.0
    weight_sum = sum(config.strategies.weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(f"Strategy weights sum to {weight_sum}, must equal 1.0")

    # Fee threshold must exceed round-trip costs
    round_trip_cost = (config.fees.spot_maker_pct + config.fees.spot_taker_pct) / 100
    if config.fees.min_profit_threshold_pct / 100 <= round_trip_cost:
        errors.append(
            f"min_profit_threshold ({config.fees.min_profit_threshold_pct}%) must exceed "
            f"round-trip fees ({round_trip_cost * 100:.2f}%)"
        )

    if errors:
        for e in errors:
            logger.error(f"CONFIG VALIDATION FAILED: {e}")
        raise ValueError(f"Configuration invalid: {'; '.join(errors)}")

    logger.info("Configuration validated successfully")
