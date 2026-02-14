"""
Unit Tests - Core Components
=============================
Tests for config validation, risk manager, exchange adapter, and trade store.
Run with: pytest tests/unit/test_core.py -v
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestConfig:
    """Test configuration loading and validation."""

    def test_config_loads_successfully(self):
        from config import load_config
        config = load_config()
        assert config is not None
        assert config.capital.total_usdt > 0

    def test_capital_calculations(self):
        from config import load_config
        config = load_config()
        expected_deployable = config.capital.total_usdt * (config.capital.max_allocation_pct / 100)
        assert abs(config.capital.deployable_usdt - expected_deployable) < 0.01

    def test_risk_limits_set(self):
        from config import load_config
        config = load_config()
        assert config.risk.kill_switch_enabled is True
        assert config.risk.stop_loss_pct > 0
        assert config.risk.daily_loss_limit_pct > 0
        assert config.risk.kill_switch_drawdown_pct > 0

    def test_strategy_weights_sum_to_one(self):
        from config import load_config
        config = load_config()
        total = sum(config.strategies.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_paper_mode_default(self):
        from config import load_config
        config = load_config()
        assert config.is_paper is True  # Must start in paper mode

    def test_trading_pairs_not_empty(self):
        from config import load_config
        config = load_config()
        assert len(config.all_trading_pairs) > 0
        assert "BTCUSDT" in config.all_trading_pairs

    def test_fee_threshold_exceeds_costs(self):
        from config import load_config
        config = load_config()
        round_trip = (config.fees.spot_maker_pct + config.fees.spot_taker_pct) / 100
        threshold = config.fees.min_profit_threshold_pct / 100
        assert threshold > round_trip

    def test_max_positions_achievable(self):
        from config import load_config
        config = load_config()
        max_possible = int(config.capital.deployable_usdt / config.positions.min_position_usdt)
        assert config.positions.max_concurrent <= max_possible


# ============================================================================
# RISK MANAGER TESTS
# ============================================================================

class TestRiskManager:
    """Test risk management logic."""

    @pytest.fixture
    def risk_setup(self):
        from config import load_config
        from core.risk.manager import RiskManager
        config = load_config()
        rm = RiskManager(config)
        return config, rm

    def _make_proposal(self, symbol="BTCUSDT", side="buy", quantity=0.0003,
                       price=97000.0, strategy="momentum", expected_profit=0.5,
                       signal=0.7):
        from core.risk.manager import TradeProposal
        return TradeProposal(
            symbol=symbol, side=side, quantity=quantity, price=price,
            strategy=strategy, expected_profit_pct=expected_profit,
            signal_strength=signal,
        )

    def test_valid_trade_approved(self, risk_setup):
        config, rm = risk_setup
        proposal = self._make_proposal()  # ~$29
        result = rm.evaluate_trade(proposal)
        assert result.approved

    def test_oversized_trade_rejected(self, risk_setup):
        config, rm = risk_setup
        proposal = self._make_proposal(quantity=0.01)  # ~$970
        result = rm.evaluate_trade(proposal)
        assert not result.approved
        assert "too_large" in result.verdict.value or "position" in result.verdict.value.lower()

    def test_below_profit_threshold_rejected(self, risk_setup):
        config, rm = risk_setup
        proposal = self._make_proposal(expected_profit=0.1)  # Below 0.3% threshold
        result = rm.evaluate_trade(proposal)
        assert not result.approved
        assert "profit_threshold" in result.verdict.value

    def test_kill_switch_blocks_all(self, risk_setup):
        config, rm = risk_setup
        rm.activate_kill_switch("test")
        proposal = self._make_proposal()
        result = rm.evaluate_trade(proposal)
        assert not result.approved
        assert "kill_switch" in result.verdict.value

    def test_kill_switch_deactivation(self, risk_setup):
        config, rm = risk_setup
        rm.activate_kill_switch("test")
        rm.deactivate_kill_switch()
        proposal = self._make_proposal()
        result = rm.evaluate_trade(proposal)
        assert result.approved

    def test_consecutive_loss_halt(self, risk_setup):
        from core.risk.manager import Position
        config, rm = risk_setup
        # Simulate consecutive losses
        for i in range(config.risk.consecutive_loss_halt):
            pos = Position(
                symbol="BTCUSDT", side="buy", quantity=0.0003,
                entry_price=97000, entry_time=f"2026-01-01T00:00:0{i}",
                strategy="momentum", stop_loss_price=96000, take_profit_price=98000,
            )
            key = f"BTCUSDT_buy_2026-01-01T00:00:0{i}"
            rm.register_position(pos)
            rm.close_position(key, exit_price=96500, pnl=-1.5)

        # Next trade should be rejected
        proposal = self._make_proposal()
        result = rm.evaluate_trade(proposal)
        assert not result.approved
        assert "consecutive" in result.verdict.value

    def test_max_concurrent_positions(self, risk_setup):
        from core.risk.manager import Position
        config, rm = risk_setup
        # Fill all position slots with positions above minimum
        for i in range(config.positions.max_concurrent):
            pos = Position(
                symbol="ETHUSDT", side="buy", quantity=0.007,
                entry_price=3000, entry_time=f"2026-01-01T00:00:{i:02d}",
                strategy="momentum", stop_loss_price=2900, take_profit_price=3100,
            )
            rm.register_position(pos)

        # Next trade should be rejected (all slots full)
        proposal = self._make_proposal(symbol="SOLUSDT", quantity=0.1, price=200,
                                       expected_profit=0.5)
        result = rm.evaluate_trade(proposal)
        assert not result.approved

    def test_status_returns_valid_data(self, risk_setup):
        config, rm = risk_setup
        status = rm.get_status()
        assert "capital" in status
        assert "daily_pnl" in status
        assert "kill_switch_active" in status
        assert status["capital"] == config.capital.total_usdt

    def test_auto_kill_switch_on_drawdown(self, risk_setup):
        from core.risk.manager import Position
        config, rm = risk_setup
        # Simulate loss exceeding kill switch threshold
        loss_needed = config.risk.kill_switch_drawdown_usdt + 1.0  # Exceed to avoid float edge
        pos = Position(
            symbol="BTCUSDT", side="buy", quantity=0.001,
            entry_price=97000, entry_time="2026-01-01T00:00:00",
            strategy="momentum", stop_loss_price=90000, take_profit_price=100000,
        )
        rm.register_position(pos)
        rm.close_position("BTCUSDT_buy_2026-01-01T00:00:00",
                          exit_price=90000, pnl=-loss_needed)

        assert rm._kill_switch_active is True


# ============================================================================
# TRADE STORE TESTS
# ============================================================================

class TestTradeStore:
    """Test trade persistence."""

    @pytest.fixture
    def store(self, tmp_path):
        from data.storage.trade_store import TradeStore
        db_path = tmp_path / "test.db"
        return TradeStore(db_url=f"sqlite:///{db_path}")

    def test_record_and_retrieve_trade(self, store):
        from core.risk.manager import TradeProposal
        from core.exchange.binance_adapter import OrderResult

        proposal = TradeProposal(
            symbol="BTCUSDT", side="buy", quantity=0.0003,
            price=97000.0, strategy="momentum",
            expected_profit_pct=0.5, signal_strength=0.7,
        )
        order = OrderResult(
            order_id="TEST_001", symbol="BTCUSDT", side="buy",
            quantity=0.0003, price=97000.0, status="FILLED",
            filled_quantity=0.0003, avg_fill_price=97000.0,
            commission=0.029, commission_asset="USDT",
            timestamp="2026-01-01T00:00:00Z", raw_response={},
        )

        trade_id = store.record_trade(proposal, order, is_paper=True)
        assert trade_id > 0

        trades = store.get_recent_trades(limit=10)
        assert len(trades) == 1
        assert trades[0]["order_id"] == "TEST_001"
        assert trades[0]["symbol"] == "BTCUSDT"

    def test_trade_count(self, store):
        assert store.get_trade_count() == 0

    def test_system_event_recording(self, store):
        store.record_system_event("test", "Unit test event", severity="info")
        # No exception = success

    def test_total_pnl_empty(self, store):
        assert store.get_total_pnl() == 0.0


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
