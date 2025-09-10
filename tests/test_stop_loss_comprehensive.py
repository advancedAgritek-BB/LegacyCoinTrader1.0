#!/usr/bin/env python3
"""
Comprehensive tests for stop loss and trailing stop loss functionality.

This test suite ensures that the stop loss system works correctly
after the fixes applied to the configuration and main trading loop.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import yaml
from pathlib import Path

from crypto_bot.risk.exit_manager import should_exit, calculate_trailing_stop
from crypto_bot.position_monitor import PositionMonitor


class TestStopLossConfiguration:
    """Test that stop loss configuration is properly set."""
    
    def test_stop_loss_pct_in_config(self):
        """Test that stop_loss_pct is present in configuration."""
        config_path = Path("crypto_bot/config.yaml")
        assert config_path.exists(), "Configuration file should exist"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        exit_cfg = config.get("exit_strategy", {})
        assert "stop_loss_pct" in exit_cfg, "stop_loss_pct should be in exit_strategy"
        assert exit_cfg["stop_loss_pct"] == 0.01, "stop_loss_pct should be 0.01 (1%)"
        
        # Test other critical settings
        assert "trailing_stop_pct" in exit_cfg, "trailing_stop_pct should be present"
        assert "take_profit_pct" in exit_cfg, "take_profit_pct should be present"
        assert "min_gain_to_trail" in exit_cfg, "min_gain_to_trail should be present"
        
        # Test real-time monitoring
        monitoring_cfg = exit_cfg.get("real_time_monitoring", {})
        assert monitoring_cfg.get("enabled", False), "Real-time monitoring should be enabled"
    
    def test_momentum_aware_exits_enabled(self):
        """Test that momentum-aware exits are enabled."""
        config_path = Path("crypto_bot/config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        exit_cfg = config.get("exit_strategy", {})
        assert exit_cfg.get("momentum_aware_exits", False), "Momentum-aware exits should be enabled"
        assert exit_cfg.get("momentum_tp_scaling", False), "Momentum TP scaling should be enabled"
        assert exit_cfg.get("momentum_trail_adjustment", False), "Momentum trail adjustment should be enabled"


class TestStopLossLogic:
    """Test stop loss logic functionality."""
    
    def test_basic_stop_loss_trigger(self):
        """Test basic stop loss trigger."""
        # Create test data
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [100, 101, 102],
            "low": [100, 101, 102],
            "close": [100, 101, 102],
            "volume": [1000, 1000, 1000]
        })
        
        entry_price = 100.0
        current_price = 98.0  # 2% below entry
        trailing_stop = 99.0  # 1% below entry
        
        config = {
            "exit_strategy": {
                "stop_loss_pct": 0.01,
                "trailing_stop_pct": 0.008,
                "take_profit_pct": 0.04
            }
        }
        
        # Test long position stop loss
        exit_signal, new_stop = should_exit(
            df, current_price, trailing_stop, config, 
            position_side="buy", entry_price=entry_price
        )
        
        assert exit_signal is True, "Should trigger stop loss for long position"
    
    def test_basic_stop_loss_trigger_no_momentum(self):
        """Test basic stop loss trigger without momentum blocking."""
        # Create test data
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [100, 101, 102],
            "low": [100, 101, 102],
            "close": [100, 101, 102],
            "volume": [1000, 1000, 1000]
        })
        
        entry_price = 100.0
        current_price = 98.0  # 2% below entry
        trailing_stop = 99.0  # 1% below entry
        
        config = {
            "exit_strategy": {
                "stop_loss_pct": 0.01,
                "trailing_stop_pct": 0.008,
                "take_profit_pct": 0.04,
                "momentum_aware_exits": False  # Disable momentum blocking
            }
        }
        
        # Test long position stop loss
        exit_signal, new_stop = should_exit(
            df, current_price, trailing_stop, config,
            position_side="buy", entry_price=entry_price
        )
        
        assert exit_signal is True, "Should trigger stop loss for long position (no momentum)"
    
    def test_basic_stop_loss_trigger_with_momentum(self):
        """Test basic stop loss trigger with momentum blocking disabled."""
        # Create test data
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [100, 101, 102],
            "low": [100, 101, 102],
            "close": [100, 101, 102],
            "volume": [1000, 1000, 1000]
        })
        
        entry_price = 100.0
        current_price = 98.0  # 2% below entry
        trailing_stop = 99.0  # 1% below entry
        
        config = {
            "exit_strategy": {
                "stop_loss_pct": 0.01,
                "trailing_stop_pct": 0.008,
                "take_profit_pct": 0.04,
                "momentum_aware_exits": True,
                "momentum_continuation": {
                    "very_strong_momentum": 0.95,  # Use correct field name
                    "breakout_threshold": 0.02
                }
            }
        }
        
        # Test long position stop loss
        exit_signal, new_stop = should_exit(
            df, current_price, trailing_stop, config,
            position_side="buy", entry_price=entry_price
        )
        
        assert exit_signal is True, "Should trigger stop loss for long position (high momentum threshold)"
    
    def test_trailing_stop_update(self):
        """Test trailing stop updates correctly."""
        # Create test data with price moving up
        df = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [100, 101, 102, 103, 104],
            "low": [100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104],
            "volume": [1000] * 5
        })
        
        entry_price = 100.0
        current_price = 103.0  # 3% above entry (below 4% take profit)
        initial_trailing_stop = 99.0
        
        config = {
            "exit_strategy": {
                "stop_loss_pct": 0.01,
                "trailing_stop_pct": 0.02,  # 2% trailing
                "min_gain_to_trail": 0.005,  # Start trailing after 0.5% gain
                "take_profit_pct": 0.04
            }
        }
        
        # Test trailing stop update
        exit_signal, new_stop = should_exit(
            df, current_price, initial_trailing_stop, config,
            position_side="buy", entry_price=entry_price
        )
        
        assert exit_signal is False, "Should not exit at profit"
        assert new_stop > initial_trailing_stop, "Trailing stop should move up"
        expected_trailing_stop = 104.0 * (1 - 0.02)  # 2% below current high
        assert abs(new_stop - expected_trailing_stop) < 0.01, "Trailing stop should be calculated correctly"
    
    def test_take_profit_trigger(self):
        """Test take profit trigger."""
        # Create test data
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [100, 101, 102],
            "low": [100, 101, 102],
            "close": [100, 101, 102],
            "volume": [1000, 1000, 1000]
        })
        
        entry_price = 100.0
        current_price = 105.0  # 5% above entry (above 4% take profit)
        trailing_stop = 99.0
        
        config = {
            "exit_strategy": {
                "stop_loss_pct": 0.01,
                "trailing_stop_pct": 0.02,
                "take_profit_pct": 0.04
            }
        }
        
        # Test take profit trigger
        exit_signal, new_stop = should_exit(
            df, current_price, trailing_stop, config,
            position_side="buy", entry_price=entry_price
        )
        
        assert exit_signal is True, "Should trigger take profit"
    
    def test_short_position_stop_loss(self):
        """Test stop loss for short positions."""
        # Create test data
        df = pd.DataFrame({
            "open": [100, 99, 98],
            "high": [100, 99, 98],
            "low": [100, 99, 98],
            "close": [100, 99, 98],
            "volume": [1000, 1000, 1000]
        })
        
        entry_price = 100.0
        current_price = 102.0  # 2% above entry (loss for short)
        trailing_stop = 101.0  # 1% above entry
        
        config = {
            "exit_strategy": {
                "stop_loss_pct": 0.01,
                "trailing_stop_pct": 0.02,
                "take_profit_pct": 0.04
            }
        }
        
        # Test short position stop loss
        exit_signal, new_stop = should_exit(
            df, current_price, trailing_stop, config,
            position_side="sell", entry_price=entry_price
        )
        
        assert exit_signal is True, "Should trigger stop loss for short position"


class TestPositionMonitorIntegration:
    """Test position monitor integration with stop loss."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange."""
        exchange = Mock()
        exchange.fetch_ticker = AsyncMock()
        exchange.watch_ticker = AsyncMock()
        return exchange
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return {
            "exit_strategy": {
                "stop_loss_pct": 0.01,
                "trailing_stop_pct": 0.02,
                "take_profit_pct": 0.04,
                "min_gain_to_trail": 0.005,
                "real_time_monitoring": {
                    "enabled": True,
                    "check_interval_seconds": 0.1,
                    "max_monitor_age_seconds": 60.0,
                    "price_update_threshold": 0.001,
                    "use_websocket_when_available": True,
                    "fallback_to_rest": True,
                    "max_execution_latency_ms": 1000
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_position_monitor_stop_loss_trigger(self, mock_exchange, test_config):
        """Test that position monitor triggers stop loss correctly."""
        # Create position monitor
        positions = {
            "BTC/USDT": {
                "side": "buy",
                "entry_price": 50000.0,
                "size": 0.1,
                "trailing_stop": 49500.0,  # 1% below entry
                "highest_price": 50000.0,
                "lowest_price": 50000.0,
                "pnl": 0.0
            }
        }
        
        monitor = PositionMonitor(
            exchange=mock_exchange,
            config=test_config,
            positions=positions,
            notifier=None
        )
        
        # Test stop loss trigger
        current_price = 49400.0  # Below stop loss
        should_exit, exit_reason = await monitor._check_exit_conditions(
            "BTC/USDT", positions["BTC/USDT"], current_price
        )
        
        assert should_exit is True, "Should trigger stop loss"
        assert exit_reason == "trailing_stop", "Should be trailing stop reason"
    
    @pytest.mark.asyncio
    async def test_position_monitor_trailing_stop_update(self, mock_exchange, test_config):
        """Test that position monitor updates trailing stops correctly."""
        positions = {
            "BTC/USDT": {
                "side": "buy",
                "entry_price": 50000.0,
                "size": 0.1,
                "trailing_stop": 49500.0,
                "highest_price": 50000.0,
                "lowest_price": 50000.0,
                "pnl": 0.0
            }
        }
        
        monitor = PositionMonitor(
            exchange=mock_exchange,
            config=test_config,
            positions=positions,
            notifier=None
        )
        
        # Test trailing stop update
        current_price = 51000.0  # 2% above entry, should update trailing stop
        
        # Update position tracking first (which updates highest price)
        await monitor._update_position_tracking("BTC/USDT", positions["BTC/USDT"], current_price)
        
        # Check that trailing stop was updated
        expected_trailing_stop = 51000.0 * (1 - 0.02)  # 2% below current price
        assert abs(positions["BTC/USDT"]["trailing_stop"] - expected_trailing_stop) < 0.01
    
    @pytest.mark.asyncio
    async def test_position_monitor_take_profit_trigger(self, mock_exchange, test_config):
        """Test that position monitor triggers take profit correctly."""
        positions = {
            "BTC/USDT": {
                "side": "buy",
                "entry_price": 50000.0,
                "size": 0.1,
                "trailing_stop": 49500.0,
                "highest_price": 50000.0,
                "lowest_price": 50000.0,
                "pnl": 0.0
            }
        }
        
        monitor = PositionMonitor(
            exchange=mock_exchange,
            config=test_config,
            positions=positions,
            notifier=None
        )
        
        # Test take profit trigger
        current_price = 52000.0  # 4% above entry, should trigger take profit
        
        should_exit, exit_reason = await monitor._check_exit_conditions(
            "BTC/USDT", positions["BTC/USDT"], current_price
        )
        
        assert should_exit is True, "Should trigger take profit"
        assert exit_reason == "take_profit", "Should be take profit reason"


class TestMainLoopIntegration:
    """Test main loop integration with stop loss."""
    
    def test_handle_exits_function_exists(self):
        """Test that handle_exits function exists in main.py."""
        main_path = Path("crypto_bot/main.py")
        assert main_path.exists(), "main.py should exist"
        
        with open(main_path, 'r') as f:
            content = f.read()
        
        assert 'async def handle_exits(ctx: BotContext) -> None:' in content, "handle_exits function should exist"
        assert 'should_exit(' in content, "should_exit should be called in handle_exits"
        assert 'ctx.position_monitor' in content, "Position monitor should be used"
    
    def test_phase_runner_includes_handle_exits(self):
        """Test that phase runner includes handle_exits."""
        main_path = Path("crypto_bot/main.py")
        
        with open(main_path, 'r') as f:
            content = f.read()
        
        assert 'handle_exits,' in content, "handle_exits should be in phase runner"
        assert 'PhaseRunner(' in content, "PhaseRunner should be used"


class TestEmergencyStopLossMonitor:
    """Test emergency stop loss monitor functionality."""
    
    def test_emergency_monitor_exists(self):
        """Test that emergency stop loss monitor exists."""
        emergency_path = Path("emergency_stop_loss_monitor.py")
        assert emergency_path.exists(), "Emergency stop loss monitor should exist"
        
        with open(emergency_path, 'r') as f:
            content = f.read()
        
        assert 'class EmergencyStopLossMonitor:' in content, "EmergencyStopLossMonitor class should exist"
        assert 'should_exit' in content, "should_exit method should exist"
        assert 'calculate_stop_loss' in content, "calculate_stop_loss method should exist"
        assert 'calculate_trailing_stop' in content, "calculate_trailing_stop method should exist"


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_all_required_settings_present(self):
        """Test that all required stop loss settings are present."""
        config_path = Path("crypto_bot/config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        exit_cfg = config.get("exit_strategy", {})
        
        # Required settings
        required_settings = [
            "stop_loss_pct",
            "trailing_stop_pct", 
            "take_profit_pct",
            "min_gain_to_trail"
        ]
        
        for setting in required_settings:
            assert setting in exit_cfg, f"{setting} should be present in exit_strategy"
            assert exit_cfg[setting] is not None, f"{setting} should not be None"
            assert isinstance(exit_cfg[setting], (int, float)), f"{setting} should be numeric"
    
    def test_real_time_monitoring_enabled(self):
        """Test that real-time monitoring is properly configured."""
        config_path = Path("crypto_bot/config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        exit_cfg = config.get("exit_strategy", {})
        monitoring_cfg = exit_cfg.get("real_time_monitoring", {})
        
        assert monitoring_cfg.get("enabled", False), "Real-time monitoring should be enabled"
        assert monitoring_cfg.get("check_interval_seconds", 0) > 0, "Check interval should be positive"
        assert monitoring_cfg.get("max_monitor_age_seconds", 0) > 0, "Max monitor age should be positive"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
