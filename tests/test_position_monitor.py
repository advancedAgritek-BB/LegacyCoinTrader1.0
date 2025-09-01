"""
Tests for the real-time position monitoring system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import time

from crypto_bot.position_monitor import PositionMonitor


@pytest.fixture
def mock_exchange():
    """Create a mock exchange for testing."""
    exchange = Mock()
    exchange.fetch_ticker = AsyncMock()
    exchange.watch_ticker = AsyncMock()
    return exchange


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "exit_strategy": {
            "real_time_monitoring": {
                "enabled": True,
                "check_interval_seconds": 0.1,  # Fast for testing
                "max_monitor_age_seconds": 60.0,
                "price_update_threshold": 0.001,
                "use_websocket_when_available": True,
                "fallback_to_rest": True,
                "max_execution_latency_ms": 1000
            },
            "trailing_stop_pct": 0.02,
            "min_gain_to_trail": 0.01,
            "take_profit_pct": 0.05
        }
    }


@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    return {
        "BTC/USDT": {
            "side": "buy",
            "entry_price": 50000.0,
            "size": 0.1,
            "trailing_stop": 0.0,
            "highest_price": 50000.0,
            "lowest_price": 50000.0,
            "pnl": 0.0
        }
    }


@pytest.fixture
def position_monitor(mock_exchange, sample_config, sample_positions):
    """Create a position monitor instance for testing."""
    return PositionMonitor(
        exchange=mock_exchange,
        config=sample_config,
        positions=sample_positions,
        notifier=None
    )


@pytest.mark.asyncio
async def test_position_monitor_initialization(position_monitor):
    """Test that the position monitor initializes correctly."""
    assert position_monitor.check_interval_seconds == 0.1
    assert position_monitor.max_monitor_age_seconds == 60.0
    assert position_monitor.price_update_threshold == 0.001
    assert position_monitor.use_websocket is True
    assert position_monitor.fallback_to_rest is True
    assert position_monitor.max_execution_latency_ms == 1000


@pytest.mark.asyncio
async def test_start_monitoring(position_monitor):
    """Test starting position monitoring."""
    symbol = "BTC/USDT"
    position = position_monitor.positions[symbol]
    
    await position_monitor.start_monitoring(symbol, position)
    
    assert symbol in position_monitor.active_monitors
    assert symbol in position_monitor.price_cache
    assert symbol in position_monitor.last_update
    assert position_monitor.monitoring_stats["positions_monitored"] == 1


@pytest.mark.asyncio
async def test_stop_monitoring(position_monitor):
    """Test stopping position monitoring."""
    symbol = "BTC/USDT"
    position = position_monitor.positions[symbol]
    
    # Start monitoring first
    await position_monitor.start_monitoring(symbol, position)
    assert symbol in position_monitor.active_monitors
    
    # Stop monitoring
    await position_monitor.stop_monitoring(symbol)
    
    assert symbol not in position_monitor.active_monitors
    assert symbol not in position_monitor.price_cache
    assert symbol not in position_monitor.last_update


@pytest.mark.asyncio
async def test_get_current_price_websocket_success(position_monitor, mock_exchange):
    """Test getting current price via WebSocket."""
    mock_exchange.watch_ticker.return_value = {"last": 51000.0}
    
    price = await position_monitor._get_current_price("BTC/USDT")
    
    assert price == 51000.0
    mock_exchange.watch_ticker.assert_called_once_with("BTC/USDT")


@pytest.mark.asyncio
async def test_get_current_price_websocket_fallback(position_monitor, mock_exchange):
    """Test WebSocket failure with REST fallback."""
    mock_exchange.watch_ticker.side_effect = Exception("WebSocket error")
    mock_exchange.fetch_ticker.return_value = {"last": 51000.0}
    
    price = await position_monitor._get_current_price("BTC/USDT")
    
    assert price == 51000.0
    mock_exchange.watch_ticker.assert_called_once_with("BTC/USDT")
    mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")


    @pytest.mark.asyncio
    async def test_update_trailing_stop_long_position(position_monitor):
        """Test trailing stop update for long position."""
        symbol = "BTC/USDT"
        position = {
            "side": "buy",
            "entry_price": 50000.0,
            "highest_price": 51000.0,
            "trailing_stop": 49000.0
        }
        
        # Price moves higher
        current_price = 52000.0
        
        # Update position tracking first (which updates highest price)
        await position_monitor._update_position_tracking(symbol, position, current_price)
        
        # Highest price should be updated
        assert position["highest_price"] == 52000.0
        # Trailing stop should be updated (2% below highest)
        expected_trailing_stop = 52000.0 * (1 - 0.02)
        assert position["trailing_stop"] == expected_trailing_stop


    @pytest.mark.asyncio
    async def test_update_trailing_stop_short_position(position_monitor):
        """Test trailing stop update for short position."""
        symbol = "BTC/USDT"
        position = {
            "side": "sell",
            "entry_price": 50000.0,
            "lowest_price": 49000.0,
            "trailing_stop": 51000.0
        }
        
        # Price moves lower
        current_price = 48000.0
        
        # Update position tracking first (which updates lowest price)
        await position_monitor._update_position_tracking(symbol, position, current_price)
        
        # Lowest price should be updated
        assert position["lowest_price"] == 48000.0
        # Trailing stop should be updated (2% above lowest)
        expected_trailing_stop = 48000.0 * (1 + 0.02)
        assert position["trailing_stop"] == expected_trailing_stop


@pytest.mark.asyncio
async def test_check_exit_conditions_trailing_stop_hit(position_monitor):
    """Test exit conditions when trailing stop is hit."""
    symbol = "BTC/USDT"
    position = {
        "side": "buy",
        "entry_price": 50000.0,
        "trailing_stop": 51000.0
    }
    
    # Price falls below trailing stop
    current_price = 50900.0
    
    should_exit, exit_reason = await position_monitor._check_exit_conditions(
        symbol, position, current_price
    )
    
    assert should_exit is True
    assert exit_reason == "trailing_stop"


@pytest.mark.asyncio
async def test_check_exit_conditions_take_profit_hit(position_monitor):
    """Test exit conditions when take profit is hit."""
    symbol = "BTC/USDT"
    position = {
        "side": "buy",
        "entry_price": 50000.0,
        "trailing_stop": 0.0
    }
    
    # Price hits take profit (5% above entry)
    current_price = 52500.0  # 5% above 50000
    
    should_exit, exit_reason = await position_monitor._check_exit_conditions(
        symbol, position, current_price
    )
    
    assert should_exit is True
    assert exit_reason == "take_profit"


@pytest.mark.asyncio
async def test_check_exit_conditions_no_exit(position_monitor):
    """Test exit conditions when no exit should occur."""
    symbol = "BTC/USDT"
    position = {
        "side": "buy",
        "entry_price": 50000.0,
        "trailing_stop": 49000.0
    }
    
    # Price is above trailing stop and below take profit
    current_price = 50500.0
    
    should_exit, exit_reason = await position_monitor._check_exit_conditions(
        symbol, position, current_price
    )
    
    assert should_exit is False
    assert exit_reason == ""


@pytest.mark.asyncio
async def test_get_monitoring_stats(position_monitor):
    """Test getting monitoring statistics."""
    stats = await position_monitor.get_monitoring_stats()
    
    expected_keys = [
        "positions_monitored", "price_updates", "trailing_stop_triggers",
        "execution_latency_ms", "missed_exits", "active_monitors",
        "monitored_symbols", "price_cache_size"
    ]
    
    for key in expected_keys:
        assert key in stats


@pytest.mark.asyncio
async def test_cleanup_old_monitors(position_monitor):
    """Test cleanup of old monitors."""
    symbol = "BTC/USDT"
    position = position_monitor.positions[symbol]
    
    # Start monitoring
    await position_monitor.start_monitoring(symbol, position)
    
    # Set last update to old time
    position_monitor.last_update[symbol] = time.time() - 400  # 400 seconds ago
    
    # Cleanup should remove the old monitor
    await position_monitor.cleanup_old_monitors()
    
    assert symbol not in position_monitor.active_monitors


@pytest.mark.asyncio
async def test_stop_all_monitoring(position_monitor):
    """Test stopping all monitoring."""
    symbols = ["BTC/USDT", "ETH/USDT"]
    
    # Start monitoring multiple positions
    for symbol in symbols:
        position = position_monitor.positions.get(symbol, {
            "side": "buy",
            "entry_price": 50000.0,
            "size": 0.1
        })
        await position_monitor.start_monitoring(symbol, position)
    
    assert len(position_monitor.active_monitors) == 2
    
    # Stop all monitoring
    await position_monitor.stop_all_monitoring()
    
    assert len(position_monitor.active_monitors) == 0


@pytest.mark.asyncio
async def test_monitoring_disabled_in_config():
    """Test that monitoring can be disabled via configuration."""
    config = {
        "exit_strategy": {
            "real_time_monitoring": {
                "enabled": False
            }
        }
    }
    
    monitor = PositionMonitor(
        exchange=Mock(),
        config=config,
        positions={},
        notifier=None
    )
    
    # Should have very long intervals when disabled
    assert monitor.check_interval_seconds == 3600.0
    assert monitor.max_monitor_age_seconds == 7200.0
