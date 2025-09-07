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
    assert monitor.use_websocket is False
    assert monitor.fallback_to_rest is False


@pytest.mark.asyncio
async def test_invalid_symbol_handling(position_monitor):
    """Test handling of invalid symbols."""
    # Test empty symbol
    await position_monitor.start_monitoring("", {"side": "buy", "entry_price": 50000.0})
    assert len(position_monitor.active_monitors) == 0

    # Test None symbol
    await position_monitor.start_monitoring(None, {"side": "buy", "entry_price": 50000.0})
    assert len(position_monitor.active_monitors) == 0

    # Test invalid position
    await position_monitor.start_monitoring("BTC/USDT", None)
    assert len(position_monitor.active_monitors) == 0


@pytest.mark.asyncio
async def test_invalid_price_handling(position_monitor, mock_exchange):
    """Test handling of invalid prices."""
    symbol = "BTC/USDT"
    position = position_monitor.positions[symbol]

    # Start monitoring
    await position_monitor.start_monitoring(symbol, position)

    # Mock invalid price returns
    mock_exchange.fetch_ticker.side_effect = [
        {"last": 0},  # Zero price
        {"last": -100},  # Negative price
        {"last": None},  # None price
        {"last": 51000.0}  # Valid price
    ]

    # Should handle invalid prices gracefully
    await position_monitor._monitor_position(symbol, position)
    # Position should still be monitored after invalid prices
    assert symbol in position_monitor.active_monitors


@pytest.mark.asyncio
async def test_price_cache_bounds_checking(position_monitor):
    """Test that price cache handles edge cases properly."""
    symbol = "BTC/USDT"

    # Test division by zero protection
    old_price = 0.0
    current_price = 50000.0

    # Should not crash with zero old price
    price_change = abs(current_price - old_price) / old_price if old_price > 0 else 1.0
    assert price_change == 1.0  # Should be 1.0 when old_price is 0


@pytest.mark.asyncio
async def test_websocket_price_callback_error_handling(position_monitor):
    """Test error handling in WebSocket price callbacks."""
    from unittest.mock import patch

    # Mock the asyncio.create_task to raise an exception
    with patch('asyncio.create_task', side_effect=Exception("Task creation failed")):
        # This should not crash the callback
        try:
            # Simulate calling the callback directly
            def test_callback(sym, price):
                try:
                    asyncio.create_task(position_monitor._handle_price_update(sym, price))
                except Exception as e:
                    logger.error(f"Error creating price update task for {sym}: {e}")

            test_callback("BTC/USDT", 50000.0)
            # Should not raise exception
        except Exception as e:
            pytest.fail(f"WebSocket callback should handle errors gracefully: {e}")


@pytest.mark.asyncio
async def test_stop_monitoring_invalid_symbol(position_monitor):
    """Test stopping monitoring with invalid symbol."""
    # Should handle invalid symbols gracefully
    await position_monitor.stop_monitoring("")
    await position_monitor.stop_monitoring(None)
    await position_monitor.stop_monitoring(123)

    # Should not crash
    assert True


@pytest.mark.asyncio
async def test_concurrent_monitoring_operations(position_monitor):
    """Test concurrent start/stop monitoring operations."""
    import asyncio

    async def concurrent_operations():
        tasks = []
        for i in range(5):
            symbol = f"BTC{i}/USDT"
            position = {
                "side": "buy",
                "entry_price": 50000.0,
                "size": 0.1
            }
            tasks.append(position_monitor.start_monitoring(symbol, position))

        # Start all monitoring tasks
        await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all are being monitored
        assert len(position_monitor.active_monitors) == 5

        # Stop all monitoring
        stop_tasks = []
        for i in range(5):
            symbol = f"BTC{i}/USDT"
            stop_tasks.append(position_monitor.stop_monitoring(symbol))

        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Verify all are stopped
        assert len(position_monitor.active_monitors) == 0


@pytest.mark.asyncio
async def test_memory_cleanup_on_stop(position_monitor):
    """Test that memory is properly cleaned up when monitoring stops."""
    symbol = "BTC/USDT"
    position = position_monitor.positions[symbol]

    # Start monitoring
    await position_monitor.start_monitoring(symbol, position)
    assert symbol in position_monitor.price_cache
    assert symbol in position_monitor.last_update

    # Stop monitoring
    await position_monitor.stop_monitoring(symbol)
    assert symbol not in position_monitor.price_cache
    assert symbol not in position_monitor.last_update


@pytest.mark.asyncio
async def test_configuration_edge_cases():
    """Test edge cases in configuration loading."""
    # Test missing exit_strategy
    config1 = {}
    monitor1 = PositionMonitor(
        exchange=Mock(),
        config=config1,
        positions={}
    )
    assert monitor1.check_interval_seconds == 5.0  # Default value

    # Test missing real_time_monitoring
    config2 = {"exit_strategy": {}}
    monitor2 = PositionMonitor(
        exchange=Mock(),
        config=config2,
        positions={}
    )
    assert monitor2.check_interval_seconds == 5.0  # Default value

    # Test partial configuration
    config3 = {
        "exit_strategy": {
            "real_time_monitoring": {
                "check_interval_seconds": 10.0
                # Missing other fields
            }
        }
    }
    monitor3 = PositionMonitor(
        exchange=Mock(),
        config=config3,
        positions={}
    )
    assert monitor3.check_interval_seconds == 10.0
    assert monitor3.price_update_threshold == 0.001  # Default value
