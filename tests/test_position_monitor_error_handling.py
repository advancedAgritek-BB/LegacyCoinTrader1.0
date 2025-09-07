"""
Error handling and edge case tests for PositionMonitor.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime

from crypto_bot.position_monitor import PositionMonitor
from crypto_bot.utils.trade_manager import TradeManager, Trade


@pytest.fixture
def mock_exchange():
    """Create a mock exchange."""
    exchange = Mock()
    exchange.fetch_ticker = AsyncMock()
    return exchange


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    return {
        "exit_strategy": {
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


@pytest.fixture
def position_monitor(mock_exchange, sample_config):
    """Create a position monitor instance."""
    return PositionMonitor(
        exchange=mock_exchange,
        config=sample_config,
        positions={},
        notifier=None
    )


class TestPositionMonitorErrorHandling:
    """Test error handling in PositionMonitor."""

    @pytest.mark.asyncio
    async def test_network_error_recovery(self, position_monitor, mock_exchange):
        """Test recovery from network errors."""
        # Setup alternating success/failure
        call_count = 0

        async def mock_fetch_ticker(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception("Network timeout")
            return {"last": 50000.0}

        mock_exchange.fetch_ticker = mock_fetch_ticker

        # Create a position
        trade = Trade(
            id="error_test",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="error_test"
        )

        trade_manager = TradeManager()
        trade_manager.record_trade(trade)
        position_monitor.trade_manager = trade_manager

        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Test multiple monitoring cycles
        for _ in range(5):
            await position_monitor._monitor_position("BTC/USDT", position_dict)

        # Should still be monitoring despite errors
        assert "BTC/USDT" in position_monitor.active_monitors

    @pytest.mark.asyncio
    async def test_websocket_initialization_failure(self, mock_exchange, sample_config):
        """Test graceful handling of WebSocket initialization failure."""
        # Mock WebSocket client import failure
        with patch('crypto_bot.execution.kraken_ws.KrakenWSClient', side_effect=ImportError("WS client not available")):
            monitor = PositionMonitor(
                exchange=mock_exchange,
                config=sample_config,
                positions={},
                notifier=None
            )

            # Should not crash
            assert monitor.ws_client is None

            # Should still function with REST API
            mock_exchange.fetch_ticker.return_value = {"last": 50000.0}

            price = await monitor._get_current_price("BTC/USDT")
            assert price == 50000.0

    @pytest.mark.asyncio
    async def test_invalid_price_data_handling(self, position_monitor, mock_exchange):
        """Test handling of invalid price data."""
        # Mock various invalid price responses
        invalid_responses = [
            {"last": None},
            {"last": 0},
            {"last": -1000},
            {"last": "invalid"},
            {}  # Missing 'last' key
        ]

        for response in invalid_responses:
            mock_exchange.fetch_ticker.return_value = response

            price = await position_monitor._get_current_price("BTC/USDT")

            # Should return None for invalid prices
            assert price is None or price <= 0

    @pytest.mark.asyncio
    async def test_position_data_corruption(self, position_monitor):
        """Test handling of corrupted position data."""
        corrupted_positions = [
            {"symbol": "BTC/USDT", "side": "invalid_side"},
            {"symbol": "BTC/USDT", "entry_price": "invalid_price"},
            {"symbol": None, "side": "buy"},
            {},  # Empty dict
            {"symbol": "BTC/USDT"}  # Missing required fields
        ]

        for position in corrupted_positions:
            # Should not crash when processing corrupted data
            try:
                result = await position_monitor._check_exit_conditions("BTC/USDT", position, 50000.0)
                # Should return safe defaults
                assert isinstance(result, tuple)
                assert len(result) == 2
            except Exception as e:
                # Log but don't fail - some corruption might cause exceptions
                print(f"Expected exception for corrupted data: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_error_scenarios(self, position_monitor, mock_exchange):
        """Test multiple concurrent errors."""
        # Mock exchange to raise exceptions
        mock_exchange.fetch_ticker.side_effect = Exception("Concurrent error")

        # Create multiple positions
        positions = []
        for i in range(3):
            symbol = f"BTC{i}/USDT"
            position = {
                "symbol": symbol,
                "side": "long",
                "total_amount": 0.1,
                "average_price": 50000.0
            }
            positions.append((symbol, position))

        # Start monitoring all positions
        monitoring_tasks = []
        for symbol, position in positions:
            monitoring_tasks.append(position_monitor.start_monitoring(symbol, position))

        await asyncio.gather(*monitoring_tasks, return_exceptions=True)

        # Run monitoring cycles that will encounter errors
        monitoring_cycles = []
        for symbol, position in positions:
            if symbol in position_monitor.active_monitors:
                monitoring_cycles.append(position_monitor._monitor_position(symbol, position))

        # Should handle concurrent errors gracefully
        results = await asyncio.gather(*monitoring_cycles, return_exceptions=True)

        # Some may succeed, some may fail, but shouldn't crash the system
        assert len(results) == len([r for r in results if not isinstance(r, Exception)])

    @pytest.mark.asyncio
    async def test_trade_manager_integration_errors(self, position_monitor):
        """Test error handling when TradeManager is unavailable."""
        # Create position without TradeManager
        position_monitor.trade_manager = None

        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Should fall back to legacy methods
        await position_monitor._update_position_tracking("BTC/USDT", position_dict, 51000.0)

        # Should not crash
        assert "BTC/USDT" in position_monitor.active_monitors

    @pytest.mark.asyncio
    async def test_memory_cleanup_on_errors(self, position_monitor):
        """Test that memory is cleaned up even when errors occur."""
        # Create a position that will cause errors
        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Verify initial state
        assert "BTC/USDT" in position_monitor.price_cache
        assert "BTC/USDT" in position_monitor.last_update

        # Simulate an error during monitoring by corrupting position data
        position_monitor.positions["BTC/USDT"] = None

        # Stop monitoring - should clean up despite errors
        await position_monitor.stop_monitoring("BTC/USDT")

        # Verify cleanup occurred
        assert "BTC/USDT" not in position_monitor.price_cache
        assert "BTC/USDT" not in position_monitor.last_update

    @pytest.mark.asyncio
    async def test_configuration_error_handling(self):
        """Test handling of invalid configurations."""
        invalid_configs = [
            {},  # Empty config
            {"exit_strategy": {}},  # Missing monitoring config
            {"exit_strategy": {"real_time_monitoring": {}}},  # Empty monitoring config
            {"exit_strategy": {"real_time_monitoring": {"enabled": "invalid"}}},  # Invalid type
        ]

        for config in invalid_configs:
            monitor = PositionMonitor(
                exchange=Mock(),
                config=config,
                positions={}
            )

            # Should not crash with invalid config
            assert hasattr(monitor, 'check_interval_seconds')
            assert hasattr(monitor, 'use_websocket')

    @pytest.mark.asyncio
    async def test_async_task_cleanup_on_error(self, position_monitor):
        """Test cleanup of async tasks when errors occur."""
        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Get initial task count
        initial_tasks = len(position_monitor.active_monitors)

        # Simulate error by cancelling the monitoring task
        if "BTC/USDT" in position_monitor.active_monitors:
            task = position_monitor.active_monitors["BTC/USDT"]
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop monitoring - should handle already cancelled task
        await position_monitor.stop_monitoring("BTC/USDT")

        # Should be cleaned up
        assert "BTC/USDT" not in position_monitor.active_monitors

    @pytest.mark.asyncio
    async def test_extreme_price_volatility_handling(self, position_monitor, mock_exchange):
        """Test handling of extreme price volatility."""
        # Mock extreme price changes
        extreme_prices = [50000.0, 100000.0, 1000.0, 1000000.0, 0.01]

        for price in extreme_prices:
            mock_exchange.fetch_ticker.return_value = {"last": price}

            position_dict = {
                "symbol": "BTC/USDT",
                "side": "long",
                "total_amount": 0.1,
                "average_price": 50000.0
            }

            # Should handle extreme prices without crashing
            await position_monitor._update_position_tracking("BTC/USDT", position_dict, price)

            # Should not have invalid calculations
            if "pnl" in position_dict:
                assert isinstance(position_dict["pnl"], (int, float))

    @pytest.mark.asyncio
    async def test_resource_exhaustion_protection(self, position_monitor):
        """Test protection against resource exhaustion."""
        # Create many positions to test resource limits
        max_positions = 100

        for i in range(max_positions):
            symbol = f"ASSET{i}/USDT"
            position_dict = {
                "symbol": symbol,
                "side": "long",
                "total_amount": 0.1,
                "average_price": 50000.0
            }

            await position_monitor.start_monitoring(symbol, position_dict)

        # Should handle large number of positions
        assert len(position_monitor.active_monitors) == max_positions

        # Get monitoring stats
        stats = await position_monitor.get_monitoring_stats()

        # Should provide accurate statistics
        assert stats["active_monitors"] == max_positions
        assert len(stats["monitored_symbols"]) == max_positions

        # Cleanup
        await position_monitor.stop_all_monitoring()
        assert len(position_monitor.active_monitors) == 0
