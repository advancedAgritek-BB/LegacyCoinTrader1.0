"""
Performance tests for the PositionMonitor.

These tests measure latency, throughput, and resource usage of the
position monitoring system under various loads.
"""

import pytest
import asyncio
import time
import statistics
from unittest.mock import Mock, AsyncMock
from decimal import Decimal
from datetime import datetime

from crypto_bot.position_monitor import PositionMonitor
from crypto_bot.utils.trade_manager import TradeManager, Trade


@pytest.fixture
def mock_exchange():
    """Create a mock exchange with configurable latency."""
    exchange = Mock()
    exchange.fetch_ticker = AsyncMock()
    exchange.watch_ticker = AsyncMock()
    return exchange


@pytest.fixture
def fast_config():
    """Create a configuration optimized for performance testing."""
    return {
        "exit_strategy": {
            "real_time_monitoring": {
                "enabled": True,
                "check_interval_seconds": 0.01,  # Very fast for performance testing
                "max_monitor_age_seconds": 60.0,
                "price_update_threshold": 0.001,
                "use_websocket_when_available": False,  # Disable for consistent testing
                "fallback_to_rest": True,
                "max_execution_latency_ms": 1000
            },
            "trailing_stop_pct": 0.02,
            "min_gain_to_trail": 0.01,
            "take_profit_pct": 0.05
        }
    }


@pytest.fixture
def trade_manager():
    """Create a trade manager instance."""
    return TradeManager()


@pytest.fixture
def position_monitor(mock_exchange, fast_config, trade_manager):
    """Create a position monitor instance for performance testing."""
    return PositionMonitor(
        exchange=mock_exchange,
        config=fast_config,
        positions={},
        notifier=None,
        trade_manager=trade_manager
    )


class TestPositionMonitorPerformance:
    """Test performance characteristics of PositionMonitor."""

    @pytest.mark.asyncio
    async def test_price_update_latency(self, position_monitor, mock_exchange):
        """Test the latency of price updates."""
        latencies = []

        # Mock fast price responses
        mock_exchange.fetch_ticker.return_value = {"last": 50000.0}

        # Create and start monitoring a position
        trade = Trade(
            id="perf_test_1",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="perf_test"
        )

        position_monitor.trade_manager.record_trade(trade)

        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Measure latency over multiple price updates
        for i in range(10):
            start_time = time.time()

            # Trigger a price monitoring cycle
            await position_monitor._monitor_position("BTC/USDT", position_dict)

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        # Performance assertions
        assert avg_latency < 100, f"Average latency too high: {avg_latency:.2f}ms"
        assert max_latency < 500, f"Max latency too high: {max_latency:.2f}ms"

        print(f"Avg latency: {avg_latency:.2f}ms, Max latency: {max_latency:.2f}ms, Min latency: {min_latency:.2f}ms")

    @pytest.mark.asyncio
    async def test_concurrent_position_monitoring(self, position_monitor, mock_exchange):
        """Test performance with multiple concurrent positions."""
        num_positions = 10
        symbols = [f"BTC{i}/USDT" for i in range(num_positions)]

        # Mock price responses
        mock_exchange.fetch_ticker.return_value = {"last": 50000.0}

        # Create multiple positions
        start_time = time.time()
        for symbol in symbols:
            trade = Trade(
                id=f"perf_test_{symbol}",
                symbol=symbol,
                side="buy",
                amount=Decimal("0.1"),
                price=Decimal("50000.0"),
                timestamp=datetime.utcnow(),
                strategy="perf_test"
            )

            position_monitor.trade_manager.record_trade(trade)

            position_dict = {
                "symbol": symbol,
                "side": "long",
                "total_amount": 0.1,
                "average_price": 50000.0
            }

            await position_monitor.start_monitoring(symbol, position_dict)

        setup_time = time.time() - start_time

        # Verify all positions are monitored
        assert len(position_monitor.active_monitors) == num_positions

        # Performance assertion for setup time
        assert setup_time < 2.0, f"Setup time too slow: {setup_time:.2f}s"

        print(f"Setup time: {setup_time:.2f}s")

    @pytest.mark.asyncio
    async def test_price_update_throughput(self, position_monitor, mock_exchange):
        """Test throughput of price updates."""
        updates_count = 0
        test_duration = 2.0  # seconds

        # Mock price responses
        mock_exchange.fetch_ticker.return_value = {"last": 50000.0}

        # Create a position
        trade = Trade(
            id="throughput_test",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="throughput_test"
        )

        position_monitor.trade_manager.record_trade(trade)

        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Count price updates over time period
        start_time = time.time()
        end_time = start_time + test_duration

        while time.time() < end_time:
            await position_monitor._monitor_position("BTC/USDT", position_dict)
            updates_count += 1

        throughput = updates_count / test_duration

        # Performance assertion
        assert throughput > 50, f"Throughput too low: {throughput:.1f} updates/sec"

        print(f"Throughput: {throughput:.1f} updates/sec")
    @pytest.mark.asyncio
    async def test_memory_usage_with_many_positions(self, position_monitor, mock_exchange):
        """Test memory usage with many positions."""
        num_positions = 50
        symbols = [f"ASSET{i}/USDT" for i in range(num_positions)]

        # Mock price responses
        mock_exchange.fetch_ticker.return_value = {"last": 100.0}

        initial_cache_size = len(position_monitor.price_cache)

        # Create many positions
        for symbol in symbols:
            trade = Trade(
                id=f"memory_test_{symbol}",
                symbol=symbol,
                side="buy",
                amount=Decimal("1.0"),
                price=Decimal("100.0"),
                timestamp=datetime.utcnow(),
                strategy="memory_test"
            )

            position_monitor.trade_manager.record_trade(trade)

            position_dict = {
                "symbol": symbol,
                "side": "long",
                "total_amount": 1.0,
                "average_price": 100.0
            }

            await position_monitor.start_monitoring(symbol, position_dict)

        # Check cache sizes
        final_cache_size = len(position_monitor.price_cache)
        expected_cache_size = initial_cache_size + num_positions

        assert final_cache_size == expected_cache_size, f"Cache size mismatch: {final_cache_size} != {expected_cache_size}"

        # Verify monitoring stats
        stats = await position_monitor.get_monitoring_stats()
        assert stats["positions_monitored"] == num_positions
        assert stats["active_monitors"] == num_positions

        print(f"Successfully monitoring {num_positions} positions")

    @pytest.mark.asyncio
    async def test_websocket_vs_rest_performance(self, position_monitor):
        """Test performance comparison between WebSocket and REST."""
        # Mock REST API
        with pytest.mock.patch.object(position_monitor.exchange, 'fetch_ticker') as mock_rest:
            mock_rest.return_value = {"last": 50000.0}

            # Create a position
            trade = Trade(
                id="ws_rest_test",
                symbol="BTC/USDT",
                side="buy",
                amount=Decimal("0.1"),
                price=Decimal("50000.0"),
                timestamp=datetime.utcnow(),
                strategy="ws_rest_test"
            )

            position_monitor.trade_manager.record_trade(trade)

            position_dict = {
                "symbol": "BTC/USDT",
                "side": "long",
                "total_amount": 0.1,
                "average_price": 50000.0
            }

            await position_monitor.start_monitoring("BTC/USDT", position_dict)

            # Test REST performance
            rest_latencies = []
            for _ in range(5):
                start_time = time.time()
                price = await position_monitor._get_current_price("BTC/USDT")
                latency = (time.time() - start_time) * 1000
                rest_latencies.append(latency)
                assert price == 50000.0

            avg_rest_latency = statistics.mean(rest_latencies)

            # Performance assertion
            assert avg_rest_latency < 200, f"REST latency too high: {avg_rest_latency:.2f}ms"

            print(f"REST latency: {avg_rest_latency:.2f}ms")
    @pytest.mark.asyncio
    async def test_position_update_frequency(self, position_monitor, mock_exchange):
        """Test how frequently positions are updated."""
        update_counts = {"price_updates": 0, "position_updates": 0}

        # Mock price responses with slight variations
        prices = [50000.0, 50001.0, 50002.0, 50001.5, 50003.0]
        mock_exchange.fetch_ticker.side_effect = [
            {"last": price} for price in prices
        ]

        # Create a position
        trade = Trade(
            id="frequency_test",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="frequency_test"
        )

        position_monitor.trade_manager.record_trade(trade)

        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Override monitoring stats to count updates
        original_price_updates = position_monitor.monitoring_stats["price_updates"]

        # Run monitoring cycles
        for _ in range(len(prices)):
            await position_monitor._monitor_position("BTC/USDT", position_dict)
            update_counts["position_updates"] += 1

        final_price_updates = position_monitor.monitoring_stats["price_updates"]
        price_update_count = final_price_updates - original_price_updates

        # Should have updated prices multiple times
        assert price_update_count > 0, "No price updates detected"

        print(f"Price updates: {price_update_count}, Position updates: {update_counts['position_updates']}")

    @pytest.mark.asyncio
    async def test_monitoring_cleanup_performance(self, position_monitor):
        """Test performance of monitoring cleanup."""
        num_positions = 20
        symbols = [f"CLEANUP{i}/USDT" for i in range(num_positions)]

        # Create and start monitoring many positions
        for symbol in symbols:
            position_dict = {
                "symbol": symbol,
                "side": "long",
                "total_amount": 0.1,
                "average_price": 50000.0
            }
            await position_monitor.start_monitoring(symbol, position_dict)

        # Verify all are monitored
        assert len(position_monitor.active_monitors) == num_positions

        # Measure cleanup time
        start_time = time.time()
        await position_monitor.stop_all_monitoring()
        cleanup_time = time.time() - start_time

        # Verify cleanup
        assert len(position_monitor.active_monitors) == 0
        assert len(position_monitor.price_cache) == 0

        # Performance assertion
        assert cleanup_time < 1.0, f"Cleanup too slow: {cleanup_time:.2f}s"

        print(f"Cleanup time: {cleanup_time:.2f}s")
    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, position_monitor, mock_exchange):
        """Test performance of error recovery."""
        error_count = 0
        recovery_count = 0

        # Mock alternating success/failure
        responses = [
            Exception("Network error"),
            {"last": 50000.0},
            Exception("Timeout"),
            {"last": 50001.0},
            {"last": 50002.0}
        ]
        mock_exchange.fetch_ticker.side_effect = responses

        # Create a position
        trade = Trade(
            id="error_recovery_test",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="error_recovery_test"
        )

        position_monitor.trade_manager.record_trade(trade)

        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Test error recovery
        start_time = time.time()
        for i in range(len(responses)):
            try:
                await position_monitor._monitor_position("BTC/USDT", position_dict)
                recovery_count += 1
            except Exception as e:
                error_count += 1
                print(f"Expected error {i}: {e}")

        total_time = time.time() - start_time

        # Should have recovered from errors
        assert recovery_count > 0, "No successful recoveries"
        assert error_count > 0, "No errors occurred (test setup issue)"

        # Performance assertion
        assert total_time < 5.0, f"Error recovery too slow: {total_time:.2f}s"

        print(f"Errors: {error_count}, Recoveries: {recovery_count}, Total time: {total_time:.2f}s")
