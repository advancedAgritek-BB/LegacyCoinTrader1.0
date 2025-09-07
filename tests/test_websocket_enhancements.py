"""
Comprehensive tests for WebSocket enhancements including health checks,
message validation, connection pooling, and error recovery.
"""

import sys
import asyncio
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta, timezone
import types

# Mock scipy for compatibility
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault("scipy.stats", types.SimpleNamespace(pearsonr=lambda *a, **k: 0))

from crypto_bot.solana.pool_ws_monitor import EnhancedPoolMonitor
from crypto_bot.execution.kraken_ws import KrakenWSClient
from crypto_bot.utils.websocket_pool import WebSocketPool, LoadBalancedWebSocketClient


class TestEnhancedPoolMonitor:
    """Test the enhanced Solana pool WebSocket monitor."""

    @pytest.fixture
    def monitor(self):
        """Create a test monitor instance."""
        return EnhancedPoolMonitor("test_key", "test_pool")

    def test_initialization(self, monitor):
        """Test monitor initialization with default config."""
        assert monitor.api_key == "test_key"
        assert monitor.pool_program == "test_pool"
        assert monitor.connection_attempts == 0
        assert monitor.is_connected is False
        assert monitor.total_messages_received == 0
        assert monitor.total_errors == 0

    def test_config_override(self):
        """Test monitor with custom configuration."""
        custom_config = {
            "reconnect": {
                "base_delay": 2.0,
                "max_delay": 120.0,
                "backoff_factor": 3.0,
                "max_attempts": 5,
            },
            "health_check": {
                "enabled": False,
                "interval": 60,
            }
        }
        monitor = EnhancedPoolMonitor("test_key", "test_pool", custom_config)
        assert monitor.reconnect_delay == 2.0
        assert monitor.max_reconnect_delay == 120.0
        assert monitor.backoff_factor == 3.0
        assert monitor.max_attempts == 5

    def test_calculate_reconnect_delay(self, monitor):
        """Test exponential backoff delay calculation."""
        monitor.connection_attempts = 0
        assert monitor._calculate_reconnect_delay() == 1.0

        monitor.connection_attempts = 1
        assert monitor._calculate_reconnect_delay() == 2.0

        monitor.connection_attempts = 2
        assert monitor._calculate_reconnect_delay() == 4.0

        # Test max delay limit
        monitor.connection_attempts = 10
        delay = monitor._calculate_reconnect_delay()
        assert delay <= monitor.max_reconnect_delay

    def test_should_attempt_reconnect(self, monitor):
        """Test reconnection attempt logic."""
        monitor.max_attempts = 5
        monitor.connection_attempts = 3
        assert monitor._should_attempt_reconnect() is True

        monitor.connection_attempts = 5
        assert monitor._should_attempt_reconnect() is False

        # Test unlimited attempts
        monitor.max_attempts = 0
        monitor.connection_attempts = 100
        assert monitor._should_attempt_reconnect() is True

    def test_validate_message(self, monitor):
        """Test message validation."""
        # Valid JSON
        valid_msg = '{"test": "data"}'
        assert monitor._validate_message(valid_msg) is True

        # Invalid JSON
        invalid_msg = '{"test": invalid}'
        assert monitor._validate_message(invalid_msg) is False

        # Oversized message
        large_msg = "x" * (monitor.config["message_validation"]["max_message_size"] + 1)
        assert monitor._validate_message(large_msg) is False

    @pytest.mark.asyncio
    async def test_handle_connection_success(self, monitor):
        """Test successful connection handling."""
        await monitor._handle_connection_success()
        assert monitor.connection_attempts == 0
        assert monitor.is_connected is True
        assert monitor.last_connection_time is not None

    @pytest.mark.asyncio
    async def test_handle_connection_failure(self, monitor):
        """Test connection failure handling."""
        error = Exception("Test error")
        monitor.connection_attempts = 2

        with patch('asyncio.sleep') as mock_sleep:
            with patch.object(monitor, '_should_attempt_reconnect', return_value=True):
                with pytest.raises(Exception):
                    await monitor._handle_connection_failure(error)

        assert monitor.is_connected is False
        assert monitor.total_errors == 1
        assert monitor.connection_attempts == 3

    def test_log_connection_stats(self, monitor):
        """Test connection statistics logging."""
        monitor.total_messages_received = 100
        monitor.total_errors = 5
        monitor.start_time = datetime.now() - timedelta(hours=2)

        # Should not raise any exceptions
        monitor._log_connection_stats()


class TestKrakenWebSocketEnhancements:
    """Test Kraken WebSocket client enhancements."""

    @pytest.fixture
    def client(self):
        """Create a test Kraken WebSocket client."""
        return KrakenWSClient()

    def test_initialization(self, client):
        """Test client initialization with health monitoring."""
        assert client.health_check_task is None
        assert "public" in client.connection_health
        assert "private" in client.connection_health
        assert client.message_stats["total_received"] == 0
        assert client.message_stats["total_sent"] == 0
        assert client.message_stats["errors"] == 0

    def test_handle_connection_error(self, client):
        """Test enhanced error handling."""
        error = Exception("timeout error")

        # Test timeout error handling
        client._handle_connection_error("public", error)
        assert client.connection_health["public"]["errors"] == 1
        assert client.connection_health["public"]["is_alive"] is False
        assert client.message_stats["errors"] == 1

    def test_validate_message(self, client):
        """Test message validation."""
        # Valid message
        valid_msg = '{"channel": "ticker", "data": []}'
        assert client._validate_message(valid_msg) is True

        # Invalid JSON
        invalid_msg = '{"channel": "ticker", "data": [}'
        assert client._validate_message(invalid_msg) is False

        # Oversized message
        large_msg = "x" * (1024 * 1024 + 1)  # Over 1MB
        assert client._validate_message(large_msg, max_size=1024*1024) is False

    def test_get_connection_health(self, client):
        """Test connection health reporting."""
        health = client.get_connection_health()

        assert "connections" in health
        assert "message_stats" in health
        assert "public" in health["connections"]
        assert "private" in health["connections"]
        assert "total_received" in health["message_stats"]

    @pytest.mark.asyncio
    async def test_health_check_loop(self, client):
        """Test health check loop functionality."""
        # Mock the sleep to speed up test
        with patch('asyncio.sleep') as mock_sleep:
            # Create a task that will be cancelled quickly
            task = asyncio.create_task(client.health_check_loop(interval=1))

            # Let it run briefly
            await asyncio.sleep(0.1)

            # Cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify sleep was called
            mock_sleep.assert_called()

    def test_start_stop_health_monitoring(self, client):
        """Test health monitoring lifecycle."""
        # Test starting
        client.start_health_monitoring(interval=1)
        assert client.health_check_task is not None
        assert not client.health_check_task.done()

        # Test stopping
        client.stop_health_monitoring()
        assert client.health_check_task.cancelled()

    def test_handle_message_with_validation(self, client):
        """Test message handling with validation."""
        # Test with valid message
        valid_message = '{"channel": "heartbeat"}'
        mock_ws = Mock()

        # Should not raise exceptions
        client._handle_message(mock_ws, valid_message)

        # Check stats were updated
        assert client.message_stats["total_received"] == 1

    def test_handle_message_with_invalid_json(self, client):
        """Test message handling with invalid JSON."""
        invalid_message = '{"channel": "heartbeat"'  # Missing closing brace
        mock_ws = Mock()

        # Should not raise exceptions
        client._handle_message(mock_ws, invalid_message)

        # Check error stats were updated
        assert client.message_stats["errors"] == 1


class TestWebSocketPool:
    """Test WebSocket connection pool functionality."""

    @pytest.fixture
    def pool(self):
        """Create a test WebSocket pool."""
        return WebSocketPool(max_connections=5, max_connections_per_host=2)

    def test_initialization(self, pool):
        """Test pool initialization."""
        assert pool.max_connections == 5
        assert pool.max_connections_per_host == 2
        assert len(pool.active_connections) == 0
        assert pool.connection_stats["total_created"] == 0
        assert pool.connection_stats["active_count"] == 0

    def test_get_pool_stats(self, pool):
        """Test pool statistics retrieval."""
        stats = pool.get_pool_stats()

        assert "total_created" in stats
        assert "total_closed" in stats
        assert "active_count" in stats
        assert "connections_by_host" in stats

    def test_health_check(self, pool):
        """Test pool health check."""
        health = pool.health_check()

        assert "healthy" in health
        assert "unhealthy" in health
        assert "details" in health
        assert health["healthy"] == 0
        assert health["unhealthy"] == 0

    def test_close_all_connections(self, pool):
        """Test closing all connections."""
        # Add some mock connections
        mock_ws = Mock()
        pool.active_connections["ws://test.com"] = [mock_ws]

        pool.close_all_connections()

        # Verify connection was closed
        mock_ws.close.assert_called_once()
        assert len(pool.active_connections) == 0


class TestLoadBalancedWebSocketClient:
    """Test load balanced WebSocket client."""

    @pytest.fixture
    def endpoints(self):
        """Test endpoints."""
        return ["ws://endpoint1.com", "ws://endpoint2.com", "ws://endpoint3.com"]

    @pytest.fixture
    def lb_client(self, endpoints):
        """Create a load balanced client."""
        return LoadBalancedWebSocketClient(endpoints)

    def test_initialization(self, lb_client, endpoints):
        """Test load balanced client initialization."""
        assert lb_client.endpoints == endpoints
        assert lb_client.current_endpoint == 0
        assert len(lb_client.connection_failures) == len(endpoints)

    def test_get_next_endpoint(self, lb_client, endpoints):
        """Test round-robin endpoint selection."""
        assert lb_client.get_next_endpoint() == endpoints[0]
        assert lb_client.get_next_endpoint() == endpoints[1]
        assert lb_client.get_next_endpoint() == endpoints[2]
        assert lb_client.get_next_endpoint() == endpoints[0]  # Wrap around

    def test_get_healthiest_endpoint(self, lb_client, endpoints):
        """Test healthiest endpoint selection."""
        # Set failure counts
        lb_client.connection_failures[endpoints[0]] = 5
        lb_client.connection_failures[endpoints[1]] = 2
        lb_client.connection_failures[endpoints[2]] = 10

        # Should return endpoint with fewest failures
        assert lb_client.get_healthiest_endpoint() == endpoints[1]

    def test_get_load_balance_stats(self, lb_client, endpoints):
        """Test load balance statistics."""
        stats = lb_client.get_load_balance_stats()

        assert "endpoints" in stats
        assert "current_endpoint" in stats
        assert "failure_counts" in stats
        assert "pool_stats" in stats
        assert stats["endpoints"] == endpoints


class TestIntegration:
    """Integration tests for WebSocket enhancements."""

    @pytest.mark.asyncio
    async def test_monitor_with_custom_config(self):
        """Test monitor with custom configuration."""
        config = {
            "reconnect": {
                "base_delay": 0.1,  # Fast for testing
                "max_delay": 1.0,
                "backoff_factor": 2.0,
                "max_attempts": 2,
            },
            "health_check": {
                "enabled": False,  # Disable for faster test
            }
        }

        monitor = EnhancedPoolMonitor("test_key", "test_pool", config)

        # Verify config was applied
        assert monitor.reconnect_delay == 0.1
        assert monitor.max_reconnect_delay == 1.0
        assert monitor.max_attempts == 2

    def test_kraken_client_with_health_monitoring(self):
        """Test Kraken client with health monitoring enabled."""
        client = KrakenWSClient()

        # Start health monitoring
        client.start_health_monitoring(interval=1)

        # Verify task was created
        assert client.health_check_task is not None

        # Stop monitoring
        client.stop_health_monitoring()

        # Verify task was cancelled
        assert client.health_check_task.cancelled()

    def test_pool_with_load_balancer(self):
        """Test connection pool with load balancer."""
        endpoints = ["ws://test1.com", "ws://test2.com"]
        pool = WebSocketPool(max_connections=3)
        lb_client = LoadBalancedWebSocketClient(endpoints, pool)

        # Verify integration
        assert lb_client.pool == pool
        assert lb_client.endpoints == endpoints

        # Test stats integration
        stats = lb_client.get_load_balance_stats()
        assert "pool_stats" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
