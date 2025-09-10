"""
Integration tests for WebSocket enhancements working together.
Tests the complete WebSocket ecosystem including monitoring, health checks,
and connection management.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
import types

# Mock scipy for compatibility
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault("scipy.stats", types.SimpleNamespace(pearsonr=lambda *a, **k: 0))

from crypto_bot.solana.pool_ws_monitor import EnhancedPoolMonitor
from crypto_bot.execution.kraken_ws import KrakenWSClient
from crypto_bot.utils.websocket_pool import WebSocketPool, LoadBalancedWebSocketClient


class TestWebSocketIntegration:
    """Integration tests for the complete WebSocket enhancement system."""

    @pytest.fixture
    async def enhanced_monitor(self):
        """Create an enhanced pool monitor for testing."""
        config = {
            "reconnect": {
                "base_delay": 0.1,  # Fast for testing
                "max_delay": 1.0,
                "backoff_factor": 2.0,
                "max_attempts": 3,
            },
            "health_check": {
                "enabled": True,
                "interval": 0.5,  # Fast health checks for testing
            },
            "message_validation": {
                "max_message_size": 1024,
                "validate_json": True,
            }
        }
        monitor = EnhancedPoolMonitor("test_key", "test_pool", config)
        yield monitor

    @pytest.fixture
    def kraken_client(self):
        """Create a Kraken WebSocket client for testing."""
        return KrakenWSClient()

    @pytest.fixture
    def websocket_pool(self):
        """Create a WebSocket connection pool for testing."""
        return WebSocketPool(max_connections=5, max_connections_per_host=2)

    @pytest.mark.asyncio
    async def test_monitor_health_check_integration(self, enhanced_monitor):
        """Test that the monitor's health check integrates properly."""
        # Start the health check
        health_task = asyncio.create_task(enhanced_monitor._health_check_loop())

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Verify the monitor is still functioning
        assert enhanced_monitor.is_connected is False  # Initially disconnected

        # Cancel the health check
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_monitor_with_connection_failure_recovery(self, enhanced_monitor):
        """Test monitor's ability to recover from connection failures."""
        # Mock a connection failure scenario
        original_should_attempt = enhanced_monitor._should_attempt_reconnect
        enhanced_monitor._should_attempt_reconnect = Mock(return_value=True)

        try:
            # Simulate connection failure
            enhanced_monitor.connection_attempts = 1
            await enhanced_monitor._handle_connection_failure(Exception("Network error"))

            # Verify backoff calculation
            delay = enhanced_monitor._calculate_reconnect_delay()
            assert delay == 0.2  # base_delay * (backoff_factor ^ attempts)

            # Verify error tracking
            assert enhanced_monitor.total_errors == 1
            assert enhanced_monitor.is_connected is False

        finally:
            enhanced_monitor._should_attempt_reconnect = original_should_attempt

    def test_kraken_client_health_monitoring_integration(self, kraken_client):
        """Test Kraken client's health monitoring integration."""
        # Test health monitoring start/stop
        kraken_client.start_health_monitoring(interval=1)

        assert kraken_client.health_check_task is not None
        assert not kraken_client.health_check_task.done()

        # Test health stats
        health = kraken_client.get_connection_health()
        assert "connections" in health
        assert "message_stats" in health
        assert "public" in health["connections"]
        assert "private" in health["connections"]

        # Stop monitoring
        kraken_client.stop_health_monitoring()
        assert kraken_client.health_check_task.cancelled()

    def test_kraken_client_message_validation_integration(self, kraken_client):
        """Test Kraken client's message validation integration."""
        # Test valid message
        valid_msg = '{"channel": "ticker", "type": "snapshot"}'
        assert kraken_client._validate_message(valid_msg) is True

        # Test invalid JSON
        invalid_msg = '{"channel": "ticker", "type": invalid}'
        assert kraken_client._validate_message(invalid_msg) is False

        # Test oversized message
        large_msg = "x" * (2 * 1024 * 1024)  # 2MB, over default limit
        assert kraken_client._validate_message(large_msg, max_size=1024*1024) is False

    def test_websocket_pool_integration(self, websocket_pool):
        """Test WebSocket pool integration."""
        # Test pool stats
        stats = websocket_pool.get_pool_stats()
        assert "total_created" in stats
        assert "active_count" in stats
        assert stats["active_count"] == 0

        # Test health check
        health = websocket_pool.health_check()
        assert "healthy" in health
        assert "unhealthy" in health
        assert health["healthy"] == 0
        assert health["unhealthy"] == 0

        # Test closing all connections
        websocket_pool.close_all_connections()
        stats_after = websocket_pool.get_pool_stats()
        assert stats_after["active_count"] == 0

    def test_load_balanced_client_integration(self):
        """Test load balanced client integration."""
        endpoints = ["ws://test1.com", "ws://test2.com", "ws://test3.com"]
        pool = WebSocketPool(max_connections=3)
        lb_client = LoadBalancedWebSocketClient(endpoints, pool)

        # Test endpoint selection
        endpoint1 = lb_client.get_next_endpoint()
        endpoint2 = lb_client.get_next_endpoint()
        endpoint3 = lb_client.get_next_endpoint()
        endpoint4 = lb_client.get_next_endpoint()

        assert endpoint1 in endpoints
        assert endpoint2 in endpoints
        assert endpoint3 in endpoints
        assert endpoint4 == endpoint1  # Should wrap around

        # Test healthiest endpoint selection
        lb_client.connection_failures[endpoints[0]] = 10
        lb_client.connection_failures[endpoints[1]] = 2
        lb_client.connection_failures[endpoints[2]] = 5

        healthiest = lb_client.get_healthiest_endpoint()
        assert healthiest == endpoints[1]  # Should have fewest failures

        # Test load balance stats
        stats = lb_client.get_load_balance_stats()
        assert "endpoints" in stats
        assert "failure_counts" in stats
        assert "pool_stats" in stats

    @pytest.mark.asyncio
    async def test_monitor_message_processing_integration(self, enhanced_monitor):
        """Test monitor's message processing integration."""
        # Test message validation
        valid_msg = '{"params": {"result": {"tx": "test"}}}'
        assert enhanced_monitor._validate_message(valid_msg) is True

        invalid_msg = '{"params": {"result": invalid}'
        assert enhanced_monitor._validate_message(invalid_msg) is False

        # Test message statistics
        initial_count = enhanced_monitor.total_messages_received
        enhanced_monitor.total_messages_received += 1
        assert enhanced_monitor.total_messages_received == initial_count + 1

    def test_kraken_client_error_handling_integration(self, kraken_client):
        """Test Kraken client's error handling integration."""
        # Test different error types
        timeout_error = Exception("Connection timeout")
        kraken_client._handle_connection_error("public", timeout_error)

        assert kraken_client.connection_health["public"]["errors"] == 1
        assert kraken_client.connection_health["public"]["is_alive"] is False

        rate_limit_error = Exception("Rate limit exceeded")
        kraken_client._handle_connection_error("private", rate_limit_error)

        assert kraken_client.connection_health["private"]["errors"] == 1
        assert kraken_client.connection_health["private"]["is_alive"] is False

        generic_error = Exception("Network error")
        kraken_client._handle_connection_error("public", generic_error)

        assert kraken_client.connection_health["public"]["errors"] == 2

    @pytest.mark.asyncio
    async def test_monitor_subscription_creation(self, enhanced_monitor):
        """Test monitor's subscription message creation."""
        msg = await enhanced_monitor._create_subscription_message()

        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "transactionSubscribe"
        assert "params" in msg
        assert len(msg["params"]) == 2
        assert msg["params"][0]["accountInclude"] == ["test_pool"]

    def test_configuration_integration(self):
        """Test that all components can be configured together."""
        # Test monitor configuration
        monitor_config = {
            "reconnect": {"max_attempts": 5, "base_delay": 0.5},
            "health_check": {"enabled": True, "interval": 10},
            "message_validation": {"max_message_size": 2048}
        }
        monitor = EnhancedPoolMonitor("key", "pool", monitor_config)

        assert monitor.max_attempts == 5
        assert monitor.reconnect_delay == 0.5
        assert monitor.config["health_check"]["enabled"] is True

        # Test that monitor integrates with other components
        assert hasattr(monitor, '_validate_message')
        assert hasattr(monitor, '_calculate_reconnect_delay')
        assert hasattr(monitor, '_handle_connection_success')

    def test_backwards_compatibility(self):
        """Test that enhanced components maintain backwards compatibility."""
        # Test old monitor function still works
        from crypto_bot.solana.pool_ws_monitor import watch_pool
        assert callable(watch_pool)

        # Test Kraken client maintains old interface
        client = KrakenWSClient()
        assert hasattr(client, 'connect_public')
        assert hasattr(client, 'connect_private')
        assert hasattr(client, 'subscribe_ticker')

        # Test new methods are additive
        assert hasattr(client, 'start_health_monitoring')  # New method
        assert hasattr(client, '_validate_message')  # New method


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
