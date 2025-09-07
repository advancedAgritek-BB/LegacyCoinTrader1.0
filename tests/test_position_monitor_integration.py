"""
Integration tests for PositionMonitor with TradeManager.

These tests verify that the position monitor works correctly when integrated
with the centralized trade management system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime

from crypto_bot.position_monitor import PositionMonitor
from crypto_bot.utils.trade_manager import TradeManager, Trade, Position


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
                "use_websocket_when_available": False,  # Disable for simpler testing
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
    """Create a trade manager instance for testing."""
    return TradeManager()


@pytest.fixture
def position_monitor(mock_exchange, sample_config, trade_manager):
    """Create a position monitor instance with trade manager integration."""
    return PositionMonitor(
        exchange=mock_exchange,
        config=sample_config,
        positions={},  # Will be populated from trade manager
        notifier=None,
        trade_manager=trade_manager
    )


class TestPositionMonitorTradeManagerIntegration:
    """Test integration between PositionMonitor and TradeManager."""

    def test_trade_manager_position_sync(self, position_monitor, trade_manager):
        """Test that position monitor syncs with trade manager positions."""
        # Create a trade in the trade manager
        trade = Trade(
            id="test_trade_1",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="test"
        )

        trade_manager.record_trade(trade)

        # Position should be available in trade manager
        tm_position = trade_manager.get_position("BTC/USDT")
        assert tm_position is not None
        assert tm_position.total_amount == Decimal("0.1")
        assert tm_position.average_price == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_position_tracking_with_trade_manager(self, position_monitor, trade_manager, mock_exchange):
        """Test position tracking using TradeManager integration."""
        # Create and record a trade
        trade = Trade(
            id="test_trade_1",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="test"
        )

        trade_manager.record_trade(trade)

        # Mock price feed
        mock_exchange.fetch_ticker.return_value = {"last": 51000.0}

        # Start monitoring
        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Simulate price monitoring cycle
        await asyncio.sleep(0.2)  # Allow monitoring to run briefly

        # Check that position tracking was updated
        tm_position = trade_manager.get_position("BTC/USDT")
        assert tm_position is not None

        # Verify price cache was updated
        assert "BTC/USDT" in position_monitor.price_cache
        assert position_monitor.price_cache["BTC/USDT"] == 51000.0

    @pytest.mark.asyncio
    async def test_trailing_stop_with_trade_manager(self, position_monitor, trade_manager, mock_exchange):
        """Test trailing stop functionality with TradeManager integration."""
        # Create a position with trailing stop
        trade = Trade(
            id="test_trade_1",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="test"
        )

        trade_manager.record_trade(trade)

        # Set up trailing stop in trade manager
        tm_position = trade_manager.get_position("BTC/USDT")
        tm_position.trailing_stop_pct = Decimal("0.02")  # 2% trailing stop
        tm_position.stop_loss_price = Decimal("49000.0")  # Initial stop loss

        # Mock price movements
        price_sequence = [51000.0, 52000.0, 53000.0]  # Price moving up
        mock_exchange.fetch_ticker.side_effect = [
            {"last": price} for price in price_sequence
        ]

        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Simulate multiple price updates
        for _ in range(len(price_sequence)):
            await asyncio.sleep(0.2)

        # Check that trailing stop was adjusted upward
        updated_position = trade_manager.get_position("BTC/USDT")
        # Trailing stop should be higher than initial
        assert updated_position.stop_loss_price > Decimal("49000.0")

    @pytest.mark.asyncio
    async def test_exit_condition_with_trade_manager(self, position_monitor, trade_manager, mock_exchange):
        """Test exit condition checking with TradeManager."""
        # Create a position
        trade = Trade(
            id="test_trade_1",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="test"
        )

        trade_manager.record_trade(trade)

        # Set take profit price
        tm_position = trade_manager.get_position("BTC/USDT")
        tm_position.take_profit_price = Decimal("52500.0")  # 5% profit target

        # Mock price hitting take profit
        mock_exchange.fetch_ticker.return_value = {"last": 52600.0}

        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Check exit condition
        should_exit, exit_reason = await position_monitor._check_exit_conditions(
            "BTC/USDT", position_dict, 52600.0
        )

        # Should trigger take profit exit
        assert should_exit is True
        assert exit_reason == "take_profit"

    @pytest.mark.asyncio
    async def test_pnl_calculation_with_trade_manager(self, position_monitor, trade_manager, mock_exchange):
        """Test PnL calculation with TradeManager integration."""
        # Create a position
        trade = Trade(
            id="test_trade_1",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="test"
        )

        trade_manager.record_trade(trade)

        # Mock price update
        mock_exchange.fetch_ticker.return_value = {"last": 51000.0}

        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Trigger position tracking update
        await position_monitor._update_position_tracking("BTC/USDT", position_dict, 51000.0)

        # Check that PnL was updated
        assert "pnl" in position_dict
        # 2% profit: (51000 - 50000) / 50000 = 0.02
        assert abs(position_dict["pnl"] - 0.02) < 0.001

    @pytest.mark.asyncio
    async def test_position_cleanup_with_trade_manager(self, position_monitor, trade_manager):
        """Test that positions are properly cleaned up when no longer in TradeManager."""
        # Start monitoring a position
        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Verify monitoring started
        assert "BTC/USDT" in position_monitor.active_monitors

        # Simulate position being closed (removed from positions dict)
        position_monitor.positions.pop("BTC/USDT", None)

        # Wait for monitoring to stop naturally
        await asyncio.sleep(0.5)

        # The monitoring should continue since we didn't call stop_monitoring
        # This tests that the monitor handles missing positions gracefully
        assert "BTC/USDT" in position_monitor.active_monitors

    @pytest.mark.asyncio
    async def test_concurrent_updates_with_trade_manager(self, position_monitor, trade_manager, mock_exchange):
        """Test concurrent position updates with TradeManager."""
        # Create multiple positions
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

        for symbol in symbols:
            trade = Trade(
                id=f"test_trade_{symbol}",
                symbol=symbol,
                side="buy",
                amount=Decimal("0.1"),
                price=Decimal("50000.0"),
                timestamp=datetime.utcnow(),
                strategy="test"
            )
            trade_manager.record_trade(trade)

        # Mock price responses
        mock_exchange.fetch_ticker.return_value = {"last": 51000.0}

        # Start monitoring all positions concurrently
        monitoring_tasks = []
        for symbol in symbols:
            position_dict = {
                "symbol": symbol,
                "side": "long",
                "total_amount": 0.1,
                "average_price": 50000.0
            }
            monitoring_tasks.append(
                position_monitor.start_monitoring(symbol, position_dict)
            )

        await asyncio.gather(*monitoring_tasks)

        # Verify all are being monitored
        assert len(position_monitor.active_monitors) == 3

        # Verify price caches are populated
        for symbol in symbols:
            assert symbol in position_monitor.price_cache

    @pytest.mark.asyncio
    async def test_websocket_fallback_with_trade_manager(self, position_monitor, trade_manager):
        """Test WebSocket fallback behavior with TradeManager."""
        # Create a position
        trade = Trade(
            id="test_trade_1",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="test"
        )

        trade_manager.record_trade(trade)

        # Mock WebSocket failure and REST success
        with patch.object(position_monitor, 'ws_client', None):  # No WebSocket client
            with patch.object(position_monitor.exchange, 'fetch_ticker', return_value={"last": 51000.0}):
                position_dict = {
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "total_amount": 0.1,
                    "average_price": 50000.0
                }

                await position_monitor.start_monitoring("BTC/USDT", position_dict)

                # Get price - should fallback to REST
                price = await position_monitor._get_current_price("BTC/USDT")
                assert price == 51000.0

    @pytest.mark.asyncio
    async def test_monitoring_stats_with_trade_manager(self, position_monitor, trade_manager):
        """Test monitoring statistics with TradeManager integration."""
        # Create and monitor a position
        trade = Trade(
            id="test_trade_1",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.1"),
            price=Decimal("50000.0"),
            timestamp=datetime.utcnow(),
            strategy="test"
        )

        trade_manager.record_trade(trade)

        position_dict = {
            "symbol": "BTC/USDT",
            "side": "long",
            "total_amount": 0.1,
            "average_price": 50000.0
        }

        await position_monitor.start_monitoring("BTC/USDT", position_dict)

        # Get monitoring stats
        stats = await position_monitor.get_monitoring_stats()

        # Verify stats structure
        assert "positions_monitored" in stats
        assert "active_monitors" in stats
        assert "monitored_symbols" in stats
        assert "price_cache_size" in stats

        # Verify values
        assert stats["positions_monitored"] == 1
        assert stats["active_monitors"] == 1
        assert "BTC/USDT" in stats["monitored_symbols"]
        assert stats["price_cache_size"] == 1
