"""
Simple performance tests for the PositionMonitor.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock
from decimal import Decimal
from datetime import datetime

from crypto_bot.position_monitor import PositionMonitor
from crypto_bot.utils.trade_manager import TradeManager, Trade


@pytest.fixture
def mock_exchange():
    """Create a mock exchange."""
    exchange = Mock()
    exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
    return exchange


@pytest.fixture
def fast_config():
    """Create a fast configuration for performance testing."""
    return {
        "exit_strategy": {
            "real_time_monitoring": {
                "enabled": True,
                "check_interval_seconds": 0.01,
                "max_monitor_age_seconds": 60.0,
                "price_update_threshold": 0.001,
                "use_websocket_when_available": False,
                "fallback_to_rest": True,
                "max_execution_latency_ms": 1000
            }
        }
    }


@pytest.mark.asyncio
async def test_basic_performance(mock_exchange, fast_config):
    """Test basic performance of position monitoring."""
    position_monitor = PositionMonitor(
        exchange=mock_exchange,
        config=fast_config,
        positions={},
        notifier=None
    )

    # Create a position
    trade = Trade(
        id="perf_test",
        symbol="BTC/USDT",
        side="buy",
        amount=Decimal("0.1"),
        price=Decimal("50000.0"),
        timestamp=datetime.utcnow(),
        strategy="perf_test"
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

    # Measure performance
    start_time = time.time()
    iterations = 10

    for _ in range(iterations):
        await position_monitor._monitor_position("BTC/USDT", position_dict)

    total_time = time.time() - start_time
    avg_time_per_iteration = total_time / iterations

    # Performance assertion
    assert avg_time_per_iteration < 0.1, f"Too slow: {avg_time_per_iteration:.3f}s per iteration"

    print(f"Performance test passed: {avg_time_per_iteration:.3f}s per iteration")
