"""
Test Configuration and Fixtures

This module provides comprehensive test configuration, fixtures, and utilities
for the modern testing infrastructure. It includes async support, database
mocking, and comprehensive test utilities.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator, Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
import tempfile
import os
from decimal import Decimal

from dependency_injector import providers

import sys
from pathlib import Path

# Add the modern/src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.config import (
    AppConfig,
    Environment,
    ExecutionMode,
    ExchangeName,
    DatabaseConfig,
    RedisConfig,
    ExchangeConfig,
    TradingConfig
)
from core.container import Container, reset_container
from domain.models import (
    TradingSymbol,
    Order,
    Position,
    Trade,
    MarketData,
    StrategySignal,
    OrderSide,
    OrderType,
    OrderStatus
)


# Test Configuration
@pytest.fixture(scope="session")
def test_config() -> AppConfig:
    """Create test configuration."""
    return AppConfig(
        environment=Environment.TESTING,
        debug=True,
        database=DatabaseConfig(
            url="sqlite+aiosqlite:///:memory:"
        ),
        redis=RedisConfig(
            host="localhost",
            port=6379,
            db=1  # Use different DB for tests
        ),
        exchange=ExchangeConfig(
            name=ExchangeName.KRAKEN,
            api_key="test_key",
            api_secret="test_secret"
        ),
        trading=TradingConfig(
            execution_mode=ExecutionMode.DRY_RUN,
            max_open_positions=5,
            position_size_pct=0.1
        )
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def container(test_config: AppConfig) -> AsyncGenerator[Container, None]:
    """Create and configure dependency injection container for tests."""
    # Reset container to clean state
    reset_container()

    # Initialize with test configuration
    container = Container()
    container.config.override(test_config)

    # Override with test implementations
    container.database_connection.override(providers.Singleton(MockDatabaseConnection))
    container.redis_cache.override(providers.Singleton(MockRedisCache))

    yield container

    # Cleanup
    container.shutdown_resources()


# Mock Implementations
class MockDatabaseConnection:
    """Mock database connection for testing."""

    def __init__(self, config=None, logger=None):
        self.config = config
        self.logger = logger
        self.connected = True
        self.data: Dict[str, Dict[str, Any]] = {}

    async def connect(self):
        """Mock connect method."""
        self.connected = True

    async def disconnect(self):
        """Mock disconnect method."""
        self.connected = False

    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None):
        """Mock execute method."""
        return {"rows_affected": 1}

    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None):
        """Mock fetch_one method."""
        return None

    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None):
        """Mock fetch_all method."""
        return []


class MockRedisCache:
    """Mock Redis cache for testing."""

    def __init__(self, config=None, logger=None):
        self.config = config
        self.logger = logger
        self.data: Dict[str, Any] = {}

    async def get(self, key: str):
        """Mock get method."""
        return self.data.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Mock set method."""
        self.data[key] = value
        return True

    async def delete(self, key: str):
        """Mock delete method."""
        return self.data.pop(key, None) is not None

    async def exists(self, key: str):
        """Mock exists method."""
        return key in self.data

    async def clear(self):
        """Mock clear method."""
        self.data.clear()
        return True


# Model Fixtures
@pytest.fixture
def sample_symbol() -> TradingSymbol:
    """Create sample trading symbol."""
    return TradingSymbol(
        symbol="BTC/USD",
        base_currency="BTC",
        quote_currency="USD",
        exchange="kraken",
        min_order_size=Decimal("0.0001"),
        price_precision=2,
        quantity_precision=8
    )


@pytest.fixture
def sample_order() -> Order:
    """Create sample order."""
    return Order(
        id="test_order_123",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("50000.00")
    )


@pytest.fixture
def sample_position() -> Position:
    """Create sample position."""
    return Position(
        id="test_position_123",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=Decimal("0.01"),
        entry_price=Decimal("50000.00"),
        current_price=Decimal("51000.00")
    )


@pytest.fixture
def sample_trade() -> Trade:
    """Create sample trade."""
    return Trade(
        id="test_trade_123",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        quantity=Decimal("0.01"),
        price=Decimal("50000.00"),
        value=Decimal("500.00"),
        pnl=Decimal("100.00"),
        pnl_percentage=2.0,
        commission=Decimal("0.50"),
        order_id="test_order_123"
    )


@pytest.fixture
def sample_market_data() -> MarketData:
    """Create sample market data."""
    return MarketData(
        symbol="BTC/USD",
        open=Decimal("50000.00"),
        high=Decimal("51000.00"),
        low=Decimal("49500.00"),
        close=Decimal("50500.00"),
        volume=Decimal("100.00"),
        exchange="kraken",
        timeframe="1h"
    )


@pytest.fixture
def sample_strategy_signal() -> StrategySignal:
    """Create sample strategy signal."""
    return StrategySignal(
        strategy_name="test_strategy",
        symbol="BTC/USD",
        signal_type="buy",
        confidence=0.85,
        strength=80.0,
        indicators={"rsi": 30.0, "macd": -0.5}
    )


# Mock Fixtures
@pytest.fixture
def mock_logger():
    """Create mock logger."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def mock_metrics():
    """Create mock metrics collector."""
    metrics = MagicMock()
    metrics.increment = MagicMock()
    metrics.gauge = MagicMock()
    metrics.histogram = MagicMock()
    return metrics


@pytest.fixture
def mock_telegram():
    """Create mock Telegram notifier."""
    telegram = MagicMock()
    telegram.notify = AsyncMock()
    telegram.send_message = AsyncMock()
    return telegram


@pytest.fixture
def mock_exchange():
    """Create mock exchange client."""
    exchange = MagicMock()
    exchange.create_order = AsyncMock(return_value={"id": "test_order_123"})
    exchange.fetch_balance = AsyncMock(return_value={"USD": {"free": 1000.0}})
    exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0})
    exchange.cancel_order = AsyncMock(return_value=True)
    return exchange


@pytest.fixture
def mock_cache():
    """Create mock cache."""
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.exists = AsyncMock(return_value=False)
    return cache


@pytest.fixture
def mock_repository():
    """Create mock repository."""
    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=None)
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock(return_value=None)
    repo.update = AsyncMock(return_value=None)
    repo.delete = AsyncMock(return_value=True)
    repo.exists = AsyncMock(return_value=False)
    return repo


# Test Database Fixtures
@pytest.fixture(scope="function")
async def test_db():
    """Create test database connection."""
    # Use in-memory SQLite for testing
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async_session = sessionmaker(engine, class_=AsyncSession)

    # Create tables (simplified for testing)
    async with engine.begin() as conn:
        # Create test tables here if needed
        pass

    yield async_session

    # Cleanup
    await engine.dispose()


# Test Utilities
@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config_file(temp_dir):
    """Create sample configuration file."""
    config_path = temp_dir / "config.yaml"
    config_data = """
environment: testing
debug: true
exchange:
  name: kraken
  api_key: test_key
  api_secret: test_secret
trading:
  execution_mode: dry_run
  max_open_positions: 5
"""
    config_path.write_text(config_data)
    return config_path


# Async Test Utilities
@pytest.fixture
async def async_sleep():
    """Utility for controlled async sleeping in tests."""
    async def sleep(seconds: float = 0.1):
        await asyncio.sleep(seconds)
    return sleep


# Exception Fixtures
@pytest.fixture
def sample_exception():
    """Create sample exception for testing."""
    return ValueError("Test exception")


# Performance Test Fixtures
@pytest.fixture
def performance_timer():
    """Create performance timer for benchmarks."""
    import time

    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.duration = self.end - self.start

    return Timer


# Integration Test Fixtures
@pytest.fixture(scope="session")
def integration_config():
    """Configuration for integration tests."""
    return {
        "use_real_api": os.getenv("USE_REAL_API", "false").lower() == "true",
        "api_timeout": 10,
        "max_retries": 3
    }


# Export all fixtures
__all__ = [
    "test_config",
    "container",
    "sample_symbol",
    "sample_order",
    "sample_position",
    "sample_trade",
    "sample_market_data",
    "sample_strategy_signal",
    "mock_logger",
    "mock_metrics",
    "mock_telegram",
    "mock_exchange",
    "mock_cache",
    "mock_repository",
    "test_db",
    "temp_dir",
    "sample_config_file",
    "async_sleep",
    "sample_exception",
    "performance_timer",
    "integration_config",
]
