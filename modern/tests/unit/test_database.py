"""
Unit Tests for Database Infrastructure

Comprehensive tests for database connection, models, and repository patterns
with proper async testing and mocking.
"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

import sys
from pathlib import Path

# Add the modern/src directory to Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from infrastructure.database import (
    DatabaseConnection,
    DatabaseRepository,
    TradingSymbolModel,
    OrderModel,
    PositionModel,
    TradeModel,
    init_database,
    get_database_connection
)
from core.config import DatabaseConfig
from domain.models import TradingSymbol, Order, Position, Trade, OrderSide, OrderType, OrderStatus


class TestDatabaseConnection:
    """Test DatabaseConnection."""

    @pytest.fixture
    def db_config(self):
        """Create test database configuration."""
        return DatabaseConfig(
            url="sqlite+aiosqlite:///:memory:",
            pool_size=5,
            max_overflow=10,
            pool_timeout=30
        )

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    @pytest.fixture
    async def db_connection(self, db_config, mock_logger):
        """Create test database connection."""
        connection = DatabaseConnection(db_config, mock_logger)

        # Mock the engine and session factory
        mock_engine = AsyncMock()
        mock_session_factory = AsyncMock()

        with patch.object(connection, '_engine', mock_engine), \
             patch.object(connection, '_session_factory', mock_session_factory), \
             patch.object(connection, '_connected', True):

            yield connection

    def test_database_connection_initialization(self, db_connection, db_config, mock_logger):
        """Test database connection initialization."""
        assert db_connection.config == db_config
        assert db_connection.logger == mock_logger
        assert not db_connection.is_connected

    @patch('src.infrastructure.database.create_async_engine')
    @patch('src.infrastructure.database.async_sessionmaker')
    async def test_connect_success(self, mock_sessionmaker, mock_create_engine, db_connection):
        """Test successful database connection."""
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine

        mock_session = AsyncMock()
        mock_sessionmaker.return_value = mock_session

        await db_connection.connect()

        assert db_connection.is_connected
        assert db_connection._engine == mock_engine
        assert db_connection._session_factory == mock_session

        mock_create_engine.assert_called_once()
        mock_sessionmaker.assert_called_once()

    async def test_connect_failure(self, db_connection):
        """Test database connection failure."""
        with patch('src.infrastructure.database.create_async_engine', side_effect=Exception("Connection failed")):
            with pytest.raises(ConnectionError, match="Database connection failed"):
                await db_connection.connect()

        assert not db_connection.is_connected

    async def test_disconnect(self, db_connection):
        """Test database disconnection."""
        mock_engine = AsyncMock()
        db_connection._engine = mock_engine
        db_connection._connected = True

        await db_connection.disconnect()

        assert not db_connection.is_connected
        mock_engine.dispose.assert_called_once()

    async def test_session_context_manager(self, db_connection):
        """Test database session context manager."""
        db_connection._connected = True

        mock_session = AsyncMock()
        db_connection._session_factory = MagicMock(return_value=mock_session)

        async with db_connection.session() as session:
            assert session == mock_session

        mock_session.close.assert_called_once()

    async def test_session_context_manager_not_connected(self, db_connection):
        """Test session context manager when not connected."""
        db_connection._connected = False

        with pytest.raises(ConnectionError, match="Database not connected"):
            async with db_connection.session():
                pass

    async def test_health_check_connected(self, db_connection):
        """Test health check when connected."""
        db_connection._connected = True

        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_result = AsyncMock()
        mock_result.fetchone.return_value = (1,)

        mock_conn.execute.return_value = mock_result
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

        db_connection._engine = mock_engine

        health = await db_connection.health_check()

        assert health["connected"] is True
        assert health["status"] == "healthy"
        assert health["query_success"] is True

    async def test_health_check_disconnected(self, db_connection):
        """Test health check when disconnected."""
        db_connection._connected = False

        health = await db_connection.health_check()

        assert health["connected"] is False
        assert health["status"] == "disconnected"


class TestDatabaseRepository:
    """Test DatabaseRepository."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock database connection."""
        connection = MagicMock()
        connection.session = MagicMock()
        return connection

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache."""
        return AsyncMock()

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    @pytest.fixture
    def repository(self, mock_connection, mock_cache, mock_logger):
        """Create test repository."""
        return DatabaseRepository(
            connection=mock_connection,
            table_name="test_table",
            model_class=TradingSymbolModel,
            cache=mock_cache,
            logger=mock_logger
        )

    async def test_get_by_id_cache_hit(self, repository, mock_cache):
        """Test get_by_id with cache hit."""
        mock_cache.get.return_value = {"id": "test_id", "symbol": "BTC/USD"}

        result = await repository.get_by_id("test_id")

        assert result == {"id": "test_id", "symbol": "BTC/USD"}
        mock_cache.get.assert_called_once_with("test_table:test_id")
        repository.connection.session.assert_not_called()

    async def test_get_by_id_cache_miss(self, repository, mock_cache, mock_connection):
        """Test get_by_id with cache miss."""
        mock_cache.get.return_value = None

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_session.get.return_value = mock_result

        mock_connection.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_connection.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repository.get_by_id("test_id")

        assert result == mock_result
        mock_cache.get.assert_called_once_with("test_table:test_id")
        mock_cache.set.assert_called_once()
        mock_session.get.assert_called_once_with(TradingSymbolModel, "test_id")

    async def test_get_all(self, repository, mock_connection):
        """Test get_all method."""
        mock_session = AsyncMock()
        mock_stmt_result = MagicMock()
        mock_scalars_result = MagicMock()
        mock_scalars_result.all.return_value = [{"id": "1"}, {"id": "2"}]

        mock_stmt_result.scalars.return_value = mock_scalars_result
        mock_session.execute.return_value = mock_stmt_result

        mock_connection.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_connection.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repository.get_all(limit=10, offset=5)

        assert result == [{"id": "1"}, {"id": "2"}]
        mock_session.query.assert_called_once_with(TradingSymbolModel)
        mock_session.execute.assert_called_once()

    async def test_create_success(self, repository, mock_connection, mock_cache):
        """Test successful entity creation."""
        mock_entity = MagicMock()
        mock_entity.id = "test_id"

        mock_session = AsyncMock()
        mock_connection.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_connection.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repository.create(mock_entity)

        assert result == mock_entity
        mock_session.add.assert_called_once_with(mock_entity)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(mock_entity)
        mock_cache.delete.assert_called_once_with("test_table:test_id")

    async def test_create_integrity_error(self, repository, mock_connection):
        """Test creation with integrity error."""
        from sqlalchemy.exc import IntegrityError

        mock_entity = MagicMock()
        mock_session = AsyncMock()
        mock_session.commit.side_effect = IntegrityError("test", "test", "test")

        mock_connection.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_connection.session.return_value.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Record already exists"):
            await repository.create(mock_entity)

        mock_session.rollback.assert_called_once()

    async def test_update_success(self, repository, mock_connection, mock_cache):
        """Test successful entity update."""
        mock_entity = MagicMock()
        mock_entity.id = "test_id"
        mock_entity.__dict__ = {"symbol": "BTC/USD", "price": 50000}

        mock_existing = MagicMock()
        mock_existing.updated_at = None

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_existing

        mock_connection.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_connection.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repository.update("test_id", mock_entity)

        assert result == mock_existing
        mock_session.get.assert_called_once_with(TradingSymbolModel, "test_id")
        mock_session.commit.assert_called_once()
        mock_cache.set.assert_called_once()

    async def test_update_not_found(self, repository, mock_connection):
        """Test update when entity not found."""
        mock_entity = MagicMock()
        mock_session = AsyncMock()
        mock_session.get.return_value = None

        mock_connection.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_connection.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repository.update("nonexistent_id", mock_entity)

        assert result is None
        mock_session.get.assert_called_once_with(TradingSymbolModel, "nonexistent_id")
        mock_session.commit.assert_not_called()

    async def test_delete_success(self, repository, mock_connection, mock_cache):
        """Test successful entity deletion."""
        mock_entity = MagicMock()
        mock_entity.id = "test_id"

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_entity

        mock_connection.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_connection.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repository.delete("test_id")

        assert result is True
        mock_session.get.assert_called_once_with(TradingSymbolModel, "test_id")
        mock_session.delete.assert_called_once_with(mock_entity)
        mock_session.commit.assert_called_once()
        mock_cache.delete.assert_called_once_with("test_table:test_id")

    async def test_delete_not_found(self, repository, mock_connection):
        """Test deletion when entity not found."""
        mock_session = AsyncMock()
        mock_session.get.return_value = None

        mock_connection.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_connection.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repository.delete("nonexistent_id")

        assert result is False
        mock_session.get.assert_called_once_with(TradingSymbolModel, "nonexistent_id")
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()

    async def test_exists_with_cache(self, repository, mock_cache):
        """Test exists method with cache."""
        mock_cache.get.return_value = True

        result = await repository.exists("test_id")

        assert result is True
        mock_cache.get.assert_called_once_with("test_table:test_id:exists")

    async def test_exists_without_cache(self, repository, mock_cache, mock_connection):
        """Test exists method without cache."""
        mock_cache.get.return_value = None

        mock_session = AsyncMock()
        mock_session.get.return_value = MagicMock()  # Entity exists

        mock_connection.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_connection.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repository.exists("test_id")

        assert result is True
        mock_cache.set.assert_called_once()


class TestDatabaseModels:
    """Test database models."""

    def test_trading_symbol_model_creation(self):
        """Test TradingSymbolModel creation."""
        model = TradingSymbolModel(
            symbol="BTC/USD",
            base_currency="BTC",
            quote_currency="USD",
            exchange="kraken",
            min_order_size=Decimal("0.0001"),
            price_precision=2,
            quantity_precision=8
        )

        assert model.symbol == "BTC/USD"
        assert model.base_currency == "BTC"
        assert model.quote_currency == "USD"
        assert model.exchange == "kraken"
        assert model.min_order_size == Decimal("0.0001")
        assert model.is_active is True

    def test_order_model_creation(self):
        """Test OrderModel creation."""
        model = OrderModel(
            id="order_123",
            symbol="BTC/USD",
            side="buy",
            type="limit",
            quantity=Decimal("0.01"),
            price=Decimal("50000.00")
        )

        assert model.id == "order_123"
        assert model.symbol == "BTC/USD"
        assert model.side == "buy"
        assert model.type == "limit"
        assert model.quantity == Decimal("0.01")
        assert model.price == Decimal("50000.00")
        assert model.status == "pending"
        assert model.filled_quantity == 0

    def test_position_model_creation(self):
        """Test PositionModel creation."""
        model = PositionModel(
            id="pos_123",
            symbol="BTC/USD",
            side="buy",
            quantity=Decimal("0.01"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("51000.00")
        )

        assert model.id == "pos_123"
        assert model.symbol == "BTC/USD"
        assert model.side == "buy"
        assert model.quantity == Decimal("0.01")
        assert model.entry_price == Decimal("50000.00")
        assert model.current_price == Decimal("51000.00")
        assert model.unrealized_pnl == Decimal("100.00")  # Calculated field
        assert model.pnl_percentage == 2.0  # Calculated field

    def test_trade_model_creation(self):
        """Test TradeModel creation."""
        model = TradeModel(
            id="trade_123",
            symbol="BTC/USD",
            side="buy",
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            value=Decimal("500.00"),
            pnl=Decimal("100.00"),
            pnl_percentage=2.0,
            commission=Decimal("0.50"),
            order_id="order_123"
        )

        assert model.id == "trade_123"
        assert model.symbol == "BTC/USD"
        assert model.side == "buy"
        assert model.quantity == Decimal("0.01")
        assert model.price == Decimal("50000.00")
        assert model.value == Decimal("500.00")
        assert model.pnl == Decimal("100.00")
        assert model.pnl_percentage == 2.0
        assert model.commission == Decimal("0.50")


class TestDatabaseIntegration:
    """Test database integration functions."""

    def test_init_database(self):
        """Test database initialization."""
        config = DatabaseConfig(url="sqlite+aiosqlite:///:memory:")

        with patch('src.infrastructure.database.DatabaseConnection') as mock_connection_class:
            mock_connection = MagicMock()
            mock_connection_class.return_value = mock_connection

            result = init_database(config)

            assert result == mock_connection
            mock_connection_class.assert_called_once_with(config, None)

    def test_get_database_connection_initialized(self):
        """Test getting database connection when initialized."""
        mock_connection = MagicMock()

        with patch('src.infrastructure.database._db_connection', mock_connection):
            result = get_database_connection()

            assert result == mock_connection

    def test_get_database_connection_not_initialized(self):
        """Test getting database connection when not initialized."""
        with patch('src.infrastructure.database._db_connection', None):
            with pytest.raises(RuntimeError, match="Database not initialized"):
                get_database_connection()


class TestRepositoryIntegration:
    """Test repository integration with domain models."""

    @pytest.fixture
    def sample_trading_symbol(self):
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
    def sample_order(self):
        """Create sample order."""
        return Order(
            id="order_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00")
        )

    @pytest.fixture
    def sample_position(self):
        """Create sample position."""
        return Position(
            id="pos_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("51000.00")
        )

    @pytest.fixture
    def sample_trade(self):
        """Create sample trade."""
        return Trade(
            id="trade_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            value=Decimal("500.00"),
            pnl=Decimal("100.00"),
            pnl_percentage=2.0,
            commission=Decimal("0.50"),
            order_id="order_123"
        )

    def test_model_to_database_conversion(self, sample_trading_symbol, sample_order, sample_position, sample_trade):
        """Test conversion from domain models to database models."""
        # These should be able to be created from domain model data
        # (In a real implementation, you'd have conversion functions)

        # Trading Symbol
        db_symbol = TradingSymbolModel(
            symbol=sample_trading_symbol.symbol,
            base_currency=sample_trading_symbol.base_currency,
            quote_currency=sample_trading_symbol.quote_currency,
            exchange=sample_trading_symbol.exchange,
            min_order_size=sample_trading_symbol.min_order_size,
            price_precision=sample_trading_symbol.price_precision,
            quantity_precision=sample_trading_symbol.quantity_precision
        )
        assert db_symbol.symbol == "BTC/USD"

        # Order
        db_order = OrderModel(
            id=sample_order.id,
            symbol=sample_order.symbol,
            side=sample_order.side.value,
            type=sample_order.type.value,
            quantity=sample_order.quantity,
            price=sample_order.price
        )
        assert db_order.id == "order_123"
        assert db_order.side == "buy"

        # Position
        db_position = PositionModel(
            id=sample_position.id,
            symbol=sample_position.symbol,
            side=sample_position.side.value,
            quantity=sample_position.quantity,
            entry_price=sample_position.entry_price,
            current_price=sample_position.current_price
        )
        assert db_position.id == "pos_123"
        assert db_position.side == "buy"

        # Trade
        db_trade = TradeModel(
            id=sample_trade.id,
            symbol=sample_trade.symbol,
            side=sample_trade.side.value,
            quantity=sample_trade.quantity,
            price=sample_trade.price,
            value=sample_trade.value,
            pnl=sample_trade.pnl,
            pnl_percentage=sample_trade.pnl_percentage,
            commission=sample_trade.commission,
            order_id=sample_trade.order_id
        )
        assert db_trade.id == "trade_123"
        assert db_trade.side == "buy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
