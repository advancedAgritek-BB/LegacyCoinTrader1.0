"""
Database Infrastructure

Enterprise-grade database layer with SQLAlchemy 2.0 async support,
connection pooling, migrations, and comprehensive error handling.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional, List, Type, Union
from datetime import datetime

from sqlalchemy import MetaData, Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError

from ..core.config import DatabaseConfig, get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


# Database Models
class TradingSymbolModel(Base):
    """Trading symbol database model."""

    __tablename__ = "trading_symbols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    base_currency: Mapped[str] = mapped_column(String(10))
    quote_currency: Mapped[str] = mapped_column(String(10))
    exchange: Mapped[str] = mapped_column(String(50), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    min_order_size: Mapped[float] = mapped_column(Float)
    max_order_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_precision: Mapped[int] = mapped_column(Integer)
    quantity_precision: Mapped[int] = mapped_column(Integer)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class OrderModel(Base):
    """Order database model."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    side: Mapped[str] = mapped_column(String(10))
    type: Mapped[str] = mapped_column(String(20))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    limit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    filled_quantity: Mapped[float] = mapped_column(Float, default=0)
    remaining_quantity: Mapped[float] = mapped_column(Float, default=0)
    average_fill_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exchange_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    client_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    commission: Mapped[float] = mapped_column(Float, default=0)
    commission_asset: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PositionModel(Base):
    """Position database model."""

    __tablename__ = "positions"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    side: Mapped[str] = mapped_column(String(10))
    quantity: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float] = mapped_column(Float)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0)
    pnl_percentage: Mapped[float] = mapped_column(Float, default=0)
    stop_loss_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trailing_stop_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exchange: Mapped[str] = mapped_column(String(50), index=True)
    strategy_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TradeModel(Base):
    """Trade database model."""

    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    side: Mapped[str] = mapped_column(String(10))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    value: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float] = mapped_column(Float)
    pnl_percentage: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float)
    order_id: Mapped[str] = mapped_column(String(100), index=True)
    exchange_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    client_order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    strategy_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    signal_strength: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MarketDataModel(Base):
    """Market data database model."""

    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    vwap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trades: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    exchange: Mapped[str] = mapped_column(String(50), index=True)
    timeframe: Mapped[str] = mapped_column(String(10), index=True)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class StrategySignalModel(Base):
    """Strategy signal database model."""

    __tablename__ = "strategy_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50), index=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    signal_type: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    strength: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    timeframe: Mapped[str] = mapped_column(String(10))
    indicators: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class PortfolioMetricsModel(Base):
    """Portfolio metrics database model."""

    __tablename__ = "portfolio_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    total_value: Mapped[float] = mapped_column(Float)
    cash_balance: Mapped[float] = mapped_column(Float)
    invested_value: Mapped[float] = mapped_column(Float)
    total_pnl: Mapped[float] = mapped_column(Float)
    total_pnl_percentage: Mapped[float] = mapped_column(Float)
    daily_pnl: Mapped[float] = mapped_column(Float)
    daily_pnl_percentage: Mapped[float] = mapped_column(Float)
    max_drawdown: Mapped[float] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volatility: Mapped[float] = mapped_column(Float)
    open_positions: Mapped[int] = mapped_column(Integer)
    winning_positions: Mapped[int] = mapped_column(Integer)
    losing_positions: Mapped[int] = mapped_column(Integer)
    win_rate: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    period_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    period_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class DatabaseConnection:
    """
    Database connection manager with connection pooling and health monitoring.

    Features:
    - Async SQLAlchemy 2.0 support
    - Connection pooling with health checks
    - Automatic reconnection on failures
    - Comprehensive error handling and logging
    """

    def __init__(self, config: DatabaseConfig, logger=None):
        """
        Initialize database connection.

        Args:
            config: Database configuration.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger or get_logger(__name__)
        self._engine = None
        self._session_factory = None
        self._connected = False

    async def connect(self) -> None:
        """
        Establish database connection with pooling.

        Raises:
            ConnectionError: If connection fails.
        """
        try:
            # Create async engine with connection pooling
            database_url = self.config.url
            if not database_url.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
                if database_url.startswith("postgresql://"):
                    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
                elif database_url.startswith("sqlite://"):
                    database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)

            self._engine = create_async_engine(
                database_url,
                poolclass=AsyncAdaptedQueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False,  # Disable SQL logging in production
                future=True  # SQLAlchemy 2.0 style
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Test connection
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._connected = True
            self.logger.info("Database connection established successfully")

        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise ConnectionError(f"Database connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._engine:
            await self._engine.dispose()
            self._connected = False
            self.logger.info("Database connection closed")

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup.

        Yields:
            AsyncSession: Database session.
        """
        if not self._connected or not self._session_factory:
            raise ConnectionError("Database not connected")

        session = self._session_factory()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.

        Returns:
            Dict[str, Any]: Health check results.
        """
        health_info = {
            "connected": self._connected,
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "timestamp": datetime.utcnow().isoformat()
        }

        if self._connected and self._engine:
            try:
                async with self._engine.begin() as conn:
                    # Simple health check query
                    result = await conn.execute("SELECT 1")
                    health_info["query_success"] = True
                    health_info["status"] = "healthy"
            except Exception as e:
                health_info["query_success"] = False
                health_info["error"] = str(e)
                health_info["status"] = "unhealthy"
        else:
            health_info["status"] = "disconnected"

        return health_info


class DatabaseRepository:
    """
    Generic database repository with CRUD operations.

    Provides a standardized interface for database operations with
    caching support and comprehensive error handling.
    """

    def __init__(self, connection: DatabaseConnection, table_name: str,
                 model_class: Type[Base], cache=None, logger=None):
        """
        Initialize repository.

        Args:
            connection: Database connection instance.
            table_name: Database table name.
            model_class: SQLAlchemy model class.
            cache: Optional cache instance.
            logger: Optional logger instance.
        """
        self.connection = connection
        self.table_name = table_name
        self.model_class = model_class
        self.cache = cache
        self.logger = logger or get_logger(__name__)

    async def get_by_id(self, id: str) -> Optional[Base]:
        """
        Get record by ID.

        Args:
            id: Record ID.

        Returns:
            Optional[Base]: Found record or None.
        """
        cache_key = f"{self.table_name}:{id}"

        # Try cache first
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

        try:
            async with self.connection.session() as session:
                result = await session.get(self.model_class, id)
                if result and self.cache:
                    await self.cache.set(cache_key, result, ttl=300)  # 5 minutes
                return result
        except SQLAlchemyError as e:
            self.logger.error(f"Database error in get_by_id: {e}")
            raise

    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Base]:
        """
        Get all records with pagination.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List[Base]: List of records.
        """
        try:
            async with self.connection.session() as session:
                stmt = session.query(self.model_class).limit(limit).offset(offset)
                result = await session.execute(stmt)
                return result.scalars().all()
        except SQLAlchemyError as e:
            self.logger.error(f"Database error in get_all: {e}")
            raise

    async def create(self, entity: Base) -> Base:
        """
        Create new record.

        Args:
            entity: Entity to create.

        Returns:
            Base: Created entity.
        """
        try:
            async with self.connection.session() as session:
                session.add(entity)
                await session.commit()
                await session.refresh(entity)

                # Invalidate cache
                if self.cache:
                    await self.cache.delete(f"{self.table_name}:{entity.id}")

                self.logger.info(f"Created {self.table_name} record: {entity.id}")
                return entity
        except IntegrityError as e:
            self.logger.error(f"Integrity error creating {self.table_name}: {e}")
            await session.rollback()
            raise ValueError(f"Record already exists or constraint violation") from e
        except SQLAlchemyError as e:
            self.logger.error(f"Database error creating {self.table_name}: {e}")
            await session.rollback()
            raise

    async def update(self, id: str, entity: Base) -> Optional[Base]:
        """
        Update existing record.

        Args:
            id: Record ID to update.
            entity: Updated entity data.

        Returns:
            Optional[Base]: Updated entity or None if not found.
        """
        try:
            async with self.connection.session() as session:
                existing = await session.get(self.model_class, id)
                if not existing:
                    return None

                # Update fields
                for key, value in entity.__dict__.items():
                    if not key.startswith('_'):
                        setattr(existing, key, value)

                existing.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(existing)

                # Update cache
                if self.cache:
                    cache_key = f"{self.table_name}:{id}"
                    await self.cache.set(cache_key, existing, ttl=300)

                self.logger.info(f"Updated {self.table_name} record: {id}")
                return existing
        except SQLAlchemyError as e:
            self.logger.error(f"Database error updating {self.table_name}: {e}")
            await session.rollback()
            raise

    async def delete(self, id: str) -> bool:
        """
        Delete record by ID.

        Args:
            id: Record ID to delete.

        Returns:
            bool: True if deleted, False if not found.
        """
        try:
            async with self.connection.session() as session:
                entity = await session.get(self.model_class, id)
                if not entity:
                    return False

                await session.delete(entity)
                await session.commit()

                # Clear cache
                if self.cache:
                    await self.cache.delete(f"{self.table_name}:{id}")

                self.logger.info(f"Deleted {self.table_name} record: {id}")
                return True
        except SQLAlchemyError as e:
            self.logger.error(f"Database error deleting {self.table_name}: {e}")
            await session.rollback()
            raise

    async def exists(self, id: str) -> bool:
        """
        Check if record exists.

        Args:
            id: Record ID to check.

        Returns:
            bool: True if exists, False otherwise.
        """
        cache_key = f"{self.table_name}:{id}:exists"

        # Try cache first
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            async with self.connection.session() as session:
                result = await session.get(self.model_class, id)
                exists = result is not None

                if self.cache:
                    await self.cache.set(cache_key, exists, ttl=300)

                return exists
        except SQLAlchemyError as e:
            self.logger.error(f"Database error in exists: {e}")
            raise


# Global database instances
_db_connection: Optional[DatabaseConnection] = None


def init_database(config: DatabaseConfig) -> DatabaseConnection:
    """
    Initialize global database connection.

    Args:
        config: Database configuration.

    Returns:
        DatabaseConnection: Initialized database connection.
    """
    global _db_connection
    logger = get_logger(__name__)

    _db_connection = DatabaseConnection(config, logger)
    return _db_connection


def get_database_connection() -> DatabaseConnection:
    """
    Get global database connection instance.

    Returns:
        DatabaseConnection: Database connection instance.

    Raises:
        RuntimeError: If database not initialized.
    """
    if _db_connection is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db_connection


# Export all database components
__all__ = [
    "Base",
    "TradingSymbolModel",
    "OrderModel",
    "PositionModel",
    "TradeModel",
    "MarketDataModel",
    "StrategySignalModel",
    "PortfolioMetricsModel",
    "DatabaseConnection",
    "DatabaseRepository",
    "init_database",
    "get_database_connection",
]
