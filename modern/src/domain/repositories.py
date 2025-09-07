"""
Repository Interfaces

This module defines the repository interfaces that abstract data access operations.
These interfaces follow the Repository pattern and provide a clean separation
between domain logic and data persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Protocol
from datetime import datetime

from .models import (
    TradingSymbol,
    Order,
    Position,
    Trade,
    MarketData,
    StrategySignal,
    PortfolioMetrics
)


class BaseRepository(Protocol):
    """Base repository protocol with common operations."""

    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Any]:
        """Get entity by ID."""
        ...

    @abstractmethod
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """Get all entities with pagination."""
        ...

    @abstractmethod
    async def create(self, entity: Any) -> Any:
        """Create new entity."""
        ...

    @abstractmethod
    async def update(self, id: str, entity: Any) -> Optional[Any]:
        """Update existing entity."""
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        ...

    @abstractmethod
    async def exists(self, id: str) -> bool:
        """Check if entity exists."""
        ...


class PositionRepository(BaseRepository, Protocol):
    """Repository interface for position operations."""

    @abstractmethod
    async def get_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        ...

    @abstractmethod
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        ...

    @abstractmethod
    async def get_positions_by_strategy(self, strategy_name: str) -> List[Position]:
        """Get positions by strategy name."""
        ...

    @abstractmethod
    async def update_pnl(self, symbol: str, pnl: float, pnl_percentage: float) -> bool:
        """Update position P&L information."""
        ...

    @abstractmethod
    async def close_position(self, symbol: str, exit_price: float) -> Optional[Position]:
        """Close position at given price."""
        ...

    @abstractmethod
    async def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        ...


class TradeRepository(BaseRepository, Protocol):
    """Repository interface for trade operations."""

    @abstractmethod
    async def get_by_symbol(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get trades by symbol."""
        ...

    @abstractmethod
    async def get_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Trade]:
        """Get trades within date range."""
        ...

    @abstractmethod
    async def get_by_strategy(self, strategy_name: str, limit: int = 100) -> List[Trade]:
        """Get trades by strategy name."""
        ...

    @abstractmethod
    async def get_profitable_trades(self, limit: int = 100) -> List[Trade]:
        """Get profitable trades."""
        ...

    @abstractmethod
    async def get_losing_trades(self, limit: int = 100) -> List[Trade]:
        """Get losing trades."""
        ...

    @abstractmethod
    async def get_total_pnl(self) -> float:
        """Get total P&L from all trades."""
        ...

    @abstractmethod
    async def get_win_rate(self) -> float:
        """Get overall win rate percentage."""
        ...


class SymbolRepository(BaseRepository, Protocol):
    """Repository interface for symbol operations."""

    @abstractmethod
    async def get_active_symbols(self) -> List[TradingSymbol]:
        """Get all active trading symbols."""
        ...

    @abstractmethod
    async def get_symbols_by_exchange(self, exchange: str) -> List[TradingSymbol]:
        """Get symbols for specific exchange."""
        ...

    @abstractmethod
    async def search_symbols(self, query: str) -> List[TradingSymbol]:
        """Search symbols by name or symbol."""
        ...

    @abstractmethod
    async def update_symbol_price(self, symbol: str, price: float) -> bool:
        """Update symbol current price."""
        ...

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> Optional[TradingSymbol]:
        """Get detailed symbol information."""
        ...


class OrderRepository(BaseRepository, Protocol):
    """Repository interface for order operations."""

    @abstractmethod
    async def get_by_symbol(self, symbol: str, limit: int = 100) -> List[Order]:
        """Get orders by symbol."""
        ...

    @abstractmethod
    async def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        ...

    @abstractmethod
    async def get_filled_orders(self, limit: int = 100) -> List[Order]:
        """Get filled orders."""
        ...

    @abstractmethod
    async def get_pending_orders(self) -> List[Order]:
        """Get pending orders."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID."""
        ...

    @abstractmethod
    async def update_order_status(self, order_id: str, status: str, filled_quantity: float = 0) -> bool:
        """Update order status and filled quantity."""
        ...


class MarketDataRepository(BaseRepository, Protocol):
    """Repository interface for market data operations."""

    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol."""
        ...

    @abstractmethod
    async def get_price_history(self, symbol: str, timeframe: str, limit: int = 100) -> List[MarketData]:
        """Get price history for symbol and timeframe."""
        ...

    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get OHLCV data for date range."""
        ...

    @abstractmethod
    async def save_market_data(self, data: MarketData) -> bool:
        """Save market data point."""
        ...

    @abstractmethod
    async def bulk_save_market_data(self, data_list: List[MarketData]) -> int:
        """Bulk save market data points."""
        ...


class StrategyRepository(BaseRepository, Protocol):
    """Repository interface for strategy operations."""

    @abstractmethod
    async def get_signals_by_symbol(self, symbol: str, limit: int = 100) -> List[StrategySignal]:
        """Get strategy signals by symbol."""
        ...

    @abstractmethod
    async def get_recent_signals(self, limit: int = 100) -> List[StrategySignal]:
        """Get recent strategy signals."""
        ...

    @abstractmethod
    async def get_signals_by_strategy(self, strategy_name: str, limit: int = 100) -> List[StrategySignal]:
        """Get signals by strategy name."""
        ...

    @abstractmethod
    async def get_strong_signals(self, min_confidence: float = 0.7) -> List[StrategySignal]:
        """Get signals above confidence threshold."""
        ...

    @abstractmethod
    async def save_signal(self, signal: StrategySignal) -> bool:
        """Save strategy signal."""
        ...

    @abstractmethod
    async def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        ...


class PortfolioRepository(BaseRepository, Protocol):
    """Repository interface for portfolio operations."""

    @abstractmethod
    async def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get current portfolio metrics."""
        ...

    @abstractmethod
    async def get_historical_metrics(self, days: int = 30) -> List[PortfolioMetrics]:
        """Get historical portfolio metrics."""
        ...

    @abstractmethod
    async def save_portfolio_metrics(self, metrics: PortfolioMetrics) -> bool:
        """Save portfolio metrics."""
        ...

    @abstractmethod
    async def get_daily_performance(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily performance data."""
        ...

    @abstractmethod
    async def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        ...


class CacheRepository(Protocol):
    """Repository interface for caching operations."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        ...

    @abstractmethod
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for cache key."""
        ...


# Export all repository interfaces
__all__ = [
    "BaseRepository",
    "PositionRepository",
    "TradeRepository",
    "SymbolRepository",
    "OrderRepository",
    "MarketDataRepository",
    "StrategyRepository",
    "PortfolioRepository",
    "CacheRepository",
]
