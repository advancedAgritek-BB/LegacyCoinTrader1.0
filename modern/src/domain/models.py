"""
Domain Models

This module defines the core domain models using Pydantic v2 for type safety,
validation, and serialization. These models represent the business entities
and their relationships.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import condecimal, confloat


class TradingSymbol(BaseModel):
    """Trading symbol model."""

    symbol: str = Field(..., min_length=1, max_length=20)
    base_currency: str = Field(..., min_length=1, max_length=10)
    quote_currency: str = Field(..., min_length=1, max_length=10)
    exchange: str = Field(..., min_length=1, max_length=20)
    is_active: bool = Field(True)
    min_order_size: Decimal = Field(..., gt=0)
    max_order_size: Optional[Decimal] = Field(None, gt=0)
    price_precision: int = Field(..., ge=0, le=18)
    quantity_precision: int = Field(..., ge=0, le=18)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("symbol")
    def validate_symbol_format(cls, v):
        """Validate symbol format (BASE/QUOTE)."""
        if "/" not in v:
            raise ValueError("Symbol must be in format BASE/QUOTE")
        return v.upper()

    @model_validator(mode="after")
    def validate_currencies(self):
        """Validate that base and quote currencies match symbol."""
        symbol = self.symbol
        base_currency = self.base_currency
        quote_currency = self.quote_currency

        if symbol and "/" in symbol:
            expected_base, expected_quote = symbol.split("/", 1)
            if base_currency and base_currency != expected_base:
                raise ValueError(f"Base currency {base_currency} doesn't match symbol {symbol}")
            if quote_currency and quote_currency != expected_quote:
                raise ValueError(f"Quote currency {quote_currency} doesn't match symbol {symbol}")

        return self


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order(BaseModel):
    """Trading order model."""

    id: str = Field(..., min_length=1, max_length=100)
    symbol: str = Field(..., min_length=1, max_length=20)
    side: OrderSide
    type: OrderType
    quantity: condecimal(gt=0, max_digits=20, decimal_places=10)
    price: Optional[condecimal(gt=0, max_digits=20, decimal_places=10)] = None
    stop_price: Optional[condecimal(gt=0, max_digits=20, decimal_places=10)] = None
    limit_price: Optional[condecimal(gt=0, max_digits=20, decimal_places=10)] = None

    # Execution details
    status: OrderStatus = Field(OrderStatus.PENDING)
    filled_quantity: condecimal(ge=0, max_digits=20, decimal_places=10) = Field(0)
    remaining_quantity: condecimal(ge=0, max_digits=20, decimal_places=10) = Field(0)
    average_fill_price: Optional[condecimal(gt=0, max_digits=20, decimal_places=10)] = None

    # Metadata
    exchange_order_id: Optional[str] = Field(None, max_length=100)
    client_order_id: Optional[str] = Field(None, max_length=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    # Fees and costs
    commission: condecimal(ge=0, max_digits=10, decimal_places=8) = Field(0)
    commission_asset: Optional[str] = Field(None, max_length=10)

    # Additional properties
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("remaining_quantity")
    @classmethod
    def calculate_remaining_quantity(cls, v, info):
        """Calculate remaining quantity based on filled quantity."""
        data = info.data
        if "quantity" in data and "filled_quantity" in data:
            return data["quantity"] - data["filled_quantity"]
        return v

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIAL]

    @property
    def fill_percentage(self) -> float:
        """Get the percentage of order that has been filled."""
        if self.quantity == 0:
            return 0.0
        return float((self.filled_quantity / self.quantity) * 100)


class Position(BaseModel):
    """Trading position model."""

    id: str = Field(..., min_length=1, max_length=100)
    symbol: str = Field(..., min_length=1, max_length=20)
    side: OrderSide
    quantity: condecimal(gt=0, max_digits=20, decimal_places=10)
    entry_price: condecimal(gt=0, max_digits=20, decimal_places=10)
    current_price: condecimal(gt=0, max_digits=20, decimal_places=10)

    # Position metrics
    unrealized_pnl: condecimal(max_digits=20, decimal_places=10) = Field(0)
    realized_pnl: condecimal(max_digits=20, decimal_places=10) = Field(0)
    pnl_percentage: confloat(ge=-100.0, le=1000.0) = Field(0.0)

    # Risk management
    stop_loss_price: Optional[condecimal(gt=0, max_digits=20, decimal_places=10)] = None
    take_profit_price: Optional[condecimal(gt=0, max_digits=20, decimal_places=10)] = None
    trailing_stop_pct: Optional[confloat(gt=0, le=50)] = None

    # Metadata
    exchange: str = Field(..., min_length=1, max_length=20)
    strategy_name: Optional[str] = Field(None, max_length=50)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Additional properties
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("unrealized_pnl")
    @classmethod
    def calculate_unrealized_pnl(cls, v, info):
        """Calculate unrealized P&L based on current vs entry price."""
        data = info.data
        if "side" in data and "quantity" in data and "entry_price" in data and "current_price" in data:
            side = data["side"]
            quantity = data["quantity"]
            entry_price = data["entry_price"]
            current_price = data["current_price"]

            if side == OrderSide.BUY:
                return (current_price - entry_price) * quantity
            else:  # SELL
                return (entry_price - current_price) * quantity
        return v

    @field_validator("pnl_percentage")
    @classmethod
    def calculate_pnl_percentage(cls, v, info):
        """Calculate P&L percentage."""
        data = info.data
        if "entry_price" in data and "current_price" in data:
            entry_price = data["entry_price"]
            current_price = data["current_price"]

            if entry_price == 0:
                return 0.0

            return float(((current_price - entry_price) / entry_price) * 100)
        return v

    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of position."""
        return self.quantity * self.current_price

    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.unrealized_pnl > 0

    @property
    def should_stop_loss(self) -> bool:
        """Check if stop loss should be triggered."""
        if not self.stop_loss_price:
            return False

        if self.side == OrderSide.BUY:
            return self.current_price <= self.stop_loss_price
        else:
            return self.current_price >= self.stop_loss_price

    @property
    def should_take_profit(self) -> bool:
        """Check if take profit should be triggered."""
        if not self.take_profit_price:
            return False

        if self.side == OrderSide.BUY:
            return self.current_price >= self.take_profit_price
        else:
            return self.current_price <= self.take_profit_price


class Trade(BaseModel):
    """Completed trade model."""

    id: str = Field(..., min_length=1, max_length=100)
    symbol: str = Field(..., min_length=1, max_length=20)
    side: OrderSide
    quantity: condecimal(gt=0, max_digits=20, decimal_places=10)
    price: condecimal(gt=0, max_digits=20, decimal_places=10)
    value: condecimal(gt=0, max_digits=20, decimal_places=10)

    # P&L information
    pnl: condecimal(max_digits=20, decimal_places=10)
    pnl_percentage: confloat(ge=-100.0, le=1000.0)
    commission: condecimal(ge=0, max_digits=10, decimal_places=8)

    # Order information
    order_id: str = Field(..., min_length=1, max_length=100)
    exchange_order_id: Optional[str] = Field(None, max_length=100)
    client_order_id: Optional[str] = Field(None, max_length=100)

    # Strategy information
    strategy_name: Optional[str] = Field(None, max_length=50)
    signal_strength: Optional[confloat(ge=0, le=1)] = None

    # Timing
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Additional properties
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("value")
    @classmethod
    def calculate_value(cls, v, info):
        """Calculate trade value from quantity and price."""
        data = info.data
        if "quantity" in data and "price" in data:
            return data["quantity"] * data["price"]
        return v

    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    @property
    def effective_price(self) -> Decimal:
        """Get the effective price including commission."""
        if self.value == 0:
            return self.price
        commission_cost = (self.commission / self.value) * self.price
        return self.price - commission_cost if self.side == OrderSide.BUY else self.price + commission_cost


class MarketData(BaseModel):
    """Market data model."""

    symbol: str = Field(..., min_length=1, max_length=20)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    open: condecimal(gt=0, max_digits=20, decimal_places=10)
    high: condecimal(gt=0, max_digits=20, decimal_places=10)
    low: condecimal(gt=0, max_digits=20, decimal_places=10)
    close: condecimal(gt=0, max_digits=20, decimal_places=10)
    volume: condecimal(ge=0, max_digits=20, decimal_places=10)

    # Additional metrics
    vwap: Optional[condecimal(gt=0, max_digits=20, decimal_places=10)] = None
    trades: Optional[int] = Field(None, ge=0)

    # Source information
    exchange: str = Field(..., min_length=1, max_length=20)
    timeframe: str = Field(..., min_length=1, max_length=10)

    @field_validator("high", "low")
    @classmethod
    def validate_price_range(cls, v, info):
        """Validate that high >= close >= low."""
        data = info.data
        field_name = info.field_name
        if "close" in data:
            close = data["close"]
            if field_name == "high" and v < close:
                raise ValueError(f"High price {v} cannot be less than close price {close}")
            if field_name == "low" and v > close:
                raise ValueError(f"Low price {v} cannot be greater than close price {close}")
        return v


class StrategySignal(BaseModel):
    """Trading strategy signal model."""

    strategy_name: str = Field(..., min_length=1, max_length=50)
    symbol: str = Field(..., min_length=1, max_length=20)
    signal_type: str = Field(..., min_length=1, max_length=20)  # buy, sell, hold
    confidence: confloat(ge=0, le=1) = Field(...)
    strength: confloat(ge=0, le=100) = Field(...)

    # Signal metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeframe: str = Field(..., min_length=1, max_length=10)

    # Technical indicators
    indicators: Dict[str, Union[float, int, str]] = Field(default_factory=dict)

    # Risk metrics
    risk_score: Optional[confloat(ge=0, le=100)] = None
    expected_return: Optional[confloat(ge=-100, le=1000)] = None
    stop_loss: Optional[condecimal(gt=0, max_digits=20, decimal_places=10)] = None
    take_profit: Optional[condecimal(gt=0, max_digits=20, decimal_places=10)] = None

    # Additional properties
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_strong_signal(self) -> bool:
        """Check if signal is strong enough for execution."""
        return self.confidence >= 0.7 and self.strength >= 70

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk-reward ratio if stop loss and take profit are set."""
        if not self.stop_loss or not self.take_profit:
            return None

        entry_price = self.indicators.get("current_price", 0)
        if not entry_price:
            return None

        if self.signal_type == "buy":
            risk = float(entry_price - self.stop_loss)
            reward = float(self.take_profit - entry_price)
        else:  # sell
            risk = float(self.stop_loss - entry_price)
            reward = float(entry_price - self.take_profit)

        return reward / risk if risk > 0 else None


class PortfolioMetrics(BaseModel):
    """Portfolio performance metrics model."""

    total_value: condecimal(ge=0, max_digits=20, decimal_places=10)
    cash_balance: condecimal(ge=0, max_digits=20, decimal_places=10)
    invested_value: condecimal(ge=0, max_digits=20, decimal_places=10)

    # Performance metrics
    total_pnl: condecimal(max_digits=20, decimal_places=10)
    total_pnl_percentage: confloat(ge=-100.0, le=1000.0)
    daily_pnl: condecimal(max_digits=20, decimal_places=10)
    daily_pnl_percentage: confloat(ge=-100.0, le=1000.0)

    # Risk metrics
    max_drawdown: confloat(ge=0, le=100) = Field(0.0)
    sharpe_ratio: Optional[confloat(ge=-10, le=10)] = None
    volatility: confloat(ge=0, le=100) = Field(0.0)

    # Position metrics
    open_positions: int = Field(0, ge=0)
    winning_positions: int = Field(0, ge=0)
    losing_positions: int = Field(0, ge=0)
    win_rate: confloat(ge=0, le=100) = Field(0.0)

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    @field_validator("win_rate")
    @classmethod
    def calculate_win_rate(cls, v, info):
        """Calculate win rate percentage."""
        data = info.data
        open_positions = data.get("open_positions", 0)
        winning_positions = data.get("winning_positions", 0)

        if open_positions == 0:
            return 0.0

        return float((winning_positions / open_positions) * 100)


# Export all models
__all__ = [
    "TradingSymbol",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Position",
    "Trade",
    "MarketData",
    "StrategySignal",
    "PortfolioMetrics",
]
