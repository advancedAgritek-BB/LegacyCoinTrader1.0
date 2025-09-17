"""Pydantic schemas for portfolio service payloads."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TradeCreate(BaseModel):
    id: str
    symbol: str
    side: str
    amount: Decimal
    price: Decimal
    timestamp: datetime
    strategy: Optional[str] = None
    exchange: Optional[str] = None
    fees: Decimal = Decimal("0")
    status: str = "filled"
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, object] = Field(default_factory=dict)


class TradeRead(TradeCreate):
    position_symbol: Optional[str] = None


class PositionBase(BaseModel):
    symbol: str
    side: str
    total_amount: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    entry_time: datetime
    last_update: datetime
    highest_price: Optional[Decimal] = None
    lowest_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    trailing_stop_pct: Optional[Decimal] = None
    metadata: Dict[str, object] = Field(default_factory=dict)
    mark_price: Optional[Decimal] = None
    is_open: bool = True


class PositionRead(PositionBase):
    trades: List[TradeRead] = Field(default_factory=list)


class BalanceRead(BaseModel):
    currency: str
    amount: Decimal
    updated_at: datetime


class RiskLimitRead(BaseModel):
    id: int
    name: str
    max_position_size: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    max_daily_loss: Optional[Decimal] = None
    value_at_risk: Optional[Decimal] = None
    metadata: Dict[str, object] = Field(default_factory=dict)


class PriceCacheEntry(BaseModel):
    symbol: str
    price: Decimal
    updated_at: datetime


class PortfolioStatistics(BaseModel):
    total_trades: int = 0
    total_volume: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    total_realized_pnl: Decimal = Decimal("0")
    last_updated: Optional[datetime] = None


class PortfolioState(BaseModel):
    trades: List[TradeRead]
    positions: List[PositionRead]
    closed_positions: List[PositionRead]
    price_cache: List[PriceCacheEntry]
    statistics: PortfolioStatistics


class PnlBreakdown(BaseModel):
    realized: Decimal
    unrealized: Decimal
    total: Decimal


class RiskCheckResult(BaseModel):
    name: str
    passed: bool
    message: str
