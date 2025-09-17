"""SQLAlchemy models for the portfolio service."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


DECIMAL_TYPE = Numeric(24, 12, asdecimal=True)


class TradeModel(Base):
    """Persisted trade information."""

    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    amount: Mapped[Decimal] = mapped_column(DECIMAL_TYPE)
    price: Mapped[Decimal] = mapped_column(DECIMAL_TYPE)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    strategy: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    exchange: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    fees: Mapped[Decimal] = mapped_column(DECIMAL_TYPE, default=Decimal("0"))
    status: Mapped[str] = mapped_column(String(16), default="filled")
    order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    client_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    position_symbol: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("positions.symbol"), nullable=True
    )

    position: Mapped["PositionModel"] = relationship(
        "PositionModel", back_populates="trades"
    )


class PositionModel(Base):
    """Persisted position information."""

    __tablename__ = "positions"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    side: Mapped[str] = mapped_column(String(8))
    total_amount: Mapped[Decimal] = mapped_column(DECIMAL_TYPE, default=Decimal("0"))
    average_price: Mapped[Decimal] = mapped_column(DECIMAL_TYPE, default=Decimal("0"))
    realized_pnl: Mapped[Decimal] = mapped_column(DECIMAL_TYPE, default=Decimal("0"))
    fees_paid: Mapped[Decimal] = mapped_column(DECIMAL_TYPE, default=Decimal("0"))
    entry_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_update: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    highest_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL_TYPE, nullable=True)
    lowest_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL_TYPE, nullable=True)
    stop_loss_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL_TYPE, nullable=True)
    take_profit_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL_TYPE, nullable=True)
    trailing_stop_pct: Mapped[Optional[Decimal]] = mapped_column(DECIMAL_TYPE, nullable=True)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    mark_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL_TYPE, nullable=True)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    trades: Mapped[list[TradeModel]] = relationship(
        "TradeModel",
        back_populates="position",
        cascade="all, delete-orphan",
        order_by="TradeModel.timestamp",
    )


class BalanceModel(Base):
    """Account balance information."""

    __tablename__ = "balances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    currency: Mapped[str] = mapped_column(String(16), unique=True)
    amount: Mapped[Decimal] = mapped_column(DECIMAL_TYPE, default=Decimal("0"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RiskLimitModel(Base):
    """Risk management configuration."""

    __tablename__ = "risk_limits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    max_position_size: Mapped[Optional[Decimal]] = mapped_column(DECIMAL_TYPE, nullable=True)
    max_drawdown: Mapped[Optional[Decimal]] = mapped_column(DECIMAL_TYPE, nullable=True)
    max_daily_loss: Mapped[Optional[Decimal]] = mapped_column(DECIMAL_TYPE, nullable=True)
    value_at_risk: Mapped[Optional[Decimal]] = mapped_column(DECIMAL_TYPE, nullable=True)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)


class PriceCacheModel(Base):
    """Latest observed prices for symbols."""

    __tablename__ = "price_cache"
    __table_args__ = (UniqueConstraint("symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True)
    price: Mapped[Decimal] = mapped_column(DECIMAL_TYPE)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PortfolioStatisticModel(Base):
    """Aggregate metrics for the portfolio."""

    __tablename__ = "portfolio_statistics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_volume: Mapped[Decimal] = mapped_column(DECIMAL_TYPE, default=Decimal("0"))
    total_fees: Mapped[Decimal] = mapped_column(DECIMAL_TYPE, default=Decimal("0"))
    total_realized_pnl: Mapped[Decimal] = mapped_column(DECIMAL_TYPE, default=Decimal("0"))
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class UserAccountModel(Base):
    """Identity and credential metadata for interactive users."""

    __tablename__ = "user_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    password_salt: Mapped[str] = mapped_column(String(128), nullable=False)
    password_iterations: Mapped[int] = mapped_column(Integer, default=210000, nullable=False)
    roles: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    password_rotated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    password_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    api_key_hash: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    api_key_salt: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    api_key_iterations: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    api_key_last_rotated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
