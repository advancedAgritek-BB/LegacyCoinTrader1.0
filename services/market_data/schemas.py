"""Pydantic models for market data service requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field, validator


class BaseExchangePayload(BaseModel):
    """Common fields shared by exchange-bound requests."""

    exchange_id: str = Field(..., description="Identifier for the exchange, e.g. 'kraken'.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Bot configuration payload.")

    @validator("exchange_id")
    def _normalize_exchange(cls, value: str) -> str:  # noqa: D401 - short validation helper
        """Ensure the exchange identifier is lower-case without whitespace."""
        return value.strip().lower()


class LoadSymbolsPayload(BaseExchangePayload):
    exclude: List[str] = Field(default_factory=list, description="Symbols to exclude from discovery.")


class SymbolListResponse(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    updated_at: datetime


class OHLCVUpdatePayload(BaseExchangePayload):
    symbols: List[str] = Field(default_factory=list)
    timeframe: str = Field(default="1h")
    limit: int = Field(default=200, ge=1)
    use_websocket: bool = Field(default=False)
    force_websocket_history: bool = Field(default=False)
    max_concurrent: Optional[int] = Field(default=None, ge=1)


class MultiOHLCVUpdatePayload(OHLCVUpdatePayload):
    additional_timeframes: Optional[List[str]] = Field(default=None)
    priority_queue_enabled: bool = Field(default=False)


class RegimeUpdatePayload(OHLCVUpdatePayload):
    df_timeframes: Optional[List[str]] = Field(default=None, description="Existing OHLCV timeframes to reuse.")


class TimeframeResponse(BaseModel):
    timeframe: str
    data: Dict[str, List[List[float]]]
    updated_at: datetime


class MultiTimeframeResponse(BaseModel):
    timeframes: Dict[str, Dict[str, List[List[float]]]]
    updated_at: datetime


class RegimeResponse(BaseModel):
    timeframes: Dict[str, Dict[str, List[List[float]]]]
    updated_at: datetime


class OrderBookPayload(BaseExchangePayload):
    symbol: str
    depth: int = Field(default=2, ge=1)


class OrderBookResponse(BaseModel):
    symbol: str
    order_book: Mapping[str, Any] | None
    updated_at: datetime


class TimeframeSecondsPayload(BaseExchangePayload):
    timeframe: str


class TimeframeSecondsResponse(BaseModel):
    timeframe: str
    seconds: int


class WebsocketRequest(BaseExchangePayload):
    symbol: str
    timeframe: str = Field(default="1m")
    limit: int = Field(default=100, ge=1)


__all__ = [
    "BaseExchangePayload",
    "LoadSymbolsPayload",
    "SymbolListResponse",
    "OHLCVUpdatePayload",
    "MultiOHLCVUpdatePayload",
    "RegimeUpdatePayload",
    "TimeframeResponse",
    "MultiTimeframeResponse",
    "RegimeResponse",
    "OrderBookPayload",
    "OrderBookResponse",
    "TimeframeSecondsPayload",
    "TimeframeSecondsResponse",
    "WebsocketRequest",
]
