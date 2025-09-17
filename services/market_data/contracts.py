from __future__ import annotations

"""Contracts for the market data microservice."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field, validator

from services.common.contracts import EventEnvelope, GrpcMethodDescriptor, HttpEndpoint


class BaseExchangePayload(BaseModel):
    """Shared fields for market data commands."""

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


class MarketDataEventPayload(BaseModel):
    """Payload describing a market data update."""

    exchange_id: str
    channel: str
    data: Dict[str, Any]
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarketDataEvent(EventEnvelope):
    event_type: str = Field(default="market-data.update", const=True)
    payload: MarketDataEventPayload


@runtime_checkable
class MarketDataService(Protocol):
    async def LoadSymbols(self, request: LoadSymbolsPayload) -> SymbolListResponse:  # noqa: N802
        ...

    async def UpdateOHLCV(self, request: OHLCVUpdatePayload) -> TimeframeResponse:  # noqa: N802
        ...

    async def UpdateRegime(self, request: RegimeUpdatePayload) -> RegimeResponse:  # noqa: N802
        ...

    async def GetOrderBook(self, request: OrderBookPayload) -> OrderBookResponse:  # noqa: N802
        ...


HTTP_CONTRACT: List[HttpEndpoint] = [
    HttpEndpoint(
        method="POST",
        path="/symbols/load",
        summary="Load tradable symbols from an exchange",
        request_model="services.market_data.contracts.LoadSymbolsPayload",
        response_model="services.market_data.contracts.SymbolListResponse",
    ),
    HttpEndpoint(
        method="POST",
        path="/ohlcv/update",
        summary="Refresh OHLCV caches for a single timeframe",
        request_model="services.market_data.contracts.OHLCVUpdatePayload",
        response_model="services.market_data.contracts.TimeframeResponse",
    ),
    HttpEndpoint(
        method="POST",
        path="/ohlcv/multi",
        summary="Refresh multiple OHLCV timeframes",
        request_model="services.market_data.contracts.MultiOHLCVUpdatePayload",
        response_model="services.market_data.contracts.MultiTimeframeResponse",
    ),
    HttpEndpoint(
        method="POST",
        path="/regime/update",
        summary="Recalculate trading regimes",
        request_model="services.market_data.contracts.RegimeUpdatePayload",
        response_model="services.market_data.contracts.RegimeResponse",
    ),
    HttpEndpoint(
        method="POST",
        path="/order-book",
        summary="Fetch order book snapshots",
        request_model="services.market_data.contracts.OrderBookPayload",
        response_model="services.market_data.contracts.OrderBookResponse",
    ),
]


GRPC_CONTRACT: List[GrpcMethodDescriptor] = [
    GrpcMethodDescriptor(
        name="LoadSymbols",
        request="services.market_data.contracts.LoadSymbolsPayload",
        response="services.market_data.contracts.SymbolListResponse",
    ),
    GrpcMethodDescriptor(
        name="UpdateOHLCV",
        request="services.market_data.contracts.OHLCVUpdatePayload",
        response="services.market_data.contracts.TimeframeResponse",
    ),
    GrpcMethodDescriptor(
        name="UpdateRegime",
        request="services.market_data.contracts.RegimeUpdatePayload",
        response="services.market_data.contracts.RegimeResponse",
    ),
    GrpcMethodDescriptor(
        name="GetOrderBook",
        request="services.market_data.contracts.OrderBookPayload",
        response="services.market_data.contracts.OrderBookResponse",
    ),
]


__all__ = [
    "BaseExchangePayload",
    "GRPC_CONTRACT",
    "HTTP_CONTRACT",
    "LoadSymbolsPayload",
    "MarketDataEvent",
    "MarketDataEventPayload",
    "MarketDataService",
    "MultiOHLCVUpdatePayload",
    "OHLCVUpdatePayload",
    "OrderBookPayload",
    "OrderBookResponse",
    "RegimeResponse",
    "RegimeUpdatePayload",
    "SymbolListResponse",
    "TimeframeResponse",
    "TimeframeSecondsPayload",
    "TimeframeSecondsResponse",
    "WebsocketRequest",
]
