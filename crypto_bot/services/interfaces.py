"""Service interface definitions for the trading bot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Protocol, Sequence


# ---------------------------------------------------------------------------
# Market data service contracts


@dataclass(slots=True)
class LoadSymbolsRequest:
    """Parameters used to discover tradable symbols on an exchange."""

    exchange: object
    exclude: Sequence[str]
    config: Optional[Mapping[str, Any]] = None


@dataclass(slots=True)
class LoadSymbolsResponse:
    """Result of a symbol discovery request."""

    symbols: list[str]


@dataclass(slots=True)
class OHLCVCacheRequest:
    """Parameters for updating a single timeframe OHLCV cache."""

    exchange: object
    cache: MutableMapping[str, Any]
    symbols: Sequence[str]
    timeframe: str = "1h"
    limit: int = 100
    use_websocket: bool = False
    force_websocket_history: bool = False
    config: Optional[Mapping[str, Any]] = None
    max_concurrent: Optional[int] = None
    notifier: Optional[object] = None


@dataclass(slots=True)
class MultiTimeframeOHLCVRequest:
    """Parameters for updating OHLCV caches across multiple timeframes."""

    exchange: object
    cache: MutableMapping[str, MutableMapping[str, Any]]
    symbols: Sequence[str]
    config: Mapping[str, Any]
    limit: int = 100
    use_websocket: bool = False
    force_websocket_history: bool = False
    max_concurrent: Optional[int] = None
    notifier: Optional[object] = None
    priority_queue: Optional[object] = None
    additional_timeframes: Optional[Sequence[str]] = None


@dataclass(slots=True)
class RegimeCacheRequest:
    """Parameters for updating regime timeframe caches."""

    exchange: object
    cache: MutableMapping[str, MutableMapping[str, Any]]
    symbols: Sequence[str]
    config: Mapping[str, Any]
    limit: int = 100
    use_websocket: bool = False
    force_websocket_history: bool = False
    max_concurrent: Optional[int] = None
    notifier: Optional[object] = None
    df_map: Optional[MutableMapping[str, MutableMapping[str, Any]]] = None


@dataclass(slots=True)
class CacheUpdateResponse:
    """Generic response for cache update operations."""

    cache: MutableMapping[str, Any]


@dataclass(slots=True)
class OrderBookRequest:
    """Parameters for fetching an order book snapshot."""

    exchange: object
    symbol: str
    depth: int = 2


@dataclass(slots=True)
class OrderBookResponse:
    """Order book snapshot data."""

    order_book: Optional[Mapping[str, Any]]


@dataclass(slots=True)
class TimeframeRequest:
    """Request conversion of timeframe to seconds."""

    exchange: Optional[object]
    timeframe: str


@dataclass(slots=True)
class TimeframeResponse:
    """Response providing timeframe length in seconds."""

    seconds: int


class MarketDataService(Protocol):
    """Protocol for market data operations."""

    async def load_symbols(self, request: LoadSymbolsRequest) -> LoadSymbolsResponse:
        ...

    async def update_ohlcv_cache(self, request: OHLCVCacheRequest) -> CacheUpdateResponse:
        ...

    async def update_multi_tf_cache(
        self, request: MultiTimeframeOHLCVRequest
    ) -> CacheUpdateResponse:
        ...

    async def update_regime_cache(self, request: RegimeCacheRequest) -> CacheUpdateResponse:
        ...

    async def fetch_order_book(self, request: OrderBookRequest) -> OrderBookResponse:
        ...

    def timeframe_seconds(self, request: TimeframeRequest) -> TimeframeResponse:
        ...


# ---------------------------------------------------------------------------
# Strategy evaluation service contracts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StrategyRequest:
    """Parameters for selecting a trading strategy."""

    regime: str
    config: Optional[Mapping[str, Any]] = None


@dataclass(slots=True)
class StrategyResponse:
    """Response containing a callable trading strategy."""

    strategy: Callable[..., Any]


@dataclass(slots=True)
class StrategyNameRequest:
    """Parameters for resolving strategy names."""

    regime: str
    mode: str


@dataclass(slots=True)
class StrategyNameResponse:
    """Response containing a strategy name."""

    name: str


class StrategyEvaluationService(Protocol):
    """Protocol for strategy evaluation helpers."""

    def select_strategy(self, request: StrategyRequest) -> StrategyResponse:
        ...

    def resolve_strategy_name(self, request: StrategyNameRequest) -> StrategyNameResponse:
        ...


# ---------------------------------------------------------------------------
# Portfolio service contracts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CreateTradeRequest:
    """Parameters for creating a Trade object."""

    symbol: str
    side: str
    amount: Any
    price: Any
    strategy: str = ""
    exchange: str = ""
    fees: Any = 0
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclass(slots=True)
class CreateTradeResponse:
    """Response containing the created trade instance."""

    trade: Any


class PortfolioService(Protocol):
    """Protocol covering trade/portfolio management helpers."""

    def create_trade(self, request: CreateTradeRequest) -> CreateTradeResponse:
        ...

    def get_trade_manager(self) -> Any:
        ...


# ---------------------------------------------------------------------------
# Execution service contracts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExchangeRequest:
    """Parameters for creating an exchange connection."""

    config: Mapping[str, Any]


@dataclass(slots=True)
class ExchangeResponse:
    """Response containing exchange connectivity objects."""

    exchange: Any
    ws_client: Optional[Any]


@dataclass(slots=True)
class TradeExecutionRequest:
    """Parameters for executing a trade through the exchange adapter."""

    exchange: Any
    ws_client: Optional[Any]
    symbol: str
    side: str
    amount: float
    notifier: Optional[Any] = None
    dry_run: bool = True
    use_websocket: bool = False
    config: Optional[Mapping[str, Any]] = None
    score: float = 0.0


@dataclass(slots=True)
class TradeExecutionResponse:
    """Response returned from trade execution."""

    order: Mapping[str, Any] | Any


class ExecutionService(Protocol):
    """Protocol for executing trades on centralized exchanges."""

    def create_exchange(self, request: ExchangeRequest) -> ExchangeResponse:
        ...

    async def execute_trade(self, request: TradeExecutionRequest) -> TradeExecutionResponse:
        ...


# ---------------------------------------------------------------------------
# Token discovery and monitoring service contracts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TokenDiscoveryRequest:
    """Parameters for discovering new tokens."""

    config: Mapping[str, Any]


@dataclass(slots=True)
class TokenDiscoveryResponse:
    """Response containing discovered token identifiers."""

    tokens: list[str]


class TokenDiscoveryService(Protocol):
    """Protocol for token discovery features."""

    async def discover_tokens(self, request: TokenDiscoveryRequest) -> TokenDiscoveryResponse:
        ...


@dataclass(slots=True)
class RecordScannerMetricsRequest:
    """Parameters for recording scanner metrics."""

    tokens: int
    latency: float
    config: Mapping[str, Any]


class MonitoringService(Protocol):
    """Protocol for recording operational metrics."""

    def record_scanner_metrics(self, request: RecordScannerMetricsRequest) -> None:
        ...


# ---------------------------------------------------------------------------
# Aggregated container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ServiceContainer:
    """Collection of service adapters used by the trading bot."""

    market_data: MarketDataService
    strategy: StrategyEvaluationService
    portfolio: PortfolioService
    execution: ExecutionService
    token_discovery: TokenDiscoveryService
    monitoring: MonitoringService

