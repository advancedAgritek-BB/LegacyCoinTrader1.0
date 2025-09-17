"""In-process adapter for market data utilities."""

from __future__ import annotations

from crypto_bot.services.interfaces import (
    CacheUpdateResponse,
    LoadSymbolsRequest,
    LoadSymbolsResponse,
    MarketDataService,
    MultiTimeframeOHLCVRequest,
    OHLCVCacheRequest,
    OrderBookRequest,
    OrderBookResponse,
    RegimeCacheRequest,
    TimeframeRequest,
    TimeframeResponse,
)
from crypto_bot.utils.market_loader import (
    fetch_order_book_async,
    load_kraken_symbols,
    timeframe_seconds,
    update_multi_tf_ohlcv_cache,
    update_ohlcv_cache,
    update_regime_tf_cache,
)


class MarketDataAdapter(MarketDataService):
    """Adapter that proxies calls to :mod:`crypto_bot.utils.market_loader`."""

    async def load_symbols(self, request: LoadSymbolsRequest) -> LoadSymbolsResponse:
        symbols = await load_kraken_symbols(
            request.exchange,
            request.exclude,
            request.config or {},
        )
        return LoadSymbolsResponse(symbols=list(symbols or []))

    async def update_ohlcv_cache(self, request: OHLCVCacheRequest) -> CacheUpdateResponse:
        cache = await update_ohlcv_cache(
            request.exchange,
            request.cache,
            request.symbols,
            timeframe=request.timeframe,
            limit=request.limit,
            use_websocket=request.use_websocket,
            force_websocket_history=request.force_websocket_history,
            config=request.config,
            max_concurrent=request.max_concurrent,
            notifier=request.notifier,
        )
        return CacheUpdateResponse(cache=cache)

    async def update_multi_tf_cache(
        self, request: MultiTimeframeOHLCVRequest
    ) -> CacheUpdateResponse:
        cache = await update_multi_tf_ohlcv_cache(
            request.exchange,
            request.cache,
            request.symbols,
            request.config,
            limit=request.limit,
            use_websocket=request.use_websocket,
            force_websocket_history=request.force_websocket_history,
            max_concurrent=request.max_concurrent,
            notifier=request.notifier,
            priority_queue=request.priority_queue,
            additional_timeframes=request.additional_timeframes,
        )
        return CacheUpdateResponse(cache=cache)

    async def update_regime_cache(self, request: RegimeCacheRequest) -> CacheUpdateResponse:
        cache = await update_regime_tf_cache(
            request.exchange,
            request.cache,
            request.symbols,
            request.config,
            limit=request.limit,
            use_websocket=request.use_websocket,
            force_websocket_history=request.force_websocket_history,
            max_concurrent=request.max_concurrent,
            notifier=request.notifier,
            df_map=request.df_map,
        )
        return CacheUpdateResponse(cache=cache)

    async def fetch_order_book(self, request: OrderBookRequest) -> OrderBookResponse:
        order_book = await fetch_order_book_async(
            request.exchange,
            request.symbol,
            request.depth,
        )
        return OrderBookResponse(order_book=order_book)

    def timeframe_seconds(self, request: TimeframeRequest) -> TimeframeResponse:
        seconds = timeframe_seconds(request.exchange, request.timeframe)
        return TimeframeResponse(seconds=seconds)
