"""HTTP-based adapter for the market data microservice."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Mapping, MutableMapping

import aiohttp
import pandas as pd

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

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _candles_to_dataframe(candles: list[list[float]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    df = pd.DataFrame(candles, columns=CANDLE_COLUMNS)
    df["timestamp"] = df["timestamp"].astype(int)
    return df


class MarketDataAdapter(MarketDataService):
    """Adapter that communicates with the market data service via HTTP."""

    def __init__(self, base_url: str | None = None, *, session: aiohttp.ClientSession | None = None):
        self._base_url = base_url or os.getenv("MARKET_DATA_SERVICE_URL", "http://localhost:8002")
        timeout = float(os.getenv("MARKET_DATA_HTTP_TIMEOUT", "60"))
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = session
        self._logger = logging.getLogger(__name__)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        session = await self._get_session()
        url = f"{self._base_url}{path}"
        try:
            async with session.request(method, url, json=payload) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise RuntimeError(f"Market data service error {response.status}: {text}")
                if "application/json" in response.headers.get("Content-Type", ""):
                    return await response.json()
                text = await response.text()
                return json.loads(text)
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Failed to reach market data service: {exc}") from exc

    @staticmethod
    def _update_timeframe_cache(cache: MutableMapping[str, Any], data: Mapping[str, list[list[float]]]) -> MutableMapping[str, Any]:
        for symbol, candles in data.items():
            df = _candles_to_dataframe(candles)
            if not df.empty:
                cache[symbol] = df
        return cache

    @staticmethod
    def _update_multi_cache(
        cache: MutableMapping[str, MutableMapping[str, Any]],
        data: Mapping[str, Mapping[str, list[list[float]]]],
    ) -> MutableMapping[str, MutableMapping[str, Any]]:
        for timeframe, symbols in data.items():
            tf_cache = cache.setdefault(timeframe, {})
            MarketDataAdapter._update_timeframe_cache(tf_cache, symbols)
            cache[timeframe] = tf_cache
        return cache

    async def load_symbols(self, request: LoadSymbolsRequest) -> LoadSymbolsResponse:
        payload = {
            "exchange_id": request.exchange_id,
            "exclude": list(request.exclude),
            "config": dict(request.config or {}),
        }
        data = await self._request("POST", "/symbols/load", payload)
        symbols = data.get("symbols", []) if isinstance(data, dict) else []
        return LoadSymbolsResponse(symbols=list(symbols))

    async def update_ohlcv_cache(self, request: OHLCVCacheRequest) -> CacheUpdateResponse:
        payload = {
            "exchange_id": request.exchange_id,
            "symbols": list(request.symbols),
            "timeframe": request.timeframe,
            "limit": request.limit,
            "use_websocket": request.use_websocket,
            "force_websocket_history": request.force_websocket_history,
            "max_concurrent": request.max_concurrent,
            "config": dict(request.config or {}),
        }
        data = await self._request("POST", "/ohlcv/update", payload)
        timeframe_data: Mapping[str, list[list[float]]] = data.get("data", {}) if isinstance(data, dict) else {}
        updated_cache = self._update_timeframe_cache(request.cache, timeframe_data)
        return CacheUpdateResponse(cache=updated_cache)

    async def update_multi_tf_cache(self, request: MultiTimeframeOHLCVRequest) -> CacheUpdateResponse:
        payload = {
            "exchange_id": request.exchange_id,
            "symbols": list(request.symbols),
            "timeframe": request.config.get("timeframes", [request.config.get("timeframe", "1h")])[0]
            if isinstance(request.config.get("timeframes"), list) and request.config.get("timeframes")
            else request.config.get("timeframe", "1h"),
            "limit": request.limit,
            "use_websocket": request.use_websocket,
            "force_websocket_history": request.force_websocket_history,
            "max_concurrent": request.max_concurrent,
            "additional_timeframes": list(request.additional_timeframes or []),
            "config": dict(request.config),
        }
        data = await self._request("POST", "/ohlcv/multi", payload)
        response_data: Mapping[str, Mapping[str, list[list[float]]]] = (
            data.get("timeframes", {}) if isinstance(data, dict) else {}
        )
        updated_cache = self._update_multi_cache(request.cache, response_data)
        return CacheUpdateResponse(cache=updated_cache)

    async def update_regime_cache(self, request: RegimeCacheRequest) -> CacheUpdateResponse:
        payload = {
            "exchange_id": request.exchange_id,
            "symbols": list(request.symbols),
            "limit": request.limit,
            "use_websocket": request.use_websocket,
            "force_websocket_history": request.force_websocket_history,
            "max_concurrent": request.max_concurrent,
            "config": dict(request.config),
            "df_timeframes": list(request.df_map.keys()) if request.df_map else None,
        }
        data = await self._request("POST", "/regime/update", payload)
        response_data: Mapping[str, Mapping[str, list[list[float]]]] = (
            data.get("timeframes", {}) if isinstance(data, dict) else {}
        )
        updated_cache = self._update_multi_cache(request.cache, response_data)
        return CacheUpdateResponse(cache=updated_cache)

    async def fetch_order_book(self, request: OrderBookRequest) -> OrderBookResponse:
        payload = {
            "exchange_id": request.exchange_id,
            "symbol": request.symbol,
            "depth": request.depth,
            "config": dict(request.config or {}),
        }
        data = await self._request("POST", "/order-book/snapshot", payload)
        order_book = data.get("order_book") if isinstance(data, dict) else None
        return OrderBookResponse(order_book=order_book)

    def timeframe_seconds(self, request: TimeframeRequest) -> TimeframeResponse:
        timeframe = request.timeframe
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        multipliers = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
            "w": 604800,
            "M": 2592000,
        }
        seconds = multipliers.get(unit)
        if seconds is None:
            raise ValueError(f"Unknown timeframe {timeframe}")
        return TimeframeResponse(seconds=value * seconds)
