"""Helpers for storing and retrieving market data in Redis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Iterable, Mapping, MutableMapping

import pandas as pd
from redis.asyncio import Redis

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _safe_exchange(exchange_id: str) -> str:
    return exchange_id.replace("/", "_")


def _cache_key(namespace: str, exchange_id: str, timeframe: str, symbol: str) -> str:
    return f"{namespace}:{_safe_exchange(exchange_id)}:{timeframe}:{_safe_symbol(symbol)}"


def _order_book_key(namespace: str, exchange_id: str, symbol: str) -> str:
    return f"{namespace}:{_safe_exchange(exchange_id)}:{_safe_symbol(symbol)}"


def _symbols_key(namespace: str, exchange_id: str) -> str:
    return f"{namespace}:{_safe_exchange(exchange_id)}"


def _candles_to_dataframe(candles: Iterable[Iterable[float]]) -> pd.DataFrame:
    data = list(candles or [])
    if not data:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    df = pd.DataFrame(data, columns=CANDLE_COLUMNS)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(int)
    return df


def _dataframe_to_candles(df: pd.DataFrame) -> list[list[float]]:
    if df is None or df.empty:
        return []
    values = df[CANDLE_COLUMNS].to_numpy(copy=True)
    candles: list[list[float]] = []
    for ts, op, high, low, close, volume in values:
        candles.append(
            [
                int(ts),
                float(op),
                float(high),
                float(low),
                float(close),
                float(volume),
            ]
        )
    return candles


async def load_timeframe_cache(
    redis_client: Redis,
    namespace: str,
    exchange_id: str,
    timeframe: str,
    symbols: Iterable[str],
) -> Dict[str, pd.DataFrame]:
    symbol_list = list(symbols)
    keys = [_cache_key(namespace, exchange_id, timeframe, sym) for sym in symbol_list]
    if not keys:
        return {}
    values = await redis_client.mget(keys)
    result: Dict[str, pd.DataFrame] = {}
    for sym, raw in zip(symbol_list, values):
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candles = payload.get("data", [])
        df = _candles_to_dataframe(candles)
        if not df.empty:
            result[sym] = df
    return result


async def store_timeframe_cache(
    redis_client: Redis,
    namespace: str,
    exchange_id: str,
    timeframe: str,
    cache: Mapping[str, pd.DataFrame],
    ttl: int,
) -> Dict[str, list[list[float]]]:
    if not cache:
        return {}
    pipe = redis_client.pipeline(transaction=False)
    now = datetime.now(timezone.utc).isoformat()
    serialized: Dict[str, list[list[float]]] = {}
    for symbol, df in cache.items():
        candles = _dataframe_to_candles(df)
        serialized[symbol] = candles
        key = _cache_key(namespace, exchange_id, timeframe, symbol)
        payload = {
            "exchange": exchange_id,
            "timeframe": timeframe,
            "symbol": symbol,
            "updated_at": now,
            "data": candles,
        }
        pipe.set(key, json.dumps(payload))
        if ttl > 0:
            pipe.expire(key, ttl)
    await pipe.execute()
    return serialized


async def load_multi_timeframe_cache(
    redis_client: Redis,
    namespace: str,
    exchange_id: str,
    timeframes: Iterable[str],
    symbols: Iterable[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    result: Dict[str, Dict[str, pd.DataFrame]] = {}
    for tf in timeframes:
        result[tf] = await load_timeframe_cache(redis_client, namespace, exchange_id, tf, symbols)
    return result


async def store_multi_timeframe_cache(
    redis_client: Redis,
    namespace: str,
    exchange_id: str,
    data: Mapping[str, Mapping[str, pd.DataFrame]],
    ttl: int,
) -> Dict[str, Dict[str, list[list[float]]]]:
    response: Dict[str, Dict[str, list[list[float]]]] = {}
    for timeframe, cache in data.items():
        serialized = await store_timeframe_cache(
            redis_client,
            namespace,
            exchange_id,
            timeframe,
            cache,
            ttl,
        )
        response[timeframe] = serialized
    return response


async def store_order_book(
    redis_client: Redis,
    namespace: str,
    exchange_id: str,
    symbol: str,
    order_book: Mapping[str, object],
    ttl: int,
) -> None:
    key = _order_book_key(namespace, exchange_id, symbol)
    payload = {
        "exchange": exchange_id,
        "symbol": symbol,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "order_book": order_book,
    }
    await redis_client.set(key, json.dumps(payload))
    if ttl > 0:
        await redis_client.expire(key, ttl)


async def store_symbols(
    redis_client: Redis,
    namespace: str,
    exchange_id: str,
    symbols: Iterable[str],
    ttl: int,
) -> None:
    key = _symbols_key(namespace, exchange_id)
    payload = {
        "exchange": exchange_id,
        "symbols": list(symbols),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await redis_client.set(key, json.dumps(payload))
    if ttl > 0:
        await redis_client.expire(key, ttl)


def prepare_local_cache(
    cache: MutableMapping[str, pd.DataFrame],
    updates: Mapping[str, list[list[float]]],
) -> MutableMapping[str, pd.DataFrame]:
    for symbol, candles in updates.items():
        df = _candles_to_dataframe(candles)
        if not df.empty:
            cache[symbol] = df
    return cache


def prepare_multi_local_cache(
    cache: MutableMapping[str, MutableMapping[str, pd.DataFrame]],
    updates: Mapping[str, Mapping[str, list[list[float]]]],
) -> MutableMapping[str, MutableMapping[str, pd.DataFrame]]:
    for timeframe, symbols in updates.items():
        tf_cache = cache.setdefault(timeframe, {})
        prepare_local_cache(tf_cache, symbols)
        cache[timeframe] = tf_cache
    return cache
