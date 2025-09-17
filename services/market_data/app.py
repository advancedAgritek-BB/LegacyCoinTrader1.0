from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status

from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging import configure_logging

from libs.execution import get_exchange
from libs.market_data import (
    AdaptiveRateLimiter,
    fetch_ohlcv_async,
    fetch_order_book_async,
    load_kraken_symbols,
    timeframe_seconds,
    update_multi_tf_ohlcv_cache,
    update_ohlcv_cache,
    update_regime_tf_cache,
)

from .config import Settings, get_settings
from .redis_cache import (
    load_multi_timeframe_cache,
    load_timeframe_cache,
    store_multi_timeframe_cache,
    store_order_book,
    store_symbols,
    store_timeframe_cache,
)
from .schemas import (
    LoadSymbolsPayload,
    MultiOHLCVUpdatePayload,
    MultiTimeframeResponse,
    OHLCVUpdatePayload,
    OrderBookPayload,
    OrderBookResponse,
    RegimeResponse,
    RegimeUpdatePayload,
    SymbolListResponse,
    TimeframeResponse,
    TimeframeSecondsPayload,
    TimeframeSecondsResponse,
)

service_settings = get_settings()
monitoring_settings = get_monitoring_settings().for_service(service_settings.app_name)
monitoring_settings = monitoring_settings.model_copy(
    update={"log_level": service_settings.log_level}
)
monitoring_settings.metrics.default_labels.setdefault("component", "market-data")
configure_logging(monitoring_settings)

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _serialize_event(data: Mapping[str, Any]) -> str:
    return json.dumps(dict(data), default=_json_default)


def _prepare_exchange_config(exchange_id: str, config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(config or {})
    exchange_cfg = cfg.get("exchange")
    if isinstance(exchange_cfg, dict):
        exchange_cfg = dict(exchange_cfg)
        exchange_cfg.setdefault("id", exchange_id)
        exchange_cfg.setdefault("name", exchange_id)
        cfg["exchange"] = exchange_cfg
    elif exchange_cfg:
        cfg["exchange"] = str(exchange_cfg)
    else:
        cfg["exchange"] = exchange_id
    # Market data service always operates in REST mode for data fetching
    cfg["use_websocket"] = False
    return cfg


async def _create_exchange(exchange_id: str, config: Optional[Mapping[str, Any]]) -> Any:
    cfg = _prepare_exchange_config(exchange_id, config)
    exchange, ws_client = get_exchange(cfg)
    if ws_client is not None:
        with suppress(Exception):
            close_ws = getattr(ws_client, "close", None)
            if close_ws:
                close_ws()
    return exchange


async def _close_exchange(exchange: Any) -> None:
    close = getattr(exchange, "close", None)
    if close is None:
        return
    if asyncio.iscoroutinefunction(close):
        with suppress(Exception):
            await close()
    else:
        with suppress(Exception):
            await asyncio.to_thread(close)


def get_settings_dependency(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = get_settings()
    return settings


def get_redis_dependency(request: Request) -> redis.Redis:
    redis_client: redis.Redis | None = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis unavailable")
    return redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.getLogger().setLevel(settings.log_level.upper())
    redis_client = redis.from_url(
        settings.redis_dsn(),
        encoding="utf-8",
        decode_responses=True,
        health_check_interval=30,
    )
    try:
        await redis_client.ping()
    except Exception as exc:  # pragma: no cover - fatal startup failure
        logger.error("Unable to connect to Redis: %s", exc)
        await redis_client.close()
        raise

    app.state.redis = redis_client
    app.state.settings = settings
    try:
        yield
    finally:
        await redis_client.close()


app = FastAPI(title="Market Data Service", lifespan=lifespan)
instrument_fastapi_app(app, settings=monitoring_settings)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readiness")
async def readiness(
    redis_client: redis.Redis = Depends(get_redis_dependency),
    settings: Settings = Depends(get_settings_dependency),
) -> Dict[str, str]:
    try:
        await redis_client.ping()
    except Exception as exc:  # pragma: no cover - readiness failure
        logger.warning("Readiness ping failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis unavailable")
    return {"status": "ready", "redis": settings.redis_dsn()}


@app.post("/symbols/load", response_model=SymbolListResponse)
async def load_symbols(
    payload: LoadSymbolsPayload,
    redis_client: redis.Redis = Depends(get_redis_dependency),
    settings: Settings = Depends(get_settings_dependency),
) -> SymbolListResponse:
    exchange = await _create_exchange(payload.exchange_id, payload.config)
    try:
        symbols = await load_kraken_symbols(
            exchange,
            payload.exclude,
            payload.config,
        )
    finally:
        await _close_exchange(exchange)
    symbols = list(symbols or [])
    await store_symbols(redis_client, payload.exchange_id, symbols, settings.cache_ttl_seconds)
    updated_at = datetime.now(timezone.utc)
    event = {
        "type": "symbols_update",
        "exchange": payload.exchange_id,
        "symbols": symbols,
        "updated_at": updated_at,
    }
    await redis_client.publish(settings.symbols_channel, _serialize_event(event))
    return SymbolListResponse(symbols=symbols, updated_at=updated_at)


@app.post("/ohlcv/update", response_model=TimeframeResponse)
async def update_ohlcv_endpoint(
    payload: OHLCVUpdatePayload,
    redis_client: redis.Redis = Depends(get_redis_dependency),
    settings: Settings = Depends(get_settings_dependency),
) -> TimeframeResponse:
    exchange = await _create_exchange(payload.exchange_id, payload.config)
    try:
        existing_cache = await load_timeframe_cache(
            redis_client,
            "ohlcv",
            payload.exchange_id,
            payload.timeframe,
            payload.symbols,
        )
        updated_cache = await update_ohlcv_cache(
            exchange,
            existing_cache,
            payload.symbols,
            timeframe=payload.timeframe,
            limit=payload.limit,
            use_websocket=payload.use_websocket,
            force_websocket_history=payload.force_websocket_history,
            config=payload.config,
            max_concurrent=payload.max_concurrent,
        )
    finally:
        await _close_exchange(exchange)

    serialized = await store_timeframe_cache(
        redis_client,
        "ohlcv",
        payload.exchange_id,
        payload.timeframe,
        updated_cache,
        settings.cache_ttl_seconds,
    )
    updated_at = datetime.now(timezone.utc)
    event = {
        "type": "ohlcv_update",
        "exchange": payload.exchange_id,
        "timeframe": payload.timeframe,
        "symbols": list(serialized.keys()),
        "updated_at": updated_at,
    }
    await redis_client.publish(settings.ohlcv_channel, _serialize_event(event))
    return TimeframeResponse(timeframe=payload.timeframe, data=serialized, updated_at=updated_at)


@app.post("/ohlcv/multi", response_model=MultiTimeframeResponse)
async def update_multi_timeframe(
    payload: MultiOHLCVUpdatePayload,
    redis_client: redis.Redis = Depends(get_redis_dependency),
    settings: Settings = Depends(get_settings_dependency),
) -> MultiTimeframeResponse:
    exchange = await _create_exchange(payload.exchange_id, payload.config)
    try:
        base_timeframes = payload.config.get("timeframes", [payload.timeframe])
        additional = payload.additional_timeframes or []
        all_timeframes = sorted({*base_timeframes, *additional})
        existing_cache = await load_multi_timeframe_cache(
            redis_client,
            "ohlcv",
            payload.exchange_id,
            all_timeframes,
            payload.symbols,
        )
        updated = await update_multi_tf_ohlcv_cache(
            exchange,
            existing_cache,
            payload.symbols,
            dict(payload.config),
            limit=payload.limit,
            use_websocket=payload.use_websocket,
            force_websocket_history=payload.force_websocket_history,
            max_concurrent=payload.max_concurrent,
            notifier=None,
            priority_queue=None,
            additional_timeframes=payload.additional_timeframes,
        )
    finally:
        await _close_exchange(exchange)

    serialized = await store_multi_timeframe_cache(
        redis_client,
        "ohlcv",
        payload.exchange_id,
        updated,
        settings.cache_ttl_seconds,
    )
    updated_at = datetime.now(timezone.utc)
    event = {
        "type": "ohlcv_multi_update",
        "exchange": payload.exchange_id,
        "timeframes": list(serialized.keys()),
        "symbols": list({sym for tf in serialized.values() for sym in tf.keys()}),
        "updated_at": updated_at,
    }
    await redis_client.publish(settings.ohlcv_channel, _serialize_event(event))
    return MultiTimeframeResponse(timeframes=serialized, updated_at=updated_at)


@app.post("/regime/update", response_model=RegimeResponse)
async def update_regime(
    payload: RegimeUpdatePayload,
    redis_client: redis.Redis = Depends(get_redis_dependency),
    settings: Settings = Depends(get_settings_dependency),
) -> RegimeResponse:
    exchange = await _create_exchange(payload.exchange_id, payload.config)
    try:
        regime_tfs = payload.config.get("regime_timeframes", [])
        if not regime_tfs:
            return RegimeResponse(timeframes={}, updated_at=datetime.now(timezone.utc))
        existing_cache = await load_multi_timeframe_cache(
            redis_client,
            "regime",
            payload.exchange_id,
            regime_tfs,
            payload.symbols,
        )
        df_timeframes = payload.df_timeframes or regime_tfs
        df_map = await load_multi_timeframe_cache(
            redis_client,
            "ohlcv",
            payload.exchange_id,
            df_timeframes,
            payload.symbols,
        )
        updated = await update_regime_tf_cache(
            exchange,
            existing_cache,
            payload.symbols,
            dict(payload.config),
            limit=payload.limit,
            use_websocket=payload.use_websocket,
            force_websocket_history=payload.force_websocket_history,
            max_concurrent=payload.max_concurrent,
            notifier=None,
            df_map=df_map,
        )
    finally:
        await _close_exchange(exchange)

    serialized = await store_multi_timeframe_cache(
        redis_client,
        "regime",
        payload.exchange_id,
        updated,
        settings.cache_ttl_seconds,
    )
    updated_at = datetime.now(timezone.utc)
    event = {
        "type": "regime_update",
        "exchange": payload.exchange_id,
        "timeframes": list(serialized.keys()),
        "symbols": list({sym for tf in serialized.values() for sym in tf.keys()}),
        "updated_at": updated_at,
    }
    await redis_client.publish(settings.regime_channel, _serialize_event(event))
    return RegimeResponse(timeframes=serialized, updated_at=updated_at)


@app.post("/order-book/snapshot", response_model=OrderBookResponse)
async def order_book_snapshot(
    payload: OrderBookPayload,
    redis_client: redis.Redis = Depends(get_redis_dependency),
    settings: Settings = Depends(get_settings_dependency),
) -> OrderBookResponse:
    exchange = await _create_exchange(payload.exchange_id, payload.config)
    try:
        order_book = await fetch_order_book_async(
            exchange,
            payload.symbol,
            payload.depth,
        )
    finally:
        await _close_exchange(exchange)
    await store_order_book(
        redis_client,
        payload.exchange_id,
        payload.symbol,
        order_book or {},
        settings.cache_ttl_seconds,
    )
    updated_at = datetime.now(timezone.utc)
    event = {
        "type": "order_book_snapshot",
        "exchange": payload.exchange_id,
        "symbol": payload.symbol,
        "updated_at": updated_at,
    }
    await redis_client.publish(settings.order_book_channel, _serialize_event(event))
    return OrderBookResponse(symbol=payload.symbol, order_book=order_book, updated_at=updated_at)


@app.post("/timeframe/seconds", response_model=TimeframeSecondsResponse)
async def timeframe_seconds_endpoint(
    payload: TimeframeSecondsPayload,
) -> TimeframeSecondsResponse:
    # ``timeframe_seconds`` gracefully handles ``None`` exchange objects
    seconds = timeframe_seconds(None, payload.timeframe)
    return TimeframeSecondsResponse(timeframe=payload.timeframe, seconds=seconds)


@app.websocket("/ws/ohlcv")
async def websocket_ohlcv(
    websocket: WebSocket,
    exchange_id: str,
    symbol: str,
    timeframe: str = "1m",
    limit: int = 100,
):
    await websocket.accept()
    settings = get_settings()
    exchange = await _create_exchange(exchange_id, {})
    rate_limiter = AdaptiveRateLimiter(
        max_requests_per_minute=settings.websocket_rate_limit_per_minute,
    )
    try:
        interval = max(
            settings.websocket_min_interval_seconds,
            float(timeframe_seconds(exchange, timeframe)),
        )
    except Exception:
        interval = settings.websocket_min_interval_seconds

    try:
        while True:
            try:
                await rate_limiter.wait_if_needed()
                candles = await fetch_ohlcv_async(
                    exchange,
                    symbol,
                    timeframe,
                    limit,
                )
                rate_limiter.record_success()
                message = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": candles or [],
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                await websocket.send_text(json.dumps(message))
            except Exception as exc:  # pragma: no cover - runtime failures
                rate_limiter.record_error()
                logger.warning("WebSocket fetch failed for %s %s: %s", symbol, timeframe, exc)
                await asyncio.sleep(settings.websocket_min_interval_seconds)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for %s %s", symbol, timeframe)
    finally:
        await _close_exchange(exchange)


__all__ = ["app"]
