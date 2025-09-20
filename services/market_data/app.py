from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
import redis.asyncio as redis
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging_compat import configure_logging

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
    CANDLE_COLUMNS,
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
monitoring_settings = monitoring_settings.clone(log_level=service_settings.log_level)
monitoring_settings.metrics.default_labels.setdefault("component", "market-data")
configure_logging(monitoring_settings)

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _serialize_event(data: Mapping[str, Any]) -> str:
    return json.dumps(dict(data), default=_json_default)


def _candles_from_dataframe(df: Optional[pd.DataFrame], limit: int) -> list[list[float]]:
    """Convert a cached OHLCV dataframe into a JSON-friendly list."""

    if df is None or df.empty:
        return []

    ordered = df.sort_values("timestamp").tail(limit)
    return [
        [
            int(row.timestamp),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        ]
        for row in ordered.itertuples(index=False)
    ]


def _dataframe_from_candles(candles: Iterable[Iterable[float]]) -> pd.DataFrame:
    """Normalise raw candle data returned by the exchange."""

    data = list(candles or [])
    if not data:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    df = pd.DataFrame(data, columns=CANDLE_COLUMNS)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(int)
    return df


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
    settings: Optional[Settings] = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = get_settings()
    return settings


def get_redis_dependency(request: Request) -> redis.Redis:
    redis_client: Optional[redis.Redis] = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis unavailable")
    return redis_client


async def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client if available, return None if not."""
    try:
        settings = get_settings()
        redis_client = redis.from_url(
            settings.redis_dsn(),
            encoding="utf-8",
            decode_responses=True,
            health_check_interval=30,
        )
        await redis_client.ping()
        return redis_client
    except Exception:
        return None


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


@app.get("/test")
async def test_endpoint() -> Dict[str, str]:
    """Simple test endpoint to verify routing is working."""
    return {"message": "Market data service is working", "timestamp": str(datetime.now(timezone.utc))}


@app.post("/batch-candles")
async def batch_candles(request: Request) -> Dict[str, Any]:
    """Get candles data for multiple symbols in a single request."""
    try:
        data = await request.json()
        symbols = data.get("symbols", [])
        limit = data.get("limit", 50)
        timeframe = data.get("timeframe", "5m")
        exchange = data.get("exchange", "kraken")
        force_fresh = data.get("force_fresh", False)

        if not symbols:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Symbols list is required")

        logger.info(f"Batch fetching candles for {len(symbols)} symbols from {exchange}")

        results: Dict[str, Any] = {}
        settings = get_settings_dependency(request)
        redis_client = getattr(request.app.state, "redis", None)

        # Seed results with cached candles when available (unless force_fresh is True)
        cached_frames = {}
        if redis_client and not force_fresh:
            try:
                cached_frames = await load_timeframe_cache(
                    redis_client,
                    "ohlcv",
                    exchange,
                    timeframe,
                    symbols,
                )
            except Exception as cache_exc:
                logger.debug("Failed to load cached candles: %s", cache_exc)
                cached_frames = {}

            for symbol, frame in (cached_frames or {}).items():
                candles = _candles_from_dataframe(frame, limit)
                if not candles:
                    continue
                results[symbol] = {
                    "symbol": symbol.upper(),
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "candles": candles,
                    "count": len(candles),
                    "source": "cache",
                }

        symbols_to_fetch = [sym for sym in symbols if sym not in results]
        frames_to_store: Dict[str, pd.DataFrame] = {}
        exchange_obj = None

        if symbols_to_fetch:
            try:
                exchange_obj, _ = get_exchange({
                    "exchange": exchange,
                    "use_websocket": False,
                })
            except Exception as exc:
                logger.error("Unable to create exchange %s: %s", exchange, exc)
                exchange_obj = None

        if not exchange_obj and symbols_to_fetch:
            logger.warning("Falling back to mock data for %d symbols (exchange unavailable)", len(symbols_to_fetch))
            for symbol in symbols_to_fetch:
                results[symbol] = _generate_fallback_chart_data(symbol, timeframe, limit)

        if exchange_obj:
            batch_size = 5
            for i in range(0, len(symbols_to_fetch), batch_size):
                batch_symbols = symbols_to_fetch[i:i + batch_size]
                for symbol in batch_symbols:
                    try:
                        candles_data = await asyncio.wait_for(
                            fetch_ohlcv_async(
                                exchange_obj,
                                symbol,
                                timeframe=timeframe,
                                limit=limit,
                            ),
                            timeout=15.0,
                        )

                        if not candles_data:
                            results[symbol] = {
                                "error": f"No data available for {symbol}",
                            }
                            continue

                        df = _dataframe_from_candles(candles_data)
                        candles = _candles_from_dataframe(df, limit)
                        if not candles:
                            results[symbol] = {
                                "error": f"Incomplete data for {symbol}",
                            }
                            continue

                        results[symbol] = {
                            "symbol": symbol.upper(),
                            "exchange": exchange,
                            "timeframe": timeframe,
                            "candles": candles,
                            "count": len(candles),
                            "source": "exchange",
                        }

                        frames_to_store[symbol] = df
                    except asyncio.TimeoutError as exc:
                        logger.warning("Timeout fetching candles for %s: %s", symbol, exc)
                        results[symbol] = _generate_fallback_chart_data(symbol, timeframe, limit)
                    except Exception as exc:
                        logger.error("Failed to fetch candles for %s: %s", symbol, exc)
                        results[symbol] = _generate_fallback_chart_data(symbol, timeframe, limit)

                if i + batch_size < len(symbols_to_fetch):
                    await asyncio.sleep(0.1)

        if frames_to_store and redis_client:
            try:
                await store_timeframe_cache(
                    redis_client,
                    "ohlcv",
                    exchange,
                    timeframe,
                    frames_to_store,
                    settings.cache_ttl_seconds,
                )
            except Exception as cache_exc:
                logger.debug("Unable to cache fetched candles: %s", cache_exc)

        if exchange_obj:
            await _close_exchange(exchange_obj)

        logger.info(
            "Batch fetch completed for %d symbols (%d cached, %d fetched)",
            len(symbols),
            len([r for r in results.values() if r.get("source") == "cache"]),
            len([r for r in results.values() if r.get("source") == "exchange"]),
        )
        return {
            "results": results,
            "total_symbols": len(symbols),
            "successful_fetches": len([r for r in results.values() if "error" not in r]),
            "failed_fetches": len([r for r in results.values() if "error" in r])
        }

    except Exception as exc:
        logger.error(f"Batch candles request failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch request failed: {str(exc)}"
        )




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


@app.get("/test-candles")
async def test_candles() -> Dict[str, Any]:
    """Test endpoint for candles."""
    return {
        "message": "Candles endpoint working",
        "data": [
            [1737148800000, 95000, 96000, 94000, 95500, 1250000],
            [1737149100000, 95500, 96500, 95000, 96200, 1180000],
        ]
    }


@app.get("/get-candles")
async def get_candles_data(symbol: str = "BTC/USD", limit: int = 50, timeframe: str = "5m", exchange: str = "kraken") -> Dict[str, Any]:
    """Get real candles data from exchange."""
    try:
        logger.info(f"Fetching real candles for {symbol} from {exchange}")

        # Get exchange instance
        exchange_obj, _ = get_exchange({
            'exchange': exchange,
            'use_websocket': False  # Use REST API for now
        })

        if not exchange_obj:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Exchange {exchange} not available"
            )

        # Fetch real OHLCV data
        candles_data = await fetch_ohlcv_async(
            exchange_obj,
            symbol,
            timeframe=timeframe,
            limit=limit
        )

        if not candles_data or len(candles_data) == 0:
            logger.warning(f"No candle data received for {symbol}")
            return {
                "symbol": symbol.upper(),
                "candles": [],
                "count": 0,
                "source": "exchange",
                "error": "No data available"
            }

        # Convert to expected format
        candles = []
        for candle in candles_data:
            if len(candle) >= 6:  # timestamp, open, high, low, close, volume
                candles.append([
                    int(candle[0]),  # timestamp
                    float(candle[1]),  # open
                    float(candle[2]),  # high
                    float(candle[3]),  # low
                    float(candle[4]),  # close
                    float(candle[5])   # volume
                ])

        logger.info(f"Successfully fetched {len(candles)} candles for {symbol}")

        return {
            "symbol": symbol.upper(),
            "exchange": exchange,
            "timeframe": timeframe,
            "candles": candles,
            "count": len(candles),
            "source": "exchange"
        }

    except Exception as exc:
        logger.error(f"Failed to fetch candles for {symbol}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to fetch real market data: {str(exc)}"
        )


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


def _generate_fallback_chart_data(symbol: str, timeframe: str, limit: int) -> Dict[str, Any]:
    """Generate fallback mock data for charts when live data is unavailable."""
    import time
    import random

    logger.info(f"Generating fallback chart data for {symbol}")

    # Generate mock candles with some realistic price movement
    base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 300
    candles = []
    current_time = int(time.time() * 1000)

    # Timeframe multipliers (in milliseconds)
    timeframe_ms = {
        "1m": 60000,
        "5m": 300000,
        "15m": 900000,
        "1h": 3600000,
        "4h": 14400000,
        "1d": 86400000
    }.get(timeframe, 300000)  # Default to 5m

    for i in range(limit):
        # Generate some realistic price movement
        price_change = random.uniform(-0.02, 0.02)  # -2% to +2%
        open_price = base_price * (1 + random.uniform(-0.1, 0.1))
        close_price = open_price * (1 + price_change)
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
        volume = random.uniform(1000, 10000)

        candle_time = current_time - (limit - i) * timeframe_ms

        candles.append([
            candle_time,  # timestamp
            round(open_price, 2),  # open
            round(high_price, 2),  # high
            round(low_price, 2),   # low
            round(close_price, 2), # close
            round(volume, 2)       # volume
        ])

        # Update base price for next candle
        base_price = close_price

    return {
        "symbol": symbol.upper(),
        "exchange": "mock",
        "timeframe": timeframe,
        "candles": candles,
        "count": len(candles),
        "source": "fallback",
        "note": "Using mock data - live data unavailable"
    }


__all__ = ["app"]
