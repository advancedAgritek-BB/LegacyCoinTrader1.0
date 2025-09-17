"""FastAPI application for the strategy engine service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable

import pandas as pd
import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Request, status

from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging import configure_logging

from libs.services.interfaces import (
    StrategyBatchRequest,
    StrategyEvaluationPayload,
    StrategyEvaluationResult,
)

from .config import get_settings
from .engine import StrategyEngine
from .queue import MarketDataSubscriber
from .schemas import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    EvaluationResultModel,
    FusedSignalModel,
    HealthResponse,
    RankedSignalModel,
)
from .storage import ModelRegistry

service_settings = get_settings()
monitoring_settings = get_monitoring_settings().for_service(service_settings.app_name)
monitoring_settings = monitoring_settings.model_copy(
    update={"log_level": service_settings.log_level}
)
monitoring_settings.metrics.default_labels.setdefault("component", "strategy-engine")
configure_logging(monitoring_settings)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.getLogger().setLevel(settings.log_level.upper())

    redis_client = redis.from_url(
        settings.redis_dsn(),
        encoding="utf-8",
        decode_responses=False,
        health_check_interval=30,
    )
    try:
        await redis_client.ping()
    except Exception as exc:  # pragma: no cover - startup failure is fatal
        logger.error("Unable to connect to Redis: %s", exc)
        await redis_client.close()
        raise

    model_store = ModelRegistry(redis_client, settings.model_key_prefix)
    engine = StrategyEngine(redis_client, settings, model_store)
    subscriber = MarketDataSubscriber(redis_client, settings.market_data_channel, engine.handle_market_event)
    await subscriber.start()

    app.state.redis = redis_client
    app.state.engine = engine
    app.state.settings = settings
    app.state.subscriber = subscriber

    try:
        yield
    finally:
        await subscriber.stop()
        await redis_client.close()


def get_engine(request: Request) -> StrategyEngine:
    engine: StrategyEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine unavailable")
    return engine


app = FastAPI(title="Strategy Engine", lifespan=lifespan)
instrument_fastapi_app(app, settings=monitoring_settings)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/readiness", response_model=HealthResponse)
async def readiness(request: Request) -> HealthResponse:
    redis_client: redis.Redis | None = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis unavailable")
    try:
        await redis_client.ping()
    except Exception as exc:  # pragma: no cover - readiness should flag failure
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis unavailable")
    return HealthResponse(status="ready")


@app.post("/evaluate/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(
    payload: BatchEvaluationRequest,
    engine: StrategyEngine = Depends(get_engine),
) -> BatchEvaluationResponse:
    request = _to_batch_request(payload)
    response = await engine.evaluate_batch(request)
    return _to_batch_response(response)


def _to_batch_request(payload: BatchEvaluationRequest) -> StrategyBatchRequest:
    items: list[StrategyEvaluationPayload] = []
    for entry in payload.requests:
        frames = {tf: _build_dataframe(candles) for tf, candles in entry.timeframes.items()}
        items.append(
            StrategyEvaluationPayload(
                symbol=entry.symbol,
                regime=entry.regime,
                mode=entry.mode,
                timeframes=frames,
                config=entry.config,
                metadata=entry.metadata,
            )
        )
    return StrategyBatchRequest(items=tuple(items), metadata=payload.metadata)


def _to_batch_response(batch: StrategyBatchResponse) -> BatchEvaluationResponse:
    results = [_to_result_model(result) for result in batch.results]
    return BatchEvaluationResponse(results=results, errors=list(batch.errors))


def _to_result_model(result: StrategyEvaluationResult) -> EvaluationResultModel:
    fused = None
    if result.fused_score is not None and result.fused_direction is not None:
        fused = FusedSignalModel(score=result.fused_score, direction=result.fused_direction)
    ranked = [
        RankedSignalModel(strategy=signal.strategy, score=signal.score, direction=signal.direction)
        for signal in result.ranked_signals
    ]
    return EvaluationResultModel(
        symbol=result.symbol,
        regime=result.regime,
        strategy=result.strategy,
        score=result.score,
        direction=result.direction,
        atr=result.atr,
        fused_signal=fused,
        ranked_signals=ranked,
        metadata=dict(result.metadata),
        cached=result.cached,
    )


def _build_dataframe(candles: Iterable[Any]) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for candle in candles:
        data = candle.model_dump() if hasattr(candle, "model_dump") else dict(candle)
        rows.append(data)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.set_index("timestamp").sort_index()
    return frame


__all__ = ["app"]
