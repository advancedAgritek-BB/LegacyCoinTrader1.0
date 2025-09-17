from __future__ import annotations

"""FastAPI application exposing the trading engine endpoints."""

"""FastAPI application exposing the trading engine endpoints."""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
import redis.asyncio as redis

from crypto_bot.solana.discovery_feed import FeedSettings, SolanaDiscoveryFeed
from crypto_bot.startup_utils import create_service_container, load_config
from crypto_bot.services.interfaces import ServiceContainer
from services.monitoring.config import get_monitoring_settings
from services.monitoring.logging import configure_logging
from services.monitoring.instrumentation import instrument_fastapi_app

from .config import Settings, get_settings
from .interface import TradingEngineInterface
from .redis_state import RedisCycleStateStore
from .scheduler import CycleScheduler
from .schemas import CycleStateResponse, RunCycleResponse, StartCycleRequest

monitoring_settings = get_monitoring_settings().for_service(get_settings().app_name)
monitoring_settings.metrics.default_labels.setdefault("component", "trading-engine")
configure_logging(monitoring_settings)

logger = logging.getLogger(__name__)


def load_trading_config() -> Dict[str, Any]:
    """Wrapper for loading the trading configuration (facilitates testing overrides)."""

    return load_config()


def build_service_container() -> ServiceContainer:
    """Construct the default service container used by the trading engine."""

    return create_service_container()


async def _maybe_start_solana_feed(config: Mapping[str, Any]) -> Optional[SolanaDiscoveryFeed]:
    if not isinstance(config, Mapping) or not config.get("enabled", False):
        return None
    settings = FeedSettings.from_dict(dict(config))
    feed = SolanaDiscoveryFeed(settings)
    try:
        await feed.start()
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to start Solana discovery feed")
        return None
    return feed


async def _shutdown_services(container: ServiceContainer) -> None:
    async def maybe_close(component: Any) -> None:
        if component is None:
            return
        for method_name in ("aclose", "close"):
            method = getattr(component, method_name, None)
            if method is None:
                continue
            try:
                result = method()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("Service component %s failed to close", component, exc_info=True)
            else:
                break

    await maybe_close(getattr(container, "market_data", None))
    await maybe_close(getattr(container, "strategy", None))
    await maybe_close(getattr(container, "execution", None))
    await maybe_close(getattr(container, "portfolio", None))
    await maybe_close(getattr(container, "token_discovery", None))


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
    except Exception as exc:  # pragma: no cover - startup failure is fatal
        logger.error("Unable to connect to Redis: %s", exc)
        await redis_client.close()
        raise

    config_data = load_trading_config()
    services = build_service_container()
    solana_feed = await _maybe_start_solana_feed(config_data.get("token_discovery_feed", {}))

    state_store = RedisCycleStateStore(redis_client, key_prefix=settings.state_key_prefix)
    await state_store.ensure_defaults(settings.default_cycle_interval)

    interface = TradingEngineInterface(
        service_container=services,
        redis_client=redis_client,
        solana_feed=solana_feed,
        base_config=config_data,
        config_loader=load_trading_config,
    )

    scheduler = CycleScheduler(
        interface=interface,
        state_store=state_store,
        default_interval=settings.default_cycle_interval,
    )

    app.state.redis = redis_client
    app.state.scheduler = scheduler
    app.state.settings = settings
    app.state.services = services
    app.state.solana_feed = solana_feed

    try:
        yield
    finally:
        await scheduler.shutdown()
        await redis_client.close()
        if solana_feed is not None:
            try:
                await solana_feed.close()
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("Failed to close Solana discovery feed", exc_info=True)
        await _shutdown_services(services)


def get_scheduler(request: Request) -> CycleScheduler:
    scheduler: Optional[CycleScheduler] = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Scheduler not ready")
    return scheduler


def get_settings_dependency(request: Request) -> Settings:
    settings: Optional[Settings] = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = get_settings()
    return settings


app = FastAPI(title="Trading Engine", lifespan=lifespan)
instrument_fastapi_app(app, settings=monitoring_settings)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}


@app.get("/readiness")
async def readiness(request: Request, settings: Settings = Depends(get_settings_dependency)) -> Dict[str, str]:
    redis_client: redis.Redis = request.app.state.redis
    try:
        await redis_client.ping()
    except Exception as exc:  # pragma: no cover - readiness should flag failure
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis unavailable")
    scheduler: Optional[CycleScheduler] = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Scheduler unavailable")
    return {"status": "ready", "interval": str(settings.default_cycle_interval)}


@app.post("/cycles/start", response_model=CycleStateResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_cycle(
    payload: StartCycleRequest,
    scheduler: CycleScheduler = Depends(get_scheduler),
) -> CycleStateResponse:
    state = await scheduler.start(
        interval_seconds=payload.interval_seconds,
        immediate=payload.immediate,
        metadata=payload.metadata,
    )
    return CycleStateResponse.from_state(state)


@app.post("/cycles/stop", response_model=CycleStateResponse)
async def stop_cycle(scheduler: CycleScheduler = Depends(get_scheduler)) -> CycleStateResponse:
    state = await scheduler.stop()
    return CycleStateResponse.from_state(state)


@app.get("/cycles/status", response_model=CycleStateResponse)
async def cycle_status(scheduler: CycleScheduler = Depends(get_scheduler)) -> CycleStateResponse:
    state = await scheduler.get_state()
    return CycleStateResponse.from_state(state)


@app.post("/cycles/run", response_model=RunCycleResponse)
async def run_cycle(
    scheduler: CycleScheduler = Depends(get_scheduler),
    payload: Optional[StartCycleRequest] = None,
) -> RunCycleResponse:
    metadata = dict(payload.metadata if payload else {})
    result = await scheduler.run_once(metadata=metadata)
    started_at = result.started_at or datetime.now(timezone.utc)
    completed_at = result.completed_at or datetime.now(timezone.utc)
    merged_metadata = {**metadata, **result.metadata}
    return RunCycleResponse(
        status="completed",
        timings=result.timings,
        started_at=started_at,
        completed_at=completed_at,
        duration=result.duration,
        metadata=merged_metadata,
    )


@app.get("/settings")
async def service_settings(settings: Settings = Depends(get_settings_dependency)) -> Dict[str, str | int]:
    return {
        "app_name": settings.app_name,
        "default_cycle_interval": settings.default_cycle_interval,
        "redis": settings.redis_dsn(),
    }


__all__ = ["app"]
