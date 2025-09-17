from __future__ import annotations

"""FastAPI application exposing the trading engine endpoints."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException, Request, status
import redis.asyncio as redis

from libs.bootstrap import load_config

from services.common.tenant import (
    TenantContext,
    TenantContextMiddleware,
    TenantLimitError,
    get_tenant_context,
)

from services.monitoring.config import get_monitoring_settings
from services.monitoring.logging import configure_logging
from services.monitoring.instrumentation import instrument_fastapi_app

from .config import Settings, get_settings
from .interface import TradingEngineInterface
from .redis_state import RedisCycleStateStore
from .scheduler import CycleScheduler
from .schemas import CycleStateResponse, RunCycleResponse, StartCycleRequest

from .clients import (
    build_paper_wallet_client,
    build_position_guard_client,
    build_risk_client,
    build_service_container,
)


monitoring_settings = get_monitoring_settings().for_service(get_settings().app_name)
monitoring_settings.metrics.default_labels.setdefault("component", "trading-engine")
configure_logging(monitoring_settings)

logger = logging.getLogger(__name__)


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

    state_store = RedisCycleStateStore(redis_client, key_prefix=settings.state_key_prefix)
    await state_store.ensure_defaults(settings.default_cycle_interval)

    base_config = load_config()

    def build_interface_for_tenant(tenant: TenantContext) -> TradingEngineInterface:
        tenant_config = tenant.apply_config(base_config)
        services = build_service_container()
        risk_manager = build_risk_client(tenant_config.get("risk", {}))
        paper_wallet = build_paper_wallet_client(tenant_config)
        position_guard = build_position_guard_client(tenant_config)
        trade_manager = None
        portfolio_service = getattr(services, "portfolio", None)
        if portfolio_service is not None:
            try:
                trade_manager = portfolio_service.get_trade_manager()
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "Trade manager initialisation failed for tenant %s",
                    tenant.tenant_id,
                    exc_info=True,
                )
        return TradingEngineInterface(
            services=services,
            config=tenant_config,
            risk_client=risk_manager,
            paper_wallet=paper_wallet,
            position_guard=position_guard,
            trade_manager=trade_manager,
        )

    scheduler = CycleScheduler(
        interface_factory=build_interface_for_tenant,
        state_store=state_store,
        default_interval=settings.default_cycle_interval,
    )

    app.state.redis = redis_client
    app.state.scheduler = scheduler
    app.state.settings = settings
    app.state.base_config = base_config

    try:
        yield
    finally:
        await scheduler.shutdown()
        await redis_client.close()


def get_scheduler(request: Request) -> CycleScheduler:
    scheduler: CycleScheduler | None = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Scheduler not ready")
    return scheduler


def get_settings_dependency(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = get_settings()
    return settings


app = FastAPI(title="Trading Engine", lifespan=lifespan)
instrument_fastapi_app(app, settings=monitoring_settings)
app.add_middleware(TenantContextMiddleware)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}


@app.get("/health/tenant", response_model=CycleStateResponse)
async def tenant_health(
    scheduler: CycleScheduler = Depends(get_scheduler),
    tenant: TenantContext = Depends(get_tenant_context),
) -> CycleStateResponse:
    state = await scheduler.get_state(tenant)
    return CycleStateResponse.from_state(state)


@app.get("/readiness")
async def readiness(request: Request, settings: Settings = Depends(get_settings_dependency)) -> Dict[str, str]:
    redis_client: redis.Redis = request.app.state.redis
    try:
        await redis_client.ping()
    except Exception as exc:  # pragma: no cover - readiness should flag failure
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis unavailable")
    scheduler: CycleScheduler | None = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Scheduler unavailable")
    return {"status": "ready", "interval": str(settings.default_cycle_interval)}


@app.post("/cycles/start", response_model=CycleStateResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_cycle(
    payload: StartCycleRequest,
    scheduler: CycleScheduler = Depends(get_scheduler),
    tenant: TenantContext = Depends(get_tenant_context),
) -> CycleStateResponse:
    try:
        state = await scheduler.start(
            tenant,
            interval_seconds=payload.interval_seconds,
            immediate=payload.immediate,
            metadata=payload.metadata,
            risk_allocation=payload.risk_allocation,
        )
    except TenantLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    return CycleStateResponse.from_state(state)


@app.post("/cycles/stop", response_model=CycleStateResponse)
async def stop_cycle(
    scheduler: CycleScheduler = Depends(get_scheduler),
    tenant: TenantContext = Depends(get_tenant_context),
) -> CycleStateResponse:
    state = await scheduler.stop(tenant)
    return CycleStateResponse.from_state(state)


@app.get("/cycles/status", response_model=CycleStateResponse)
async def cycle_status(
    scheduler: CycleScheduler = Depends(get_scheduler),
    tenant: TenantContext = Depends(get_tenant_context),
) -> CycleStateResponse:
    state = await scheduler.get_state(tenant)
    return CycleStateResponse.from_state(state)


@app.post("/cycles/run", response_model=RunCycleResponse)
async def run_cycle(
    scheduler: CycleScheduler = Depends(get_scheduler),
    payload: StartCycleRequest | None = None,
    tenant: TenantContext = Depends(get_tenant_context),
) -> RunCycleResponse:
    metadata = dict(payload.metadata if payload else {})
    risk_allocation = payload.risk_allocation if payload else None
    try:
        result = await scheduler.run_once(
            tenant,
            metadata=metadata,
            risk_allocation=risk_allocation,
        )
    except TenantLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
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
