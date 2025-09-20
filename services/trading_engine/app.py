from __future__ import annotations

"""FastAPI application exposing the trading engine endpoints."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Union

from fastapi import Depends, FastAPI, HTTPException, Request, status
import redis.asyncio as redis

from libs.bootstrap import load_config

from services.common.tenant import get_tenant_registry, get_tenant_context, TenantContextMiddleware
from services.monitoring.config import get_monitoring_settings
from services.monitoring.logging_compat import configure_logging
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

# Import enhanced scanning integration
try:
    import sys
    from pathlib import Path
    # Add crypto_bot to path
    crypto_bot_path = Path(__file__).parent.parent.parent / "crypto_bot"
    if str(crypto_bot_path) not in sys.path:
        sys.path.insert(0, str(crypto_bot_path))
    
    from crypto_bot.enhanced_scan_integration import start_enhanced_scan_integration, stop_enhanced_scan_integration
    ENHANCED_SCANNING_AVAILABLE = True
except ImportError as exc:
    logger.warning(f"Enhanced scan integration not available: {exc}")
    ENHANCED_SCANNING_AVAILABLE = False


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

    config = load_config()
    services = build_service_container()

    risk_manager = build_risk_client(config.get("risk", {}))
    logger.info(f"Building paper wallet with config execution_mode: {config.get('execution_mode')}")
    paper_wallet = build_paper_wallet_client(config)
    logger.info(f"Paper wallet created: {paper_wallet}")
    position_guard = build_position_guard_client(config)

    trade_manager = None
    portfolio_service = getattr(services, "portfolio", None)
    if portfolio_service is not None:
        try:
            trade_manager = portfolio_service.get_trade_manager()
        except Exception:  # pragma: no cover - defensive
            logger.warning("Trade manager initialisation failed", exc_info=True)

    interface = TradingEngineInterface(
        services=services,
        config=config,
        risk_client=risk_manager,
        paper_wallet=paper_wallet,
        position_guard=position_guard,
        trade_manager=trade_manager,
    )

    scheduler = CycleScheduler(
        interface_factory=lambda ctx: interface,
        state_store=state_store,
        default_interval=settings.default_cycle_interval,
    )

    app.state.redis = redis_client
    app.state.scheduler = scheduler
    app.state.settings = settings
    app.state.trading_interface = interface
    app.state.service_container = services

    auto_start_env = os.getenv("AUTO_START_TRADING") or os.getenv("AUTO_START_TRADING_ENGINE")
    config_flag = config.get("auto_start_trading")
    if config_flag is None:
        should_auto_start = True
    else:
        should_auto_start = str(config_flag).strip().lower() not in {"0", "false", "no"}
    if auto_start_env is not None:
        should_auto_start = auto_start_env.strip().lower() not in {"0", "false", "no"}

    # Start enhanced scan integration if available
    enhanced_integration_started = False
    if ENHANCED_SCANNING_AVAILABLE:
        try:
            logger.info("Starting enhanced scan integration...")
            await start_enhanced_scan_integration(config)
            enhanced_integration_started = True
            logger.info("Enhanced scan integration started successfully")
        except Exception as exc:
            logger.warning(f"Failed to start enhanced scan integration: {exc}")

    if should_auto_start:
        try:
            # Get default tenant context for auto-startup
            registry = get_tenant_registry()
            tenant_context = registry.get("alpha")  # Use alpha as default tenant

            persisted_state = await state_store.load_state()
            interval = persisted_state.interval_seconds or settings.default_cycle_interval
            immediate = persisted_state.last_run_completed_at is None
            await scheduler.start(
                tenant_context,
                interval_seconds=interval,
                immediate=immediate,
                metadata={
                    "source": "auto_start",
                    "resume": bool(persisted_state.running),
                },
            )
        except Exception:  # pragma: no cover - defensive startup
            logger.exception("Failed to auto-start trading cycle scheduler")

    try:
        yield
    finally:
        # Stop enhanced scan integration if it was started
        if enhanced_integration_started and ENHANCED_SCANNING_AVAILABLE:
            try:
                logger.info("Stopping enhanced scan integration...")
                await stop_enhanced_scan_integration()
                logger.info("Enhanced scan integration stopped")
            except Exception as exc:
                logger.warning(f"Error stopping enhanced scan integration: {exc}")
        
        await scheduler.shutdown()
        await interface.shutdown()
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

# Add tenant context middleware
app.add_middleware(TenantContextMiddleware)

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
    scheduler: CycleScheduler | None = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Scheduler unavailable")
    return {"status": "ready", "interval": str(settings.default_cycle_interval)}


@app.post("/cycles/start", response_model=CycleStateResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_cycle(
    payload: StartCycleRequest,
    scheduler: CycleScheduler = Depends(get_scheduler),
    tenant_context = Depends(get_tenant_context),
) -> CycleStateResponse:
    state = await scheduler.start(
        tenant_context,
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
    payload: Union[StartCycleRequest, None] = None,
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
async def service_settings(settings: Settings = Depends(get_settings_dependency)) -> Dict[str, Union[str, int]]:
    return {
        "app_name": settings.app_name,
        "default_cycle_interval": settings.default_cycle_interval,
        "redis": settings.redis_dsn(),
    }


__all__ = ["app"]
