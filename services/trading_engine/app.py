from __future__ import annotations

"""FastAPI application exposing the trading engine endpoints."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
import redis.asyncio as redis

from libs.execution import get_exchange
from libs.models import OpenPositionGuard, PaperWallet
from libs.risk import RiskManager
from libs.bootstrap import create_service_container, load_config

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
    services = create_service_container()

    try:
        exchange, ws_client = get_exchange(config)
    except Exception:  # pragma: no cover - connectivity issues are surfaced via logs
        logger.warning("Exchange initialisation failed", exc_info=True)
        exchange = None
        ws_client = None

    try:
        risk_manager: Optional[RiskManager] = RiskManager.from_config(config.get("risk", {}))
    except Exception:  # pragma: no cover - risk manager is optional
        logger.warning("Risk manager initialisation failed", exc_info=True)
        risk_manager = None

    position_guard = OpenPositionGuard(
        config.get("max_open_trades")
        or config.get("paper_wallet", {}).get("max_open_trades", 5)
    )

    paper_wallet: Optional[PaperWallet] = None
    if str(config.get("execution_mode", "dry_run")).lower() == "dry_run":
        wallet_cfg = config.get("paper_wallet", {})
        initial_balance = wallet_cfg.get("initial_balance")
        if initial_balance is None:
            initial_balance = config.get("risk", {}).get("starting_balance", 0.0)
        try:
            paper_wallet = PaperWallet(
                float(initial_balance or 0.0),
                max_open_trades=int(wallet_cfg.get("max_open_trades", 5)),
                allow_short=bool(wallet_cfg.get("allow_short", True)),
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("Paper wallet initialisation failed", exc_info=True)
            paper_wallet = None

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
        exchange=exchange,
        ws_client=ws_client,
        risk_manager=risk_manager,
        paper_wallet=paper_wallet,
        position_guard=position_guard,
        trade_manager=trade_manager,
    )

    scheduler = CycleScheduler(
        interface=interface,
        state_store=state_store,
        default_interval=settings.default_cycle_interval,
    )

    app.state.redis = redis_client
    app.state.scheduler = scheduler
    app.state.settings = settings
    app.state.trading_interface = interface
    app.state.service_container = services

    try:
        yield
    finally:
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
    payload: StartCycleRequest | None = None,
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
