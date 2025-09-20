"""FastAPI application exposing token discovery endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Request, status

try:
    from services.monitoring.config import get_monitoring_settings
    from services.monitoring.instrumentation import instrument_fastapi_app
    from services.monitoring.logging_compat import configure_logging
    MONITORING_AVAILABLE = True
except ImportError:
    # Fallback when monitoring service is not available
    MONITORING_AVAILABLE = False
    
    class MockMonitoringSettings:
        def __init__(self):
            self.log_level = "INFO"
            self.metrics = MockMetrics()
        
        def for_service(self, name):
            return self
        
        def clone(self, **kwargs):
            return self
    
    class MockMetrics:
        def __init__(self):
            self.default_labels = {}
    
    def get_monitoring_settings():
        return MockMonitoringSettings()
    
    def instrument_fastapi_app(app, settings=None):
        pass
    
    def configure_logging(settings):
        import logging
        logging.basicConfig(level=logging.INFO)

from config import Settings, get_settings
from publisher import DiscoveryPublisher
from scanner import TokenDiscoveryCoordinator
from schemas import (
    DiscoveryResponse,
    Opportunity,
    OpportunityResponse,
    ScanRequest,
    ScoreRequest,
    ScoreResponse,
    StatusResponse,
)

# Configure file logging for the token discovery service
import os
from pathlib import Path
from crypto_bot.utils.logger import setup_logger

# Ensure log directory exists
log_dir = Path("/app/crypto_bot/logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logger with file output for all token discovery related loggers
logger = setup_logger(
    "services.token_discovery",
    log_file=log_dir / "token_discovery.log",
    to_console=True
)

# Also setup for crypto_bot token discovery loggers
crypto_logger = setup_logger(
    "crypto_bot.solana",
    log_file=log_dir / "token_discovery.log",
    to_console=True
)

service_settings = get_settings()
service_identifier = service_settings.app_name.replace(" ", "-").lower()
monitoring_settings = get_monitoring_settings().for_service(service_identifier)
monitoring_settings = monitoring_settings.clone(log_level=service_settings.log_level)
monitoring_settings.metrics.default_labels.setdefault("component", "token-discovery")
configure_logging(monitoring_settings)

# Use the configured logger
logger = logging.getLogger("token_discovery")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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

    publisher = DiscoveryPublisher(redis_client, settings)
    coordinator = TokenDiscoveryCoordinator(settings, publisher)
    await coordinator.start()

    app.state.settings = settings
    app.state.redis = redis_client
    app.state.publisher = publisher
    app.state.coordinator = coordinator

    try:
        yield
    finally:
        await coordinator.shutdown()
        await publisher.close()
        await redis_client.close()


def get_settings_dependency(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = get_settings()
    return settings


def get_coordinator(request: Request) -> TokenDiscoveryCoordinator:
    coordinator: TokenDiscoveryCoordinator | None = getattr(
        request.app.state, "coordinator", None
    )
    if coordinator is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return coordinator


def get_redis_client(request: Request) -> redis.Redis:
    redis_client: redis.Redis | None = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return redis_client


def _opportunity_from_dict(data: dict[str, Any]) -> Opportunity:
    return Opportunity(
        token=str(data.get("symbol") or data.get("token")),
        score=float(data.get("score", 0.0)),
        source=str(data.get("source", "enhanced")),
        metadata={
            key: value
            for key, value in data.items()
            if key not in {"symbol", "token", "score", "source"}
        },
    )


app = FastAPI(title="Token Discovery Service", lifespan=lifespan)
instrument_fastapi_app(app, settings=monitoring_settings)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
async def status_endpoint(
    coordinator: TokenDiscoveryCoordinator = Depends(get_coordinator),
    settings: Settings = Depends(get_settings_dependency),
    redis_client: redis.Redis = Depends(get_redis_client),
) -> StatusResponse:
    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:  # pragma: no cover - defensive fallback
        redis_ok = False

    kafka_ok = bool(settings.kafka_enabled)
    stats = coordinator.get_status()
    return StatusResponse(
        status="ok",
        redis_connected=redis_ok,
        kafka_connected=kafka_ok,
        last_basic_scan=stats.get("last_basic_scan"),
        last_enhanced_scan=stats.get("last_enhanced_scan"),
        tokens_cached=stats.get("tokens_cached", 0),
        opportunities_cached=stats.get("opportunities_cached", 0),
        last_cex_scan=stats.get("last_cex_scan"),
        cex_tokens_cached=stats.get("cex_tokens_cached", 0),
    )


@app.post("/scan/basic", response_model=DiscoveryResponse)
async def trigger_basic_scan(
    request: ScanRequest,
    coordinator: TokenDiscoveryCoordinator = Depends(get_coordinator),
) -> DiscoveryResponse:
    tokens = await coordinator.run_basic_scan(limit=request.limit)
    return DiscoveryResponse(
        tokens=tokens,
        metadata={"source": request.source or "basic", "count": len(tokens)},
    )


@app.post("/scan/enhanced", response_model=OpportunityResponse)
async def trigger_enhanced_scan(
    request: ScanRequest,
    coordinator: TokenDiscoveryCoordinator = Depends(get_coordinator),
) -> OpportunityResponse:
    opportunities = await coordinator.run_enhanced_scan(limit=request.limit)
    payload = [_opportunity_from_dict(item) for item in opportunities]
    return OpportunityResponse(
        opportunities=payload,
        metadata={"source": request.source or "enhanced", "count": len(payload)},
    )


@app.post("/scan/cex", response_model=DiscoveryResponse)
async def trigger_cex_scan(
    request: ScanRequest,
    coordinator: TokenDiscoveryCoordinator = Depends(get_coordinator),
    settings: Settings = Depends(get_settings_dependency),
) -> DiscoveryResponse:
    tokens = await coordinator.run_cex_scan(limit=request.limit)
    listings = await coordinator.get_latest_cex_listings()
    exchange = settings.cex_exchange
    return DiscoveryResponse(
        tokens=tokens,
        metadata={
            "source": request.source or f"cex:{exchange.lower()}",
            "count": len(tokens),
            "exchange": exchange,
            "listings": listings,
        },
    )


@app.get("/tokens/latest", response_model=DiscoveryResponse)
async def latest_tokens(
    coordinator: TokenDiscoveryCoordinator = Depends(get_coordinator),
) -> DiscoveryResponse:
    tokens = await coordinator.get_latest_tokens()
    return DiscoveryResponse(tokens=tokens, metadata={"count": len(tokens)})


@app.get("/cex/latest", response_model=DiscoveryResponse)
async def latest_cex_tokens(
    coordinator: TokenDiscoveryCoordinator = Depends(get_coordinator),
    settings: Settings = Depends(get_settings_dependency),
) -> DiscoveryResponse:
    tokens = await coordinator.get_latest_cex_tokens()
    listings = await coordinator.get_latest_cex_listings()
    return DiscoveryResponse(
        tokens=tokens,
        metadata={
            "count": len(tokens),
            "exchange": settings.cex_exchange,
            "listings": listings,
        },
    )


@app.get("/opportunities/top", response_model=OpportunityResponse)
async def top_opportunities(
    limit: int = 10,
    coordinator: TokenDiscoveryCoordinator = Depends(get_coordinator),
) -> OpportunityResponse:
    opportunities = await coordinator.get_latest_opportunities(limit=limit)
    payload = [_opportunity_from_dict(item) for item in opportunities]
    return OpportunityResponse(opportunities=payload, metadata={"count": len(payload)})


@app.post("/opportunities/score", response_model=ScoreResponse)
async def score_tokens(
    request: ScoreRequest,
    coordinator: TokenDiscoveryCoordinator = Depends(get_coordinator),
) -> ScoreResponse:
    opportunities = await coordinator.score_tokens(request.tokens)
    payload = [_opportunity_from_dict(item) for item in opportunities]
    return ScoreResponse(opportunities=payload, metadata={"count": len(payload)})


__all__ = ["app"]
