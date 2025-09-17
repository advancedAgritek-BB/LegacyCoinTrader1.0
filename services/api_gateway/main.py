from __future__ import annotations

import asyncio
import logging
from typing import Dict

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis import asyncio as redis_asyncio

from .auth import AuthManager, TokenPayload
from .config import GatewaySettings, ServiceRouteConfig, load_gateway_settings
from .proxy import ProxyGateway
from .rate_limiter import RateLimitResult, RateLimiter

LOGGER = logging.getLogger("api_gateway")


def create_app() -> FastAPI:
    settings = load_gateway_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    app = FastAPI(
        title="LegacyCoinTrader API Gateway",
        description="Unified entry point for LegacyCoinTrader microservices",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )

    state = GatewayState(settings=settings)
    app.state.gateway_state = state

    register_events(app, state)
    register_routes(app, state)

    return app


class GatewayState:
    """Holds long-lived resources for the FastAPI application."""

    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings
        self.redis: redis_asyncio.Redis | None = None
        self.http_client: httpx.AsyncClient | None = None
        self.auth_manager: AuthManager | None = None
        self.rate_limiter: RateLimiter | None = None
        self.proxy_gateway: ProxyGateway | None = None


async def get_state(request: Request) -> GatewayState:
    return request.app.state.gateway_state


def register_events(app: FastAPI, state: GatewayState) -> None:
    @app.on_event("startup")
    async def startup_event() -> None:
        LOGGER.info("Starting API Gateway")
        state.redis = await _init_redis(state.settings)
        state.http_client = httpx.AsyncClient(timeout=state.settings.http_client_timeout)
        state.auth_manager = AuthManager(state.settings)
        state.rate_limiter = RateLimiter(
            state.redis,
            state.settings.default_rate_limit_per_minute,
            state.settings.rate_limit_window_seconds,
        )
        state.proxy_gateway = ProxyGateway(
            state.settings,
            state.http_client,
        )

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        LOGGER.info("Shutting down API Gateway")
        if state.http_client:
            await state.http_client.aclose()
        if state.redis:
            await state.redis.close()


async def _init_redis(settings: GatewaySettings) -> redis_asyncio.Redis | None:
    try:
        redis_client = redis_asyncio.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )
        await redis_client.ping()
        LOGGER.info("Connected to Redis at %s:%s", settings.redis_host, settings.redis_port)
        return redis_client
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Redis unavailable - using in-memory rate limiting: %s", exc)
        return None


def register_routes(app: FastAPI, state: GatewayState) -> None:
    @app.get("/health", tags=["Health"])
    async def gateway_health() -> JSONResponse:
        health_results: Dict[str, Dict[str, object]] = {}
        overall_healthy = True
        if state.proxy_gateway:
            tasks = [
                state.proxy_gateway.check_service_health(route)
                for route in state.settings.service_routes.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for route, result in zip(state.settings.service_routes.values(), results):
                if isinstance(result, Exception):
                    overall_healthy = False
                    health_results[route.name] = {
                        "healthy": False,
                        "error": str(result),
                        "service": route.name,
                    }
                else:
                    overall_healthy = overall_healthy and result.get("healthy", False)
                    health_results[route.name] = result

        status = "healthy" if overall_healthy else "degraded"
        payload = {
            "status": status,
            "redis": {
                "healthy": state.rate_limiter.uses_redis if state.rate_limiter else False,
            },
            "services": health_results,
            "routes": {
                name: {
                    "prefix": route.prefix,
                    "target": route.url,
                    "auth": route.allowed_auth_modes,
                    "rate_limit_per_minute": route.rate_limit_per_minute,
                }
                for name, route in state.settings.service_routes.items()
            },
        }
        return JSONResponse(payload)

    @app.get("/routes", tags=["Configuration"])
    async def list_routes() -> Dict[str, object]:
        return {
            name: {
                "prefix": route.prefix,
                "target": route.url,
                "auth": route.allowed_auth_modes,
                "rate_limit_per_minute": route.rate_limit_per_minute,
            }
            for name, route in state.settings.service_routes.items()
        }

    def _register_proxy_endpoint(route: ServiceRouteConfig) -> None:
        async def auth_and_rate_limit(
            request: Request,
            state: GatewayState = Depends(get_state),
        ) -> TokenPayload:
            assert state.auth_manager is not None
            assert state.rate_limiter is not None

            token = await state.auth_manager.authenticate_request(
                request, route.allowed_auth_modes
            )
            identifier = f"{route.name}:{token.rate_limit_key}"
            result = await state.rate_limiter.check(identifier, route.rate_limit_per_minute)
            if not result.allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(result.retry_after or state.settings.rate_limit_window_seconds),
                        "X-RateLimit-Limit": str(result.limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )
            request.state.rate_limit_result = result
            return token

        async def proxy_endpoint(
            request: Request,
            path: str = "",
            token: TokenPayload = Depends(auth_and_rate_limit),
            state: GatewayState = Depends(get_state),
        ):
            assert state.proxy_gateway is not None
            response = await state.proxy_gateway.proxy_request(route, path, request, token)
            rate_limit_result: RateLimitResult = getattr(request.state, "rate_limit_result", None)
            if rate_limit_result:
                response.headers["X-RateLimit-Limit"] = str(rate_limit_result.limit)
                response.headers["X-RateLimit-Remaining"] = str(rate_limit_result.remaining)
                response.headers["X-RateLimit-Reset"] = str(rate_limit_result.reset_after)
            return response

        app.router.add_api_route(
            route.prefix,
            proxy_endpoint,
            methods=list(route.methods),
        )
        app.router.add_api_route(
            f"{route.prefix}/{{path:path}}",
            proxy_endpoint,
            methods=list(route.methods),
        )

    for service_route in state.settings.service_routes.values():
        _register_proxy_endpoint(service_route)


app = create_app()

