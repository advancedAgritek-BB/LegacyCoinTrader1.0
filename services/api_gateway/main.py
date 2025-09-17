from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis import asyncio as redis_asyncio

from services.api_gateway.contracts import (
    ApiKeyValidationRequest,
    ApiKeyValidationResponse,
    AuthenticationEvent,
    AuthenticationPayload,
    HTTP_CONTRACT,
    PasswordRotationRequest,
    PasswordRotationResponse,
    TokenRequest,
    TokenResponse,
)
from services.common.contracts import ServiceMetadata
from services.common.discovery import (
    ServiceDiscoveryClient,
    ServiceDiscoveryConfig,
    ServiceDiscoveryError,
)
from services.common.messaging import RedisEventBus

from .auth import AuthManager, TokenPayload
from .config import GatewaySettings, ServiceRouteConfig, load_gateway_settings
from .identity import (
    ApiKeyRotationRequiredError,
    ApiKeyValidationError,
    IdentityService,
    InactiveAccountError,
    InvalidCredentialsError,
    PasswordExpiredError,
)
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
        self.identity_service: IdentityService | None = None
        self.event_bus: RedisEventBus | None = None
        self.metadata: ServiceMetadata | None = None
        self.service_discovery: ServiceDiscoveryClient | None = None


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
        state.identity_service = IdentityService(state.settings)
        if state.redis is not None:
            state.event_bus = RedisEventBus(
                state.redis,
                channel_prefix=state.settings.event_channel_prefix,
                service_name=state.settings.service_name,
            )
        state.metadata = _build_service_metadata(state.settings)
        discovery_config = ServiceDiscoveryConfig(
            metadata=state.metadata,
            backend=state.settings.service_discovery_backend,
            consul_url=state.settings.service_discovery_url,
            consul_token=state.settings.service_discovery_token,
            namespace=state.settings.service_discovery_namespace,
            datacenter=state.settings.service_discovery_datacenter,
            register=state.settings.enable_service_registration,
            check_interval=state.settings.discovery_check_interval,
            check_timeout=state.settings.discovery_check_timeout,
            deregister_after=state.settings.discovery_deregister_after,
            additional_tags=state.settings.service_discovery_tags,
        )
        state.service_discovery = ServiceDiscoveryClient(discovery_config)
        try:
            await state.service_discovery.register()
        except ServiceDiscoveryError as exc:  # pragma: no cover - external dependency
            LOGGER.warning("Service discovery registration failed: %s", exc)

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        LOGGER.info("Shutting down API Gateway")
        if state.service_discovery:
            await state.service_discovery.close()
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


def _build_service_metadata(settings: GatewaySettings) -> ServiceMetadata:
    """Construct the service metadata object used for discovery registration."""

    return ServiceMetadata(
        name=settings.service_name,
        version=settings.service_version,
        host=settings.host,
        port=settings.port,
        scheme=settings.service_scheme,
        tags=list(settings.service_discovery_tags),
        health_endpoint=settings.health_endpoint,
        readiness_endpoint=settings.readiness_endpoint,
        metrics_endpoint=settings.metrics_endpoint,
    )


def register_routes(app: FastAPI, state: GatewayState) -> None:
    def _get_identity_service(state: GatewayState) -> IdentityService:
        if state.identity_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Identity service unavailable",
            )
        return state.identity_service

    def _token_has_required_roles(token: TokenPayload, required_roles: List[str]) -> bool:
        if not required_roles:
            return True
        scopes = set(token.scopes or [])
        if "admin" in scopes or "internal" in scopes:
            return True
        if token.token_type == "service" and token.service_name:
            if token.service_name in required_roles:
                return True
        for role in required_roles:
            if role in scopes or f"service:{role}" in scopes:
                return True
        return False

    async def _emit_auth_event(state: GatewayState, payload: AuthenticationPayload) -> None:
        if not state.event_bus:
            return
        event = AuthenticationEvent(source=state.settings.service_name, payload=payload)
        try:
            await state.event_bus.publish(state.settings.auth_event_channel, event)
        except Exception as exc:  # pragma: no cover - Redis issues handled gracefully
            LOGGER.warning("Failed to publish authentication event: %s", exc)

    @app.post("/auth/token", tags=["Authentication"], response_model=TokenResponse)
    async def issue_access_token(
        payload: TokenRequest,
        request: Request,
        state: GatewayState = Depends(get_state),
    ) -> TokenResponse:
        del request  # request metadata reserved for future auditing
        identity_service = _get_identity_service(state)
        try:
            issued = identity_service.issue_token(payload.username, payload.password)
        except InvalidCredentialsError:
            await _emit_auth_event(
                state,
                AuthenticationPayload(
                    username=payload.username,
                    successful=False,
                    roles=[],
                    subject="auth-token",
                ),
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            ) from None
        except InactiveAccountError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except PasswordExpiredError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

        response = TokenResponse(
            access_token=issued.access_token,
            expires_at=issued.expires_at,
            username=issued.username,
            roles=issued.roles,
            password_expires_at=issued.password_expires_at,
        )
        await _emit_auth_event(
            state,
            AuthenticationPayload(
                username=response.username,
                successful=True,
                roles=response.roles,
                subject="auth-token",
            ),
        )
        return response

    @app.post(
        "/auth/password/rotate",
        tags=["Authentication"],
        response_model=PasswordRotationResponse,
    )
    async def rotate_password(
        payload: PasswordRotationRequest,
        state: GatewayState = Depends(get_state),
    ) -> PasswordRotationResponse:
        identity_service = _get_identity_service(state)
        try:
            identity = identity_service.rotate_password(
                payload.username, payload.current_password, payload.new_password
            )
        except InvalidCredentialsError as exc:
            await _emit_auth_event(
                state,
                AuthenticationPayload(
                    username=payload.username,
                    successful=False,
                    roles=[],
                    subject="password-rotation",
                ),
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        except InactiveAccountError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

        response = PasswordRotationResponse(
            username=identity.username,
            roles=identity.roles,
            password_rotated_at=identity.password_rotated_at,
            password_expires_at=identity.password_expires_at,
        )
        await _emit_auth_event(
            state,
            AuthenticationPayload(
                username=response.username,
                successful=True,
                roles=response.roles,
                subject="password-rotation",
            ),
        )
        return response

    @app.post(
        "/auth/api-key/validate",
        tags=["Authentication"],
        response_model=ApiKeyValidationResponse,
    )
    async def validate_api_key(
        payload: ApiKeyValidationRequest,
        state: GatewayState = Depends(get_state),
    ) -> ApiKeyValidationResponse:
        identity_service = _get_identity_service(state)
        try:
            identity = identity_service.validate_api_key(payload.api_key)
        except ApiKeyRotationRequiredError as exc:
            await _emit_auth_event(
                state,
                AuthenticationPayload(
                    username=payload.api_key,
                    successful=False,
                    roles=[],
                    subject="api-key",
                ),
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except InactiveAccountError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except ApiKeyValidationError:
            await _emit_auth_event(
                state,
                AuthenticationPayload(
                    username=payload.api_key,
                    successful=False,
                    roles=[],
                    subject="api-key",
                ),
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            ) from None

        response = ApiKeyValidationResponse(
            username=identity.username,
            roles=identity.roles,
            api_key_last_rotated_at=identity.api_key_last_rotated_at,
        )
        await _emit_auth_event(
            state,
            AuthenticationPayload(
                username=response.username,
                successful=True,
                roles=response.roles,
                subject="api-key",
            ),
        )
        return response

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
                    "required_roles": route.required_roles,
                }
                for name, route in state.settings.service_routes.items()
            },
            "event_bus": {
                "enabled": state.event_bus is not None,
                "channel_prefix": state.settings.event_channel_prefix,
            },
            "service_discovery": {"enabled": state.service_discovery is not None},
        }
        if state.metadata:
            payload["metadata"] = state.metadata.model_dump()
        if state.service_discovery:
            try:
                registrations = await state.service_discovery.list_services()
                payload["service_discovery"].update(
                    {"registered": len(registrations)}
                )
            except ServiceDiscoveryError as exc:
                payload["service_discovery"]["error"] = str(exc)
        return JSONResponse(payload)

    @app.get("/routes", tags=["Configuration"])
    async def list_routes() -> Dict[str, object]:
        return {
            name: {
                "prefix": route.prefix,
                "target": route.url,
                "auth": route.allowed_auth_modes,
                "rate_limit_per_minute": route.rate_limit_per_minute,
                "required_roles": route.required_roles,
            }
            for name, route in state.settings.service_routes.items()
        }

    @app.get("/contracts", tags=["Discovery"])
    async def describe_contracts() -> Dict[str, object]:
        return {"http": [endpoint.model_dump() for endpoint in HTTP_CONTRACT]}

    @app.get("/discovery/self", tags=["Discovery"])
    async def discovery_self(state: GatewayState = Depends(get_state)) -> Dict[str, object]:
        metadata = state.metadata or _build_service_metadata(state.settings)
        return metadata.model_dump()

    @app.get("/discovery/services", tags=["Discovery"])
    async def discovery_services(state: GatewayState = Depends(get_state)) -> Dict[str, object]:
        if not state.service_discovery:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service discovery disabled",
            )
        try:
            registrations = await state.service_discovery.list_services()
        except ServiceDiscoveryError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return {"services": [registration.model_dump() for registration in registrations]}

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
            if not _token_has_required_roles(token, route.required_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient role privileges",
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

