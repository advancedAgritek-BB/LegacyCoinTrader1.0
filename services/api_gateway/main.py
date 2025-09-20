from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr
from redis import asyncio as redis_asyncio

from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging_compat import configure_logging

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
from .service_auth import ServiceTokenManager

LOGGER = logging.getLogger("api_gateway")


class TokenRequest(BaseModel):
    """Incoming credentials for token issuance."""

    username: constr(min_length=1, strip_whitespace=True)
    password: constr(min_length=1)


class TokenResponse(BaseModel):
    """JWT payload returned to authenticated clients."""

    access_token: str
    token_type: str = Field(default="bearer")
    expires_at: datetime
    username: str
    roles: List[str]
    password_expires_at: Optional[datetime] = None


class PasswordRotationRequest(BaseModel):
    """Payload for a password rotation request."""

    username: constr(min_length=1, strip_whitespace=True)
    current_password: constr(min_length=1)
    new_password: constr(min_length=8)


class PasswordRotationResponse(BaseModel):
    """Metadata returned after a successful password rotation."""

    username: str
    roles: List[str]
    password_rotated_at: datetime
    password_expires_at: Optional[datetime] = None


class ApiKeyValidationRequest(BaseModel):
    """Payload for API key validation."""

    api_key: constr(min_length=1)


class ApiKeyValidationResponse(BaseModel):
    """Response returned when an API key is validated."""

    username: str
    roles: List[str]
    api_key_last_rotated_at: Optional[datetime] = None


def create_app() -> FastAPI:
    settings = load_gateway_settings()
    monitoring_settings = get_monitoring_settings().for_service("api-gateway")
    monitoring_settings = monitoring_settings.clone(log_level=settings.log_level)
    monitoring_settings.metrics.default_labels.setdefault("component", "api-gateway")
    configure_logging(monitoring_settings)
    logging.getLogger().setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Configure TLS/HTTPS if enabled
    tls_enabled = os.getenv("TLS_ENABLED", "false").lower() == "true"
    if tls_enabled and settings.tls.enabled and settings.tls.is_valid():
        LOGGER.info("🔐 TLS/HTTPS enabled for API Gateway")
        LOGGER.info(f"   📜 Certificate: {settings.tls.cert_file}")
        LOGGER.info(f"   🔑 Private Key: {settings.tls.key_file}")
    elif tls_enabled:
        LOGGER.warning("⚠️  TLS enabled but certificates not found or invalid")
        LOGGER.warning("   Please run: python generate_tls_certificates.py")

    app = FastAPI(
        title="LegacyCoinTrader API Gateway",
        description="Unified entry point for LegacyCoinTrader microservices",
        version="1.0.0",
    )
    instrument_fastapi_app(app, settings=monitoring_settings)

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
        self.redis: Union[redis_asyncio.Redis, None] = None
        self.http_client: Union[httpx.AsyncClient, None] = None
        self.auth_manager: Union[AuthManager, None] = None
        self.rate_limiter: Union[RateLimiter, None] = None
        self.proxy_gateway: Union[ProxyGateway, None] = None
        self.identity_service: Union[IdentityService, None] = None
        self.service_token_manager: Union[ServiceTokenManager, None] = None


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
        state.service_token_manager = ServiceTokenManager(
            state.settings.service_auth,
            state.redis
        )

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        LOGGER.info("Shutting down API Gateway")
        if state.http_client:
            await state.http_client.aclose()
        if state.redis:
            await state.redis.close()


async def _init_redis(settings: GatewaySettings) -> Union[redis_asyncio.Redis, None]:
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
        # Allow anonymous tokens when authentication is disabled
        if token.token_type == "anonymous":
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
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            ) from None
        except InactiveAccountError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except PasswordExpiredError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

        return TokenResponse(
            access_token=issued.access_token,
            expires_at=issued.expires_at,
            username=issued.username,
            roles=issued.roles,
            password_expires_at=issued.password_expires_at,
        )

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
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        except InactiveAccountError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

        return PasswordRotationResponse(
            username=identity.username,
            roles=identity.roles,
            password_rotated_at=identity.password_rotated_at,
            password_expires_at=identity.password_expires_at,
        )

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
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except InactiveAccountError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except ApiKeyValidationError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            ) from None

        return ApiKeyValidationResponse(
            username=identity.username,
            roles=identity.roles,
            api_key_last_rotated_at=identity.api_key_last_rotated_at,
        )

    @app.post("/auth/service-token/generate", tags=["Service Authentication"])
    async def generate_service_token(
        service_name: str,
        state: GatewayState = Depends(get_state),
    ) -> Dict[str, str]:
        """Generate a new service token for inter-service authentication."""
        if not state.service_token_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service token manager not available"
            )

        try:
            token = await state.service_token_manager.generate_service_token(service_name)
            return {
                "service_name": service_name,
                "token": token,
                "expires_in_days": state.settings.service_auth.token_rotation_days,
            }
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    @app.post("/auth/service-token/validate", tags=["Service Authentication"])
    async def validate_service_token(
        service_name: str,
        token: str,
        state: GatewayState = Depends(get_state),
    ) -> Dict[str, bool]:
        """Validate a service token."""
        if not state.service_token_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service token manager not available"
            )

        is_valid = await state.service_token_manager.validate_service_token(service_name, token)
        return {"valid": is_valid, "service_name": service_name}

    @app.post("/auth/service-token/rotate", tags=["Service Authentication"])
    async def rotate_service_token(
        service_name: str,
        state: GatewayState = Depends(get_state),
    ) -> Dict[str, str]:
        """Rotate (regenerate) a service token."""
        if not state.service_token_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service token manager not available"
            )

        try:
            new_token = await state.service_token_manager.rotate_service_token(service_name)
            return {
                "service_name": service_name,
                "new_token": new_token,
                "rotated_at": datetime.now().isoformat(),
            }
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    @app.get("/auth/service-tokens", tags=["Service Authentication"])
    async def list_service_tokens(state: GatewayState = Depends(get_state)) -> Dict[str, List[str]]:
        """List all services with active tokens."""
        if not state.service_token_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service token manager not available"
            )

        active_services = await state.service_token_manager.list_active_services()
        return {"active_services": active_services}

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
                "required_roles": route.required_roles,
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
