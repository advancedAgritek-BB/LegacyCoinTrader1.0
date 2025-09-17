from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr
from redis import asyncio as redis_asyncio

from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging import configure_logging

from .audit import AuditClient
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
from .oidc import OidcConfiguration, OidcValidator
from .proxy import ProxyGateway
from .rate_limiter import RateLimitResult, RateLimiter
from .tenant import TenantContext, TenantServiceClient, TenantUsageTracker

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
    monitoring_settings = monitoring_settings.model_copy(
        update={"log_level": settings.log_level}
    )
    monitoring_settings.metrics.default_labels.setdefault("component", "api-gateway")
    configure_logging(monitoring_settings)
    logging.getLogger().setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

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
        self.redis: redis_asyncio.Redis | None = None
        self.http_client: httpx.AsyncClient | None = None
        self.auth_manager: AuthManager | None = None
        self.rate_limiter: RateLimiter | None = None
        self.proxy_gateway: ProxyGateway | None = None
        self.identity_service: IdentityService | None = None
        self.oidc_validator: OidcValidator | None = None
        self.tenant_service: TenantServiceClient | None = None
        self.usage_tracker: TenantUsageTracker | None = None
        self.audit_client: AuditClient | None = None


async def get_state(request: Request) -> GatewayState:
    return request.app.state.gateway_state


def register_events(app: FastAPI, state: GatewayState) -> None:
    @app.on_event("startup")
    async def startup_event() -> None:
        LOGGER.info("Starting API Gateway")
        state.redis = await _init_redis(state.settings)
        state.http_client = httpx.AsyncClient(timeout=state.settings.http_client_timeout)
        if state.settings.oidc_issuer and state.settings.oidc_jwks_url:
            oidc_config = OidcConfiguration(
                issuer=state.settings.oidc_issuer,
                jwks_url=state.settings.oidc_jwks_url,
                audience=state.settings.oidc_audience,
            )
            state.oidc_validator = OidcValidator(oidc_config, state.http_client)
        state.auth_manager = AuthManager(state.settings, oidc_validator=state.oidc_validator)
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
        tenant_route = state.settings.service_routes.get("tenant_management")
        state.tenant_service = TenantServiceClient(
            tenant_route,
            state.http_client,
            cache_ttl=state.settings.tenant_metadata_ttl,
        )
        state.usage_tracker = TenantUsageTracker(
            state.redis,
            kafka_bootstrap_servers=state.settings.kafka_bootstrap_servers,
        )
        audit_route = state.settings.service_routes.get("audit")
        state.audit_client = AuditClient(
            state.http_client,
            audit_route,
            service_name="api-gateway",
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
        roles = set(token.roles or [])
        if {"admin", "internal"} & scopes or "admin" in roles:
            return True
        if token.token_type == "service" and token.service_name:
            if token.service_name in required_roles:
                return True
        for role in required_roles:
            if (
                role in scopes
                or role in roles
                or f"service:{role}" in scopes
                or f"service:{role}" in roles
            ):
                return True
        return False

    def _validate_tenant_entitlements(
        token: TokenPayload, tenant_context: Optional[TenantContext]
    ) -> None:
        if not tenant_context or token.token_type not in {"jwt", "oidc"}:
            return

        requested_scopes = set(token.tenant_scopes or token.scopes or [])
        if requested_scopes:
            allowed_scopes = set(tenant_context.scopes or requested_scopes)
            missing_scopes = requested_scopes - allowed_scopes
            if missing_scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Tenant scope policy does not permit the requested operation",
                )

        if tenant_context.roles:
            allowed_roles = set(tenant_context.roles)
            requested_roles = set(token.roles or [])
            missing_roles = requested_roles - allowed_roles
            if missing_roles and "admin" not in requested_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Tenant role policy does not permit the requested operation",
                )

    async def _emit_audit_event(
        request: Request,
        *,
        state: GatewayState,
        event_type: str,
        status_code: int,
        token: Optional[TokenPayload] = None,
        tenant_context: Optional[TenantContext] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        if not state.audit_client:
            return

        active_token = token or getattr(request.state, "auth_token", None)
        active_tenant = tenant_context or getattr(request.state, "tenant_context", None)
        route_name = getattr(request.state, "route_name", request.url.path)
        event = state.audit_client.build_event(
            event_type=event_type,
            route=str(route_name),
            method=request.method,
            path=str(request.url.path),
            status_code=status_code,
            tenant_id=(
                active_tenant.tenant_id
                if active_tenant
                else (active_token.tenant_id if active_token else None)
            ),
            tenant_plan=(
                active_tenant.plan
                if active_tenant
                else (active_token.tenant_plan if active_token else None)
            ),
            tenant_slug=active_tenant.slug if active_tenant else None,
            actor=active_token.subject if active_token else None,
            actor_roles=list(active_token.roles if active_token and active_token.roles else []),
            scopes=list(active_token.scopes if active_token and active_token.scopes else []),
            request_id=request.headers.get("X-Request-ID"),
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        await state.audit_client.emit(event)

    @app.post("/auth/token", tags=["Authentication"], response_model=TokenResponse)
    async def issue_access_token(
        payload: TokenRequest,
        request: Request,
        state: GatewayState = Depends(get_state),
    ) -> TokenResponse:
        identity_service = _get_identity_service(state)
        try:
            issued = identity_service.issue_token(payload.username, payload.password)
        except InvalidCredentialsError:
            await _emit_audit_event(
                request,
                state=state,
                event_type="auth.issue-token",
                status_code=status.HTTP_401_UNAUTHORIZED,
                metadata={"username": payload.username, "reason": "invalid-credentials"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            ) from None
        except InactiveAccountError as exc:
            await _emit_audit_event(
                request,
                state=state,
                event_type="auth.issue-token",
                status_code=status.HTTP_403_FORBIDDEN,
                metadata={"username": payload.username, "reason": "inactive"},
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except PasswordExpiredError as exc:
            await _emit_audit_event(
                request,
                state=state,
                event_type="auth.issue-token",
                status_code=status.HTTP_403_FORBIDDEN,
                metadata={"username": payload.username, "reason": "password-expired"},
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

        response_payload = TokenResponse(
            access_token=issued.access_token,
            expires_at=issued.expires_at,
            username=issued.username,
            roles=issued.roles,
            password_expires_at=issued.password_expires_at,
        )
        await _emit_audit_event(
            request,
            state=state,
            event_type="auth.issue-token",
            status_code=status.HTTP_200_OK,
            metadata={"username": issued.username},
        )
        return response_payload

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
            await _emit_audit_event(
                request,
                state=state,
                event_type="auth.rotate-password",
                status_code=status.HTTP_401_UNAUTHORIZED,
                metadata={"username": payload.username, "reason": "invalid-credentials"},
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        except InactiveAccountError as exc:
            await _emit_audit_event(
                request,
                state=state,
                event_type="auth.rotate-password",
                status_code=status.HTTP_403_FORBIDDEN,
                metadata={"username": payload.username, "reason": "inactive"},
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

        response_payload = PasswordRotationResponse(
            username=identity.username,
            roles=identity.roles,
            password_rotated_at=identity.password_rotated_at,
            password_expires_at=identity.password_expires_at,
        )
        await _emit_audit_event(
            request,
            state=state,
            event_type="auth.rotate-password",
            status_code=status.HTTP_200_OK,
            metadata={"username": identity.username},
        )
        return response_payload

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
            await _emit_audit_event(
                request,
                state=state,
                event_type="auth.validate-api-key",
                status_code=status.HTTP_403_FORBIDDEN,
                metadata={"reason": "rotation-required"},
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except InactiveAccountError as exc:
            await _emit_audit_event(
                request,
                state=state,
                event_type="auth.validate-api-key",
                status_code=status.HTTP_403_FORBIDDEN,
                metadata={"reason": "inactive"},
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except ApiKeyValidationError:
            await _emit_audit_event(
                request,
                state=state,
                event_type="auth.validate-api-key",
                status_code=status.HTTP_401_UNAUTHORIZED,
                metadata={"reason": "invalid-api-key"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            ) from None

        response_payload = ApiKeyValidationResponse(
            username=identity.username,
            roles=identity.roles,
            api_key_last_rotated_at=identity.api_key_last_rotated_at,
        )
        await _emit_audit_event(
            request,
            state=state,
            event_type="auth.validate-api-key",
            status_code=status.HTTP_200_OK,
            metadata={"username": identity.username},
        )
        return response_payload

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
                    "metadata": route.metadata,
                }
                for name, route in state.settings.service_routes.items()
            },
            "tenant_service": {
                "configured": state.tenant_service is not None,
                "cache_ttl": state.settings.tenant_metadata_ttl,
            },
            "oidc": {
                "issuer": state.settings.oidc_issuer,
                "jwks_url": state.settings.oidc_jwks_url,
            },
            "audit": {"configured": state.audit_client is not None},
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
                "metadata": route.metadata,
            }
            for name, route in state.settings.service_routes.items()
        }

    async def _require_admin_token(
        request: Request, state: GatewayState = Depends(get_state)
    ) -> TokenPayload:
        assert state.auth_manager is not None
        token = await state.auth_manager.authenticate_request(
            request, ["oidc", "jwt", "service"]
        )
        if not _token_has_required_roles(token, ["admin"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator privileges required",
            )
        request.state.auth_token = token
        return token

    @app.get("/admin/routes", tags=["Administration"])
    async def admin_route_discovery(
        request: Request,
        token: TokenPayload = Depends(_require_admin_token),
        state: GatewayState = Depends(get_state),
    ) -> Dict[str, object]:
        tenant_service = state.tenant_service
        tenants = await tenant_service.list_tenants() if tenant_service else []
        request.state.route_name = "admin.routes"
        response = {
            "routes": {
                name: {
                    "prefix": route.prefix,
                    "target": route.url,
                    "auth": route.allowed_auth_modes,
                    "rate_limit_per_minute": route.rate_limit_per_minute,
                    "required_roles": route.required_roles,
                    "metadata": route.metadata,
                }
                for name, route in state.settings.service_routes.items()
            },
            "tenants": [
                {
                    "tenant_id": ctx.tenant_id,
                    "slug": ctx.slug,
                    "name": ctx.name,
                    "plan": ctx.plan,
                    "rate_limit_per_minute": ctx.rate_limit_per_minute,
                    "burst_limit": ctx.burst_limit,
                    "burst_window_seconds": ctx.burst_window_seconds,
                    "route_limits": ctx.route_limits,
                    "burst_limits": ctx.burst_limits,
                    "scopes": ctx.scopes,
                    "roles": ctx.roles,
                }
                for ctx in tenants
            ],
        }
        await _emit_audit_event(
            request,
            state=state,
            event_type="admin.route-discovery",
            status_code=200,
            token=token,
            metadata={"tenant_count": len(response["tenants"])},
        )
        return response

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
            request.state.auth_token = token
            request.state.route_name = route.name

            tenant_context: Optional[TenantContext] = None
            if state.tenant_service:
                try:
                    claims = token.claims if isinstance(token.claims, dict) else None
                    tenant_context = await state.tenant_service.resolve(
                        token.tenant_id, claims=claims
                    )
                except Exception as exc:  # pragma: no cover - defensive guardrail
                    LOGGER.warning("Unable to resolve tenant metadata for %s: %s", token.subject, exc)

            if tenant_context:
                token.tenant_id = token.tenant_id or tenant_context.tenant_id
                token.tenant_plan = token.tenant_plan or tenant_context.plan
            _validate_tenant_entitlements(token, tenant_context)

            identifier = f"{route.name}:{token.rate_limit_key}"
            route_result = await state.rate_limiter.check(
                identifier, route.rate_limit_per_minute
            )
            if not route_result.allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(
                            route_result.retry_after or state.settings.rate_limit_window_seconds
                        ),
                        "X-RateLimit-Limit": str(route_result.limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            tenant_result: Optional[RateLimitResult] = None
            burst_result: Optional[RateLimitResult] = None
            tenant_limit: Optional[int] = None
            tenant_burst_limit: Optional[int] = None
            combined_usage: Optional[int] = None

            if tenant_context:
                base_limit = tenant_context.rate_limit_per_minute or route.rate_limit_per_minute
                tenant_limit = tenant_context.limit_for_route(route.name, base_limit)
                if tenant_limit and tenant_limit > 0:
                    tenant_identifier = f"tenant:{tenant_context.tenant_id}:{route.name}"
                    tenant_result = await state.rate_limiter.check(tenant_identifier, tenant_limit)
                    if not tenant_result.allowed:
                        raise HTTPException(
                            status_code=429,
                            detail="Tenant plan rate limit exceeded",
                            headers={
                                "Retry-After": str(
                                    tenant_result.retry_after or state.settings.rate_limit_window_seconds
                                ),
                                "X-Tenant-RateLimit-Limit": str(tenant_result.limit),
                                "X-Tenant-RateLimit-Remaining": "0",
                            },
                        )

                tenant_burst_limit = tenant_context.burst_for_route(
                    route.name, tenant_context.burst_limit
                )
                if tenant_burst_limit and tenant_context.burst_window_seconds > 0:
                    if not tenant_limit or tenant_burst_limit > tenant_limit:
                        burst_identifier = (
                            f"tenant:{tenant_context.tenant_id}:{route.name}:burst"
                        )
                        burst_result = await state.rate_limiter.check(
                            burst_identifier,
                            tenant_burst_limit,
                            window_seconds=tenant_context.burst_window_seconds,
                        )
                        if not burst_result.allowed:
                            raise HTTPException(
                                status_code=429,
                                detail="Tenant burst capacity exceeded",
                                headers={
                                    "Retry-After": str(
                                        burst_result.retry_after or tenant_context.burst_window_seconds
                                    ),
                                    "X-Tenant-Burst-Limit": str(burst_result.limit),
                                    "X-Tenant-Burst-Remaining": "0",
                                },
                            )

                if tenant_limit and tenant_limit > 0 and state.usage_tracker:
                    combined_usage = await state.usage_tracker.combined_usage(
                        tenant_context.tenant_id, route.name
                    )
                    if combined_usage >= tenant_limit:
                        raise HTTPException(
                            status_code=429,
                            detail="Tenant aggregate usage exceeded plan limits",
                        )

            request.state.rate_limit_result = route_result
            request.state.tenant_rate_limit_result = tenant_result
            request.state.tenant_burst_result = burst_result
            request.state.tenant_rate_limit_value = tenant_limit
            request.state.tenant_burst_limit_value = tenant_burst_limit
            request.state.tenant_usage_combined = combined_usage
            request.state.tenant_context = tenant_context
            return token

        async def proxy_endpoint(
            request: Request,
            path: str = "",
            token: TokenPayload = Depends(auth_and_rate_limit),
            state: GatewayState = Depends(get_state),
        ):
            assert state.proxy_gateway is not None
            start_time = time.perf_counter()
            tenant_context: Optional[TenantContext] = getattr(request.state, "tenant_context", None)
            try:
                response = await state.proxy_gateway.proxy_request(route, path, request, token)
            except HTTPException as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000
                await _emit_audit_event(
                    request,
                    state=state,
                    event_type="proxy.request",
                    status_code=exc.status_code,
                    token=token,
                    tenant_context=tenant_context,
                    latency_ms=latency_ms,
                    metadata={"detail": exc.detail} if exc.detail else {},
                )
                raise
            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000
                await _emit_audit_event(
                    request,
                    state=state,
                    event_type="proxy.request",
                    status_code=500,
                    token=token,
                    tenant_context=tenant_context,
                    latency_ms=latency_ms,
                    metadata={"error": str(exc)},
                )
                raise

            latency_ms = (time.perf_counter() - start_time) * 1000
            rate_limit_result: RateLimitResult = getattr(request.state, "rate_limit_result", None)
            if rate_limit_result:
                response.headers["X-RateLimit-Limit"] = str(rate_limit_result.limit)
                response.headers["X-RateLimit-Remaining"] = str(rate_limit_result.remaining)
                response.headers["X-RateLimit-Reset"] = str(rate_limit_result.reset_after)

            tenant_rate_result: Optional[RateLimitResult] = getattr(
                request.state, "tenant_rate_limit_result", None
            )
            if tenant_rate_result:
                response.headers["X-Tenant-RateLimit-Limit"] = str(tenant_rate_result.limit)
                response.headers["X-Tenant-RateLimit-Remaining"] = str(tenant_rate_result.remaining)
                response.headers["X-Tenant-RateLimit-Reset"] = str(tenant_rate_result.reset_after)

            tenant_burst_result: Optional[RateLimitResult] = getattr(
                request.state, "tenant_burst_result", None
            )
            if tenant_burst_result:
                response.headers["X-Tenant-Burst-Limit"] = str(tenant_burst_result.limit)
                response.headers["X-Tenant-Burst-Remaining"] = str(tenant_burst_result.remaining)
                response.headers["X-Tenant-Burst-Reset"] = str(tenant_burst_result.reset_after)

            tenant_usage_value: Optional[int] = getattr(
                request.state, "tenant_usage_combined", None
            )
            if tenant_context and state.usage_tracker:
                tenant_usage_value = await state.usage_tracker.increment(
                    tenant_context.tenant_id, route.name
                )
            if tenant_usage_value is not None:
                response.headers["X-Tenant-Usage"] = str(tenant_usage_value)

            metadata = {
                "route": route.name,
                "tenant_limit": getattr(request.state, "tenant_rate_limit_value", None),
                "tenant_burst_limit": getattr(request.state, "tenant_burst_limit_value", None),
            }
            if tenant_usage_value is not None:
                metadata["tenant_usage"] = tenant_usage_value
            metadata = {key: value for key, value in metadata.items() if value is not None}
            await _emit_audit_event(
                request,
                state=state,
                event_type="proxy.request",
                status_code=response.status_code,
                token=token,
                tenant_context=tenant_context,
                latency_ms=latency_ms,
                metadata=metadata,
            )
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

