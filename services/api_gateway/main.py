from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr
from redis import asyncio as redis_asyncio

from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging import configure_logging

from .audit import AuditClient, AuditEvent
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
from .tenant import TenantPlan, TenantServiceClient, TenantUsageTracker

LOGGER = logging.getLogger("api_gateway")


class TokenRequest(BaseModel):
    """Incoming credentials for token issuance."""

    username: constr(min_length=1, strip_whitespace=True)
    password: constr(min_length=1)


class TokenResponse(BaseModel):
    """JWT payload returned to authenticated clients."""

    access_token: str
    refresh_token: str
    token_type: str = Field(default="bearer")
    expires_at: datetime
    issued_at: datetime
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
        self.tenant_service: TenantServiceClient | None = None
        self.tenant_usage_tracker: TenantUsageTracker | None = None
        self.audit_client: AuditClient | None = None
        self.tenant_default_plan: TenantPlan | None = None


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
        state.identity_service = IdentityService(state.settings, state.http_client)
        default_tenant = state.settings.identity_default_tenant or "global"
        state.tenant_default_plan = TenantPlan.from_defaults(
            tenant_id=default_tenant,
            plan=state.settings.tenant_default_plan,
            rate_limit_per_minute=state.settings.tenant_default_rate_limit_per_minute,
            burst_limit=state.settings.tenant_default_burst_limit,
            burst_window_seconds=state.settings.tenant_default_burst_window_seconds,
        )
        state.tenant_service = TenantServiceClient(
            state.settings.tenant_service_url,
            cache_seconds=state.settings.tenant_metadata_cache_seconds,
            service_token=state.settings.tenant_service_token,
            service_token_header=state.settings.tenant_service_token_header,
            timeout=state.settings.tenant_service_timeout,
            http_client=state.http_client,
        )
        state.tenant_usage_tracker = TenantUsageTracker(
            redis_client=state.redis,
            kafka_bootstrap_servers=state.settings.kafka_bootstrap_servers,
            kafka_topic=state.settings.kafka_usage_topic,
        )
        await state.tenant_usage_tracker.start()
        state.audit_client = AuditClient(
            state.settings.audit_service_url,
            service_token=state.settings.audit_service_token,
            service_token_header=state.settings.audit_service_token_header,
            tenant_header=state.settings.identity_tenant_header,
            timeout=state.settings.audit_service_timeout,
            http_client=state.http_client,
        )

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        LOGGER.info("Shutting down API Gateway")
        if state.http_client:
            await state.http_client.aclose()
        if state.redis:
            await state.redis.close()
        if state.tenant_usage_tracker:
            await state.tenant_usage_tracker.stop()
        if state.tenant_service:
            await state.tenant_service.close()
        if state.audit_client:
            await state.audit_client.close()


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

    def _tenant_fallback_plan(tenant_id: str) -> TenantPlan:
        template = state.tenant_default_plan
        if template is not None:
            return TenantPlan(
                tenant_id=tenant_id,
                plan=template.plan,
                rate_limit_per_minute=template.rate_limit_per_minute,
                burst_limit=template.burst_limit,
                burst_window_seconds=template.burst_window_seconds,
                route_overrides=dict(template.route_overrides),
                burst_overrides={
                    key: dict(value) for key, value in template.burst_overrides.items()
                },
                metadata=dict(template.metadata),
            )
        return TenantPlan.from_defaults(
            tenant_id=tenant_id,
            plan=state.settings.tenant_default_plan,
            rate_limit_per_minute=state.settings.tenant_default_rate_limit_per_minute,
            burst_limit=state.settings.tenant_default_burst_limit,
            burst_window_seconds=state.settings.tenant_default_burst_window_seconds,
        )

    async def _get_tenant_plan(tenant_id: str) -> TenantPlan:
        fallback_plan = _tenant_fallback_plan(tenant_id)
        if state.tenant_service is None:
            return fallback_plan
        try:
            return await state.tenant_service.get_tenant_plan(tenant_id, fallback=fallback_plan)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            LOGGER.warning("Falling back to default tenant plan for %s: %s", tenant_id, exc)
            return fallback_plan

    def _resolve_tenant_id(request: Request, token: TokenPayload) -> str:
        header_value = request.headers.get(state.settings.identity_tenant_header, "") or ""
        header_value = header_value.strip()
        if header_value and token.tenant_id and header_value.lower() != token.tenant_id.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Token tenant does not match request tenant header",
            )
        if token.tenant_id:
            return token.tenant_id
        if header_value:
            return header_value
        if state.settings.identity_default_tenant:
            return state.settings.identity_default_tenant
        if state.tenant_default_plan:
            return state.tenant_default_plan.tenant_id
        return "global"

    async def _record_usage(
        *,
        tenant_id: str,
        route_name: str,
        result: RateLimitResult,
        burst_result: Optional[RateLimitResult],
        allowed: bool,
        plan: TenantPlan,
        token: TokenPayload,
    ) -> None:
        tracker = state.tenant_usage_tracker
        if tracker is None:
            return
        metadata = {
            "plan": plan.plan,
            "token_type": token.token_type,
            "subject": token.subject,
        }
        burst_limit = burst_result.limit if burst_result else None
        burst_allowed = burst_result.allowed if burst_result else None
        burst_remaining = burst_result.remaining if burst_result else None
        await tracker.record_request(
            tenant_id=tenant_id,
            route_name=route_name,
            rate_limit=result.limit,
            allowed=allowed,
            remaining=result.remaining,
            burst_limit=burst_limit,
            burst_allowed=burst_allowed,
            burst_remaining=burst_remaining,
            metadata=metadata,
        )

    async def _emit_audit_event(
        *,
        tenant_id: str,
        actor: str,
        action: str,
        resource: str,
        outcome: str,
        severity: str = "info",
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        client = state.audit_client
        if client is None:
            return
        tenant_value = tenant_id or state.settings.identity_default_tenant or "global"
        event = AuditEvent(
            tenant_id=tenant_value,
            actor=actor,
            action=action,
            resource=resource,
            source=state.settings.audit_event_source,
            outcome=outcome,
            severity=severity,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )
        try:
            await client.emit(event)
        except Exception:  # pragma: no cover - defensive guardrail
            LOGGER.debug("Failed to emit audit event", exc_info=True)

    def _token_has_required_roles(token: TokenPayload, required_roles: List[str]) -> bool:
        if not required_roles:
            return True
        scopes = set(token.scopes or [])
        roles = set(token.roles or [])
        if {"admin", "internal"} & (scopes | roles):
            return True
        if token.token_type == "service" and token.service_name:
            if token.service_name in required_roles:
                return True
        for role in required_roles:
            if (
                role in scopes
                or f"service:{role}" in scopes
                or role in roles
                or f"service:{role}" in roles
            ):
                return True
        return False

    @app.post("/auth/token", tags=["Authentication"], response_model=TokenResponse)
    async def issue_access_token(
        payload: TokenRequest,
        request: Request,
        state: GatewayState = Depends(get_state),
    ) -> TokenResponse:
        tenant_id = (
            request.headers.get(state.settings.identity_tenant_header, "") or ""
        ).strip() or state.settings.identity_default_tenant or "global"
        identity_service = _get_identity_service(state)
        try:
            issued = await identity_service.issue_token(payload.username, payload.password)
        except InvalidCredentialsError:
            await _emit_audit_event(
                tenant_id=tenant_id,
                actor=payload.username,
                action="auth.token.issue",
                resource="identity",
                outcome="denied",
                severity="warning",
                metadata={"reason": "invalid_credentials"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            ) from None
        except InactiveAccountError as exc:
            await _emit_audit_event(
                tenant_id=tenant_id,
                actor=payload.username,
                action="auth.token.issue",
                resource="identity",
                outcome="denied",
                severity="warning",
                metadata={"reason": "inactive_account"},
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except PasswordExpiredError as exc:
            await _emit_audit_event(
                tenant_id=tenant_id,
                actor=payload.username,
                action="auth.token.issue",
                resource="identity",
                outcome="denied",
                severity="warning",
                metadata={"reason": "password_expired"},
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

        refresh_token = issued.refresh_token or ""
        await _emit_audit_event(
            tenant_id=tenant_id,
            actor=payload.username,
            action="auth.token.issue",
            resource="identity",
            outcome="success",
            metadata={
                "roles": issued.roles,
                "expires_at": issued.expires_at.isoformat(),
            },
        )
        return TokenResponse(
            access_token=issued.access_token,
            refresh_token=refresh_token,
            expires_at=issued.expires_at,
            issued_at=issued.issued_at,
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
            identity = await identity_service.rotate_password(
                payload.username, payload.current_password, payload.new_password
            )
        except InvalidCredentialsError as exc:
            await _emit_audit_event(
                tenant_id=state.settings.identity_default_tenant or "global",
                actor=payload.username,
                action="auth.password.rotate",
                resource="identity",
                outcome="denied",
                severity="warning",
                metadata={"reason": "invalid_credentials"},
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        except InactiveAccountError as exc:
            await _emit_audit_event(
                tenant_id=state.settings.identity_default_tenant or "global",
                actor=payload.username,
                action="auth.password.rotate",
                resource="identity",
                outcome="denied",
                severity="warning",
                metadata={"reason": "inactive_account"},
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

        await _emit_audit_event(
            tenant_id=identity.tenant or state.settings.identity_default_tenant or "global",
            actor=payload.username,
            action="auth.password.rotate",
            resource="identity",
            outcome="success",
            metadata={"roles": identity.roles},
        )
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
            identity = await identity_service.validate_api_key(payload.api_key)
        except ApiKeyRotationRequiredError as exc:
            await _emit_audit_event(
                tenant_id=state.settings.identity_default_tenant or "global",
                actor="api-key",
                action="auth.api_key.validate",
                resource="identity",
                outcome="denied",
                severity="warning",
                metadata={"reason": "rotation_required"},
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except InactiveAccountError as exc:
            await _emit_audit_event(
                tenant_id=state.settings.identity_default_tenant or "global",
                actor="api-key",
                action="auth.api_key.validate",
                resource="identity",
                outcome="denied",
                severity="warning",
                metadata={"reason": "inactive_account"},
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except ApiKeyValidationError:
            await _emit_audit_event(
                tenant_id=state.settings.identity_default_tenant or "global",
                actor="api-key",
                action="auth.api_key.validate",
                resource="identity",
                outcome="denied",
                severity="warning",
                metadata={"reason": "invalid_api_key"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            ) from None

        await _emit_audit_event(
            tenant_id=identity.tenant or state.settings.identity_default_tenant or "global",
            actor=identity.username,
            action="auth.api_key.validate",
            resource="identity",
            outcome="success",
            metadata={"roles": identity.roles},
        )
        return ApiKeyValidationResponse(
            username=identity.username,
            roles=identity.roles,
            api_key_last_rotated_at=identity.api_key_last_rotated_at,
        )

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

    @app.get("/admin/routes", tags=["Administration"])
    async def admin_route_catalog(
        request: Request,
        tenant: Optional[str] = None,
        state: GatewayState = Depends(get_state),
    ) -> Dict[str, Any]:
        assert state.auth_manager is not None
        token = await state.auth_manager.authenticate_request(request, ["jwt", "service"])
        if not _token_has_required_roles(token, ["admin"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator privileges required",
            )
        tenant_id = tenant or _resolve_tenant_id(request, token)
        tenant_plan = await _get_tenant_plan(tenant_id)
        routes: Dict[str, Any] = {}
        for name, route in state.settings.service_routes.items():
            limit = tenant_plan.limit_for_route(name, route.rate_limit_per_minute)
            burst_limit, burst_window = tenant_plan.burst_for_route(
                name,
                state.settings.tenant_default_burst_limit,
                state.settings.tenant_default_burst_window_seconds,
            )
            routes[name] = {
                "prefix": route.prefix,
                "target": route.url,
                "auth": route.allowed_auth_modes,
                "required_roles": route.required_roles,
                "tenant_limit_per_minute": limit,
                "burst_limit": burst_limit,
                "burst_window_seconds": burst_window,
            }
        await _emit_audit_event(
            tenant_id=tenant_id,
            actor=token.subject,
            action="admin.routes.inspect",
            resource="gateway",
            outcome="success",
            metadata={"plan": tenant_plan.plan},
        )
        return {"tenant": tenant_id, "plan": tenant_plan.plan, "routes": routes}

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
            tenant_id = _resolve_tenant_id(request, token)
            tenant_plan = await _get_tenant_plan(tenant_id)
            effective_limit = tenant_plan.limit_for_route(
                route.name, route.rate_limit_per_minute
            )
            identifier = f"{tenant_id}:{route.name}:{token.rate_limit_key}"
            result = await state.rate_limiter.check(
                identifier,
                effective_limit,
                namespace="tenant",
            )
            if not result.allowed:
                await _record_usage(
                    tenant_id=tenant_id,
                    route_name=route.name,
                    result=result,
                    burst_result=None,
                    allowed=False,
                    plan=tenant_plan,
                    token=token,
                )
                await _emit_audit_event(
                    tenant_id=tenant_id,
                    actor=token.subject,
                    action="gateway.rate_limit",
                    resource=route.name,
                    outcome="denied",
                    severity="warning",
                    metadata={
                        "limit": result.limit,
                        "remaining": result.remaining,
                        "plan": tenant_plan.plan,
                        "type": "steady",
                    },
                )
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(
                            result.retry_after or state.settings.rate_limit_window_seconds
                        ),
                        "X-RateLimit-Limit": str(result.limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(result.reset_after),
                    },
                )

            burst_result: Optional[RateLimitResult] = None
            burst_limit, burst_window = tenant_plan.burst_for_route(
                route.name,
                state.settings.tenant_default_burst_limit,
                state.settings.tenant_default_burst_window_seconds,
            )
            if burst_limit > effective_limit:
                burst_identifier = f"{tenant_id}:{route.name}"
                burst_result = await state.rate_limiter.check(
                    burst_identifier,
                    burst_limit,
                    window_seconds=burst_window,
                    namespace="burst",
                )
                if not burst_result.allowed:
                    await _record_usage(
                        tenant_id=tenant_id,
                        route_name=route.name,
                        result=result,
                        burst_result=burst_result,
                        allowed=False,
                        plan=tenant_plan,
                        token=token,
                    )
                    await _emit_audit_event(
                        tenant_id=tenant_id,
                        actor=token.subject,
                        action="gateway.rate_limit",
                        resource=route.name,
                        outcome="denied",
                        severity="warning",
                        metadata={
                            "limit": burst_result.limit,
                            "remaining": burst_result.remaining,
                            "plan": tenant_plan.plan,
                            "type": "burst",
                        },
                    )
                    raise HTTPException(
                        status_code=429,
                        detail="Burst limit exceeded",
                        headers={
                            "Retry-After": str(
                                burst_result.retry_after
                                or state.settings.tenant_default_burst_window_seconds
                            ),
                            "X-RateLimit-Limit": str(burst_result.limit),
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(burst_result.reset_after),
                        },
                    )

            await _record_usage(
                tenant_id=tenant_id,
                route_name=route.name,
                result=result,
                burst_result=burst_result,
                allowed=True,
                plan=tenant_plan,
                token=token,
            )
            request.state.rate_limit_result = result
            if burst_result is not None:
                request.state.burst_rate_limit_result = burst_result
            request.state.tenant_plan = tenant_plan
            request.state.tenant_id = tenant_id
            request.state.tenant_roles = token.roles
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
            burst_result: Optional[RateLimitResult] = getattr(
                request.state, "burst_rate_limit_result", None
            )
            if burst_result:
                response.headers["X-RateLimit-Burst-Limit"] = str(burst_result.limit)
                response.headers["X-RateLimit-Burst-Remaining"] = str(burst_result.remaining)
                response.headers["X-RateLimit-Burst-Reset"] = str(burst_result.reset_after)

            tenant_id = getattr(request.state, "tenant_id", None) or token.tenant_id or state.settings.identity_default_tenant or "global"
            tenant_plan: Optional[TenantPlan] = getattr(request.state, "tenant_plan", None)
            outcome = "success" if response.status_code < 400 else "error"
            metadata = {
                "route": route.name,
                "method": request.method,
                "status": response.status_code,
                "plan": tenant_plan.plan if tenant_plan else state.settings.tenant_default_plan,
                "limit": rate_limit_result.limit if rate_limit_result else None,
                "remaining": rate_limit_result.remaining if rate_limit_result else None,
                "burst_limit": burst_result.limit if burst_result else None,
                "burst_remaining": burst_result.remaining if burst_result else None,
            }
            await _emit_audit_event(
                tenant_id=tenant_id,
                actor=token.subject,
                action="gateway.proxy",
                resource=f"{route.name}:{path or '/'}",
                outcome=outcome,
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

