from __future__ import annotations

import logging
import uuid
from typing import Dict

import httpx
from fastapi import Request
from starlette.responses import Response

from libs.resilience import (
    call_market_data_service,
    call_strategy_engine_service,
    call_execution_service,
    call_token_discovery_service,
    call_with_resilience,
    ResilienceConfig,
    CircuitBreakerConfig,
    RetryConfig,
)
from .auth import TokenPayload
from .config import GatewaySettings, ServiceRouteConfig

LOGGER = logging.getLogger(__name__)


class ProxyGateway:
    """Performs HTTP proxying to downstream services."""

    def __init__(self, settings: GatewaySettings, http_client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.http_client = http_client

    async def proxy_request(
        self,
        route: ServiceRouteConfig,
        path: str,
        request: Request,
        token: TokenPayload,
    ) -> Response:
        target_url = route.build_target_url(path)

        LOGGER.debug(
            "Proxying %s %s to %s", request.method, request.url.path, target_url
        )

        headers = self._prepare_headers(request, route, token)
        body = await request.body()

        # Use resilience patterns based on service type
        if route.name == "market-data":
            upstream_response = await self._proxy_with_market_data_resilience(
                request, target_url, headers, body
            )
        elif route.name == "strategy-engine":
            upstream_response = await self._proxy_with_strategy_resilience(
                request, target_url, headers, body
            )
        elif route.name == "execution":
            upstream_response = await self._proxy_with_execution_resilience(
                request, target_url, headers, body
            )
        elif route.name == "token-discovery":
            upstream_response = await self._proxy_with_token_discovery_resilience(
                request, target_url, headers, body
            )
        else:
            # Generic resilience for other services
            upstream_response = await self._proxy_with_generic_resilience(
                route.name, request, target_url, headers, body
            )

        return self._build_response(upstream_response)

    async def _proxy_with_market_data_resilience(
        self, request: Request, target_url: str, headers: Dict[str, str], body: bytes
    ) -> httpx.Response:
        """Proxy with market data specific resilience settings."""
        async def market_data_call():
            return await self.http_client.request(
                request.method,
                target_url,
                content=body or None,
                params=request.query_params,
                headers=headers,
                timeout=10.0,  # Shorter timeout for market data
            )

        return await call_market_data_service(market_data_call)

    async def _proxy_with_strategy_resilience(
        self, request: Request, target_url: str, headers: Dict[str, str], body: bytes
    ) -> httpx.Response:
        """Proxy with strategy engine specific resilience settings."""
        async def strategy_call():
            return await self.http_client.request(
                request.method,
                target_url,
                content=body or None,
                params=request.query_params,
                headers=headers,
                timeout=30.0,  # Longer timeout for strategy evaluation
            )

        return await call_strategy_engine_service(strategy_call)

    async def _proxy_with_execution_resilience(
        self, request: Request, target_url: str, headers: Dict[str, str], body: bytes
    ) -> httpx.Response:
        """Proxy with execution service specific resilience settings."""
        async def execution_call():
            return await self.http_client.request(
                request.method,
                target_url,
                content=body or None,
                params=request.query_params,
                headers=headers,
                timeout=15.0,  # Medium timeout for execution
            )

        return await call_execution_service(execution_call)

    async def _proxy_with_token_discovery_resilience(
        self, request: Request, target_url: str, headers: Dict[str, str], body: bytes
    ) -> httpx.Response:
        """Proxy with token discovery specific resilience settings."""

        async def token_discovery_call():
            return await self.http_client.request(
                request.method,
                target_url,
                content=body or None,
                params=request.query_params,
                headers=headers,
                timeout=20.0,
            )

        return await call_token_discovery_service(token_discovery_call)

    async def _proxy_with_generic_resilience(
        self, service_name: str, request: Request, target_url: str, headers: Dict[str, str], body: bytes
    ) -> httpx.Response:
        """Proxy with generic resilience settings."""
        config = ResilienceConfig(
            service_name=service_name,
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                timeout=self.settings.http_client_timeout
            ),
            retry_policy=RetryConfig(
                max_attempts=3,
                initial_delay=1.0,
                max_delay=10.0
            )
        )

        async def generic_call():
            return await self.http_client.request(
                request.method,
                target_url,
                content=body or None,
                params=request.query_params,
                headers=headers,
                timeout=self.settings.http_client_timeout,
            )

        return await call_with_resilience(service_name, generic_call, resilience_config=config)

    async def check_service_health(self, route: ServiceRouteConfig) -> Dict[str, object]:
        url = route.build_target_url(route.health_endpoint.lstrip("/"))
        try:
            response = await self.http_client.get(url, timeout=5)
            healthy = response.status_code < 500
            return {
                "healthy": healthy,
                "status_code": response.status_code,
                "service": route.name,
                "target": url,
            }
        except httpx.HTTPError as exc:
            LOGGER.warning("Service %s health check failed: %s", route.name, exc)
            return {
                "healthy": False,
                "error": str(exc),
                "service": route.name,
                "target": url,
            }

    def _prepare_headers(
        self, request: Request, route: ServiceRouteConfig, token: TokenPayload
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }

        client_host = request.client.host if request.client else ""
        existing_xff = request.headers.get("x-forwarded-for")
        if existing_xff:
            headers["X-Forwarded-For"] = f"{existing_xff}, {client_host}" if client_host else existing_xff
        elif client_host:
            headers["X-Forwarded-For"] = client_host

        headers["X-Forwarded-Proto"] = request.url.scheme
        headers["X-Forwarded-Host"] = request.url.hostname or "api-gateway"
        headers.setdefault("X-Request-ID", request.headers.get("X-Request-ID", str(uuid.uuid4())))

        if token.token_type == "jwt":
            headers["X-Authenticated-User"] = token.subject
            if token.scopes:
                headers["X-User-Scopes"] = ",".join(token.scopes)
            if token.roles:
                headers["X-User-Roles"] = ",".join(token.roles)
            elif token.scopes:
                headers.setdefault("X-User-Roles", ",".join(token.scopes))
            if token.raw_token:
                headers.setdefault("Authorization", f"Bearer {token.raw_token}")
        elif token.token_type == "service":
            headers["X-Service-Caller"] = token.service_name or token.subject

        tenant_id = getattr(request.state, "tenant_id", None) or token.tenant_id
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
        tenant_plan = getattr(request.state, "tenant_plan", None)
        if tenant_plan and getattr(tenant_plan, "plan", None):
            headers["X-Tenant-Plan"] = tenant_plan.plan
        tenant_roles = getattr(request.state, "tenant_roles", None)
        if tenant_roles:
            headers["X-Tenant-Roles"] = ",".join(sorted(set(str(role) for role in tenant_roles)))

        if route.service_token:
            headers["X-Service-Token"] = route.service_token

        headers.setdefault("X-Gateway-Version", "1.0.0")
        return headers

    @staticmethod
    def _build_response(upstream_response: httpx.Response) -> Response:
        excluded = {
            "content-encoding",
            "transfer-encoding",
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "upgrade",
        }
        headers = {
            key: value for key, value in upstream_response.headers.items() if key.lower() not in excluded
        }
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=headers,
            media_type=upstream_response.headers.get("content-type"),
        )
