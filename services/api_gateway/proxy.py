from __future__ import annotations

import logging
import uuid
from typing import Dict

import httpx
from fastapi import Request
from starlette.responses import Response

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

        upstream_response = await self.http_client.request(
            request.method,
            target_url,
            content=body or None,
            params=request.query_params,
            headers=headers,
            timeout=self.settings.http_client_timeout,
        )

        return self._build_response(upstream_response)

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
                headers.setdefault("X-User-Roles", ",".join(token.scopes))
            if token.raw_token:
                headers.setdefault("Authorization", f"Bearer {token.raw_token}")
        elif token.token_type == "service":
            headers["X-Service-Caller"] = token.service_name or token.subject

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

