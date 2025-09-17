from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from .config import ServiceRouteConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AuditEvent:
    """Structured audit payload forwarded to the compliance pipeline."""

    event_type: str
    timestamp: float
    route: str
    method: str
    path: str
    status_code: int
    tenant_id: Optional[str]
    tenant_plan: Optional[str]
    tenant_slug: Optional[str]
    actor: Optional[str]
    actor_roles: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)
    request_id: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditClient:
    """Thin wrapper around the audit service HTTP API."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        route_config: Optional[ServiceRouteConfig],
        *,
        service_name: str,
    ) -> None:
        self.http_client = http_client
        self.route_config = route_config
        self.service_name = service_name

    async def emit(self, event: AuditEvent) -> None:
        if not self.route_config:
            LOGGER.debug("Audit route not configured; skipping event %s", event.event_type)
            return

        url = self.route_config.build_target_url("events")
        headers = {"Content-Type": "application/json", "X-Audit-Source": self.service_name}
        if self.route_config.service_token:
            headers["X-Service-Token"] = self.route_config.service_token

        try:
            response = await self.http_client.post(
                url,
                json=asdict(event),
                headers=headers,
                timeout=5,
            )
            if response.status_code >= 400:
                LOGGER.warning(
                    "Audit service responded with %s for event %s", response.status_code, event.event_type
                )
        except httpx.HTTPError as exc:
            LOGGER.warning("Failed to emit audit event %s: %s", event.event_type, exc)

    def build_event(
        self,
        *,
        event_type: str,
        route: str,
        method: str,
        path: str,
        status_code: int,
        tenant_id: Optional[str],
        tenant_plan: Optional[str],
        tenant_slug: Optional[str],
        actor: Optional[str],
        actor_roles: List[str],
        scopes: List[str],
        request_id: Optional[str],
        latency_ms: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        return AuditEvent(
            event_type=event_type,
            timestamp=time.time(),
            route=route,
            method=method,
            path=path,
            status_code=status_code,
            tenant_id=tenant_id,
            tenant_plan=tenant_plan,
            tenant_slug=tenant_slug,
            actor=actor,
            actor_roles=list(actor_roles or []),
            scopes=list(scopes or []),
            request_id=request_id,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )


__all__ = ["AuditClient", "AuditEvent"]
