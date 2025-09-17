from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AuditEvent:
    """Structured audit payload emitted by the API gateway."""

    tenant_id: str
    actor: str
    action: str
    resource: str
    source: str
    outcome: str
    severity: str = "info"
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_payload(self) -> Dict[str, Any]:
        payload = {
            "tenant": self.tenant_id,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "source": self.source,
            "outcome": self.outcome,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        if self.correlation_id:
            payload["correlation_id"] = self.correlation_id
        return payload


class AuditClient:
    """HTTP client responsible for delivering audit events."""

    def __init__(
        self,
        base_url: Optional[str],
        *,
        service_token: Optional[str],
        service_token_header: str,
        tenant_header: str,
        timeout: float,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/") if base_url else None
        self._service_token = service_token
        self._service_token_header = service_token_header
        self._tenant_header = tenant_header
        self._timeout = timeout
        self._client = http_client
        self._owns_client = False
        if self._client is None and self._base_url:
            self._client = httpx.AsyncClient(timeout=timeout)
            self._owns_client = True

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()

    async def emit(self, event: AuditEvent) -> None:
        if not self._base_url or not self._client:
            LOGGER.debug("Audit service not configured; skipping event %s", event.action)
            return

        url = f"{self._base_url}/api/v1/audit/events"
        headers = {"Content-Type": "application/json"}
        if self._service_token and self._service_token_header:
            headers[self._service_token_header] = self._service_token
        if self._tenant_header and event.tenant_id:
            headers[self._tenant_header] = event.tenant_id
        try:
            response = await self._client.post(
                url,
                json=event.as_payload(),
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.warning("Failed to deliver audit event %s: %s", event.action, exc)


__all__ = ["AuditClient", "AuditEvent"]
