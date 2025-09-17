from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from redis import asyncio as redis_asyncio

from .config import ServiceRouteConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TenantContext:
    """Snapshot of tenant configuration relevant to request handling."""

    tenant_id: str
    slug: Optional[str]
    name: Optional[str]
    plan: str
    rate_limit_per_minute: int
    burst_limit: Optional[int]
    burst_window_seconds: int
    route_limits: Dict[str, int] = field(default_factory=dict)
    burst_limits: Dict[str, int] = field(default_factory=dict)
    scopes: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def limit_for_route(self, route_name: str, default: Optional[int]) -> Optional[int]:
        return self.route_limits.get(route_name, default)

    def burst_for_route(self, route_name: str, default: Optional[int]) -> Optional[int]:
        return self.burst_limits.get(route_name, default)


class TenantServiceClient:
    """Client for retrieving tenant metadata from the tenant-management service."""

    def __init__(
        self,
        route_config: Optional[ServiceRouteConfig],
        http_client: httpx.AsyncClient,
        *,
        cache_ttl: int = 60,
    ) -> None:
        self.route_config = route_config
        self.http_client = http_client
        self.cache_ttl = max(5, cache_ttl)
        self._cache: Dict[str, Tuple[TenantContext, float]] = {}
        self._cache_lock = asyncio.Lock()

    async def resolve(
        self, tenant_id: Optional[str], *, claims: Optional[Dict[str, Any]] = None
    ) -> Optional[TenantContext]:
        tenant_identifier = tenant_id or self._extract_tenant_from_claims(claims)
        if not tenant_identifier:
            return None

        cached = self._cache.get(tenant_identifier)
        now = time.monotonic()
        if cached and cached[1] > now:
            return cached[0]

        tenant = await self._fetch_tenant_metadata(tenant_identifier)
        if tenant is None and claims:
            tenant = self._build_context_from_claims(tenant_identifier, claims)

        if tenant is not None:
            async with self._cache_lock:
                self._cache[tenant_identifier] = (tenant, now + self.cache_ttl)
        return tenant

    async def list_tenants(self) -> List[TenantContext]:
        if not self.route_config:
            return []
        url = self.route_config.build_target_url("tenants")
        headers = self._auth_headers()
        try:
            response = await self.http_client.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.warning("Unable to list tenants from %s: %s", url, exc)
            return []

        payload = response.json()
        items: Iterable[Dict[str, Any]]
        if isinstance(payload, dict) and "items" in payload and isinstance(payload["items"], list):
            items = [item for item in payload["items"] if isinstance(item, dict)]
        elif isinstance(payload, list):
            items = [item for item in payload if isinstance(item, dict)]
        else:
            items = []

        contexts: List[TenantContext] = []
        for item in items:
            context = self._parse_context(item)
            if context:
                contexts.append(context)
        return contexts

    async def _fetch_tenant_metadata(self, tenant_id: str) -> Optional[TenantContext]:
        if not self.route_config:
            return None
        path = f"tenants/{tenant_id}"
        url = self.route_config.build_target_url(path)
        headers = self._auth_headers()
        try:
            response = await self.http_client.get(url, headers=headers, timeout=5)
        except httpx.HTTPError as exc:
            LOGGER.warning("Tenant lookup failed for %s: %s", tenant_id, exc)
            return None

        if response.status_code == 404:
            return None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            LOGGER.warning("Tenant metadata request returned %s: %s", response.status_code, exc)
            return None

        data = response.json()
        return self._parse_context(data)

    def _parse_context(self, payload: Dict[str, Any]) -> Optional[TenantContext]:
        tenant_payload = payload.get("tenant") if isinstance(payload.get("tenant"), dict) else payload
        tenant_id = tenant_payload.get("tenant_id") or tenant_payload.get("id") or tenant_payload.get("slug")
        if not tenant_id:
            return None

        plan = str(tenant_payload.get("plan") or tenant_payload.get("subscription_plan") or "standard")
        rate_limit = int(tenant_payload.get("rate_limit_per_minute") or tenant_payload.get("rate_limit") or 0)
        burst_config = tenant_payload.get("burst") or {}
        burst_limit = burst_config.get("requests") or tenant_payload.get("burst_limit")
        burst_window = burst_config.get("window_seconds") or tenant_payload.get("burst_window_seconds") or 60

        route_limits_raw = tenant_payload.get("route_limits") or tenant_payload.get("plan_route_limits") or {}
        burst_limits_raw = tenant_payload.get("burst_limits") or {}

        route_limits = self._normalise_mapping(route_limits_raw)
        burst_limits = self._normalise_mapping(burst_limits_raw)

        scopes = self._normalise_list(
            tenant_payload.get("scopes")
            or payload.get("scopes")
            or tenant_payload.get("permissions")
        )
        roles = self._normalise_list(
            tenant_payload.get("roles")
            or payload.get("roles")
        )

        slug = tenant_payload.get("slug") or tenant_payload.get("tenant_slug")
        name = tenant_payload.get("name") or payload.get("name")

        metadata = {
            key: value
            for key, value in tenant_payload.items()
            if key
            not in {
                "tenant_id",
                "id",
                "slug",
                "plan",
                "subscription_plan",
                "rate_limit_per_minute",
                "rate_limit",
                "scopes",
                "permissions",
                "roles",
                "burst",
                "burst_limit",
                "burst_limits",
                "burst_window_seconds",
                "route_limits",
                "plan_route_limits",
                "name",
            }
        }

        return TenantContext(
            tenant_id=str(tenant_id),
            slug=str(slug) if slug else None,
            name=str(name) if name else None,
            plan=plan,
            rate_limit_per_minute=rate_limit,
            burst_limit=int(burst_limit) if burst_limit else None,
            burst_window_seconds=int(burst_window),
            route_limits=route_limits,
            burst_limits=burst_limits,
            scopes=scopes,
            roles=roles,
            metadata=metadata,
        )

    def _build_context_from_claims(
        self, tenant_id: str, claims: Dict[str, Any]
    ) -> Optional[TenantContext]:
        plan = str(claims.get("tenant_plan") or claims.get("plan") or "standard")
        rate_limit = int(claims.get("tenant_rate_limit") or 0)
        burst_limit = claims.get("tenant_burst_limit")
        burst_window = int(claims.get("tenant_burst_window") or 60)
        scopes = self._normalise_list(
            claims.get("tenant_scopes") or claims.get("scopes")
        )
        roles = self._normalise_list(claims.get("roles"))
        slug = claims.get("tenant_slug")
        name = claims.get("tenant_name")

        return TenantContext(
            tenant_id=tenant_id,
            slug=str(slug) if slug else None,
            name=str(name) if name else None,
            plan=plan,
            rate_limit_per_minute=rate_limit,
            burst_limit=int(burst_limit) if burst_limit else None,
            burst_window_seconds=burst_window,
            route_limits={},
            burst_limits={},
            scopes=scopes,
            roles=roles,
            metadata={"source": "token"},
        )

    def _extract_tenant_from_claims(self, claims: Optional[Dict[str, Any]]) -> Optional[str]:
        if not claims:
            return None
        candidate = claims.get("tenant_id") or claims.get("tenant")
        if candidate:
            return str(candidate)
        return None

    def _auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.route_config and self.route_config.service_token:
            headers["X-Service-Token"] = self.route_config.service_token
        return headers

    @staticmethod
    def _normalise_list(value: Optional[Any]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item for item in value.replace(",", " ").split() if item]
        if isinstance(value, Iterable):
            return [str(item) for item in value if str(item)]
        return []

    @staticmethod
    def _normalise_mapping(value: Optional[Dict[str, Any]]) -> Dict[str, int]:
        if not isinstance(value, dict):
            return {}
        result: Dict[str, int] = {}
        for key, raw_value in value.items():
            try:
                result[str(key)] = int(raw_value)
            except (TypeError, ValueError):
                continue
        return result


class TenantUsageTracker:
    """Tracks tenant usage across Redis counters and a Kafka-mirrored buffer."""

    def __init__(
        self,
        redis_client: Optional[redis_asyncio.Redis],
        *,
        kafka_bootstrap_servers: Optional[str] = None,
    ) -> None:
        self.redis = redis_client
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self._kafka_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _redis_key(tenant_id: str, route_name: str) -> str:
        return f"rate-limit:tenant:{tenant_id}:{route_name}"

    async def combined_usage(self, tenant_id: str, route_name: str) -> int:
        redis_count = 0
        if self.redis:
            try:
                raw_value = await self.redis.get(self._redis_key(tenant_id, route_name))
                if raw_value:
                    redis_count = int(raw_value)
            except Exception as exc:  # pragma: no cover - redis unavailable
                LOGGER.debug("Unable to read redis counter for %s/%s: %s", tenant_id, route_name, exc)
        async with self._lock:
            kafka_count = self._kafka_counts.get(f"{tenant_id}:{route_name}", 0)
        return max(redis_count, kafka_count)

    async def increment(self, tenant_id: str, route_name: str, amount: int = 1) -> int:
        key = f"{tenant_id}:{route_name}"
        async with self._lock:
            self._kafka_counts[key] = self._kafka_counts.get(key, 0) + amount
            return self._kafka_counts[key]


__all__ = [
    "TenantContext",
    "TenantServiceClient",
    "TenantUsageTracker",
]
