from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import httpx
from redis import asyncio as redis_asyncio

try:  # pragma: no cover - optional dependency
    from aiokafka import AIOKafkaProducer
except Exception:  # pragma: no cover - optional dependency
    AIOKafkaProducer = None  # type: ignore[misc,assignment]

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TenantPlan:
    """Describes rate limit policy and plan metadata for a tenant."""

    tenant_id: str
    plan: str
    rate_limit_per_minute: int
    burst_limit: int
    burst_window_seconds: int
    route_overrides: Dict[str, int] = field(default_factory=dict)
    burst_overrides: Dict[str, Dict[str, int]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_defaults(
        cls,
        tenant_id: str,
        plan: str,
        rate_limit_per_minute: int,
        burst_limit: int,
        burst_window_seconds: int,
    ) -> "TenantPlan":
        return cls(
            tenant_id=tenant_id,
            plan=plan,
            rate_limit_per_minute=rate_limit_per_minute,
            burst_limit=burst_limit,
            burst_window_seconds=burst_window_seconds,
        )

    def limit_for_route(self, route_name: str, fallback: int) -> int:
        """Return the effective steady-state rate limit for a route."""

        if route_name in self.route_overrides:
            return int(self.route_overrides[route_name])
        if self.rate_limit_per_minute > 0:
            return int(self.rate_limit_per_minute)
        return int(fallback)

    def burst_for_route(
        self,
        route_name: str,
        fallback_limit: int,
        fallback_window_seconds: int,
    ) -> tuple[int, int]:
        """Return the burst limit/window for the route."""

        override = self.burst_overrides.get(route_name) or {}
        limit = int(override.get("limit", self.burst_limit or fallback_limit))
        window = int(override.get("window_seconds", self.burst_window_seconds or fallback_window_seconds))
        return max(limit, 0), max(window, 1)


class TenantServiceClient:
    """Client wrapper for retrieving tenant metadata from the tenant service."""

    def __init__(
        self,
        base_url: Optional[str],
        *,
        cache_seconds: int,
        service_token: Optional[str],
        service_token_header: str,
        timeout: float,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/") if base_url else None
        self._cache_seconds = max(cache_seconds, 0)
        self._service_token = service_token
        self._service_token_header = service_token_header
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._cache: Dict[str, tuple[TenantPlan, float]] = {}
        self._client = http_client
        self._owns_client = False
        if self._client is None and self._base_url:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()

    async def get_tenant_plan(
        self,
        tenant_id: str,
        *,
        fallback: TenantPlan,
    ) -> TenantPlan:
        """Return tenant metadata, applying caching and fallback behaviour."""

        tenant_id = tenant_id or fallback.tenant_id
        now = time.monotonic()
        async with self._lock:
            cached = self._cache.get(tenant_id)
            if cached and cached[1] > now:
                return cached[0]

        if not self._base_url or not self._client:
            plan = self._clone_plan(fallback, tenant_id)
            await self._store_plan(tenant_id, plan)
            return plan

        url = f"{self._base_url}/api/v1/tenants/{tenant_id}"
        headers = {"Accept": "application/json"}
        if self._service_token and self._service_token_header:
            headers[self._service_token_header] = self._service_token
        try:
            response = await self._client.get(url, headers=headers, timeout=self._timeout)
        except httpx.HTTPError as exc:
            LOGGER.warning("Failed to fetch tenant metadata for %s: %s", tenant_id, exc)
            plan = self._clone_plan(fallback, tenant_id)
            await self._store_plan(tenant_id, plan, stale=True)
            return plan

        if response.status_code == 404:
            LOGGER.info("Tenant %s not recognised by tenant service; using fallback", tenant_id)
            plan = self._clone_plan(fallback, tenant_id)
            await self._store_plan(tenant_id, plan)
            return plan

        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.warning("Tenant service error for %s: %s", tenant_id, exc)
            plan = self._clone_plan(fallback, tenant_id)
            await self._store_plan(tenant_id, plan, stale=True)
            return plan

        payload = response.json()
        plan = self._plan_from_payload(tenant_id, payload, fallback)
        await self._store_plan(tenant_id, plan)
        return plan

    async def invalidate(self, tenant_id: str) -> None:
        """Invalidate a cached tenant entry."""

        async with self._lock:
            self._cache.pop(tenant_id, None)

    async def _store_plan(self, tenant_id: str, plan: TenantPlan, stale: bool = False) -> None:
        ttl = time.monotonic() + self._cache_seconds if self._cache_seconds else float("inf")
        if stale:
            ttl = time.monotonic() + min(self._cache_seconds or 30, 30)
        async with self._lock:
            self._cache[tenant_id] = (plan, ttl)

    def _clone_plan(self, template: TenantPlan, tenant_id: str) -> TenantPlan:
        return TenantPlan(
            tenant_id=tenant_id,
            plan=template.plan,
            rate_limit_per_minute=template.rate_limit_per_minute,
            burst_limit=template.burst_limit,
            burst_window_seconds=template.burst_window_seconds,
            route_overrides=dict(template.route_overrides),
            burst_overrides={key: dict(value) for key, value in template.burst_overrides.items()},
            metadata=dict(template.metadata),
        )

    def _plan_from_payload(
        self,
        tenant_id: str,
        payload: Dict[str, object],
        fallback: TenantPlan,
    ) -> TenantPlan:
        plan_info = payload.get("plan") if isinstance(payload, dict) else None
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        route_overrides = payload.get("route_overrides") if isinstance(payload, dict) else None
        burst_overrides = payload.get("burst_overrides") if isinstance(payload, dict) else None

        def _as_int(value: object, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return int(default)

        rate_limit = fallback.rate_limit_per_minute
        burst_limit = fallback.burst_limit
        burst_window = fallback.burst_window_seconds
        plan_name = fallback.plan

        if isinstance(plan_info, dict):
            rate_limit = _as_int(plan_info.get("rate_limit_per_minute"), rate_limit)
            burst_limit = _as_int(plan_info.get("burst_limit"), burst_limit)
            burst_window = _as_int(plan_info.get("burst_window_seconds"), burst_window)
            plan_name = str(plan_info.get("name") or plan_name)

        overrides: Dict[str, int] = {}
        if isinstance(route_overrides, dict):
            for key, value in route_overrides.items():
                overrides[str(key)] = _as_int(value, rate_limit)

        burst_data: Dict[str, Dict[str, int]] = {}
        if isinstance(burst_overrides, dict):
            for key, value in burst_overrides.items():
                if isinstance(value, dict):
                    burst_data[str(key)] = {
                        "limit": _as_int(value.get("limit"), burst_limit),
                        "window_seconds": _as_int(value.get("window_seconds"), burst_window),
                    }

        if not isinstance(metadata, dict):
            metadata = {}

        return TenantPlan(
            tenant_id=tenant_id,
            plan=plan_name,
            rate_limit_per_minute=rate_limit,
            burst_limit=burst_limit,
            burst_window_seconds=burst_window,
            route_overrides=overrides,
            burst_overrides=burst_data,
            metadata=metadata,
        )


class TenantUsageTracker:
    """Record per-tenant usage statistics using Redis and optional Kafka."""

    def __init__(
        self,
        *,
        redis_client: Optional[redis_asyncio.Redis],
        kafka_bootstrap_servers: Optional[str],
        kafka_topic: str,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.redis = redis_client
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self._loop = loop or asyncio.get_event_loop()
        self._producer: Optional[AIOKafkaProducer] = None
        self._kafka_lock = asyncio.Lock()

    async def start(self) -> None:
        if not self.kafka_bootstrap_servers or AIOKafkaProducer is None:
            return
        async with self._kafka_lock:
            if self._producer is not None:
                return
            producer = AIOKafkaProducer(
                bootstrap_servers=self.kafka_bootstrap_servers,
                loop=self._loop,
                value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            )
            try:
                await producer.start()
                self._producer = producer
                LOGGER.info(
                    "Tenant usage tracker connected to Kafka topic %s", self.kafka_topic
                )
            except Exception as exc:  # pragma: no cover - dependent on runtime env
                LOGGER.warning("Unable to start Kafka producer: %s", exc)

    async def stop(self) -> None:
        async with self._kafka_lock:
            producer = self._producer
            self._producer = None
        if producer is not None:
            try:
                await producer.stop()
            except Exception:  # pragma: no cover - dependent on runtime env
                LOGGER.debug("Failed to stop Kafka producer", exc_info=True)

    async def record_request(
        self,
        *,
        tenant_id: str,
        route_name: str,
        rate_limit: int,
        allowed: bool,
        remaining: int,
        burst_limit: Optional[int] = None,
        burst_allowed: Optional[bool] = None,
        burst_remaining: Optional[int] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """Update Redis counters and publish an optional Kafka event."""

        metadata = metadata or {}
        if self.redis is not None:
            try:
                key = f"tenant-usage:{tenant_id}"
                await self.redis.hincrby(key, f"{route_name}:requests", 1)
                await self.redis.hset(key, f"{route_name}:remaining", max(remaining, 0))
                if burst_limit is not None and burst_remaining is not None:
                    await self.redis.hset(key, f"{route_name}:burst_remaining", max(burst_remaining, 0))
                await self.redis.expire(key, 86_400)
            except Exception:  # pragma: no cover - defensive guardrail
                LOGGER.debug("Failed to persist tenant usage counters", exc_info=True)

        producer = self._producer
        if producer is None:
            return

        event = {
            "tenant": tenant_id,
            "route": route_name,
            "allowed": allowed,
            "rate_limit": rate_limit,
            "remaining": remaining,
            "burst": {
                "limit": burst_limit,
                "allowed": burst_allowed,
                "remaining": burst_remaining,
            },
            "metadata": metadata,
            "timestamp": time.time(),
        }
        try:
            await producer.send_and_wait(self.kafka_topic, event)
        except Exception:  # pragma: no cover - dependent on runtime env
            LOGGER.debug("Failed to publish tenant usage metric", exc_info=True)


__all__ = [
    "TenantPlan",
    "TenantServiceClient",
    "TenantUsageTracker",
]
