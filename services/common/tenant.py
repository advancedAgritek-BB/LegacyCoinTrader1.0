"""Shared tenant context utilities for multi-tenant services.

This module provides a common registry and FastAPI middleware that extracts a
tenant identifier from inbound requests, loads the associated configuration and
risk policies, and exposes the resolved context to downstream service layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from services.configuration.managed_config_service import deep_merge


class TenantConfigurationError(RuntimeError):
    """Raised when tenant configuration cannot be loaded or parsed."""


class TenantNotFoundError(KeyError):
    """Raised when the requested tenant does not exist in the registry."""


class TenantLimitError(RuntimeError):
    """Raised when a tenant level limit (cycles, risk budget, etc.) is exceeded."""


@dataclass(frozen=True)
class TenantRiskPolicy:
    """Declarative risk controls configured for a tenant."""

    max_active_cycles: int = 1
    risk_budget: Optional[float] = None

    def validate_cycle_start(self, active_cycles: int) -> None:
        """Ensure the tenant has capacity for another active cycle."""

        if self.max_active_cycles <= 0:
            raise TenantLimitError("Tenant is not allowed to run trading cycles")
        if active_cycles >= self.max_active_cycles:
            raise TenantLimitError(
                "Tenant has reached the maximum number of concurrent trading cycles"
            )

    def validate_allocation(self, allocation: Optional[float]) -> float:
        """Validate a requested risk allocation, returning the normalised value."""

        value = float(allocation or 0.0)
        if value < 0:
            raise TenantLimitError("Risk allocation cannot be negative")
        if self.risk_budget is not None and value > self.risk_budget:
            raise TenantLimitError(
                "Requested risk allocation exceeds the tenant risk budget"
            )
        return value

    def validate_additional_allocation(self, current: float, delta: float) -> float:
        """Validate increasing an allocation by ``delta`` returning the new total."""

        if delta < 0:
            raise TenantLimitError("Risk allocation increment cannot be negative")
        total = float(current) + float(delta)
        if self.risk_budget is not None and total > self.risk_budget:
            raise TenantLimitError(
                "Risk budget exhausted for this tenant"
            )
        return total


@dataclass(frozen=True)
class TenantExecutionConfig:
    """Execution specific overrides for a tenant."""

    overrides: Mapping[str, Any] = field(default_factory=dict)
    credentials: Mapping[str, Any] = field(default_factory=dict)
    network_segment: Optional[str] = None
    failover_endpoints: tuple[str, ...] = ()


@dataclass(frozen=True)
class TenantConfiguration:
    """Materialised tenant configuration loaded from ``config/tenants.yaml``."""

    tenant_id: str
    name: str
    redis_namespace: str
    risk: TenantRiskPolicy
    execution: TenantExecutionConfig
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TenantContext:
    """Lightweight accessor that exposes tenant metadata to services."""

    __slots__ = {"_config"}

    def __init__(self, config: TenantConfiguration) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def tenant_id(self) -> str:
        return self._config.tenant_id

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def risk_policy(self) -> TenantRiskPolicy:
        return self._config.risk

    @property
    def execution_config(self) -> TenantExecutionConfig:
        return self._config.execution

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def redis_namespace(self, namespace: Optional[str] = None) -> str:
        """Return a namespaced Redis prefix for the tenant."""

        base = self._config.redis_namespace.format(tenant=self.tenant_id)
        if namespace:
            return f"{base}:{namespace}"
        return base

    def redis_key(self, *parts: str) -> str:
        """Compose a fully qualified Redis key scoped to the tenant."""

        segments = [self.redis_namespace()]
        segments.extend(part for part in parts if part)
        return ":".join(segments)

    def channel(self, base: str) -> str:
        """Return a tenant-scoped pub/sub channel name."""

        channel = base.rstrip(":")
        return f"{channel}:{self.tenant_id}"

    def apply_config(self, base_config: Mapping[str, Any]) -> Dict[str, Any]:
        """Merge tenant overrides with a base configuration mapping."""

        merged = deep_merge(dict(base_config), dict(self._config.config_overrides))
        merged.setdefault("tenant", {})
        if isinstance(merged["tenant"], Mapping):
            merged["tenant"] = dict(merged["tenant"])
            merged["tenant"].setdefault("id", self.tenant_id)
            merged["tenant"].setdefault("name", self.name)
        return merged

    def execution_overrides(self, config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Merge execution overrides and credentials with a request config."""

        merged = deep_merge(dict(config or {}), dict(self.execution_config.overrides))
        if self.execution_config.credentials:
            merged.setdefault("credentials", {})
            merged["credentials"] = deep_merge(
                dict(merged["credentials"]), dict(self.execution_config.credentials)
            )
        if self.execution_config.network_segment:
            merged.setdefault("network_segment", self.execution_config.network_segment)
        if self.execution_config.failover_endpoints:
            merged.setdefault(
                "failover_endpoints", list(self.execution_config.failover_endpoints)
            )
        merged.setdefault("tenant_id", self.tenant_id)
        return merged

    def enrich_metadata(
        self, metadata: Optional[Mapping[str, Any]], *, base: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """Merge tenant details into metadata dictionaries."""

        merged: Dict[str, Any] = dict(base or {})
        if metadata:
            merged.update(dict(metadata))
        merged.setdefault("tenant_id", self.tenant_id)
        merged.setdefault("tenant", {"id": self.tenant_id, "name": self.name})
        return merged


class TenantRegistry:
    """In-memory registry of tenant configurations backed by YAML."""

    def __init__(self, *, path: Path | str | None = None) -> None:
        manifest_path = Path(path) if path else DEFAULT_TENANT_CONFIG_PATH
        if not manifest_path.exists():
            raise TenantConfigurationError(
                f"Tenant configuration file not found: {manifest_path}"
            )
        try:
            raw = yaml.safe_load(manifest_path.read_text()) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - configuration error
            raise TenantConfigurationError(str(exc)) from exc
        tenants_data = raw.get("tenants", {}) if isinstance(raw, Mapping) else {}
        if not isinstance(tenants_data, Mapping):
            raise TenantConfigurationError("Tenant configuration must contain a mapping")
        self._configs: Dict[str, TenantConfiguration] = {}
        for tenant_id, payload in tenants_data.items():
            if not isinstance(payload, Mapping):
                continue
            config = self._parse_tenant(str(tenant_id), payload)
            self._configs[config.tenant_id] = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_tenant(
        self, tenant_id: str, data: Mapping[str, Any]
    ) -> TenantConfiguration:
        name = str(data.get("name", tenant_id))
        namespace_template = str(data.get("redis_namespace", "tenant:{tenant}"))
        risk_cfg = data.get("risk", {}) or {}
        risk_policy = TenantRiskPolicy(
            max_active_cycles=int(risk_cfg.get("max_active_cycles", 1)),
            risk_budget=(
                float(risk_cfg["risk_budget"])
                if risk_cfg.get("risk_budget") is not None
                else None
            ),
        )
        execution_cfg = data.get("execution", {}) or {}
        exec_config = TenantExecutionConfig(
            overrides=dict(execution_cfg.get("overrides", {})),
            credentials=dict(execution_cfg.get("credentials", {})),
            network_segment=execution_cfg.get("network_segment"),
            failover_endpoints=tuple(execution_cfg.get("failover_endpoints", []) or ()),
        )
        config_overrides = dict(data.get("config_overrides", data.get("config", {})) or {})
        metadata = dict(data.get("metadata", {}))
        return TenantConfiguration(
            tenant_id=tenant_id,
            name=name,
            redis_namespace=namespace_template,
            risk=risk_policy,
            execution=exec_config,
            config_overrides=config_overrides,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def tenants(self) -> Iterable[TenantConfiguration]:
        return tuple(self._configs.values())

    def get(self, tenant_id: str) -> TenantContext:
        try:
            config = self._configs[tenant_id]
        except KeyError as exc:
            raise TenantNotFoundError(tenant_id) from exc
        return TenantContext(config)


DEFAULT_TENANT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "tenants.yaml"
)


@lru_cache
def get_tenant_registry() -> TenantRegistry:
    """Return a cached tenant registry instance."""

    return TenantRegistry()


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Middleware that attaches a :class:`TenantContext` to each request."""

    def __init__(
        self,
        app,
        *,
        header_name: str = "x-tenant-id",
        registry: TenantRegistry | None = None,
    ) -> None:
        super().__init__(app)
        self._header_name = header_name.lower()
        self._registry = registry or get_tenant_registry()

    async def dispatch(self, request: Request, call_next):
        tenant_id = self._extract_tenant_id(request)
        if not tenant_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "Missing tenant identifier in request headers"},
            )
        try:
            context = self._registry.get(tenant_id)
        except TenantNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Unknown tenant '{tenant_id}'"},
            )
        request.state.tenant_context = context
        return await call_next(request)

    def _extract_tenant_id(self, request: Request) -> Optional[str]:
        header_value = request.headers.get(self._header_name)
        if header_value:
            return header_value.strip()
        # Support canonical header casing (X-Tenant-ID)
        alt_value = request.headers.get("X-Tenant-ID")
        if alt_value:
            return alt_value.strip()
        # Allow query parameter fallback for non-HTTP transports (e.g. WebSockets)
        query_value = request.query_params.get("tenant_id")
        if query_value:
            return query_value.strip()
        return None


def get_tenant_context(request: Request) -> TenantContext:
    """FastAPI dependency that returns the tenant context for a request."""

    context: TenantContext | None = getattr(request.state, "tenant_context", None)
    if context is None:
        raise HTTPException(
            status_code=400,
            detail="Tenant context is not available for this request",
        )
    return context


class TenantContextClient:
    """Utility for background jobs to resolve :class:`TenantContext` instances."""

    def __init__(self, registry: TenantRegistry | None = None) -> None:
        self._registry = registry or get_tenant_registry()

    def get(self, tenant_id: str) -> TenantContext:
        return self._registry.get(tenant_id)


__all__ = [
    "TenantConfigurationError",
    "TenantContext",
    "TenantContextClient",
    "TenantContextMiddleware",
    "TenantLimitError",
    "TenantNotFoundError",
    "TenantRegistry",
    "TenantRiskPolicy",
    "get_tenant_context",
    "get_tenant_registry",
]

