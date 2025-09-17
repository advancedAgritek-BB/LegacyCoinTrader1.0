from __future__ import annotations

"""Service discovery helpers for the LegacyCoinTrader microservices."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import httpx

from .contracts import HealthCheckResult, ServiceMetadata, ServiceRegistration

LOGGER = logging.getLogger(__name__)


class ServiceDiscoveryError(RuntimeError):
    """Raised when a service discovery backend interaction fails."""


@dataclass(slots=True)
class ServiceDiscoveryConfig:
    """Configuration describing how a service participates in discovery."""

    metadata: ServiceMetadata
    backend: str = "consul"
    consul_url: str = "http://localhost:8500"
    consul_token: Optional[str] = None
    namespace: Optional[str] = None
    datacenter: Optional[str] = None
    register: bool = True
    check_interval: int = 15
    check_timeout: int = 5
    deregister_after: int = 60
    request_timeout: float = 5.0
    additional_tags: Iterable[str] = field(default_factory=list)

    def service_id(self) -> str:
        base = self.metadata.name.replace(" ", "-")
        if self.namespace:
            base = f"{self.namespace}-{base}"
        return f"{base}-{self.metadata.host}-{self.metadata.port}".replace("/", "-")

    def combined_tags(self) -> List[str]:
        tags = list(self.metadata.tags)
        tags.extend(tag for tag in self.additional_tags if tag not in tags)
        if self.namespace and self.namespace not in tags:
            tags.append(self.namespace)
        return tags


class ServiceDiscoveryClient:
    """Thin asynchronous client for registering services with discovery backends."""

    def __init__(
        self,
        config: ServiceDiscoveryConfig,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.config = config
        timeout = httpx.Timeout(config.request_timeout, connect=config.request_timeout)
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._registered = False

    async def __aenter__(self) -> "ServiceDiscoveryClient":
        await self.register()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - FastAPI handles errors
        await self.close()

    async def register(self) -> None:
        if not self.config.register or self.config.backend == "none":
            LOGGER.info("Service discovery registration disabled for %s", self.config.metadata.name)
            return
        if self.config.backend.lower() == "consul":
            await self._register_consul()
        elif self.config.backend.lower() == "kubernetes":
            # Kubernetes deployments typically rely on declarative manifests. We simply log.
            LOGGER.info(
                "Kubernetes service discovery selected for %s – assuming manifest-driven registration",
                self.config.metadata.name,
            )
            self._registered = True
        else:
            LOGGER.warning("Unknown service discovery backend '%s'", self.config.backend)

    async def deregister(self) -> None:
        if not self._registered:
            return
        if self.config.backend.lower() == "consul":
            await self._deregister_consul()
        else:
            LOGGER.debug("Skipping deregistration for backend %s", self.config.backend)
        self._registered = False

    async def list_services(self) -> List[ServiceRegistration]:
        if self.config.backend.lower() == "consul":
            return await self._list_consul_services()
        if self._registered:
            return [
                ServiceRegistration(metadata=self.config.metadata, status="registered", checks=[])
            ]
        return []

    async def get_service(self, service_name: str) -> List[ServiceRegistration]:
        services = await self.list_services()
        return [service for service in services if service.metadata.name == service_name]

    async def close(self) -> None:
        await self.deregister()
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Consul helpers
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.config.consul_token:
            headers["X-Consul-Token"] = self.config.consul_token
        if self.config.namespace:
            headers["X-Consul-Namespace"] = self.config.namespace
        if self.config.datacenter:
            headers["X-Consul-Datacenter"] = self.config.datacenter
        return headers

    async def _register_consul(self) -> None:
        payload = {
            "ID": self.config.service_id(),
            "Name": self.config.metadata.name,
            "Tags": self.config.combined_tags(),
            "Port": self.config.metadata.port,
            "Address": self.config.metadata.host,
            "Meta": {
                "version": self.config.metadata.version,
                "scheme": self.config.metadata.scheme,
                "health": self.config.metadata.health_endpoint,
                "readiness": self.config.metadata.readiness_endpoint,
            },
        }
        if self.config.metadata.metrics_endpoint:
            payload["Meta"]["metrics"] = self.config.metadata.metrics_endpoint
        checks = [
            {
                "Name": f"{self.config.metadata.name}-http-health",
                "HTTP": self.config.metadata.health_url(),
                "Method": "GET",
                "Interval": f"{self.config.check_interval}s",
                "Timeout": f"{self.config.check_timeout}s",
                "DeregisterCriticalServiceAfter": f"{self.config.deregister_after}s",
            }
        ]
        if self.config.metadata.grpc_port:
            checks.append(
                {
                    "Name": f"{self.config.metadata.name}-grpc",
                    "GRPC": f"{self.config.metadata.host}:{self.config.metadata.grpc_port}",
                    "GRPCUseTLS": self.config.metadata.scheme == "https",
                    "Interval": f"{self.config.check_interval}s",
                }
            )
        payload["Checks"] = checks

        try:
            url = f"{self.config.consul_url.rstrip('/')}/v1/agent/service/register"
            response = await self._client.put(url, json=payload, headers=self._headers())
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failures hard to reproduce in tests
            raise ServiceDiscoveryError(f"Failed to register service with Consul: {exc}") from exc
        self._registered = True
        LOGGER.info("Registered %s with Consul at %s", self.config.metadata.name, self.config.consul_url)

    async def _deregister_consul(self) -> None:
        try:
            url = f"{self.config.consul_url.rstrip('/')}/v1/agent/service/deregister/{self.config.service_id()}"
            response = await self._client.put(url, headers=self._headers())
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - depends on runtime environment
            raise ServiceDiscoveryError(f"Failed to deregister service from Consul: {exc}") from exc
        LOGGER.info("Deregistered %s from Consul", self.config.metadata.name)

    async def _list_consul_services(self) -> List[ServiceRegistration]:
        try:
            services_url = f"{self.config.consul_url.rstrip('/')}/v1/agent/services"
            services_response = await self._client.get(services_url, headers=self._headers())
            services_response.raise_for_status()
            raw_services = services_response.json()

            checks_url = f"{self.config.consul_url.rstrip('/')}/v1/agent/checks"
            checks_response = await self._client.get(checks_url, headers=self._headers())
            checks_response.raise_for_status()
            raw_checks = checks_response.json()
        except httpx.HTTPError as exc:  # pragma: no cover - best effort discovery
            raise ServiceDiscoveryError(f"Failed to query Consul services: {exc}") from exc

        checks_by_service: Dict[str, List[HealthCheckResult]] = {}
        for check in raw_checks.values():
            service_id = check.get("ServiceID")
            if not service_id:
                continue
            result = HealthCheckResult(
                status=str(check.get("Status", "unknown")),
                details={
                    "name": check.get("Name"),
                    "output": check.get("Output"),
                    "check_id": check.get("CheckID"),
                },
            )
            checks_by_service.setdefault(service_id, []).append(result)

        services: List[ServiceRegistration] = []
        for entry in raw_services.values():
            meta = entry.get("Meta") or {}
            metadata = ServiceMetadata(
                name=str(entry.get("Service") or entry.get("ID") or "unknown"),
                host=str(entry.get("Address") or "127.0.0.1"),
                port=int(entry.get("Port") or 0),
                tags=list(entry.get("Tags") or []),
                version=str(meta.get("version", "unknown")),
                scheme=str(meta.get("scheme", "http")),
                health_endpoint=str(meta.get("health", "/health")),
                readiness_endpoint=str(meta.get("readiness", "/readiness")),
                metrics_endpoint=meta.get("metrics"),
            )
            checks = checks_by_service.get(entry.get("ID"), [])
            status = "passing"
            if any(check.status not in {"passing", "up"} for check in checks):
                status = "critical"
            services.append(ServiceRegistration(metadata=metadata, status=status, checks=checks))
        return services


__all__ = [
    "ServiceDiscoveryClient",
    "ServiceDiscoveryConfig",
    "ServiceDiscoveryError",
]
