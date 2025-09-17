from __future__ import annotations

"""Common data contracts shared across LegacyCoinTrader microservices."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ServiceMetadata(BaseModel):
    """Basic information required to register a service with a discovery backend."""

    name: str
    version: str = Field(default="1.0.0", description="Semantic version of the service implementation.")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=0)
    scheme: str = Field(default="http", pattern=r"https?")
    namespace: Optional[str] = Field(default=None)
    tags: List[str] = Field(default_factory=list)
    health_endpoint: str = Field(default="/health")
    readiness_endpoint: str = Field(default="/readiness")
    metrics_endpoint: Optional[str] = Field(default=None)
    grpc_port: Optional[int] = Field(default=None)

    @validator("health_endpoint", "readiness_endpoint", "metrics_endpoint", pre=True)
    def _ensure_leading_slash(cls, value: Optional[str]) -> Optional[str]:  # noqa: D401 - small helper
        """Make sure endpoints are expressed as absolute paths."""

        if value is None:
            return None
        if not value.startswith("/"):
            return f"/{value}"
        return value

    def base_url(self) -> str:
        """Return the canonical URL for the HTTP interface."""

        return f"{self.scheme}://{self.host}:{self.port}".rstrip(":0")

    def health_url(self) -> str:
        return f"{self.base_url()}{self.health_endpoint}"

    def readiness_url(self) -> str:
        return f"{self.base_url()}{self.readiness_endpoint}"


class HealthCheckResult(BaseModel):
    """Represents the outcome of a health probe for a service."""

    status: str = Field(default="unknown")
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = Field(default_factory=dict)


class ServiceRegistration(BaseModel):
    """Snapshot of a registered service within the discovery backend."""

    metadata: ServiceMetadata
    status: str = Field(default="unknown")
    checks: List[HealthCheckResult] = Field(default_factory=list)


class HttpEndpoint(BaseModel):
    """Declarative description of a REST/HTTP endpoint exposed by a service."""

    method: str = Field(default="GET")
    path: str = Field(..., description="Relative URL path of the endpoint.")
    summary: Optional[str] = Field(default=None)
    request_model: Optional[str] = Field(default=None, description="Qualified name of the request model.")
    response_model: Optional[str] = Field(default=None, description="Qualified name of the response model.")


class GrpcMethodDescriptor(BaseModel):
    """Metadata about a gRPC method exposed by a microservice."""

    name: str
    request: str
    response: str
    client_streaming: bool = Field(default=False)
    server_streaming: bool = Field(default=False)


class EventEnvelope(BaseModel):
    """Standard event structure for inter-service messaging."""

    event_type: str
    source: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    emitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(default="1.0")
    correlation_id: Optional[str] = Field(default=None)
    subject: Optional[str] = Field(default=None)

    def enrich(self, **metadata: Any) -> "EventEnvelope":
        """Return a copy of the event with additional metadata merged into the payload."""

        merged = dict(self.payload)
        merged.update(metadata)
        return self.copy(update={"payload": merged})


__all__ = [
    "EventEnvelope",
    "GrpcMethodDescriptor",
    "HealthCheckResult",
    "HttpEndpoint",
    "ServiceMetadata",
    "ServiceRegistration",
]
