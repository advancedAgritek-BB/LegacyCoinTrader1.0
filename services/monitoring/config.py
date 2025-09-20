"""Configuration models for the monitoring subsystem."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class MetricsSettings(BaseModel):
    """Settings controlling Prometheus metrics exporters."""

    enabled: bool = Field(default=True)
    namespace: str = Field(default="legacycoin")
    default_labels: Dict[str, str] = Field(default_factory=dict)
    exporter_host: str = Field(default="0.0.0.0")
    exporter_port: int = Field(default=9000)


class OpenSearchSettings(BaseModel):
    """Configuration for centralised log shipping into OpenSearch."""

    enabled: bool = Field(default=True)
    host: str = Field(default="localhost")
    port: int = Field(default=9200)
    index: str = Field(default="legacycoin-logs")
    username: Optional[str] = Field(default=None)
    password: Optional[str] = Field(default=None)
    use_ssl: bool = Field(default=False)
    verify_certs: bool = Field(default=False)
    timeout: int = Field(default=5)


class TracingSettings(BaseModel):
    """Configuration for OpenTelemetry tracing export."""

    enabled: bool = Field(default=True)
    endpoint: str = Field(default="http://localhost:4318")
    headers: Dict[str, str] = Field(default_factory=dict)
    service_namespace: str = Field(default="legacycoin")


def _coerce_setting(value, model_cls):  # pragma: no cover - simple helper
    if isinstance(value, model_cls):
        return value
    if isinstance(value, FieldInfo):
        return model_cls()
    if isinstance(value, dict):
        return model_cls(**value)
    if hasattr(value, "model_dump"):
        return model_cls(**value.model_dump())
    return model_cls()


class MonitoringSettings(BaseSettings):
    """Composite configuration used to instrument services."""

    model_config = SettingsConfigDict(
        env_prefix="MONITORING_",
        env_nested_delimiter="__",
        extra="allow",
    )

    service_name: str = Field(default="legacycoin-service")
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    correlation_header: str = Field(default="X-Correlation-ID")
    service_role: str = Field(default="unspecified")
    default_tenant: str = Field(default="global")
    tenant_header: str = Field(default="X-Tenant-ID")
    service_role_header: str = Field(default="X-Service-Role")
    slo: Optional[Any] = Field(default=None)
    compliance: Optional[Any] = Field(default=None)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)

    def clone(self, **update: Any) -> "MonitoringSettings":
        payload = self.model_dump()
        payload.update(update)
        return _normalise_monitoring_settings(MonitoringSettings(**payload))

    def for_service(self, service_name: str, *, environment: Optional[str] = None) -> "MonitoringSettings":
        """Return a copy of the settings with a service-specific name."""

        update: Dict[str, str] = {"service_name": service_name}
        if environment is not None:
            update["environment"] = environment
        return self.clone(**update)


def _normalise_monitoring_settings(settings: MonitoringSettings) -> MonitoringSettings:
    settings.metrics = _coerce_setting(settings.metrics, MetricsSettings)
    settings.opensearch = _coerce_setting(settings.opensearch, OpenSearchSettings)
    settings.tracing = _coerce_setting(settings.tracing, TracingSettings)
    if not isinstance(settings.service_name, str):
        settings.service_name = str(getattr(settings.service_name, "default", settings.service_name) or "legacycoin-service")
    if not isinstance(settings.environment, str):
        settings.environment = str(getattr(settings.environment, "default", settings.environment) or "development")
    if not isinstance(settings.log_level, str):
        default = getattr(settings.log_level, "default", "INFO")
        settings.log_level = str(default or "INFO")
    if not isinstance(settings.correlation_header, str):
        default = getattr(settings.correlation_header, "default", "X-Correlation-ID")
        settings.correlation_header = str(default or "X-Correlation-ID")
    return settings


@lru_cache(maxsize=8)
def get_monitoring_settings() -> MonitoringSettings:
    """Return cached monitoring settings loaded from the environment."""
    return _normalise_monitoring_settings(MonitoringSettings())


__all__ = [
    "MetricsSettings",
    "OpenSearchSettings",
    "TracingSettings",
    "MonitoringSettings",
    "get_monitoring_settings",
]
