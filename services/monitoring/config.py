"""Configuration models for the monitoring subsystem."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from pydantic import BaseModel, Field
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


class SLOSettings(BaseModel):
    """Targets and window definitions for tenant SLO calculations."""

    window_seconds: int = Field(default=300, ge=60)
    latency_target_ms: float = Field(default=500.0, ge=0.0)
    error_rate_target: float = Field(default=0.01, ge=0.0)
    throughput_target_rps: float = Field(default=1.0, ge=0.0)
    rto_target_seconds: float = Field(default=900.0, ge=0.0)
    rpo_target_seconds: float = Field(default=300.0, ge=0.0)


class ComplianceSettings(BaseModel):
    """Controls for audit/compliance metric exports."""

    enabled: bool = Field(default=True)
    secrets_rotated_env: str = Field(default="SECRETS_ROTATED_AT")
    fallback_timestamp: Optional[str] = Field(default=None)
    max_secret_age_days: float = Field(default=90.0, ge=0.0)


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
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    slo: SLOSettings = Field(default_factory=SLOSettings)
    compliance: ComplianceSettings = Field(default_factory=ComplianceSettings)

    def for_service(
        self,
        service_name: str,
        *,
        environment: Optional[str] = None,
        service_role: Optional[str] = None,
        default_tenant: Optional[str] = None,
    ) -> "MonitoringSettings":
        """Return a copy of the settings with a service-specific name."""

        update: Dict[str, str] = {"service_name": service_name}
        if environment is not None:
            update["environment"] = environment
        if service_role is not None:
            update["service_role"] = service_role
        if default_tenant is not None:
            update["default_tenant"] = default_tenant
        return self.model_copy(update=update)


@lru_cache(maxsize=8)
def get_monitoring_settings() -> MonitoringSettings:
    """Return cached monitoring settings loaded from the environment."""

    return MonitoringSettings()


__all__ = [
    "MetricsSettings",
    "OpenSearchSettings",
    "TracingSettings",
    "SLOSettings",
    "ComplianceSettings",
    "MonitoringSettings",
    "get_monitoring_settings",
]
