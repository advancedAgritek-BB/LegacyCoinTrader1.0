"""Prometheus instrumentation helpers."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

from crypto_bot.utils.logger import get_service_role, get_tenant_id

from .config import ComplianceSettings, MetricsSettings, SLOSettings
from .slo import SyntheticMonitor, TenantSLOAggregator, TenantSLOSnapshot


@dataclass
class HttpMetrics:
    """Track HTTP metrics, tenant SLOs and synthetic guard-rails for a service."""

    service_name: str
    environment: str
    settings: MetricsSettings
    slo_settings: SLOSettings
    compliance_settings: ComplianceSettings
    default_tenant: str
    default_service_role: str
    registry: CollectorRegistry = REGISTRY
    request_counter: Counter = field(init=False)
    error_counter: Counter = field(init=False)
    latency_histogram: Histogram = field(init=False)
    slo_latency_gauge: Gauge = field(init=False)
    slo_error_rate_gauge: Gauge = field(init=False)
    slo_throughput_gauge: Gauge = field(init=False)
    synthetic_status_gauge: Gauge = field(init=False)
    synthetic_latency_gauge: Gauge = field(init=False)
    synthetic_rto_gauge: Gauge = field(init=False)
    synthetic_rpo_gauge: Gauge = field(init=False)
    compliance_rotation_timestamp_gauge: Gauge = field(init=False)
    compliance_rotation_age_gauge: Gauge = field(init=False)
    slo_aggregator: TenantSLOAggregator = field(init=False)
    synthetic_monitor: SyntheticMonitor = field(init=False)
    _static_labels: Dict[str, str] = field(init=False)
    _secrets_rotation_epoch: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.slo_aggregator = TenantSLOAggregator(
            window_seconds=self.slo_settings.window_seconds,
            latency_target_ms=self.slo_settings.latency_target_ms,
            error_rate_target=self.slo_settings.error_rate_target,
            throughput_target_rps=self.slo_settings.throughput_target_rps,
        )
        self.synthetic_monitor = SyntheticMonitor(
            rto_target_seconds=self.slo_settings.rto_target_seconds,
            rpo_target_seconds=self.slo_settings.rpo_target_seconds,
        )
        self._static_labels = {
            "service": self.service_name,
            "environment": self.environment,
        }
        if self.settings.default_labels:
            self._static_labels.update(self.settings.default_labels)
        self._secrets_rotation_epoch = self._resolve_secrets_rotation_epoch()
        self._initialise_metrics(self.registry)
        if self._secrets_rotation_epoch is not None:
            self._publish_compliance_metrics(
                self.default_tenant, self.default_service_role
            )

    def _initialise_metrics(self, registry: CollectorRegistry) -> None:
        extra = []
        if self.settings.default_labels:
            extra = sorted(self.settings.default_labels.keys())
        base = ["service", "environment", *extra, "tenant", "service_role"]
        request_labels = [*base, "method", "route", "status"]
        latency_labels = [*base, "method", "route"]
        slo_labels = base
        synthetic_labels = [*base, "check"]
        namespace = self.settings.namespace
        try:
            self.request_counter = Counter(
                "http_requests_total",
                "Total count of HTTP requests.",
                request_labels,
                namespace=namespace,
                registry=registry,
            )
            self.error_counter = Counter(
                "http_request_errors_total",
                "Total count of failed HTTP requests.",
                request_labels,
                namespace=namespace,
                registry=registry,
            )
            self.latency_histogram = Histogram(
                "http_request_duration_seconds",
                "Latency of HTTP requests in seconds.",
                latency_labels,
                namespace=namespace,
                registry=registry,
            )
            self.slo_latency_gauge = Gauge(
                "tenant_latency_p95_seconds",
                "95th percentile latency per tenant/service role over the SLO window.",
                slo_labels,
                namespace=namespace,
                registry=registry,
            )
            self.slo_error_rate_gauge = Gauge(
                "tenant_error_rate",
                "Error rate per tenant/service role over the SLO window.",
                slo_labels,
                namespace=namespace,
                registry=registry,
            )
            self.slo_throughput_gauge = Gauge(
                "tenant_throughput_rps",
                "Average request throughput per tenant/service role (requests per second).",
                slo_labels,
                namespace=namespace,
                registry=registry,
            )
            self.synthetic_status_gauge = Gauge(
                "synthetic_check_status",
                "Synthetic check compliance status (1 = passing).",
                synthetic_labels,
                namespace=namespace,
                registry=registry,
            )
            self.synthetic_latency_gauge = Gauge(
                "synthetic_check_latency_seconds",
                "Latency of synthetic checks in seconds.",
                synthetic_labels,
                namespace=namespace,
                registry=registry,
            )
            self.synthetic_rto_gauge = Gauge(
                "synthetic_check_recovery_time_seconds",
                "Observed recovery time from synthetic checks.",
                synthetic_labels,
                namespace=namespace,
                registry=registry,
            )
            self.synthetic_rpo_gauge = Gauge(
                "synthetic_check_data_lag_seconds",
                "Observed data lag from synthetic checks.",
                synthetic_labels,
                namespace=namespace,
                registry=registry,
            )
            self.compliance_rotation_timestamp_gauge = Gauge(
                "compliance_secrets_rotation_timestamp_seconds",
                "Timestamp of the last secrets rotation for compliance tracking.",
                slo_labels,
                namespace=namespace,
                registry=registry,
            )
            self.compliance_rotation_age_gauge = Gauge(
                "compliance_secrets_rotation_age_days",
                "Age in days since the last secrets rotation event.",
                slo_labels,
                namespace=namespace,
                registry=registry,
            )
        except ValueError:
            fresh_registry = CollectorRegistry()
            self.registry = fresh_registry
            self._initialise_metrics(fresh_registry)
            return
        self.registry = registry

    def _context_labels(self, tenant: str, service_role: str) -> Dict[str, str]:
        labels = dict(self._static_labels)
        labels.update({"tenant": tenant, "service_role": service_role})
        return labels

    def _resolve_secrets_rotation_epoch(self) -> Optional[float]:
        if not self.compliance_settings.enabled:
            return None
        env_key = self.compliance_settings.secrets_rotated_env
        value = os.getenv(env_key)
        if not value and self.compliance_settings.fallback_timestamp:
            value = self.compliance_settings.fallback_timestamp
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                return None
        return parsed.astimezone(timezone.utc).timestamp()

    def _publish_compliance_metrics(self, tenant: str, service_role: str) -> None:
        if self._secrets_rotation_epoch is None:
            return
        labels = self._context_labels(tenant, service_role)
        now = time.time()
        self.compliance_rotation_timestamp_gauge.labels(**labels).set(
            self._secrets_rotation_epoch
        )
        age_days = max((now - self._secrets_rotation_epoch) / 86400.0, 0.0)
        self.compliance_rotation_age_gauge.labels(**labels).set(age_days)

    def _update_slo_gauges(self, snapshot: TenantSLOSnapshot) -> None:
        labels = self._context_labels(snapshot.tenant, snapshot.service_role)
        self.slo_latency_gauge.labels(**labels).set(snapshot.p95_latency_ms / 1000.0)
        self.slo_error_rate_gauge.labels(**labels).set(snapshot.error_rate)
        self.slo_throughput_gauge.labels(**labels).set(snapshot.throughput_rps)

    def observe(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration: float,
        extra_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record metrics for a completed HTTP request."""

        overrides = dict(extra_labels or {})
        tenant = overrides.pop("tenant", None) or get_tenant_id() or self.default_tenant
        service_role = (
            overrides.pop("service_role", None)
            or get_service_role()
            or self.default_service_role
        )
        context = self._context_labels(tenant, service_role)
        if overrides:
            context.update(overrides)

        base = {**context, "method": method, "route": route}
        counter_labels = {**base, "status": str(status_code)}
        self.request_counter.labels(**counter_labels).inc()
        if status_code >= 500:
            self.error_counter.labels(**counter_labels).inc()
        self.latency_histogram.labels(**base).observe(duration)

        snapshot = self.slo_aggregator.add_sample(
            tenant=tenant,
            service_role=service_role,
            method=method,
            route=route,
            status_code=status_code,
            duration=duration,
        )
        self._update_slo_gauges(snapshot)
        self._publish_compliance_metrics(tenant, service_role)

    def record_synthetic_check(
        self,
        *,
        name: str,
        status: bool,
        latency_ms: float,
        tenant: Optional[str] = None,
        service_role: Optional[str] = None,
        recovery_time_seconds: Optional[float] = None,
        data_lag_seconds: Optional[float] = None,
    ) -> Dict[str, object]:
        tenant_id = tenant or self.default_tenant
        service_role_id = service_role or self.default_service_role
        check = self.synthetic_monitor.record_check(
            name=name,
            tenant=tenant_id,
            service_role=service_role_id,
            status=status,
            latency_seconds=latency_ms / 1000.0,
            recovery_time_seconds=recovery_time_seconds,
            data_lag_seconds=data_lag_seconds,
        )
        labels = self._context_labels(tenant_id, service_role_id)
        metric_labels = {**labels, "check": name}
        self.synthetic_status_gauge.labels(**metric_labels).set(1.0 if check.overall_ok else 0.0)
        self.synthetic_latency_gauge.labels(**metric_labels).set(check.latency_seconds)
        self.synthetic_rto_gauge.labels(**metric_labels).set(
            check.recovery_time_seconds or 0.0
        )
        self.synthetic_rpo_gauge.labels(**metric_labels).set(check.data_lag_seconds or 0.0)
        self._publish_compliance_metrics(tenant_id, service_role_id)
        return check.to_dict()

    def get_slo_snapshots(self) -> List[Dict[str, object]]:
        return [snapshot.to_dict() for snapshot in self.slo_aggregator.get_all_snapshots()]

    def get_slo_snapshot(
        self, tenant: str, service_role: Optional[str] = None
    ) -> Optional[Dict[str, object]]:
        snapshot = self.slo_aggregator.get_snapshot(tenant, service_role)
        if snapshot is None:
            return None
        return snapshot.to_dict()

    def get_synthetic_checks(self) -> List[Dict[str, object]]:
        return [check.to_dict() for check in self.synthetic_monitor.get_all()]

    def get_synthetic_check(
        self, name: str, tenant: str, service_role: str
    ) -> Optional[Dict[str, object]]:
        check = self.synthetic_monitor.get_check(name, tenant, service_role)
        if check is None:
            return None
        return check.to_dict()

    def time(self, method: str, route: str):  # pragma: no cover - simple helper
        start = time.perf_counter()

        def _finish(status_code: int, extra: Optional[Dict[str, str]] = None) -> None:
            self.observe(
                method=method,
                route=route,
                status_code=status_code,
                duration=time.perf_counter() - start,
                extra_labels=extra,
            )

        return _finish


__all__ = ["HttpMetrics"]
