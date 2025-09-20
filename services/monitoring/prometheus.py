"""Prometheus instrumentation helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from prometheus_client import Counter, Histogram, REGISTRY, CollectorRegistry

from .config import MetricsSettings


@dataclass
class HttpMetrics:
    """Track HTTP metrics for a specific service."""

    service_name: str
    environment: str
    settings: MetricsSettings
    registry: CollectorRegistry = REGISTRY

    def __post_init__(self) -> None:
        # Handle FieldInfo for default_labels
        default_labels = getattr(self.settings, 'default_labels', {})
        if hasattr(default_labels, 'default'):
            default_labels = default_labels.default
        
        self._default_label_names = sorted(default_labels.keys())
        self._default_label_values = default_labels.copy()
        self._initialise_metrics(self.registry)

    def _resolve_default_label(self, name: str) -> str:
        if name in self._default_label_values:
            return str(self._default_label_values[name])
        return ""

    def _initialise_metrics(self, registry: CollectorRegistry) -> None:
        extra = list(self._default_label_names)
        labels = ["service", "environment", *extra, "method", "route", "status"]
        latency_labels = ["service", "environment", *extra, "method", "route"]
        namespace = self.settings.namespace
        try:
            self.request_counter = Counter(
                "http_requests_total",
                "Total count of HTTP requests.",
                labels,
                namespace=namespace,
                registry=registry,
            )
            self.error_counter = Counter(
                "http_request_errors_total",
                "Total count of failed HTTP requests.",
                labels,
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
        except ValueError:
            # Fall back to a dedicated registry when another service already registered
            # the same metrics in the default collector.
            fresh_registry = CollectorRegistry()
            self.registry = fresh_registry
            self._initialise_metrics(fresh_registry)
            return
        self.registry = registry

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

        base_labels = {
            "service": self.service_name,
            "environment": self.environment,
            "method": method,
            "route": route,
        }
        for name in self._default_label_names:
            base_labels.setdefault(name, self._resolve_default_label(name))
        if extra_labels:
            base_labels.update(extra_labels)

        labels = {**base_labels, "status": str(status_code)}
        self.request_counter.labels(**labels).inc()
        if status_code >= 500:
            self.error_counter.labels(**labels).inc()
        self.latency_histogram.labels(**base_labels).observe(duration)

    def time(self, method: str, route: str) -> callable[[int, Optional[Dict[str, str]]], None]:  # pragma: no cover - simple helper
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
