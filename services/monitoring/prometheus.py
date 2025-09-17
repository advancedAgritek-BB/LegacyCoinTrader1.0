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
        self._initialise_metrics(self.registry)

    def _initialise_metrics(self, registry: CollectorRegistry) -> None:
        extra = []
        if self.settings.default_labels:
            extra = sorted(self.settings.default_labels.keys())
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
        if self.settings.default_labels:
            base_labels.update(self.settings.default_labels)
        if extra_labels:
            base_labels.update(extra_labels)

        labels = {**base_labels, "status": str(status_code)}
        self.request_counter.labels(**labels).inc()
        if status_code >= 500:
            self.error_counter.labels(**labels).inc()
        self.latency_histogram.labels(**base_labels).observe(duration)

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
