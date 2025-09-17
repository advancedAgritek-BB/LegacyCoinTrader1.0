"""OpenTelemetry tracing helpers."""

from __future__ import annotations

from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from .config import MonitoringSettings

_TRACING_CONFIGURED = False


def _build_resource(settings: MonitoringSettings) -> Resource:
    attributes = {
        "service.name": settings.service_name,
        "service.namespace": settings.tracing.service_namespace,
        "deployment.environment": settings.environment,
        "legacy.service_role": settings.service_role,
        "legacy.default_tenant": settings.default_tenant,
    }
    attributes.update(settings.metrics.default_labels)
    return Resource.create(attributes)


def configure_tracing(settings: MonitoringSettings) -> Optional[Tracer]:
    """Configure OTLP tracing exporter if enabled."""

    global _TRACING_CONFIGURED
    if not settings.tracing.enabled:
        return None

    endpoint = settings.tracing.endpoint.rstrip("/") + "/v1/traces"
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers=settings.tracing.headers,
    )

    provider = TracerProvider(resource=_build_resource(settings))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACING_CONFIGURED = True
    return trace.get_tracer(settings.service_name)


__all__ = ["configure_tracing"]
