"""OpenTelemetry tracing helpers."""

from __future__ import annotations

import logging
import os
from typing import Optional

try:  # Optional dependency guard for environments without OpenTelemetry
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Tracer
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except Exception:  # pragma: no cover - executed when opentelemetry is not installed
    trace = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    Tracer = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]

from .config import MonitoringSettings

LOGGER = logging.getLogger(__name__)

_TRACING_CONFIGURED = False


def _build_resource(settings: MonitoringSettings) -> Resource:
    if Resource is None:  # pragma: no cover - guarded by configure_tracing
        raise RuntimeError("OpenTelemetry Resource class is not available")
    
    # Handle FieldInfo for various settings
    service_name = getattr(settings, 'service_name', 'legacycoin-service')
    if hasattr(service_name, 'default'):
        service_name = service_name.default
    
    environment = getattr(settings, 'environment', 'development')
    if hasattr(environment, 'default'):
        environment = environment.default
    
    service_namespace = getattr(settings.tracing, 'service_namespace', 'legacycoin')
    if hasattr(service_namespace, 'default'):
        service_namespace = service_namespace.default
    
    default_labels = getattr(settings.metrics, 'default_labels', {})
    if hasattr(default_labels, 'default'):
        default_labels = default_labels.default
    
    attributes = {
        "service.name": service_name,
        "service.namespace": service_namespace,
        "deployment.environment": environment,
    }
    attributes.update(default_labels)
    return Resource.create(attributes)


def configure_tracing(settings: MonitoringSettings) -> Optional[Tracer]:
    """Configure OTLP tracing exporter if enabled."""

    global _TRACING_CONFIGURED

    # Handle FieldInfo for tracing settings
    tracing_enabled = False
    try:
        tracing_enabled = getattr(settings.tracing, 'enabled', False)
        if hasattr(tracing_enabled, 'default'):
            tracing_enabled = tracing_enabled.default
    except AttributeError:
        tracing_enabled = False
    
    if not tracing_enabled:
        return None

    otel_exporter = os.getenv("OTEL_TRACES_EXPORTER", "").strip().lower()
    if otel_exporter and otel_exporter in {"none", "disabled", "off"}:
        LOGGER.info(
            "Skipping OpenTelemetry configuration because OTEL_TRACES_EXPORTER=%s",
            otel_exporter,
        )
        return None

    if None in {trace, Resource, TracerProvider, BatchSpanProcessor, OTLPSpanExporter}:
        return None

    # Handle FieldInfo for tracing endpoint and headers
    tracing_endpoint = getattr(settings.tracing, 'endpoint', 'http://localhost:4317')
    if hasattr(tracing_endpoint, 'default'):
        tracing_endpoint = tracing_endpoint.default
    
    tracing_headers = getattr(settings.tracing, 'headers', {})
    if hasattr(tracing_headers, 'default'):
        tracing_headers = tracing_headers.default
    
    endpoint = tracing_endpoint.rstrip("/") + "/v1/traces"

    try:
        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers=tracing_headers,
        )
    except Exception as exc:  # pragma: no cover - only hit when exporter misconfigured
        LOGGER.warning(
            "Failed to initialise OTLP exporter for %s: %s", endpoint, exc
        )
        return None

    try:
        provider = TracerProvider(resource=_build_resource(settings))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception as exc:  # pragma: no cover - defensive guard against SDK issues
        LOGGER.warning("Failed to configure OpenTelemetry tracing: %s", exc)
        return None

    _TRACING_CONFIGURED = True
    return trace.get_tracer(settings.service_name)


__all__ = ["configure_tracing"]
