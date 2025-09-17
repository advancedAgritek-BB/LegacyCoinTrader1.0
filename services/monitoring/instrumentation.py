"""Instrumentation helpers for FastAPI and Flask services."""

from __future__ import annotations

import contextlib
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import Response as FastAPIResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from crypto_bot.utils.logger import clear_correlation_id, get_correlation_id, set_correlation_id

from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode

from .config import MonitoringSettings, get_monitoring_settings
from .logging import configure_logging
from .prometheus import HttpMetrics
from .tracing import configure_tracing

try:  # pragma: no cover - optional dependency guard for Flask environments
    from flask import Flask, Response as FlaskResponse, g, request
except Exception:  # pragma: no cover - flask may not be installed in some test contexts
    Flask = None  # type: ignore
    FlaskResponse = None  # type: ignore
    g = None  # type: ignore
    request = None  # type: ignore


def _resolve_route(scope_route: object, path: str) -> str:
    route_path = getattr(scope_route, "path", None)
    if isinstance(route_path, str):
        return route_path
    return path


class FastAPIMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware that instruments incoming FastAPI requests."""

    def __init__(
        self,
        app,
        *,
        metrics: HttpMetrics,
        header_name: str,
        tracer: Optional[Tracer],
    ) -> None:
        super().__init__(app)
        self.metrics = metrics
        self.header_name = header_name
        self.tracer = tracer

    async def dispatch(self, request: Request, call_next):
        correlation_id = set_correlation_id(request.headers.get(self.header_name))
        route = _resolve_route(request.scope.get("route"), request.url.path)

        if route == "/metrics":
            try:
                response = await call_next(request)
            finally:
                clear_correlation_id()
            response.headers[self.header_name] = correlation_id
            return response

        method = request.method.upper()
        start_time = time.perf_counter()

        span_cm = (
            self.tracer.start_as_current_span(f"{method} {route}")
            if self.tracer is not None
            else contextlib.nullcontext()
        )

        with span_cm as span:
            if span is not None:
                span.set_attribute("http.method", method)
                span.set_attribute("http.route", route)
                span.set_attribute("http.url", str(request.url))
                span.set_attribute("http.scheme", request.url.scheme)
                if request.client:
                    span.set_attribute("http.client_ip", request.client.host)
                span.set_attribute("legacy.correlation_id", correlation_id)
            try:
                response: Response = await call_next(request)
            except Exception as exc:  # pragma: no cover - defensive logging
                duration = time.perf_counter() - start_time
                self.metrics.observe(method=method, route=route, status_code=500, duration=duration)
                if span is not None:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                clear_correlation_id()
                raise
            status_code = getattr(response, "status_code", 200)
            if span is not None:
                span.set_attribute("http.status_code", status_code)

        duration = time.perf_counter() - start_time
        self.metrics.observe(method=method, route=route, status_code=status_code, duration=duration)
        response.headers[self.header_name] = correlation_id
        clear_correlation_id()
        return response


def _register_metrics_route(app: FastAPI, registry: CollectorRegistry) -> None:
    for route in app.router.routes:
        if getattr(route, "path", None) == "/metrics":
            return

    @app.get("/metrics")
    async def metrics() -> FastAPIResponse:  # pragma: no cover - simple endpoint
        return FastAPIResponse(
            content=generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST,
        )


def instrument_fastapi_app(
    app: FastAPI,
    settings: Optional[MonitoringSettings] = None,
    *,
    service_name: Optional[str] = None,
    environment: Optional[str] = None,
) -> MonitoringSettings:
    """Instrument a FastAPI application with observability features."""

    if getattr(app.state, "observability_instrumented", False):
        return app.state.observability_settings  # type: ignore[attr-defined]

    base_settings = settings or get_monitoring_settings()
    if service_name is not None:
        base_settings = base_settings.for_service(service_name, environment=environment)
    elif environment is not None:
        base_settings = base_settings.model_copy(update={"environment": environment})

    configure_logging(base_settings)
    tracer = configure_tracing(base_settings)
    metrics = HttpMetrics(
        service_name=base_settings.service_name,
        environment=base_settings.environment,
        settings=base_settings.metrics,
    )

    app.add_middleware(
        FastAPIMonitoringMiddleware,
        metrics=metrics,
        header_name=base_settings.correlation_header,
        tracer=tracer,
    )
    _register_metrics_route(app, metrics.registry)
    app.state.observability_instrumented = True
    app.state.observability_settings = base_settings
    app.state.observability_metrics = metrics
    return base_settings


def _flask_has_route(app: "Flask", rule: str) -> bool:
    return any(getattr(route, "rule", None) == rule for route in app.url_map.iter_rules())


def instrument_flask_app(
    app: "Flask",
    settings: Optional[MonitoringSettings] = None,
    *,
    service_name: Optional[str] = None,
    environment: Optional[str] = None,
) -> MonitoringSettings:
    """Instrument a Flask application with observability capabilities."""

    if Flask is None:  # pragma: no cover - happens during linting without Flask installed
        raise RuntimeError("Flask is not available; cannot instrument frontend")

    if app.config.get("OBSERVABILITY_INSTRUMENTED"):
        return app.config["OBSERVABILITY_SETTINGS"]

    base_settings = settings or get_monitoring_settings()
    if service_name is not None:
        base_settings = base_settings.for_service(service_name, environment=environment)
    elif environment is not None:
        base_settings = base_settings.model_copy(update={"environment": environment})

    configure_logging(base_settings)
    tracer = configure_tracing(base_settings)
    metrics = HttpMetrics(
        service_name=base_settings.service_name,
        environment=base_settings.environment,
        settings=base_settings.metrics,
    )
    header_name = base_settings.correlation_header

    @app.before_request  # type: ignore[misc]
    def _before_request() -> None:  # pragma: no cover - request lifecycle hook
        correlation_id = set_correlation_id(request.headers.get(header_name))
        g._observed = False
        g._route = request.url_rule.rule if request.url_rule else request.path
        g._method = request.method.upper()
        g._start_time = time.perf_counter()
        g._correlation_id = correlation_id
        if tracer is not None:
            span_cm = tracer.start_as_current_span(f"{g._method} {g._route}")
            g._span_cm = span_cm
            span = span_cm.__enter__()
            g._span = span
            span.set_attribute("http.method", g._method)
            span.set_attribute("http.route", g._route)
            span.set_attribute("legacy.correlation_id", correlation_id)
        else:
            g._span_cm = None
            g._span = None

    @app.after_request  # type: ignore[misc]
    def _after_request(response: "FlaskResponse") -> "FlaskResponse":  # pragma: no cover - lifecycle hook
        correlation_id = getattr(g, "_correlation_id", get_correlation_id())
        route = getattr(g, "_route", request.path)
        method = getattr(g, "_method", request.method.upper())
        duration = time.perf_counter() - getattr(g, "_start_time", time.perf_counter())
        metrics.observe(method=method, route=route, status_code=response.status_code, duration=duration)
        response.headers[header_name] = correlation_id
        span = getattr(g, "_span", None)
        span_cm = getattr(g, "_span_cm", None)
        if span is not None:
            span.set_attribute("http.status_code", response.status_code)
        if span_cm is not None:
            span_cm.__exit__(None, None, None)
            g._span_cm = None
        clear_correlation_id()
        g._observed = True
        return response

    @app.teardown_request  # type: ignore[misc]
    def _teardown_request(exc: Optional[BaseException]) -> None:  # pragma: no cover - lifecycle hook
        if exc is not None and not getattr(g, "_observed", False):
            correlation_id = getattr(g, "_correlation_id", get_correlation_id())
            route = getattr(g, "_route", request.path)
            method = getattr(g, "_method", request.method.upper())
            duration = time.perf_counter() - getattr(g, "_start_time", time.perf_counter())
            metrics.observe(method=method, route=route, status_code=500, duration=duration)
            span = getattr(g, "_span", None)
            span_cm = getattr(g, "_span_cm", None)
            if span is not None:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            if span_cm is not None:
                span_cm.__exit__(type(exc), exc, getattr(exc, "__traceback__", None))
        clear_correlation_id()

    if not _flask_has_route(app, "/metrics"):
        @app.route("/metrics")  # type: ignore[misc]
        def metrics_endpoint() -> "FlaskResponse":  # pragma: no cover - simple endpoint
            payload = generate_latest(metrics.registry)
            return FlaskResponse(payload, mimetype=CONTENT_TYPE_LATEST)

    app.config["OBSERVABILITY_INSTRUMENTED"] = True
    app.config["OBSERVABILITY_SETTINGS"] = base_settings
    app.config["OBSERVABILITY_METRICS"] = metrics
    return base_settings


__all__ = [
    "FastAPIMonitoringMiddleware",
    "instrument_fastapi_app",
    "instrument_flask_app",
]
