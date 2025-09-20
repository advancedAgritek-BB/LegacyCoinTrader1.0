"""Instrumentation helpers for FastAPI and Flask services."""

from __future__ import annotations

import contextlib
import time
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import Response as FastAPIResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from pydantic import BaseModel, Field

from crypto_bot.utils.logger import (
    clear_correlation_id,
    clear_observability_context,
    get_correlation_id,
    set_correlation_id,
    set_observability_context,
)

try:  # Optional dependency: OpenTelemetry is not always available in tests
    from opentelemetry.trace import Tracer
    from opentelemetry.trace.status import Status, StatusCode
except Exception:  # pragma: no cover - executed when opentelemetry is absent
    Tracer = Any  # type: ignore[assignment]

    class StatusCode:  # Minimal stand-in used for status assignment
        OK = "OK"
        ERROR = "ERROR"

    class Status:  # type: ignore[override]
        def __init__(self, status_code: str, description: Optional[str] = None) -> None:
            self.status_code = status_code
            self.description = description

from .config import MonitoringSettings, get_monitoring_settings
from .logging_utils import configure_logging
from .prometheus import HttpMetrics
from .tracing import configure_tracing

try:  # pragma: no cover - optional dependency guard for Flask environments
    from flask import Flask, Response as FlaskResponse, g, jsonify, request
except Exception:  # pragma: no cover - flask may not be installed in some test contexts
    Flask = None  # type: ignore
    FlaskResponse = None  # type: ignore
    g = None  # type: ignore
    jsonify = None  # type: ignore
    request = None  # type: ignore


class SyntheticCheckPayload(BaseModel):
    """Request payload for submitting synthetic check results."""

    name: str
    status: bool
    latency_ms: float = Field(ge=0)
    tenant: Optional[str] = Field(default=None)
    service_role: Optional[str] = Field(default=None)
    recovery_time_seconds: Optional[float] = Field(default=None, ge=0)
    data_lag_seconds: Optional[float] = Field(default=None, ge=0)


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
        tenant_header: str,
        service_role_header: str,
        default_tenant: str,
        default_service_role: str,
    ) -> None:
        super().__init__(app)
        self.metrics = metrics
        self.header_name = header_name
        self.tracer = tracer
        self.tenant_header = tenant_header
        self.service_role_header = service_role_header
        self.default_tenant = default_tenant
        self.default_service_role = default_service_role

    async def dispatch(self, request: Request, call_next):
        correlation_id = set_correlation_id(request.headers.get(self.header_name))
        route = _resolve_route(request.scope.get("route"), request.url.path)

        tenant_value = request.headers.get(self.tenant_header, "") or ""
        service_role_value = request.headers.get(self.service_role_header, "") or ""
        tenant_id, service_role = set_observability_context(
            tenant_id=tenant_value.strip() or self.default_tenant,
            service_role=service_role_value.strip() or self.default_service_role,
        )
        extra_labels = {"tenant": tenant_id, "service_role": service_role}

        if route == "/metrics":
            try:
                response = await call_next(request)
                if hasattr(response, "headers"):
                    response.headers[self.header_name] = correlation_id
                return response
            finally:
                clear_observability_context()
                clear_correlation_id()

        method = request.method.upper()
        start_time = time.perf_counter()

        span_cm = (
            self.tracer.start_as_current_span(f"{method} {route}")
            if self.tracer is not None
            else contextlib.nullcontext()
        )

        response: Optional[Response] = None
        status_code = 500
        span = None
        try:
            with span_cm as span:
                if span is not None:
                    span.set_attribute("http.method", method)
                    span.set_attribute("http.route", route)
                    span.set_attribute("http.url", str(request.url))
                    span.set_attribute("http.scheme", request.url.scheme)
                    if request.client:
                        span.set_attribute("http.client_ip", request.client.host)
                    span.set_attribute("legacy.correlation_id", correlation_id)
                    span.set_attribute("legacy.tenant_id", tenant_id)
                    span.set_attribute("legacy.service_role", service_role)
                try:
                    response = await call_next(request)
                except Exception as exc:  # pragma: no cover - defensive logging
                    duration = time.perf_counter() - start_time
                    self.metrics.observe(
                        method=method,
                        route=route,
                        status_code=500,
                        duration=duration,
                        extra_labels=extra_labels,
                    )
                    if span is not None:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                status_code = getattr(response, "status_code", 200)
                if span is not None:
                    span.set_attribute("http.status_code", status_code)
            duration = time.perf_counter() - start_time
            self.metrics.observe(
                method=method,
                route=route,
                status_code=status_code,
                duration=duration,
                extra_labels=extra_labels,
            )
            if response is not None:
                response.headers[self.header_name] = correlation_id
            return response
        finally:
            clear_observability_context()
            clear_correlation_id()


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


def _register_fastapi_observability_routes(
    app: FastAPI, metrics: HttpMetrics, settings: MonitoringSettings
) -> None:
    if getattr(app.state, "observability_routes_registered", False):
        return

    router = APIRouter(prefix="/observability", tags=["observability"])

    @router.get("/slo")
    async def slo_overview() -> list[dict[str, object]]:
        return metrics.get_slo_snapshots()

    @router.get("/slo/{tenant_id}")
    async def slo_for_tenant(
        tenant_id: str, service_role: Optional[str] = None
    ) -> dict[str, object]:
        snapshot = metrics.get_slo_snapshot(
            tenant_id, service_role or settings.service_role
        )
        if snapshot is None:
            raise HTTPException(status_code=404, detail="SLO data not found for tenant")
        return snapshot

    @router.get("/synthetic-checks")
    async def list_synthetic_checks() -> list[dict[str, object]]:
        return metrics.get_synthetic_checks()

    @router.get("/synthetic-checks/{check_name}")
    async def get_synthetic_check(
        check_name: str,
        tenant: Optional[str] = None,
        service_role: Optional[str] = None,
    ) -> dict[str, object]:
        tenant_id = tenant or settings.default_tenant
        service_role_id = service_role or settings.service_role
        record = metrics.get_synthetic_check(check_name, tenant_id, service_role_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Synthetic check '{check_name}' not found for tenant "
                    f"{tenant_id} ({service_role_id})."
                ),
            )
        return record

    @router.post("/synthetic-checks", status_code=201)
    async def post_synthetic_check(payload: SyntheticCheckPayload) -> dict[str, object]:
        return metrics.record_synthetic_check(
            name=payload.name,
            status=payload.status,
            latency_ms=payload.latency_ms,
            tenant=payload.tenant or settings.default_tenant,
            service_role=payload.service_role or settings.service_role,
            recovery_time_seconds=payload.recovery_time_seconds,
            data_lag_seconds=payload.data_lag_seconds,
        )

    app.include_router(router)
    app.state.observability_routes_registered = True


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
        base_settings = base_settings.clone(environment=environment)

    configure_logging(base_settings)
    tracer = configure_tracing(base_settings)
    metrics = HttpMetrics(
        service_name=base_settings.service_name,
        environment=base_settings.environment,
        settings=base_settings.metrics,
        slo_settings=base_settings.slo,
        compliance_settings=base_settings.compliance,
        default_tenant=base_settings.default_tenant,
        default_service_role=base_settings.service_role,
    )

    app.add_middleware(
        FastAPIMonitoringMiddleware,
        metrics=metrics,
        header_name=base_settings.correlation_header,
        tracer=tracer,
        tenant_header=base_settings.tenant_header,
        service_role_header=base_settings.service_role_header,
        default_tenant=base_settings.default_tenant,
        default_service_role=base_settings.service_role,
    )
    _register_metrics_route(app, metrics.registry)
    _register_fastapi_observability_routes(app, metrics, base_settings)
    app.state.observability_instrumented = True
    app.state.observability_settings = base_settings
    app.state.observability_metrics = metrics
    app.state.observability_slo = metrics.slo_aggregator
    app.state.observability_synthetic = metrics.synthetic_monitor
    return base_settings


def _flask_has_route(app: "Flask", rule: str) -> bool:
    return any(getattr(route, "rule", None) == rule for route in app.url_map.iter_rules())


def _register_flask_observability_routes(
    app: "Flask", metrics: HttpMetrics, settings: MonitoringSettings
) -> None:
    if Flask is None or jsonify is None:
        return

    if not _flask_has_route(app, "/observability/slo"):

        @app.route("/observability/slo", methods=["GET"])
        def flask_slo_overview() -> "FlaskResponse":  # pragma: no cover - framework glue
            return jsonify(metrics.get_slo_snapshots())

    if not _flask_has_route(app, "/observability/slo/<tenant_id>"):

        @app.route("/observability/slo/<tenant_id>", methods=["GET"])
        def flask_slo_tenant(tenant_id: str) -> "FlaskResponse":  # pragma: no cover
            service_role = request.args.get("service_role", settings.service_role)
            snapshot = metrics.get_slo_snapshot(tenant_id, service_role)
            if snapshot is None:
                return jsonify({"detail": "SLO data not found"}), 404
            return jsonify(snapshot)

    if not _flask_has_route(app, "/observability/synthetic-checks"):

        @app.route("/observability/synthetic-checks", methods=["GET", "POST"])
        def flask_synthetic_checks() -> "FlaskResponse":  # pragma: no cover - glue code
            if request.method == "GET":
                return jsonify(metrics.get_synthetic_checks())

            payload = request.get_json(force=True, silent=True) or {}
            try:
                name = str(payload["name"])
                status = bool(payload["status"])
                latency_ms = float(payload["latency_ms"])
            except (KeyError, TypeError, ValueError):
                return jsonify({"detail": "Invalid synthetic check payload"}), 400

            tenant = str(payload.get("tenant") or settings.default_tenant)
            service_role = str(payload.get("service_role") or settings.service_role)

            recovery_time = payload.get("recovery_time_seconds")
            data_lag = payload.get("data_lag_seconds")
            try:
                recovery_time_val = (
                    None if recovery_time is None else float(recovery_time)
                )
                data_lag_val = None if data_lag is None else float(data_lag)
            except (TypeError, ValueError):
                return jsonify({"detail": "Invalid recovery/data lag values"}), 400

            record = metrics.record_synthetic_check(
                name=name,
                status=status,
                latency_ms=latency_ms,
                tenant=tenant,
                service_role=service_role,
                recovery_time_seconds=recovery_time_val,
                data_lag_seconds=data_lag_val,
            )
            return jsonify(record), 201

    if not _flask_has_route(app, "/observability/synthetic-checks/<check_name>"):

        @app.route("/observability/synthetic-checks/<check_name>", methods=["GET"])
        def flask_synthetic_check_detail(check_name: str) -> "FlaskResponse":
            tenant = request.args.get("tenant", settings.default_tenant)
            service_role = request.args.get("service_role", settings.service_role)
            record = metrics.get_synthetic_check(check_name, tenant, service_role)
            if record is None:
                return (
                    jsonify(
                        {
                            "detail": (
                                f"Synthetic check '{check_name}' not found for tenant "
                                f"{tenant} ({service_role})."
                            )
                        }
                    ),
                    404,
                )
            return jsonify(record)


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
        base_settings = base_settings.clone(environment=environment)

    configure_logging(base_settings)
    tracer = configure_tracing(base_settings)
    metrics = HttpMetrics(
        service_name=base_settings.service_name,
        environment=base_settings.environment,
        settings=base_settings.metrics,
        slo_settings=base_settings.slo,
        compliance_settings=base_settings.compliance,
        default_tenant=base_settings.default_tenant,
        default_service_role=base_settings.service_role,
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
        tenant_value = request.headers.get(base_settings.tenant_header, "") or ""
        service_role_value = (
            request.headers.get(base_settings.service_role_header, "") or ""
        )
        tenant_id, service_role = set_observability_context(
            tenant_id=tenant_value.strip() or base_settings.default_tenant,
            service_role=service_role_value.strip() or base_settings.service_role,
        )
        g._tenant_id = tenant_id
        g._service_role = service_role
        if tracer is not None:
            span_cm = tracer.start_as_current_span(f"{g._method} {g._route}")
            g._span_cm = span_cm
            span = span_cm.__enter__()
            g._span = span
            span.set_attribute("http.method", g._method)
            span.set_attribute("http.route", g._route)
            span.set_attribute("legacy.correlation_id", correlation_id)
            span.set_attribute("legacy.tenant_id", tenant_id)
            span.set_attribute("legacy.service_role", service_role)
        else:
            g._span_cm = None
            g._span = None

    @app.after_request  # type: ignore[misc]
    def _after_request(response: "FlaskResponse") -> "FlaskResponse":  # pragma: no cover - lifecycle hook
        correlation_id = getattr(g, "_correlation_id", get_correlation_id())
        route = getattr(g, "_route", request.path)
        method = getattr(g, "_method", request.method.upper())
        tenant_id = getattr(g, "_tenant_id", base_settings.default_tenant)
        service_role = getattr(g, "_service_role", base_settings.service_role)
        duration = time.perf_counter() - getattr(g, "_start_time", time.perf_counter())
        if route != "/metrics":
            metrics.observe(
                method=method,
                route=route,
                status_code=response.status_code,
                duration=duration,
                extra_labels={"tenant": tenant_id, "service_role": service_role},
            )
        response.headers[header_name] = correlation_id
        span = getattr(g, "_span", None)
        span_cm = getattr(g, "_span_cm", None)
        if span is not None:
            span.set_attribute("http.status_code", response.status_code)
        if span_cm is not None:
            span_cm.__exit__(None, None, None)
            g._span_cm = None
        clear_observability_context()
        clear_correlation_id()
        g._observed = True
        return response

    @app.teardown_request  # type: ignore[misc]
    def _teardown_request(exc: Optional[BaseException]) -> None:  # pragma: no cover - lifecycle hook
        if exc is not None and not getattr(g, "_observed", False):
            correlation_id = getattr(g, "_correlation_id", get_correlation_id())
            route = getattr(g, "_route", request.path)
            method = getattr(g, "_method", request.method.upper())
            tenant_id = getattr(g, "_tenant_id", base_settings.default_tenant)
            service_role = getattr(g, "_service_role", base_settings.service_role)
            duration = time.perf_counter() - getattr(g, "_start_time", time.perf_counter())
            metrics.observe(
                method=method,
                route=route,
                status_code=500,
                duration=duration,
                extra_labels={"tenant": tenant_id, "service_role": service_role},
            )
            span = getattr(g, "_span", None)
            span_cm = getattr(g, "_span_cm", None)
            if span is not None:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            if span_cm is not None:
                span_cm.__exit__(type(exc), exc, getattr(exc, "__traceback__", None))
        clear_observability_context()
        clear_correlation_id()

    if not _flask_has_route(app, "/metrics"):
        @app.route("/metrics")  # type: ignore[misc]
        def metrics_endpoint() -> "FlaskResponse":  # pragma: no cover - simple endpoint
            payload = generate_latest(metrics.registry)
            return FlaskResponse(payload, mimetype=CONTENT_TYPE_LATEST)

    _register_flask_observability_routes(app, metrics, base_settings)
    app.config["OBSERVABILITY_INSTRUMENTED"] = True
    app.config["OBSERVABILITY_SETTINGS"] = base_settings
    app.config["OBSERVABILITY_METRICS"] = metrics
    app.config["OBSERVABILITY_SLO"] = metrics.slo_aggregator
    app.config["OBSERVABILITY_SYNTHETIC"] = metrics.synthetic_monitor
    return base_settings


__all__ = [
    "FastAPIMonitoringMiddleware",
    "instrument_fastapi_app",
    "instrument_flask_app",
]
