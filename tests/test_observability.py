"""Tests covering the monitoring utilities and instrumentation."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from flask import Flask

from services.monitoring.config import (
    MetricsSettings,
    MonitoringSettings,
    OpenSearchSettings,
    TracingSettings,
)
from services.monitoring.instrumentation import instrument_fastapi_app, instrument_flask_app


def _make_settings(service: str) -> MonitoringSettings:
    return MonitoringSettings(
        service_name=service,
        environment="test",
        metrics=MetricsSettings(namespace="legacycoin_test"),
        opensearch=OpenSearchSettings(enabled=False),
        tracing=TracingSettings(enabled=False),
    )


def test_fastapi_instrumentation_emits_metrics_and_headers() -> None:
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    instrument_fastapi_app(app, settings=_make_settings("fastapi-test"))

    with TestClient(app) as client:
        response = client.get("/ping")
        assert response.status_code == 200
        correlation_id = response.headers.get("X-Correlation-ID")
        assert correlation_id and len(correlation_id) == 32

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        body = metrics.text
        assert "legacycoin_test_http_requests_total" in body
        assert "fastapi-test" in body


def test_flask_instrumentation_adds_metrics_route_and_header() -> None:
    app = Flask(__name__)

    @app.route("/ping")
    def ping() -> tuple[str, int]:
        return "pong", 200

    instrument_flask_app(app, settings=_make_settings("flask-test"))

    client = app.test_client()
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID")

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "legacycoin_test_http_requests_total" in metrics.get_data(as_text=True)


def test_core_services_expose_metrics_routes() -> None:
    from services.trading_engine.app import app as trading_app

    try:
        from services.portfolio.rest_api import app as portfolio_app
    except Exception as exc:  # pragma: no cover - optional dependency chain
        pytest.skip(f"Portfolio service unavailable: {exc}")

    assert getattr(trading_app.state, "observability_instrumented", False)
    assert any(route.path == "/metrics" for route in trading_app.routes)

    assert getattr(portfolio_app.state, "observability_instrumented", False)
    assert any(route.path == "/metrics" for route in portfolio_app.routes)


def test_frontend_instrumentation_enabled() -> None:
    frontend_module = pytest.importorskip("frontend.app")
    frontend_app = getattr(frontend_module, "app", None)
    assert frontend_app is not None
    assert frontend_app.config.get("OBSERVABILITY_INSTRUMENTED", False)
    assert any(rule.rule == "/metrics" for rule in frontend_app.url_map.iter_rules())
