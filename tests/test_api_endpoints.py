"""Tests covering the high-level API endpoints exposed by the dashboard."""

from __future__ import annotations

from typing import Iterable

import pytest

from frontend.app import app, get_open_positions


@pytest.fixture(scope="module")
def api_client():
    """Return a Flask test client with testing configuration enabled."""

    # Disable rate limiting and ensure Prometheus metrics use the expected labels.
    app._rate_limiter_enabled = False
    if hasattr(app, "monitoring_settings"):
        app.monitoring_settings.metrics.default_labels.clear()
    metrics = app.config.get("OBSERVABILITY_METRICS")
    if metrics is not None:
        metrics.settings.default_labels.clear()

    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


def _build_position(
    *,
    symbol: str,
    side: str,
    amount: float,
    entry_price: float,
    current_price: float,
    entry_time: str,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "entry_price": entry_price,
        "current_price": current_price,
        "entry_time": entry_time,
    }


def _rows_to_payload(rows: Iterable[tuple[str, str, float, float, str]]) -> list[dict[str, object]]:
    return [
        _build_position(
            symbol=row[0],
            side=row[1],
            amount=1.0,
            entry_price=row[2],
            current_price=row[3],
            entry_time=row[4],
        )
        for row in rows
    ]


def test_get_open_positions_deduplicates_latest_entries(monkeypatch, database_connection):
    """Ensure duplicate symbols are reduced to their most recent entry."""

    sample_rows = [
        ("BTC/USD", "long", 100.0, 105.0, "2024-01-01T00:00:00Z"),
        ("ETH/USD", "long", 50.0, 49.0, "2024-01-02T00:00:00Z"),
        ("BTC/USD", "long", 102.0, 106.0, "2024-01-03T00:00:00Z"),
    ]
    with database_connection:
        database_connection.executemany(
            "INSERT INTO positions(symbol, side, entry_price, current_price, entry_time) VALUES (?, ?, ?, ?, ?)",
            sample_rows,
        )

    payload = _rows_to_payload(database_connection.execute(
        "SELECT symbol, side, entry_price, current_price, entry_time FROM positions"
    ).fetchall())

    monkeypatch.setattr("frontend.app.fetch_positions_from_service", lambda: payload)

    positions = get_open_positions()
    assert {p["symbol"] for p in positions} == {"BTC/USD", "ETH/USD"}
    btc = next(item for item in positions if item["symbol"] == "BTC/USD")
    assert btc["entry_price"] == pytest.approx(102.0)
    assert btc["current_price"] == pytest.approx(106.0)


def test_api_open_positions_returns_payload(api_client, monkeypatch):
    sample_positions = [
        {
            "symbol": "BTC/USD",
            "side": "long",
            "amount": 1.0,
            "entry_price": 100.0,
            "current_price": 105.0,
            "entry_time": "2024-01-01T00:00:00Z",
        }
    ]
    monkeypatch.setattr("frontend.app.get_open_positions", lambda: sample_positions)

    response = api_client.get("/api/open-positions")
    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload, list)
    assert payload == sample_positions


def test_api_open_positions_handles_failure(api_client, monkeypatch):
    monkeypatch.setattr("frontend.app.get_open_positions", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    response = api_client.get("/api/open-positions")
    assert response.status_code == 502
    payload = response.get_json()
    assert payload["error"]


def test_api_bot_status_detects_running_process(api_client, monkeypatch):
    class DummyProcess:
        def __init__(self, *, cmdline: list[str]):
            self.info = {"name": "python", "cmdline": cmdline}

    monkeypatch.setattr(
        "psutil.process_iter",
        lambda _: [DummyProcess(cmdline=["python", "crypto_bot/main.py"])],
    )

    response = api_client.get("/api/bot-status")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["data"]["bot_running"] is True


def test_api_bot_status_reports_idle(api_client, monkeypatch):
    monkeypatch.setattr("psutil.process_iter", lambda _: [])

    response = api_client.get("/api/bot-status")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["data"]["bot_running"] is False


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {"status": "ok"}
