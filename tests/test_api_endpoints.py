"""Tests covering the high-level API endpoints exposed by the dashboard."""

from __future__ import annotations

from typing import Iterable

import pytest

from frontend.app import (
    ApiGatewayError,
    POSITION_DUST_THRESHOLD,
    app,
    get_open_positions,
)


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
        (" btc/usd ", "BUY", 100.0, 105.0, "2024-01-01T00:00:00Z"),
        ("eth/usd", "long", 50.0, 49.0, "2024-01-02T00:00:00Z"),
        ("BTC/USD", "LONG", 102.0, 106.0, "2024-01-03T00:00:00Z"),
    ]
    with database_connection:
        database_connection.executemany(
            "INSERT INTO positions(symbol, side, entry_price, current_price, entry_time) VALUES (?, ?, ?, ?, ?)",
            sample_rows,
        )

    payload = _rows_to_payload(database_connection.execute(
        "SELECT symbol, side, entry_price, current_price, entry_time FROM positions"
    ).fetchall())

    monkeypatch.setattr("frontend.app.get_gateway_json", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr("frontend.app.post_gateway_json", lambda *_args, **_kwargs: {"results": {}})

    positions = get_open_positions()
    assert {p["symbol"] for p in positions} == {"BTC/USD", "ETH/USD"}
    btc = next(item for item in positions if item["symbol"] == "BTC/USD")
    assert btc["entry_price"] == pytest.approx(102.0)
    assert btc["current_price"] == pytest.approx(106.0)
    assert btc["side"] == "long"


def test_get_open_positions_skips_closed_and_dust_positions(monkeypatch):
    """Positions flagged closed or with dust amounts should be ignored."""

    payload = [
        {
            "symbol": "XRP/USD",
            "side": "long",
            "total_amount": POSITION_DUST_THRESHOLD / 10,
            "average_price": 0.55,
            "current_price": 0.57,
            "entry_time": "2024-01-04T00:00:00Z",
        },
        {
            "symbol": "ADA/USD",
            "side": "sell",
            "total_amount": 25.0,
            "average_price": 0.45,
            "current_price": 0.40,
            "is_open": False,
            "entry_time": "2024-01-05T00:00:00Z",
        },
        {
            "symbol": "SOL/USD",
            "side": "short",
            "total_amount": 3.0,
            "average_price": 100.0,
            "current_price": 95.0,
            "entry_time": "2024-01-06T00:00:00Z",
        },
    ]

    monkeypatch.setattr("frontend.app.get_gateway_json", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr("frontend.app.post_gateway_json", lambda *_args, **_kwargs: {"results": {}})

    positions = get_open_positions()
    assert [pos["symbol"] for pos in positions] == ["SOL/USD"]
    assert positions[0]["side"] == "short"
    assert positions[0]["trend_direction"] == "DOWNWARD"


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


def test_api_wallet_balance_returns_summary(api_client, monkeypatch):
    summary = {
        "balance": 12500.0,
        "total_pnl": 2500.0,
        "realized_pnl": 1000.0,
        "unrealized_pnl": 1500.0,
        "initial_balance": 10000.0,
        "pnl_percentage": 25.0,
    }
    monkeypatch.setattr("frontend.app.calculate_wallet_pnl", lambda: summary)

    response = api_client.get("/api/wallet-balance")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["balance"] == summary["balance"]
    assert payload["total_pnl"] == summary["total_pnl"]


def test_api_wallet_balance_handles_gateway_error(api_client, monkeypatch):
    def _raise():
        raise ApiGatewayError("down")

    monkeypatch.setattr("frontend.app.calculate_wallet_pnl", lambda: _raise())

    response = api_client.get("/api/wallet-balance")
    assert response.status_code == 502
    payload = response.get_json()
    assert payload["success"] is False


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
