import json
from pathlib import Path
from typing import Any
from frontend import app
import sys


def _write_state(
    tmp_path: Path,
    avg_price: float,
    current_price: float,
    amount: float = 1.0,
    side: str = "long",
) -> None:
    logs_dir = tmp_path / "crypto_bot" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "positions": {
            "BTC/USD": {
                "symbol": "BTC/USD",
                "total_amount": amount,
                "average_price": avg_price,
                "side": side,
                "entry_time": "2024-01-01T00:00:00+00:00",
                "last_update": "2024-01-01T00:00:00+00:00",
                "realized_pnl": 0.0,
                "fees_paid": 0.0,
                "trades": []
            }
        },
        "price_cache": {
            "BTC/USD": current_price
        },
        "statistics": {
            "total_realized_pnl": 0.0
        }
    }
    (logs_dir / "trade_manager_state.json").write_text(json.dumps(state))


def _seed_required_files(tmp_path: Path, monkeypatch: Any) -> None:
    # Point module-level paths to tmp files to avoid reading real files
    app_module = sys.modules['frontend.app']
    trade_file = tmp_path / "trades.csv"
    trade_file.write_text("symbol,side,amount,price\n")
    monkeypatch.setattr(app_module, "TRADE_FILE", trade_file)

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "strategy_allocation:\n  trend_bot: 0.5\n  grid_bot: 0.5\n"
    )
    monkeypatch.setattr(app_module, "CONFIG_FILE", cfg)

    regime_file = tmp_path / "regime.txt"
    regime_file.write_text("trending\nsideways\n")
    monkeypatch.setattr(app_module, "REGIME_FILE", regime_file)


def test_dashboard_open_position_cards_show_nonzero_pnl(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)
    _seed_required_files(tmp_path, monkeypatch)
    # Entry 100 -> current 110 => +$10.00, +10.00%
    _write_state(
        tmp_path,
        avg_price=100.0,
        current_price=110.0,
        amount=1.0,
        side="long",
    )

    client = app.test_client()

    # Act
    resp = client.get("/dashboard")

    # Assert
    assert resp.status_code == 200
    body = resp.data
    assert b"Open Positions" in body
    assert b"+10.00%" in body
    assert b"+$10.00" in body


def test_api_open_positions_returns_pnl_fields(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)
    _seed_required_files(tmp_path, monkeypatch)
    _write_state(
        tmp_path,
        avg_price=100.0,
        current_price=110.0,
        amount=2.0,
        side="long",
    )

    client = app.test_client()

    # Act
    resp = client.get("/api/open-positions")

    # Assert
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    found = next((p for p in data if p.get("symbol") == "BTC/USD"), None)
    assert found is not None
    # amount=2, pnl_value should be (110-100)*2 = 20; pct = 20/(100*2)*100 = 10
    assert abs(found.get("pnl_value") - 20.0) < 1e-6
    assert abs(found.get("pnl") - 10.0) < 1e-6


def test_api_sell_position_closes_position(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)
    _seed_required_files(tmp_path, monkeypatch)
    # Create a long position: entry 100, current 110, size 1.0
    _write_state(
        tmp_path,
        avg_price=100.0,
        current_price=110.0,
        amount=1.0,
        side="long",
    )

    client = app.test_client()

    # Ensure position exists
    resp_positions_before = client.get("/api/open-positions")
    assert resp_positions_before.status_code == 200
    data_before = resp_positions_before.get_json()
    assert any(p.get("symbol") == "BTC/USD" for p in data_before)

    # Act: sell position
    resp_sell = client.post(
        "/api/sell-position",
        json={"symbol": "BTC/USD", "amount": 1.0},
    )
    assert resp_sell.status_code == 200
    sell_json = resp_sell.get_json()
    assert sell_json.get("success") is True

    # Assert: position is closed (no longer in open positions)
    resp_positions_after = client.get("/api/open-positions")
    assert resp_positions_after.status_code == 200
    data_after = resp_positions_after.get_json()
    assert not any(p.get("symbol") == "BTC/USD" for p in data_after)

