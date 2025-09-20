import pytest

from frontend import app as frontend_app


def test_fetch_positions_from_service_uses_price_cache(monkeypatch):
    payload = {
        "positions": [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "total_amount": "1.0",
                "average_price": "20000",
                "entry_time": "2024-01-01T00:00:00Z",
                "mark_price": None,
            }
        ],
        "price_cache": [
            {
                "symbol": "BTC/USDT",
                "price": "21000",
            }
        ],
    }

    monkeypatch.setattr(
        frontend_app, "get_gateway_json", lambda path, **kwargs: payload
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("post_gateway_json should not be called when price cache is present")

    monkeypatch.setattr(frontend_app, "post_gateway_json", _unexpected)

    positions = frontend_app.fetch_positions_from_service()

    assert positions, "expected at least one position"
    position = positions[0]
    assert position["current_price"] == pytest.approx(21000.0)
    assert position["pnl_value"] == pytest.approx(1000.0)
    assert position["pnl"] == pytest.approx(5.0)
