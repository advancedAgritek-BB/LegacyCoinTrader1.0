import pytest
from fastapi.testclient import TestClient

from frontend.api import ApiGatewayError, app


@pytest.fixture()
def fastapi_client():
    return TestClient(app)


def test_positions_returns_gateway_payload(monkeypatch, fastapi_client):
    sample = [{"symbol": "BTC/USD", "side": "long"}]

    async def _mock_fetch(path):
        return sample

    monkeypatch.setattr("frontend.api.async_get_gateway_json", _mock_fetch)

    response = fastapi_client.get("/positions")
    assert response.status_code == 200
    assert response.json() == sample


def test_positions_handles_gateway_errors(monkeypatch, fastapi_client):

    async def _raise(path):
        raise ApiGatewayError("down")

    monkeypatch.setattr("frontend.api.async_get_gateway_json", _raise)

    response = fastapi_client.get("/positions")
    assert response.status_code == 200
    assert response.json() == []
