import sys
import types
from typing import Any, Dict

if "pydantic_settings" not in sys.modules:  # pragma: no cover - import hook
    module = types.ModuleType("pydantic_settings")

    class _BaseSettings:  # pylint: disable=too-few-public-methods
        model_config: Dict[str, Any] = {}

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._data = kwargs

        def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return dict(self._data)

    module.BaseSettings = _BaseSettings
    module.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = module

import httpx
import pytest

from crypto_bot.services.adapters.execution import ExecutionAdapter
from crypto_bot.services.interfaces import TradeExecutionRequest
from services.execution.app import create_app
from services.execution.models import ExchangeSession


class DummyExchange:
    nonce = None


async def _success_trade(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return {"id": "order-123", "status": "closed"}


def _stub_session(*args: Any, **kwargs: Any) -> ExchangeSession:
    return ExchangeSession(exchange=DummyExchange(), ws_client=None, config_hash="stub")


@pytest.mark.asyncio
async def test_execute_trade_success(monkeypatch):
    monkeypatch.setenv("EXECUTION_SERVICE_SECRET", "supersecret")
    monkeypatch.setattr(
        "services.execution.exchange.ExchangeFactory.create_session",
        lambda self, config, credentials, nonce_manager: _stub_session(),
    )
    monkeypatch.setattr("services.execution.service.execute_trade_async", _success_trade)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        adapter = ExecutionAdapter(base_url="http://test", secret="supersecret", http_client=client)
        request = TradeExecutionRequest(
            exchange=None,
            ws_client=None,
            symbol="BTC/USD",
            side="buy",
            amount=1.0,
            config={"ack_timeout": 2.0, "fill_timeout": 2.0},
        )
        response = await adapter.execute_trade(request)
        assert response.order["id"] == "order-123"
        await adapter.aclose()


@pytest.mark.asyncio
async def test_execute_trade_failure(monkeypatch):
    monkeypatch.setenv("EXECUTION_SERVICE_SECRET", "supersecret")
    monkeypatch.setattr(
        "services.execution.exchange.ExchangeFactory.create_session",
        lambda self, config, credentials, nonce_manager: _stub_session(),
    )

    async def _error_trade(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr("services.execution.service.execute_trade_async", _error_trade)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        adapter = ExecutionAdapter(base_url="http://test", secret="supersecret", http_client=client)
        request = TradeExecutionRequest(
            exchange=None,
            ws_client=None,
            symbol="ETH/USD",
            side="sell",
            amount=0.5,
            config={"ack_timeout": 2.0, "fill_timeout": 2.0},
        )
        response = await adapter.execute_trade(request)
        assert response.order == {}
        await adapter.aclose()

