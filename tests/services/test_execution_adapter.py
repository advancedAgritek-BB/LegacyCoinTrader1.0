import asyncio
from typing import Any, Dict

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

MODULE_PATH = Path(__file__).resolve().parents[2] / "crypto_bot" / "services" / "adapters" / "execution.py"
MODULE_SPEC = importlib.util.spec_from_file_location("execution_adapter_test", MODULE_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader  # sanity check for test setup
execution_module = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = execution_module
MODULE_SPEC.loader.exec_module(execution_module)

ExecutionAdapter = execution_module.ExecutionAdapter
ExecutionApiClient = execution_module.ExecutionApiClient

from crypto_bot.services.interfaces import ExchangeRequest, TradeExecutionRequest
from services.execution.config import get_execution_api_settings


class DummyExchange:
    id = "dummy"

    def close(self) -> None:  # pragma: no cover - cleanup helper
        pass


@pytest.fixture(autouse=True)
def clear_execution_settings_cache(monkeypatch):
    monkeypatch.setenv("EXECUTION_SERVICE_TOKEN", "test-token")
    monkeypatch.setenv("EXECUTION_SERVICE_SIGNING_KEY", "test-secret")
    get_execution_api_settings.cache_clear()


@pytest_asyncio.fixture
async def execution_app(monkeypatch):
    from services.execution.app import create_app

    async def fake_execute_trade_async(
        exchange: Any,
        ws_client: Any,
        symbol: str,
        side: str,
        amount: float,
        **_: Any,
    ) -> Dict[str, Any]:
        return {
            "id": "order-1",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "status": "closed",
        }

    def fake_get_exchange(config):
        return DummyExchange(), None

    monkeypatch.setattr(
        "services.execution.exchange.get_exchange",
        fake_get_exchange,
    )
    monkeypatch.setattr(
        "services.execution.service.execute_trade_async",
        fake_execute_trade_async,
    )
    app = create_app()
    return app


@pytest_asyncio.fixture
async def execution_client(execution_app):
    async with execution_app.router.lifespan_context(execution_app):
        transport = httpx.ASGITransport(app=execution_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver/api/v1/execution"
        ) as http_client:
            yield http_client


@pytest.mark.asyncio
async def test_adapter_executes_trade_via_http(execution_client):
    client = ExecutionApiClient(
        base_url="http://testserver/api/v1/execution",
        service_token="test-token",
        signing_key="test-secret",
        ack_timeout=2.0,
        fill_timeout=2.0,
        client=execution_client,
    )
    metadata = await client.ensure_exchange_async({"client_prefix": "unit"})
    assert metadata["status"] == "ready"
    adapter = ExecutionAdapter(client=client)
    request = TradeExecutionRequest(
        exchange=None,
        ws_client=None,
        symbol="BTC/USD",
        side="buy",
        amount=1.25,
        dry_run=False,
        use_websocket=False,
        config={"client_prefix": "unit"},
    )
    result = await adapter.execute_trade(request)
    assert result.order["id"] == "order-1"
    assert result.order["symbol"] == "BTC/USD"


@pytest.mark.asyncio
async def test_ack_and_fill_events_available_after_submission(execution_client):
    client = ExecutionApiClient(
        base_url="http://testserver/api/v1/execution",
        service_token="test-token",
        signing_key="test-secret",
        ack_timeout=2.0,
        fill_timeout=2.0,
        client=execution_client,
    )
    payload = {
        "symbol": "ETH/USD",
        "side": "sell",
        "amount": 2.5,
        "client_order_id": "test-ack-1",
        "dry_run": False,
        "use_websocket": False,
        "score": 0.0,
        "config": {"client_prefix": "unit"},
        "metadata": {"source": "tests"},
    }
    await client.submit_order(payload)
    await asyncio.sleep(0)  # allow acknowledgement task to run
    ack = await client.wait_for_ack("test-ack-1", timeout=2.0)
    assert ack["accepted"] is True
    fill = await client.wait_for_fill("test-ack-1", timeout=2.0)
    assert fill["success"] is True
    assert fill["order"]["side"] == "sell"
