from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import importlib.util
import pytest
import requests

from crypto_bot.services.interfaces import CreateTradeRequest
from services.portfolio.schemas import (
    PnlBreakdown,
    PortfolioState,
    PortfolioStatistics,
    PositionRead,
    PriceCacheEntry,
    RiskCheckResult,
    TradeRead,
)


spec = importlib.util.spec_from_file_location(
    "_portfolio_adapter_test_module",
    Path(__file__).resolve().parents[2] / "crypto_bot" / "services" / "adapters" / "portfolio.py",
)
assert spec is not None and spec.loader is not None
portfolio_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portfolio_module)
PortfolioAdapter = portfolio_module.PortfolioAdapter


class FakePortfolioService:
    """In-memory simulation of the portfolio-service REST API."""

    def __init__(self) -> None:
        now = datetime(2024, 1, 1, 12, 0, 0)
        trade = TradeRead(
            id="trade-1",
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.5"),
            price=Decimal("20000"),
            timestamp=now,
            strategy="momentum",
            exchange="binance",
            fees=Decimal("1"),
            status="filled",
            order_id="order-1",
            client_order_id="client-1",
            metadata={"source": "test"},
        )
        self.position = PositionRead(
            symbol="BTC/USDT",
            side="long",
            total_amount=Decimal("0.5"),
            average_price=Decimal("20000"),
            realized_pnl=Decimal("0"),
            fees_paid=Decimal("1"),
            entry_time=now,
            last_update=now,
            highest_price=Decimal("20500"),
            lowest_price=Decimal("19500"),
            stop_loss_price=None,
            take_profit_price=None,
            trailing_stop_pct=None,
            metadata={"source": "test"},
            mark_price=Decimal("20500"),
            is_open=True,
            trades=[trade],
        )
        self.state = PortfolioState(
            trades=[trade],
            positions=[self.position],
            closed_positions=[],
            price_cache=[
                PriceCacheEntry(symbol="BTC/USDT", price=Decimal("20500"), updated_at=now)
            ],
            statistics=PortfolioStatistics(
                total_trades=1,
                total_volume=Decimal("10000"),
                total_fees=Decimal("1"),
                total_realized_pnl=Decimal("100"),
                last_updated=now,
            ),
        )
        self.pnl = PnlBreakdown(
            realized=Decimal("100"),
            unrealized=Decimal("50"),
            total=Decimal("150"),
        )
        self.risk = [
            RiskCheckResult(name="max_position", passed=True, message="Within limits")
        ]
        self.trade_payloads: list[dict[str, Any]] = []
        self.updated_prices: list[dict[str, str]] = []
        self.pnl_requests: list[dict[str, str]] = []

    def _response(self, payload: Any, status_code: int = 200) -> requests.Response:
        response = requests.Response()
        response.status_code = status_code
        if payload is None:
            response._content = b""
        else:
            response._content = json.dumps(payload).encode()
            response.headers["Content-Type"] = "application/json"
        response.url = "http://portfolio.test"
        return response

    def handle(
        self,
        method: str,
        url: str,
        timeout: Any = None,
        params: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        **_: Any,
    ) -> requests.Response:
        path = urlparse(url).path
        if method == "POST" and path == "/trades":
            assert json is not None
            self.trade_payloads.append(json)
            return self._response(self.position.model_dump(mode="json"), status_code=201)
        if method == "GET" and path == "/state":
            return self._response(self.state.model_dump(mode="json"))
        if method == "GET" and path == "/positions":
            return self._response(
                [pos.model_dump(mode="json") for pos in self.state.positions]
            )
        if method == "POST" and path == "/prices":
            assert params is not None
            self.updated_prices.append(params)
            return self._response(self.position.model_dump(mode="json"))
        if method == "GET" and path == "/pnl":
            if params:
                self.pnl_requests.append(params)
            return self._response(self.pnl.model_dump(mode="json"))
        if method == "GET" and path == "/risk":
            return self._response([entry.model_dump(mode="json") for entry in self.risk])
        raise AssertionError(f"Unexpected request: {method} {path}")


@pytest.fixture
def fake_portfolio_service(monkeypatch: pytest.MonkeyPatch) -> FakePortfolioService:
    server = FakePortfolioService()

    def fake_request(method: str, url: str, **kwargs: Any) -> requests.Response:
        return server.handle(method, url, **kwargs)

    monkeypatch.setattr(
        "services.portfolio.clients.rest.requests.request", fake_request
    )
    return server


def test_adapter_records_trades_via_rest(monkeypatch: pytest.MonkeyPatch, fake_portfolio_service: FakePortfolioService) -> None:
    monkeypatch.setenv("PORTFOLIO_SERVICE_URL", "http://portfolio.test")
    monkeypatch.setenv("PORTFOLIO_SERVICE_TIMEOUT", "7.5")
    monkeypatch.setenv("PORTFOLIO_SERVICE_CONNECT_TIMEOUT", "2.0")
    monkeypatch.setenv("PORTFOLIO_SERVICE_READ_TIMEOUT", "3.0")

    adapter = PortfolioAdapter()
    response = adapter.create_trade(
        CreateTradeRequest(
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.5"),
            price=Decimal("21000"),
            strategy="momentum",
            exchange="binance",
            fees=Decimal("0.1"),
            order_id="exchange-order",
            client_order_id="client-order",
            metadata={"source": "unit-test"},
        )
    )

    assert response.trade.symbol == "BTC/USDT"
    assert fake_portfolio_service.trade_payloads, "trade should be sent to the REST API"
    payload = fake_portfolio_service.trade_payloads[0]
    assert payload["symbol"] == "BTC/USDT"
    assert Decimal(payload["amount"]) == Decimal("0.5")
    assert adapter._client.base_url == "http://portfolio.test"
    assert adapter._client.timeout == (2.0, 3.0)


def test_adapter_fetches_state_and_positions(
    monkeypatch: pytest.MonkeyPatch, fake_portfolio_service: FakePortfolioService
) -> None:
    monkeypatch.setenv("PORTFOLIO_SERVICE_URL", "http://portfolio.test")
    adapter = PortfolioAdapter()

    state = adapter.get_state()
    assert state.statistics.total_trades == fake_portfolio_service.state.statistics.total_trades

    positions = adapter.list_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "BTC/USDT"


def test_adapter_updates_prices_and_queries_pnl(
    monkeypatch: pytest.MonkeyPatch, fake_portfolio_service: FakePortfolioService
) -> None:
    monkeypatch.setenv("PORTFOLIO_SERVICE_URL", "http://portfolio.test")
    adapter = PortfolioAdapter()

    adapter.update_price("BTC/USDT", Decimal("21500"))
    assert fake_portfolio_service.updated_prices
    assert fake_portfolio_service.updated_prices[0]["symbol"] == "BTC/USDT"
    assert fake_portfolio_service.updated_prices[0]["price"] == "21500"

    pnl = adapter.compute_pnl(symbol="BTC/USDT")
    assert fake_portfolio_service.pnl_requests == [{"symbol": "BTC/USDT"}]
    assert pnl.total == fake_portfolio_service.pnl.total

    risk = adapter.check_risk()
    assert risk[0].name == fake_portfolio_service.risk[0].name
