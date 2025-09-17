"""REST client for the portfolio service."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

import requests

from ..config import get_service_base_url
from ..schemas import PnlBreakdown, PortfolioState, PositionRead, RiskCheckResult, TradeCreate


class PortfolioRestClient:
    """Lightweight REST client for interacting with the portfolio service."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
        self.base_url = base_url or get_service_base_url()
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        response = requests.request(method, url, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_state(self) -> PortfolioState:
        payload = self._request("GET", "/state")
        return PortfolioState.model_validate(payload)

    def put_state(self, state: PortfolioState) -> PortfolioState:
        payload = self._request("PUT", "/state", json=state.model_dump())
        return PortfolioState.model_validate(payload)

    def record_trade(self, trade: TradeCreate) -> PositionRead:
        payload = self._request("POST", "/trades", json=trade.model_dump(mode="json"))
        return PositionRead.model_validate(payload)

    def update_price(self, symbol: str, price: Decimal) -> Optional[PositionRead]:
        payload = self._request(
            "POST",
            "/prices",
            params={"symbol": symbol, "price": str(price)},
        )
        return PositionRead.model_validate(payload) if payload else None

    def compute_pnl(self, symbol: Optional[str] = None) -> PnlBreakdown:
        params = {"symbol": symbol} if symbol else None
        payload = self._request("GET", "/pnl", params=params)
        return PnlBreakdown.model_validate(payload)

    def check_risk(self) -> list[RiskCheckResult]:
        payload = self._request("GET", "/risk")
        return [RiskCheckResult.model_validate(item) for item in payload]

    def list_positions(self) -> list[PositionRead]:
        payload = self._request("GET", "/positions")
        return [PositionRead.model_validate(item) for item in payload]
