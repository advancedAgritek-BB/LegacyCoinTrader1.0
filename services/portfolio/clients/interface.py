"""High level client facade for the portfolio service."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from ..schemas import PnlBreakdown, PortfolioState, PositionRead, RiskCheckResult, TradeCreate
from .rest import PortfolioRestClient


class PortfolioServiceClient:
    """Facade that hides the REST/gRPC transport selection."""

    def __init__(self, use_grpc: bool = False):
        # For now prefer REST. gRPC hooks can be added when needed.
        self.rest_client = PortfolioRestClient()
        self.use_grpc = use_grpc
        if use_grpc:
            from .grpc import PortfolioGrpcClient

            self.grpc_client = PortfolioGrpcClient()
        else:
            self.grpc_client = None

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------
    def _client(self):
        if self.use_grpc and self.grpc_client is not None:
            return self.grpc_client
        return self.rest_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_state(self) -> PortfolioState:
        return self._client().get_state()

    def put_state(self, state: PortfolioState) -> PortfolioState:
        return self._client().put_state(state)

    def record_trade(self, trade: TradeCreate) -> PositionRead:
        return self._client().record_trade(trade)

    def update_price(self, symbol: str, price: Decimal) -> Optional[PositionRead]:
        return self._client().update_price(symbol, price)

    def compute_pnl(self, symbol: Optional[str] = None) -> PnlBreakdown:
        return self._client().compute_pnl(symbol)

    def check_risk(self) -> list[RiskCheckResult]:
        return self._client().check_risk()

    def list_positions(self) -> list[PositionRead]:
        return self._client().list_positions()
