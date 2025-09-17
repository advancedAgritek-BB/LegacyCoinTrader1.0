"""HTTP-backed portfolio adapter."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Tuple

from crypto_bot.services.interfaces import (
    CreateTradeRequest,
    CreateTradeResponse,
    PortfolioService,
)
from crypto_bot.utils.trade_manager import Trade
from services.portfolio.clients.rest import PortfolioRestClient
from services.portfolio.schemas import (
    PnlBreakdown,
    PortfolioState,
    PositionRead,
    RiskCheckResult,
    TradeCreate,
)

_DEFAULT_TIMEOUT = 10.0
TimeoutValue = float | Tuple[float, float]


def _ensure_decimal(value: Any) -> Decimal:
    """Convert arbitrary numeric types to :class:`Decimal`."""

    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _env_timeout(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"Invalid value for {name!r}: {value!r}") from exc


def _resolve_timeout(
    timeout: Optional[float],
    connect_timeout: Optional[float],
    read_timeout: Optional[float],
) -> TimeoutValue:
    """Determine the timeout configuration for HTTP requests."""

    env_timeout = _env_timeout("PORTFOLIO_SERVICE_TIMEOUT")
    env_connect = _env_timeout("PORTFOLIO_SERVICE_CONNECT_TIMEOUT")
    env_read = _env_timeout("PORTFOLIO_SERVICE_READ_TIMEOUT")

    connect = connect_timeout if connect_timeout is not None else env_connect
    read = read_timeout if read_timeout is not None else env_read

    if connect is not None or read is not None:
        base = timeout if timeout is not None else env_timeout
        default = base if base is not None else _DEFAULT_TIMEOUT
        connect_value = connect if connect is not None else default
        read_value = read if read is not None else default
        return (connect_value, read_value)

    if timeout is not None:
        return timeout
    if env_timeout is not None:
        return env_timeout
    return _DEFAULT_TIMEOUT


class RemoteTradeManager:
    """Minimal trade manager facade backed by the portfolio service."""

    def __init__(self, client: PortfolioRestClient):
        self._client = client
        self._state: Optional[PortfolioState] = None
        self.trades: list[Any] = []
        self.positions: list[PositionRead] = []
        self.closed_positions: list[PositionRead] = []
        self.price_cache: dict[str, Decimal] = {}
        self.total_trades: int = 0
        self.total_volume: Decimal = Decimal("0")
        self.total_fees: Decimal = Decimal("0")
        self.total_realized_pnl: Decimal = Decimal("0")
        self.refresh()

    def refresh(self) -> None:
        """Fetch the latest state from the remote service."""

        state = self._client.get_state()
        self.update_from_state(state)

    def update_from_state(self, state: PortfolioState) -> None:
        """Apply a freshly retrieved :class:`PortfolioState`."""

        self._state = state
        self.trades = list(state.trades)
        self.positions = list(state.positions)
        self.closed_positions = list(state.closed_positions)
        self.price_cache = {
            entry.symbol: Decimal(str(entry.price)) for entry in state.price_cache
        }
        stats = state.statistics
        self.total_trades = stats.total_trades
        self.total_volume = Decimal(str(stats.total_volume))
        self.total_fees = Decimal(str(stats.total_fees))
        self.total_realized_pnl = Decimal(str(stats.total_realized_pnl))

    def get_state(self) -> PortfolioState:
        if self._state is None:
            self.refresh()
        assert self._state is not None
        return self._state

    def get_all_positions(self) -> list[PositionRead]:
        return list(self.positions)

    def get_position(self, symbol: str) -> Optional[PositionRead]:
        for position in self.positions:
            if position.symbol == symbol:
                return position
        for position in self.closed_positions:
            if position.symbol == symbol:
                return position
        return None

    def record_trade(self, trade: Trade) -> str:
        payload = TradeCreate(
            id=trade.id,
            symbol=trade.symbol,
            side=trade.side,
            amount=trade.amount,
            price=trade.price,
            timestamp=trade.timestamp,
            strategy=trade.strategy or None,
            exchange=trade.exchange or None,
            fees=trade.fees,
            status=trade.status,
            order_id=trade.order_id,
            client_order_id=trade.client_order_id,
            metadata=trade.metadata,
        )
        self._client.record_trade(payload)
        self.refresh()
        return trade.id

    def update_price(self, symbol: str, price: Decimal) -> Optional[PositionRead]:
        result = self._client.update_price(symbol, price)
        self.refresh()
        return result

    def save_state(self) -> PortfolioState:
        state = self.get_state()
        self._client.put_state(state)
        return state


class PortfolioAdapter(PortfolioService):
    """Adapter that proxies portfolio operations to the remote service."""

    def __init__(
        self,
        client: Optional[PortfolioRestClient] = None,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
        read_timeout: Optional[float] = None,
    ) -> None:
        if client is None:
            resolved_base_url = base_url or os.getenv("PORTFOLIO_SERVICE_URL")
            resolved_timeout = _resolve_timeout(timeout, connect_timeout, read_timeout)
            self._client = PortfolioRestClient(
                base_url=resolved_base_url, timeout=resolved_timeout
            )
        else:
            self._client = client
        self._trade_manager: Optional[RemoteTradeManager] = None

    def _trade_manager_refresh(self) -> None:
        if self._trade_manager is not None:
            self._trade_manager.refresh()

    def create_trade(self, request: CreateTradeRequest) -> CreateTradeResponse:
        metadata = dict(request.metadata) if request.metadata is not None else {}
        trade = Trade(
            id=str(uuid.uuid4()),
            symbol=request.symbol,
            side=request.side,
            amount=_ensure_decimal(request.amount),
            price=_ensure_decimal(request.price),
            timestamp=datetime.utcnow(),
            strategy=request.strategy,
            exchange=request.exchange,
            fees=_ensure_decimal(request.fees or 0),
            order_id=request.order_id,
            client_order_id=request.client_order_id,
            metadata=metadata,
        )

        payload = TradeCreate(
            id=trade.id,
            symbol=trade.symbol,
            side=trade.side,
            amount=trade.amount,
            price=trade.price,
            timestamp=trade.timestamp,
            strategy=trade.strategy or None,
            exchange=trade.exchange or None,
            fees=trade.fees,
            status=trade.status,
            order_id=trade.order_id,
            client_order_id=trade.client_order_id,
            metadata=trade.metadata,
        )
        self._client.record_trade(payload)
        self._trade_manager_refresh()
        return CreateTradeResponse(trade=trade)

    def get_state(self) -> PortfolioState:
        state = self._client.get_state()
        if self._trade_manager is not None:
            self._trade_manager.update_from_state(state)
        return state

    def list_positions(self) -> list[PositionRead]:
        return self._client.list_positions()

    def update_price(self, symbol: str, price: Any) -> Optional[PositionRead]:
        decimal_price = _ensure_decimal(price)
        result = self._client.update_price(symbol, decimal_price)
        self._trade_manager_refresh()
        return result

    def compute_pnl(self, symbol: Optional[str] = None) -> PnlBreakdown:
        return self._client.compute_pnl(symbol)

    def check_risk(self) -> list[RiskCheckResult]:
        return self._client.check_risk()

    def get_trade_manager(self) -> RemoteTradeManager:
        if self._trade_manager is None:
            self._trade_manager = RemoteTradeManager(self._client)
        else:
            self._trade_manager.refresh()
        return self._trade_manager
