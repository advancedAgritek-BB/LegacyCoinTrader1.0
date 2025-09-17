"""In-process adapter for portfolio and trade management."""

from __future__ import annotations

from crypto_bot.services.interfaces import (
    CreateTradeRequest,
    CreateTradeResponse,
    PortfolioService,
)
from crypto_bot.utils.trade_manager import create_trade, get_trade_manager


class PortfolioAdapter(PortfolioService):
    """Adapter that exposes the TradeManager helpers via a protocol."""

    def create_trade(self, request: CreateTradeRequest) -> CreateTradeResponse:
        metadata = dict(request.metadata) if request.metadata is not None else None
        trade = create_trade(
            symbol=request.symbol,
            side=request.side,
            amount=request.amount,
            price=request.price,
            strategy=request.strategy,
            exchange=request.exchange,
            fees=request.fees,
            order_id=request.order_id,
            client_order_id=request.client_order_id,
            metadata=metadata,
        )
        return CreateTradeResponse(trade=trade)

    def get_trade_manager(self):  # type: ignore[override]
        return get_trade_manager()
