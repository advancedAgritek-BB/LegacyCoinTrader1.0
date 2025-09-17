"""In-process adapter for centralized exchange execution."""

from __future__ import annotations

from crypto_bot.execution.cex_executor import execute_trade_async, get_exchange
from crypto_bot.services.interfaces import (
    ExchangeRequest,
    ExchangeResponse,
    ExecutionService,
    TradeExecutionRequest,
    TradeExecutionResponse,
)


class ExecutionAdapter(ExecutionService):
    """Adapter for :mod:`crypto_bot.execution.cex_executor`."""

    def create_exchange(self, request: ExchangeRequest) -> ExchangeResponse:
        exchange, ws_client = get_exchange(dict(request.config))
        return ExchangeResponse(exchange=exchange, ws_client=ws_client)

    async def execute_trade(self, request: TradeExecutionRequest) -> TradeExecutionResponse:
        order = await execute_trade_async(
            request.exchange,
            request.ws_client,
            request.symbol,
            request.side,
            request.amount,
            notifier=request.notifier,
            dry_run=request.dry_run,
            use_websocket=request.use_websocket,
            config=dict(request.config or {}),
            score=request.score,
        )
        return TradeExecutionResponse(order=order)
