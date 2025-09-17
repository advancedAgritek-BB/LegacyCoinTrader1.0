"""Adapter delegating trade execution to the execution microservice."""

from __future__ import annotations

import asyncio
import json
import logging
from threading import Lock
from typing import Any, Mapping

from services.execution import ExecutionService as CoreExecutionService
from services.execution import ExecutionServiceConfig, OrderRequest

from crypto_bot.services.interfaces import (
    ExchangeRequest,
    ExchangeResponse,
    ExecutionService,
    TradeExecutionRequest,
    TradeExecutionResponse,
)

logger = logging.getLogger(__name__)


class ExecutionAdapter(ExecutionService):
    """Execution adapter backed by :mod:`services.execution`."""

    def __init__(self) -> None:
        self._services: dict[str, CoreExecutionService] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _config_key(self, config: Mapping[str, Any] | None) -> str:
        payload = json.dumps(dict(config or {}), sort_keys=True, default=str)
        return payload

    def _get_service(self, config: Mapping[str, Any] | None) -> CoreExecutionService:
        key = self._config_key(config)
        with self._lock:
            service = self._services.get(key)
            if service is None:
                service = CoreExecutionService(ExecutionServiceConfig.from_mapping(config or {}))
                # Ensure connectivity is established eagerly for deterministic errors
                service.ensure_session()
                self._services[key] = service
        return service

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def create_exchange(self, request: ExchangeRequest) -> ExchangeResponse:
        service = self._get_service(request.config)
        exchange, ws_client = service.create_exchange()
        return ExchangeResponse(exchange=exchange, ws_client=ws_client)

    async def execute_trade(self, request: TradeExecutionRequest) -> TradeExecutionResponse:
        service = self._get_service(request.config)
        client_order_id = service.generate_client_order_id()
        ack_subscription = service.subscribe_acks()
        fill_subscription = service.subscribe_fills()
        try:
            order_request = OrderRequest(
                symbol=request.symbol,
                side=request.side,
                amount=request.amount,
                client_order_id=client_order_id,
                dry_run=request.dry_run,
                use_websocket=request.use_websocket,
                score=request.score,
                config=dict(request.config or {}),
                notifier=request.notifier,
                metadata={"source": "crypto_bot"},
            )
            await service.submit_order(order_request)
            ack = await self._await_ack(ack_subscription, client_order_id, request.config)
            if not ack.accepted:
                logger.warning("Order %s rejected: %s", client_order_id, ack.reason)
                return TradeExecutionResponse(order={})
            fill = await self._await_fill(fill_subscription, client_order_id, request.config)
            if not fill.success or not fill.order:
                logger.warning("Order %s failed: %s", client_order_id, fill.error)
                return TradeExecutionResponse(order=fill.order or {})
            return TradeExecutionResponse(order=fill.order)
        finally:
            ack_subscription.close()
            fill_subscription.close()

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    async def _await_ack(
        self,
        subscription,
        client_order_id: str,
        config: Mapping[str, Any] | None,
    ):
        timeout = float((config or {}).get("ack_timeout", 15.0))
        while True:
            ack = await asyncio.wait_for(subscription.get(), timeout=timeout)
            if ack.client_order_id != client_order_id:
                continue
            return ack

    async def _await_fill(
        self,
        subscription,
        client_order_id: str,
        config: Mapping[str, Any] | None,
    ):
        timeout = float((config or {}).get("fill_timeout", 60.0))
        while True:
            fill = await asyncio.wait_for(subscription.get(), timeout=timeout)
            if fill.client_order_id != client_order_id:
                continue
            return fill
