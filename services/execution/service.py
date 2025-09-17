"""Implementation of the execution microservice."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from threading import Lock
from typing import Any, Dict, Mapping, MutableMapping, Optional

from crypto_bot.execution.cex_executor import execute_trade_async
from crypto_bot.utils.telegram import TelegramNotifier, send_message

from .config import ExecutionServiceConfig
from .exchange import ExchangeFactory
from .message_bus import AsyncTopic, TopicSubscription
from .models import OrderAck, OrderFill, OrderRequest
from .nonce import NonceManager
from .secrets import SecretLoader

logger = logging.getLogger("services.execution")


class ExecutionService:
    """Coordinates exchange connectivity and order lifecycle events."""

    def __init__(
        self,
        config: ExecutionServiceConfig | Mapping[str, Any] | None,
        *,
        secret_loader: Optional[SecretLoader] = None,
    ) -> None:
        self._config = ExecutionServiceConfig.from_mapping(config)
        self._secret_loader = secret_loader or SecretLoader()
        self._nonce_manager = NonceManager()
        self._exchange_factory = ExchangeFactory(self._secret_loader)
        self._session_lock = Lock()
        self._session = None
        self._orders: MutableMapping[str, OrderRequest] = {}
        self._results: MutableMapping[str, OrderFill] = {}
        self._order_lock = asyncio.Lock()
        self._ack_topic = AsyncTopic[OrderAck]()
        self._fill_topic = AsyncTopic[OrderFill]()
        self._telegram_notifier: Optional[TelegramNotifier] = None
        self._init_notifier()

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def _init_notifier(self) -> None:
        if not self._config.telegram.enabled:
            return
        token = self._secret_loader.load_secret(self._config.telegram.token)
        chat_id = self._secret_loader.load_secret(self._config.telegram.chat_id)
        if token and chat_id:
            try:
                self._telegram_notifier = TelegramNotifier(token, chat_id)
            except Exception as exc:  # pragma: no cover - depends on telegram pkg
                logger.warning("Failed to create Telegram notifier: %s", exc)

    def ensure_session(self):
        """Ensure an exchange session is available and return it."""
        with self._session_lock:
            if self._session is None:
                credentials = self._secret_loader.load_credentials(self._config.credentials)
                self._session = self._exchange_factory.create_session(
                    self._config.exchange,
                    credentials,
                    self._nonce_manager,
                )
        return self._session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe_acks(self) -> TopicSubscription[OrderAck]:
        return self._ack_topic.subscribe()

    def subscribe_fills(self) -> TopicSubscription[OrderFill]:
        return self._fill_topic.subscribe()

    def generate_client_order_id(self, prefix: str | None = None) -> str:
        base = prefix or self._config.exchange.get("client_prefix") or "exec"
        return f"{base}-{uuid.uuid4().hex}"

    async def submit_order(self, request: OrderRequest) -> str:
        session = self.ensure_session()
        metadata = {
            "symbol": request.symbol,
            "side": request.side,
            "amount": request.amount,
            "dry_run": request.dry_run,
        }
        async with self._order_lock:
            if request.client_order_id in self._results:
                ack = OrderAck(
                    client_order_id=request.client_order_id,
                    accepted=True,
                    reason="duplicate",
                    metadata=metadata,
                )
                await self._ack_topic.publish(ack)
                await self._fill_topic.publish(self._results[request.client_order_id])
                return request.client_order_id
            if request.client_order_id in self._orders:
                return request.client_order_id
            self._orders[request.client_order_id] = request
        ack = OrderAck(client_order_id=request.client_order_id, accepted=True, metadata=metadata)
        await self._ack_topic.publish(ack)
        await self._post_ack_callbacks(ack)
        asyncio.create_task(self._execute_order(session, request))
        return request.client_order_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_order(self, session, request: OrderRequest) -> None:
        notifier = request.notifier or self._telegram_notifier
        merged_config: Dict[str, Any] = dict(self._config.exchange)
        merged_config.update(dict(request.config))
        try:
            order = await execute_trade_async(
                session.exchange,
                session.ws_client,
                request.symbol,
                request.side,
                request.amount,
                notifier=notifier,
                dry_run=request.dry_run if request.dry_run is not None else self._config.dry_run,
                use_websocket=request.use_websocket or self._config.use_websocket,
                config=merged_config,
                score=request.score,
            )
            fill = OrderFill(
                client_order_id=request.client_order_id,
                success=True,
                order=order,
                metadata={
                    "symbol": request.symbol,
                    "side": request.side,
                    "amount": request.amount,
                    "dry_run": request.dry_run,
                },
            )
        except Exception as exc:  # pragma: no cover - network side effects
            logger.exception("Order execution failed: %s", exc)
            fill = OrderFill(
                client_order_id=request.client_order_id,
                success=False,
                error=str(exc),
                metadata={
                    "symbol": request.symbol,
                    "side": request.side,
                    "amount": request.amount,
                    "dry_run": request.dry_run,
                },
            )
        async with self._order_lock:
            self._orders.pop(request.client_order_id, None)
            self._results[request.client_order_id] = fill
            self._cleanup_results()
        await self._fill_topic.publish(fill)
        await self._post_fill_callbacks(fill)

    async def _post_ack_callbacks(self, ack: OrderAck) -> None:
        if not self._config.telegram.enabled:
            return
        if self._telegram_notifier is not None:
            message = (
                f"\u2705 Order accepted {ack.metadata.get('side')} {ack.metadata.get('amount')}"
                f" {ack.metadata.get('symbol')} (id {ack.client_order_id})"
            )
            try:
                err = self._telegram_notifier.notify(message)
                if err:
                    logger.warning("Telegram notification error: %s", err)
            except Exception as exc:  # pragma: no cover - telegram optional
                logger.warning("Failed to send Telegram acknowledgement: %s", exc)
        else:
            token = self._secret_loader.load_secret(self._config.telegram.token)
            chat_id = self._secret_loader.load_secret(self._config.telegram.chat_id)
            if token and chat_id:
                message = (
                    f"\u2705 Order accepted {ack.metadata.get('side')} {ack.metadata.get('amount')}"
                    f" {ack.metadata.get('symbol')} (id {ack.client_order_id})"
                )
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, send_message, token, chat_id, message)

    async def _post_fill_callbacks(self, fill: OrderFill) -> None:
        if not self._config.monitoring.enabled:
            return
        status = "filled" if fill.success else "failed"
        logger.info(
            "order.%s symbol=%s side=%s amount=%s id=%s",  # monitoring log line
            status,
            fill.metadata.get("symbol"),
            fill.metadata.get("side"),
            fill.metadata.get("amount"),
            fill.client_order_id,
        )

    def _cleanup_results(self) -> None:
        ttl = max(0.0, self._config.idempotency_ttl)
        if not ttl:
            return
        threshold = time.time() - ttl
        stale = [oid for oid, fill in self._results.items() if fill.timestamp < threshold]
        for oid in stale:
            self._results.pop(oid, None)

    # Compatibility helpers -------------------------------------------------

    def create_exchange(self):
        """Return the underlying exchange and WebSocket client."""
        session = self.ensure_session()
        return session.exchange, session.ws_client
