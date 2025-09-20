"""Implementation of the execution microservice."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from threading import Lock
from typing import Any, Dict, Mapping, MutableMapping, Optional

from libs.execution import execute_trade_async
from libs.notifications import TelegramNotifier, send_message

from crypto_bot.services.adapters.market_data import MarketDataAdapter

from .config import ExecutionServiceConfig
from .exchange import ExchangeFactory
from .message_bus import AsyncTopic, TopicSubscription
from .models import OrderAck, OrderFill, OrderRequest
from .nonce import NonceManager
from .secret_loader import SecretLoader

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
        self._market_data = MarketDataAdapter()
        self._session_lock = Lock()
        self._session = None
        self._orders: MutableMapping[str, OrderRequest] = {}
        self._results: MutableMapping[str, OrderFill] = {}
        self._acks: MutableMapping[str, OrderAck] = {}
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
                exchange_config = dict(self._config.exchange)
                failover_endpoints = exchange_config.pop("failover_endpoints", None)
                if failover_endpoints:
                    endpoint = self._select_failover_endpoint(failover_endpoints)
                    if endpoint:
                        exchange_config.setdefault("api_endpoint", endpoint)
                        exchange_config.setdefault("rest_base_url", endpoint)
                credentials = self._secret_loader.load_credentials(self._config.credentials)
                self._session = self._exchange_factory.create_session(
                    exchange_config,
                    credentials,
                    self._nonce_manager,
                )
        return self._session

    @staticmethod
    def _select_failover_endpoint(endpoints: Any) -> Optional[str]:
        if isinstance(endpoints, str):
            return endpoints
        try:
            for candidate in endpoints or []:
                if candidate:
                    return str(candidate)
        except TypeError:  # pragma: no cover - defensive
            return None
        return None

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
        # For dry-run orders, execute immediately and return - no need for async event system
        if request.dry_run:
            session = self.ensure_session()
            try:
                # For dry-run trades, create a dummy disabled notifier if none is available
                notifier = request.notifier or self._telegram_notifier
                if notifier is None:
                    from crypto_bot.utils.telegram import TelegramNotifier
                    notifier = TelegramNotifier(enabled=False)

                order = await execute_trade_async(
                    session.exchange,
                    session.ws_client,
                    request.symbol,
                    request.side,
                    request.amount,
                    notifier=notifier,
                    dry_run=True,
                    use_websocket=request.use_websocket or self._config.use_websocket,
                    config=dict(self._config.exchange),
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
                        "dry_run": True,
                    },
                )
                # Store the fill for any future queries
                async with self._order_lock:
                    self._results[request.client_order_id] = fill
                    self._cleanup_results()
                return request.client_order_id
            except Exception as exc:
                logger.exception("Dry-run order execution failed: %s", exc)
                fill = OrderFill(
                    client_order_id=request.client_order_id,
                    success=False,
                    error=str(exc),
                    metadata={
                        "symbol": request.symbol,
                        "side": request.side,
                        "amount": request.amount,
                        "dry_run": True,
                    },
                )
                async with self._order_lock:
                    self._results[request.client_order_id] = fill
                    self._cleanup_results()
                return request.client_order_id

        # For live orders, use the async event system
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
        self._acks[request.client_order_id] = ack
        await self._ack_topic.publish(ack)
        await self._post_ack_callbacks(ack)
        task = asyncio.create_task(self._execute_order(session, request))
        task.add_done_callback(lambda t: t.exception() and logger.error(f"Order execution task failed: {t.exception()}", exc_info=t.exception()))
        return request.client_order_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_order(self, session, request: OrderRequest) -> None:
        logger.info(f"Starting order execution for {request.client_order_id} ({'dry-run' if request.dry_run else 'live'})")
        notifier = request.notifier or self._telegram_notifier
        # For dry-run trades, create a dummy disabled notifier if none is available
        is_dry_run = request.dry_run if request.dry_run is not None else self._config.dry_run
        if notifier is None and is_dry_run:
            from crypto_bot.utils.telegram import TelegramNotifier
            notifier = TelegramNotifier(enabled=False)

        merged_config: Dict[str, Any] = dict(self._config.exchange)
        merged_config.update(dict(request.config))

        # For dry-run trades, get current price from market data service
        if is_dry_run:
            try:
                # Try to get current price from market data service
                ticker_response = await self._market_data.get_ticker(request.symbol)
                if ticker_response and hasattr(ticker_response, 'close'):
                    merged_config['last_price'] = ticker_response.close
                    merged_config['price'] = ticker_response.close
                elif ticker_response and isinstance(ticker_response, dict):
                    # Try different possible price fields
                    for price_field in ['close', 'last', 'price']:
                        if price_field in ticker_response:
                            merged_config['last_price'] = ticker_response[price_field]
                            merged_config['price'] = ticker_response[price_field]
                            break
            except Exception as e:
                logger.debug(f"Could not fetch price for {request.symbol} from market data service: {e}")

        # For dry-run trades, still use real exchange for price fetching but ensure no orders are placed
        exchange = session.exchange
        ws_client = session.ws_client

        try:
            order = await execute_trade_async(
                exchange,
                ws_client,
                request.symbol,
                request.side,
                request.amount,
                notifier=notifier,
                dry_run=is_dry_run,
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
        logger.info(f"Stored fill for {request.client_order_id}: success={fill.success}, order_keys={list(fill.order.keys()) if fill.order else 'None'}")
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
        stale_acks = [oid for oid, ack in self._acks.items() if ack.timestamp < threshold]
        for oid in stale_acks:
            self._acks.pop(oid, None)

    # Compatibility helpers -------------------------------------------------

    def create_exchange(self):
        """Return the underlying exchange and WebSocket client."""
        session = self.ensure_session()
        return session.exchange, session.ws_client

    def pop_ack(self, client_order_id: str) -> Optional[OrderAck]:
        """Return and remove the cached acknowledgement for ``client_order_id``."""

        return self._acks.pop(client_order_id, None)

    def get_fill(self, client_order_id: str) -> Optional[OrderFill]:
        """Return the cached fill for ``client_order_id`` if available."""

        return self._results.get(client_order_id)
