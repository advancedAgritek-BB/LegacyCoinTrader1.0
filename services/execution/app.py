"""ASGI application exposing the execution service over HTTP."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Mapping, MutableMapping, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from .config import ExecutionServiceConfig
from .models import OrderAck, OrderFill, OrderRequest
from .service import ExecutionService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------


def _canonical_query(request: Request) -> str:
    query = request.url.query
    return query or ""


async def authenticate_request(request: Request) -> None:
    """Validate service-to-service authentication headers."""

    secret = os.getenv("EXECUTION_SERVICE_SECRET")
    if not secret:
        return  # Authentication disabled – primarily for local development/tests.

    signature = request.headers.get("X-Execution-Signature")
    timestamp = request.headers.get("X-Execution-Timestamp")
    if not signature or not timestamp:
        logger.warning("Missing authentication headers for execution request")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing auth headers")

    try:
        ts_value = float(timestamp)
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid timestamp") from exc

    if abs(time.time() - ts_value) > 60:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="stale request")

    body = await request.body()
    path = request.url.path
    method = request.method.upper()
    query = _canonical_query(request)
    message = b"".join([
        timestamp.encode(),
        method.encode(),
        path.encode(),
        b"?" + query.encode() if query else b"",
        body,
    ])
    expected = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        logger.warning("Invalid signature for execution request %s %s", method, path)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature")


# ---------------------------------------------------------------------------
# Registry & event channels
# ---------------------------------------------------------------------------


def _hash_config(config: Mapping[str, Any] | None) -> str:
    payload = json.dumps(dict(config or {}), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class OrderEventChannel:
    """Fan-out helper that buffers acknowledgement/fill events per order."""

    def __init__(self, service: ExecutionService, client_order_id: str) -> None:
        self.service = service
        self.client_order_id = client_order_id
        self.queue: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue()
        self._tasks: list[asyncio.Task[Any]] = []
        self._closed = False

    def start(self) -> None:
        ack_sub = self.service.subscribe_acks()
        fill_sub = self.service.subscribe_fills()
        self._tasks.append(asyncio.create_task(self._forward(ack_sub, "ack")))
        self._tasks.append(asyncio.create_task(self._forward(fill_sub, "fill")))

    async def _forward(self, subscription, event_type: str) -> None:
        try:
            while True:
                event = await subscription.get()
                if event.client_order_id != self.client_order_id:
                    continue
                payload = serialize_event(event)
                payload["type"] = event_type
                await self.queue.put(payload)
                if event_type == "fill":
                    break
                if event_type == "ack" and not getattr(event, "accepted", True):
                    break
                if event_type == "ack":
                    break
        finally:
            subscription.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):  # pragma: no cover - defensive
                await task


class ExecutionServiceRegistry:
    """Manage :class:`ExecutionService` instances keyed by configuration."""

    def __init__(self) -> None:
        self._services: MutableMapping[str, ExecutionService] = {}
        self._channels: MutableMapping[str, OrderEventChannel] = {}
        self._lock = asyncio.Lock()

    async def get_service(self, config: Mapping[str, Any] | None) -> ExecutionService:
        key = _hash_config(config)
        async with self._lock:
            service = self._services.get(key)
            if service is None:
                service = ExecutionService(ExecutionServiceConfig.from_mapping(config))
                # Establish connectivity immediately for deterministic errors.
                service.ensure_session()
                self._services[key] = service
            return service

    async def start_order_channel(self, service: ExecutionService, client_order_id: str) -> OrderEventChannel:
        async with self._lock:
            channel = self._channels.get(client_order_id)
            if channel is None:
                channel = OrderEventChannel(service, client_order_id)
                channel.start()
                self._channels[client_order_id] = channel
            return channel

    async def get_channel(self, client_order_id: str) -> Optional[OrderEventChannel]:
        async with self._lock:
            return self._channels.get(client_order_id)

    async def clear_channel(self, client_order_id: str) -> None:
        async with self._lock:
            channel = self._channels.pop(client_order_id, None)
        if channel is not None:
            await channel.close()

    async def aclose(self) -> None:
        async with self._lock:
            channels = list(self._channels.items())
            self._channels.clear()
        for order_id, channel in channels:
            logger.debug("Closing order channel %s", order_id)
            await channel.close()


# ---------------------------------------------------------------------------
# Pydantic models for HTTP payloads
# ---------------------------------------------------------------------------


class ExchangeCreatePayload(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


class ExchangeCreateResponse(BaseModel):
    status: str
    config_hash: str


class OrderSubmitPayload(BaseModel):
    symbol: str
    side: str
    amount: float
    client_order_id: Optional[str] = None
    dry_run: Optional[bool] = None
    use_websocket: Optional[bool] = None
    score: Optional[float] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrderSubmitResponse(BaseModel):
    client_order_id: str


class OrderEventResponse(BaseModel):
    type: str
    data: Dict[str, Any]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def serialize_event(event: OrderAck | OrderFill) -> dict[str, Any]:
    payload = asdict(event)
    payload.pop("notifier", None)
    return {"data": payload}


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app(registry: ExecutionServiceRegistry | None = None) -> FastAPI:
    app = FastAPI(title="Execution Service", version="1.0.0")
    app.state.registry = registry or ExecutionServiceRegistry()

    @app.on_event("shutdown")
    async def _shutdown_registry() -> None:  # pragma: no cover - teardown
        await app.state.registry.aclose()

    @app.post("/api/v1/execution/exchanges", response_model=ExchangeCreateResponse)
    async def create_exchange(
        payload: ExchangeCreatePayload,
        _auth: None = Depends(authenticate_request),
    ) -> ExchangeCreateResponse:
        service = await app.state.registry.get_service(payload.config)
        service.ensure_session()
        config_hash = _hash_config(payload.config)
        return ExchangeCreateResponse(status="ready", config_hash=config_hash)

    @app.post("/api/v1/execution/orders", response_model=OrderSubmitResponse)
    async def submit_order(
        payload: OrderSubmitPayload,
        _auth: None = Depends(authenticate_request),
    ) -> OrderSubmitResponse:
        service = await app.state.registry.get_service(payload.config)
        client_order_id = payload.client_order_id or service.generate_client_order_id()
        await app.state.registry.start_order_channel(service, client_order_id)
        order_request = OrderRequest(
            symbol=payload.symbol,
            side=payload.side,
            amount=payload.amount,
            client_order_id=client_order_id,
            dry_run=payload.dry_run if payload.dry_run is not None else service._config.dry_run,  # type: ignore[attr-defined]
            use_websocket=bool(payload.use_websocket),
            score=payload.score or 0.0,
            config=dict(payload.config),
            metadata=dict(payload.metadata),
        )
        await service.submit_order(order_request)
        return OrderSubmitResponse(client_order_id=client_order_id)

    @app.get("/api/v1/execution/orders/{client_order_id}/events", response_model=OrderEventResponse)
    async def next_order_event(
        client_order_id: str,
        timeout: float = 15.0,
        _auth: None = Depends(authenticate_request),
    ) -> OrderEventResponse:
        channel = await app.state.registry.get_channel(client_order_id)
        if channel is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown order")
        try:
            event = await asyncio.wait_for(channel.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="no events available")
        event_type = event.get("type")
        data = event.get("data", {})
        if event_type == "fill" or (event_type == "ack" and not data.get("accepted", True)):
            await app.state.registry.clear_channel(client_order_id)
        return OrderEventResponse(type=event_type, data=data)

    return app


# Convenience global for ``uvicorn`` style imports.
app = create_app()

