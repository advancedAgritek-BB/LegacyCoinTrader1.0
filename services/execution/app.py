"""FastAPI application exposing the execution service over HTTP."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from asyncio import Lock
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict, Mapping, MutableMapping, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging import configure_logging

from .config import ExecutionApiSettings, ExecutionServiceConfig, get_execution_api_settings
from .models import OrderAck, OrderFill, OrderRequest
from .service import ExecutionService

LOGGER = logging.getLogger("services.execution.api")

_execution_settings = get_execution_api_settings()
monitoring_settings = get_monitoring_settings().for_service("execution-service")
monitoring_settings = monitoring_settings.model_copy(
    update={"log_level": _execution_settings.log_level}
)
monitoring_settings.metrics.default_labels.setdefault("component", "execution")
configure_logging(monitoring_settings)


class ExchangeCreatePayload(BaseModel):
    """Incoming payload for creating (or ensuring) an exchange session."""

    config: Optional[Dict[str, Any]] = Field(default=None)


class ExchangeCreateResponse(BaseModel):
    """Metadata returned after ensuring an exchange session exists."""

    status: str = Field(default="ready")
    config_hash: str
    exchange_id: Optional[str] = None


class OrderSubmitPayload(BaseModel):
    """Incoming payload describing an order submission."""

    symbol: str
    side: str
    amount: float
    client_order_id: Optional[str] = None
    dry_run: Optional[bool] = None
    use_websocket: Optional[bool] = None
    score: Optional[float] = Field(default=None)
    config: Optional[Dict[str, Any]] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class OrderSubmitResponse(BaseModel):
    """Response returned after an order submission request."""

    client_order_id: str
    status: str = Field(default="submitted")


class OrderEventResponse(BaseModel):
    """Single acknowledgement or fill event payload."""

    type: str = Field(pattern=r"^(ack|fill)$")
    data: Dict[str, Any]


class ExecutionApplicationState:
    """Holds cached execution service instances keyed by configuration."""

    def __init__(self, settings: ExecutionApiSettings) -> None:
        self.settings = settings
        self._services: Dict[str, ExecutionService] = {}
        self._service_lock = Lock()
        self._order_index: Dict[str, str] = {}
        self._order_lock = Lock()
        self._ack_cache: Dict[str, OrderAck] = {}
        self._fill_cache: Dict[str, OrderFill] = {}
        self._order_tasks: Dict[str, list[asyncio.Task[Any]]] = {}

    @staticmethod
    def _config_key(config: Optional[Mapping[str, Any]]) -> str:
        payload = dict(config or {})
        return json.dumps(payload, sort_keys=True, default=str)

    async def get_service(
        self, config: Optional[Mapping[str, Any]]
    ) -> tuple[str, ExecutionService]:
        key = self._config_key(config)
        async with self._service_lock:
            service = self._services.get(key)
            if service is None:
                service = ExecutionService(ExecutionServiceConfig.from_mapping(config or {}))
                service.ensure_session()
                self._services[key] = service
        return key, service

    async def register_order(
        self, client_order_id: str, config_key: str, service: ExecutionService
    ) -> None:
        async with self._order_lock:
            self._order_index[client_order_id] = config_key
            self._ack_cache.pop(client_order_id, None)
            self._fill_cache.pop(client_order_id, None)
        ack_task = asyncio.create_task(self._capture_ack(service, client_order_id))
        fill_task = asyncio.create_task(self._capture_fill(service, client_order_id))
        async with self._order_lock:
            self._order_tasks[client_order_id] = [ack_task, fill_task]
        await asyncio.sleep(0)

    async def resolve_order_service(self, client_order_id: str) -> ExecutionService:
        async with self._order_lock:
            config_key = self._order_index.get(client_order_id)
        if not config_key:
            raise KeyError(client_order_id)
        async with self._service_lock:
            service = self._services.get(config_key)
        if service is None:
            raise KeyError(client_order_id)
        return service

    async def clear_order(self, client_order_id: str) -> None:
        tasks: list[asyncio.Task[Any]]
        async with self._order_lock:
            self._order_index.pop(client_order_id, None)
            self._ack_cache.pop(client_order_id, None)
            self._fill_cache.pop(client_order_id, None)
            tasks = self._order_tasks.pop(client_order_id, [])
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task

    async def consume_ack(self, client_order_id: str) -> Optional[OrderAck]:
        async with self._order_lock:
            return self._ack_cache.pop(client_order_id, None)

    async def consume_fill(self, client_order_id: str) -> Optional[OrderFill]:
        async with self._order_lock:
            return self._fill_cache.pop(client_order_id, None)

    async def _capture_ack(self, service: ExecutionService, order_id: str) -> None:
        subscription = service.subscribe_acks()
        try:
            while True:
                ack = await subscription.get()
                if ack.client_order_id != order_id:
                    continue
                async with self._order_lock:
                    self._ack_cache[order_id] = ack
                return
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        finally:
            subscription.close()

    async def _capture_fill(self, service: ExecutionService, order_id: str) -> None:
        subscription = service.subscribe_fills()
        try:
            while True:
                fill = await subscription.get()
                if fill.client_order_id != order_id:
                    continue
                async with self._order_lock:
                    self._fill_cache[order_id] = fill
                return
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        finally:
            subscription.close()

    async def aclose(self) -> None:
        async with self._service_lock:
            services = list(self._services.values())
            self._services.clear()
        async with self._order_lock:
            pending = list(self._order_tasks.values())
            self._order_tasks.clear()
        for tasks in pending:
            for task in tasks:
                task.cancel()
        for tasks in pending:
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task
        for service in services:
            session = getattr(service, "_session", None)
            if session is None:
                continue
            exchange = getattr(session, "exchange", None)
            close = getattr(exchange, "close", None)
            if callable(close):
                if asyncio.iscoroutinefunction(close):  # pragma: no branch - defensive
                    try:
                        await close()
                    except Exception:  # pragma: no cover - best effort cleanup
                        LOGGER.debug("Failed to close exchange cleanly", exc_info=True)
                else:
                    try:
                        await asyncio.to_thread(close)
                    except Exception:  # pragma: no cover - best effort cleanup
                        LOGGER.debug("Failed to close exchange cleanly", exc_info=True)
            ws_client = getattr(session, "ws_client", None)
            ws_close = getattr(ws_client, "close", None)
            if callable(ws_close):
                try:
                    result = ws_close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # pragma: no cover - best effort cleanup
                    LOGGER.debug("Failed to close websocket client", exc_info=True)


def _ack_to_dict(ack: OrderAck) -> Dict[str, Any]:
    return {
        "client_order_id": ack.client_order_id,
        "accepted": ack.accepted,
        "reason": ack.reason,
        "timestamp": ack.timestamp,
        "metadata": dict(ack.metadata),
    }


def _fill_to_dict(fill: OrderFill) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "client_order_id": fill.client_order_id,
        "success": fill.success,
        "error": fill.error,
        "timestamp": fill.timestamp,
        "metadata": dict(fill.metadata),
    }
    if fill.order is not None:
        payload["order"] = jsonable_encoder(fill.order)
    return payload


async def get_state(request: Request) -> ExecutionApplicationState:
    state: Optional[ExecutionApplicationState] = getattr(
        request.app.state, "execution_state", None
    )
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Execution service unavailable",
        )
    return state


async def _verify_signature(
    request: Request, settings: ExecutionApiSettings
) -> None:
    if not settings.signing_key:
        return
    provided = request.headers.get("x-request-signature")
    timestamp_header = request.headers.get("x-request-timestamp")
    if not provided or not timestamp_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing signature")
    try:
        timestamp = int(timestamp_header)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature timestamp") from exc
    now = int(time.time())
    if abs(now - timestamp) > settings.signature_ttl_seconds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Signature expired")
    body = await request.body()
    payload = f"{timestamp}|{request.method.upper()}|{request.url.path}|{body.decode('utf-8')}"
    expected = hmac.new(
        settings.signing_key.encode("utf-8"), payload.encode("utf-8"), digestmod="sha256"
    ).hexdigest()
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")


async def authorize_request(
    request: Request, state: ExecutionApplicationState = Depends(get_state)
) -> None:
    settings = state.settings
    token = request.headers.get("x-service-token")
    if settings.service_token and not hmac.compare_digest(
        settings.service_token, token or ""
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service token")
    await _verify_signature(request, settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_execution_api_settings()
    logging.getLogger().setLevel(settings.log_level.upper())
    state = ExecutionApplicationState(settings)
    app.state.execution_state = state
    try:
        yield
    finally:
        await state.aclose()


def create_app() -> FastAPI:
    settings = get_execution_api_settings()
    app = FastAPI(title="Execution Service", lifespan=lifespan)
    instrument_fastapi_app(app, settings=monitoring_settings)
    router = APIRouter(prefix=settings.base_path)

    @router.post(
        "/exchanges",
        response_model=ExchangeCreateResponse,
        dependencies=[Depends(authorize_request)],
    )
    async def create_exchange_endpoint(
        payload: ExchangeCreatePayload, state: ExecutionApplicationState = Depends(get_state)
    ) -> ExchangeCreateResponse:
        config = payload.config or {}
        _, service = await state.get_service(config)
        session = service.ensure_session()
        exchange = getattr(session.exchange, "id", None)
        return ExchangeCreateResponse(
            status="ready",
            config_hash=session.config_hash,
            exchange_id=str(exchange) if exchange else None,
        )

    @router.post(
        "/orders",
        response_model=OrderSubmitResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(authorize_request)],
    )
    async def submit_order_endpoint(
        payload: OrderSubmitPayload, state: ExecutionApplicationState = Depends(get_state)
    ) -> OrderSubmitResponse:
        config = payload.config or {}
        config_key, service = await state.get_service(config)
        client_order_id = payload.client_order_id or service.generate_client_order_id()
        await state.register_order(client_order_id, config_key, service)
        metadata: MutableMapping[str, Any] = dict(payload.metadata or {})
        metadata.setdefault("source", "execution-api")
        request_payload = OrderRequest(
            symbol=payload.symbol,
            side=payload.side,
            amount=payload.amount,
            client_order_id=client_order_id,
            dry_run=payload.dry_run if payload.dry_run is not None else service._config.dry_run,
            use_websocket=
                payload.use_websocket if payload.use_websocket is not None else service._config.use_websocket,
            score=payload.score or 0.0,
            config=dict(config),
            metadata=metadata,
        )
        try:
            await service.submit_order(request_payload)
        except Exception:
            await state.clear_order(client_order_id)
            LOGGER.exception("Order submission failed for %s", client_order_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Order submission failed",
            )
        return OrderSubmitResponse(client_order_id=client_order_id)

    @router.get(
        "/orders/{client_order_id}/events",
        response_model=OrderEventResponse,
        dependencies=[Depends(authorize_request)],
    )
    async def poll_order_events(
        client_order_id: str,
        wait_for: str = "ack",
        timeout: Optional[float] = None,
        state: ExecutionApplicationState = Depends(get_state),
    ) -> OrderEventResponse:
        if wait_for not in {"ack", "fill"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported event type")
        try:
            service = await state.resolve_order_service(client_order_id)
        except KeyError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown order")
        if wait_for == "ack":
            cached_ack = await state.consume_ack(client_order_id)
            if cached_ack is not None:
                return OrderEventResponse(type="ack", data=_ack_to_dict(cached_ack))
            ack_from_service = service.pop_ack(client_order_id)
            if ack_from_service is not None:
                return OrderEventResponse(type="ack", data=_ack_to_dict(ack_from_service))
        else:
            cached_fill = await state.consume_fill(client_order_id)
            if cached_fill is not None:
                await state.clear_order(client_order_id)
                return OrderEventResponse(type="fill", data=_fill_to_dict(cached_fill))
            fill_from_service = service.get_fill(client_order_id)
            if fill_from_service is not None:
                await state.clear_order(client_order_id)
                return OrderEventResponse(type="fill", data=_fill_to_dict(fill_from_service))
        subscription = (
            service.subscribe_acks() if wait_for == "ack" else service.subscribe_fills()
        )
        timeout_value = timeout
        if timeout_value is None:
            timeout_value = (
                state.settings.ack_timeout if wait_for == "ack" else state.settings.fill_timeout
            )
        try:
            while True:
                event = await asyncio.wait_for(subscription.get(), timeout=timeout_value)
                if event.client_order_id != client_order_id:
                    continue
                if isinstance(event, OrderAck):
                    return OrderEventResponse(type="ack", data=_ack_to_dict(event))
                if isinstance(event, OrderFill):
                    await state.clear_order(client_order_id)
                    return OrderEventResponse(type="fill", data=_fill_to_dict(event))
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Timeout waiting for {wait_for}",
            ) from exc
        finally:
            subscription.close()

    app.include_router(router)

    @app.get("/health")
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readiness")
    async def readiness(state: ExecutionApplicationState = Depends(get_state)) -> Dict[str, Any]:
        return {
            "status": "ready",
            "services": len(state._services),
        }

    return app


app = create_app()

__all__ = [
    "app",
    "create_app",
]
