"""HTTP client adapter for the execution microservice."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Mapping, MutableMapping, Optional

import httpx

from crypto_bot.services.interfaces import (
    ExchangeRequest,
    ExchangeResponse,
    ExecutionService,
    TradeExecutionRequest,
    TradeExecutionResponse,
)

logger = logging.getLogger(__name__)


def _hash_config(config: Mapping[str, Any] | None) -> str:
    payload = json.dumps(dict(config or {}), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class ExecutionAdapter(ExecutionService):
    """Async HTTP adapter talking to the execution microservice."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        secret: Optional[str] = None,
        key: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        env_url = os.getenv("EXECUTION_SERVICE_URL")
        self._base_url = (base_url or env_url or "http://localhost:8006").rstrip("/")
        self._secret = secret or os.getenv("EXECUTION_SERVICE_SECRET")
        self._key = key or os.getenv("EXECUTION_SERVICE_KEY")
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._backoff = max(0.0, backoff)
        self._client = http_client or httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._owns_client = http_client is None
        self._sessions: MutableMapping[str, str] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_query(self, params: Optional[Mapping[str, Any]]) -> str:
        if not params:
            return ""
        if isinstance(params, str):
            return params
        # Preserve deterministic ordering for signing
        items = []
        for key, value in sorted(params.items(), key=lambda kv: str(kv[0])):
            if isinstance(value, (list, tuple)):
                for element in value:
                    items.append((str(key), str(element)))
            else:
                items.append((str(key), str(value)))
        return str(httpx.QueryParams(items))

    def _serialize_json(self, payload: Mapping[str, Any] | None) -> bytes:
        if payload is None:
            return b""
        return json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str).encode()

    def _sign_headers(self, method: str, path: str, query: str, body: bytes) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._secret:
            timestamp = str(int(time.time()))
            canonical = path
            if query:
                canonical = f"{path}?{query}"
            message = b"".join([
                timestamp.encode(),
                method.upper().encode(),
                canonical.encode(),
                body,
            ])
            signature = hmac.new(self._secret.encode(), message, hashlib.sha256).hexdigest()
            headers["X-Execution-Timestamp"] = timestamp
            headers["X-Execution-Signature"] = signature
            if self._key:
                headers["X-Execution-Key"] = self._key
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> httpx.Response:
        body = self._serialize_json(json_payload)
        query = self._build_query(params)
        # httpx requires params separately; provide raw string for deterministic signing.
        url_params = query if query else None
        for attempt in range(self._max_retries + 1):
            headers = self._sign_headers(method, path, query, body)
            if body and "Content-Type" not in headers:
                headers["Content-Type"] = "application/json"
            try:
                response = await self._client.request(
                    method,
                    path,
                    content=body if body else None,
                    params=url_params,
                    headers=headers,
                )
                response.raise_for_status()
                return response
            except httpx.RequestError as exc:
                if attempt >= self._max_retries:
                    raise
                await asyncio.sleep(self._backoff * (2 ** attempt))
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code >= 500 and attempt < self._max_retries:
                    await asyncio.sleep(self._backoff * (2 ** attempt))
                    continue
                raise
        raise RuntimeError("unreachable")

    async def _ensure_exchange(self, config: Mapping[str, Any] | None) -> str:
        key = _hash_config(config)
        if key in self._sessions:
            return self._sessions[key]
        payload = {"config": dict(config or {})}
        response = await self._request("POST", "/api/v1/execution/exchanges", json_payload=payload)
        data = response.json()
        self._sessions[key] = data.get("config_hash", key)
        return self._sessions[key]

    async def _next_event(self, client_order_id: str, timeout: float) -> dict[str, Any]:
        params = {"timeout": timeout}
        try:
            response = await self._request(
                "GET",
                f"/api/v1/execution/orders/{client_order_id}/events",
                params=params,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 408:
                raise asyncio.TimeoutError("execution event timeout") from exc
            raise
        return response.json()

    # ------------------------------------------------------------------
    # ExecutionService protocol implementation
    # ------------------------------------------------------------------

    def create_exchange(self, request: ExchangeRequest) -> ExchangeResponse:
        # Exchange connections are managed remotely; return a placeholder response.
        return ExchangeResponse(exchange={"status": "managed"}, ws_client=None)

    async def execute_trade(self, request: TradeExecutionRequest) -> TradeExecutionResponse:
        config = dict(request.config or {})
        await self._ensure_exchange(config)
        client_order_id = str(uuid.uuid4())
        payload = {
            "symbol": request.symbol,
            "side": request.side,
            "amount": request.amount,
            "client_order_id": client_order_id,
            "dry_run": request.dry_run,
            "use_websocket": request.use_websocket,
            "score": request.score,
            "config": config,
            "metadata": {"source": "crypto_bot"},
        }
        response = await self._request("POST", "/api/v1/execution/orders", json_payload=payload)
        data = response.json()
        client_order_id = data.get("client_order_id", client_order_id)

        ack_timeout = float(config.get("ack_timeout", 15.0))
        ack_event = await self._next_event(client_order_id, ack_timeout)
        if ack_event.get("type") != "ack":
            logger.warning("Execution service returned unexpected event %s for ack", ack_event.get("type"))
        ack_data = ack_event.get("data", {})
        if not ack_data.get("accepted", True):
            logger.warning(
                "Order %s rejected by execution service: %s",
                client_order_id,
                ack_data.get("reason", "unknown"),
            )
            return TradeExecutionResponse(order={})

        fill_timeout = float(config.get("fill_timeout", 60.0))
        fill_event = await self._next_event(client_order_id, fill_timeout)
        fill_data = fill_event.get("data", {})
        if not fill_data.get("success", False) or not fill_data.get("order"):
            logger.warning(
                "Order %s failed via execution service: %s",
                client_order_id,
                fill_data.get("error", "unknown"),
            )
            return TradeExecutionResponse(order=fill_data.get("order") or {})
        return TradeExecutionResponse(order=fill_data.get("order"))


__all__ = ["ExecutionAdapter"]

