"""Adapter delegating trade execution to the dedicated HTTP microservice."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
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

_DEFAULT_BASE_URL = "http://execution:8006/api/v1/execution"
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class ExecutionApiError(RuntimeError):
    """Generic failure returned by the execution service API."""


class ExecutionAuthError(ExecutionApiError):
    """Raised when authentication with the execution service fails."""


class ExecutionTimeoutError(ExecutionApiError):
    """Raised when the API reports a timeout while waiting for an event."""


@dataclass(slots=True)
class _RequestOptions:
    method: str
    path: str
    body: bytes
    params: Optional[Mapping[str, Any]]
    expected: tuple[int, ...]
    event: Optional[str]
    request_timeout: Optional[float]


class ExecutionApiClient:
    """Minimal asynchronous HTTP client for the execution microservice."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        service_token: Optional[str] = None,
        signing_key: Optional[str] = None,
        timeout: Optional[float] = None,
        ack_timeout: Optional[float] = None,
        fill_timeout: Optional[float] = None,
        retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        sync_client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url or os.getenv("EXECUTION_SERVICE_URL", _DEFAULT_BASE_URL)
        self._url = httpx.URL(self.base_url)
        self._base_path = self._url.raw_path.decode("utf-8") if self._url.raw_path else self._url.path
        if not self._base_path:
            self._base_path = ""
        self.service_token = service_token or os.getenv("EXECUTION_SERVICE_TOKEN", "")
        signing_secret = signing_key or os.getenv("EXECUTION_SERVICE_SIGNING_KEY")
        self._signing_key = signing_secret.encode("utf-8") if signing_secret else None
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("EXECUTION_SERVICE_TIMEOUT", "10"))
        )
        self.ack_timeout = (
            ack_timeout
            if ack_timeout is not None
            else float(os.getenv("EXECUTION_SERVICE_ACK_TIMEOUT", "15"))
        )
        self.fill_timeout = (
            fill_timeout
            if fill_timeout is not None
            else float(os.getenv("EXECUTION_SERVICE_FILL_TIMEOUT", "60"))
        )
        self.retries = (
            retries if retries is not None else int(os.getenv("EXECUTION_SERVICE_RETRIES", "3"))
        )
        self.retry_backoff = (
            retry_backoff
            if retry_backoff is not None
            else float(os.getenv("EXECUTION_SERVICE_RETRY_BACKOFF", "0.5"))
        )
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        self._owns_client = client is None
        self._sync_client = sync_client
        self._timeout_margin = 1.0

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Core HTTP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_body(payload: Optional[Mapping[str, Any]]) -> bytes:
        if payload is None:
            return b""
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def _full_path(self, path: str) -> str:
        suffix = path if path.startswith("/") else f"/{path}"
        base = self._base_path.rstrip("/")
        full = f"{base}{suffix}" if base else suffix
        if not full.startswith("/"):
            full = f"/{full}"
        return full

    def _build_headers(self, options: _RequestOptions) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if options.body:
            headers["Content-Type"] = "application/json"
        if self.service_token:
            headers["X-Service-Token"] = self.service_token
        if self._signing_key is not None:
            timestamp = str(int(time.time()))
            message = f"{timestamp}|{options.method.upper()}|{self._full_path(options.path)}|{options.body.decode('utf-8')}"
            signature = hmac.new(self._signing_key, message.encode("utf-8"), hashlib.sha256).hexdigest()
            headers["X-Request-Timestamp"] = timestamp
            headers["X-Request-Signature"] = signature
        return headers

    def _prepare_request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
        expected: tuple[int, ...] = (200,),
        event: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> _RequestOptions:
        body = self._serialize_body(json_payload)
        return _RequestOptions(
            method=method,
            path=path,
            body=body,
            params=params,
            expected=expected,
            event=event,
            request_timeout=request_timeout,
        )

    def _extract_error(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, Mapping) and "detail" in payload:
                detail = payload["detail"]
                if isinstance(detail, (str, bytes)):
                    return detail if isinstance(detail, str) else detail.decode("utf-8")
        except ValueError:
            pass
        return response.text or f"HTTP {response.status_code}"

    async def _request(self, options: _RequestOptions) -> httpx.Response:
        backoff = self.retry_backoff
        for attempt in range(max(1, self.retries)):
            headers = self._build_headers(options)
            try:
                response = await self._client.request(
                    options.method,
                    options.path,
                    params=options.params,
                    content=options.body or None,
                    headers=headers,
                    timeout=options.request_timeout or self.timeout,
                )
            except httpx.RequestError as exc:
                if attempt >= self.retries - 1:
                    raise ExecutionApiError(f"Execution service request failed: {exc}") from exc
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            if response.status_code in options.expected:
                return response
            if response.status_code in {401, 403}:
                raise ExecutionAuthError(self._extract_error(response))
            if response.status_code == 504 and options.event:
                raise ExecutionTimeoutError(f"Timeout waiting for {options.event}")
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            raise ExecutionApiError(self._extract_error(response))
        raise ExecutionApiError("Execution service request exhausted retries")

    def _request_sync(self, options: _RequestOptions) -> httpx.Response:
        client = self._sync_client or httpx.Client(base_url=self.base_url, timeout=self.timeout)
        owns_client = self._sync_client is None
        backoff = self.retry_backoff
        try:
            for attempt in range(max(1, self.retries)):
                headers = self._build_headers(options)
                try:
                    response = client.request(
                        options.method,
                        options.path,
                        params=options.params,
                        content=options.body or None,
                        headers=headers,
                        timeout=options.request_timeout or self.timeout,
                    )
                except httpx.RequestError as exc:
                    if attempt >= self.retries - 1:
                        raise ExecutionApiError(f"Execution service request failed: {exc}") from exc
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                if response.status_code in options.expected:
                    return response
                if response.status_code in {401, 403}:
                    raise ExecutionAuthError(self._extract_error(response))
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise ExecutionApiError(self._extract_error(response))
        finally:
            if owns_client:
                client.close()
        raise ExecutionApiError("Execution service request exhausted retries")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_exchange(self, config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
        options = self._prepare_request(
            "POST",
            "/exchanges",
            json_payload={"config": dict(config or {})},
            expected=(200,),
        )
        response = self._request_sync(options)
        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - defensive guardrail
            raise ExecutionApiError("Invalid JSON returned from execution service") from exc
        return payload

    async def ensure_exchange_async(self, config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
        options = self._prepare_request(
            "POST",
            "/exchanges",
            json_payload={"config": dict(config or {})},
            expected=(200,),
        )
        response = await self._request(options)
        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - defensive guardrail
            raise ExecutionApiError("Invalid JSON returned from execution service") from exc
        return payload

    async def submit_order(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        options = self._prepare_request(
            "POST",
            "/orders",
            json_payload=payload,
            expected=(202,),
        )
        response = await self._request(options)
        return response.json() if response.content else {}

    async def wait_for_ack(self, client_order_id: str, timeout: Optional[float]) -> Mapping[str, Any]:
        params: Dict[str, Any] = {"wait_for": "ack"}
        if timeout is not None:
            params["timeout"] = timeout
        options = self._prepare_request(
            "GET",
            f"/orders/{client_order_id}/events",
            params=params,
            expected=(200,),
            event="ack",
            request_timeout=(timeout or self.ack_timeout) + self._timeout_margin,
        )
        response = await self._request(options)
        data = response.json()
        return data.get("data", data)

    async def wait_for_fill(self, client_order_id: str, timeout: Optional[float]) -> Mapping[str, Any]:
        params: Dict[str, Any] = {"wait_for": "fill"}
        if timeout is not None:
            params["timeout"] = timeout
        options = self._prepare_request(
            "GET",
            f"/orders/{client_order_id}/events",
            params=params,
            expected=(200,),
            event="fill",
            request_timeout=(timeout or self.fill_timeout) + self._timeout_margin,
        )
        response = await self._request(options)
        data = response.json()
        return data.get("data", data)


class ExecutionAdapter(ExecutionService):
    """Execution adapter backed by the HTTP execution microservice."""

    def __init__(self, client: Optional[ExecutionApiClient] = None) -> None:
        self._client = client or ExecutionApiClient()

    @staticmethod
    def _generate_client_order_id(config: Optional[Mapping[str, Any]]) -> str:
        base = "exec"
        if isinstance(config, Mapping):
            prefix = config.get("client_prefix")
            if isinstance(prefix, str) and prefix:
                base = prefix
            else:
                exchange_cfg = config.get("exchange")
                if isinstance(exchange_cfg, Mapping):
                    nested_prefix = exchange_cfg.get("client_prefix")
                    if isinstance(nested_prefix, str) and nested_prefix:
                        base = nested_prefix
        return f"{base}-{uuid.uuid4().hex}"

    @staticmethod
    def _resolve_timeout(config: Optional[Mapping[str, Any]], key: str, default: float) -> float:
        if not isinstance(config, Mapping):
            return default
        value = config.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - invalid config
            logger.warning("Invalid %s override: %r", key, value)
            return default

    def create_exchange(self, request: ExchangeRequest) -> ExchangeResponse:
        metadata = self._client.ensure_exchange(request.config)
        return ExchangeResponse(exchange=metadata, ws_client=None)

    async def execute_trade(self, request: TradeExecutionRequest) -> TradeExecutionResponse:
        config: MutableMapping[str, Any] = dict(request.config or {})
        client_order_id = self._generate_client_order_id(config)
        order_payload: Dict[str, Any] = {
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
        submission = await self._client.submit_order(order_payload)
        client_order_id = submission.get("client_order_id", client_order_id)
        ack_timeout = self._resolve_timeout(config, "ack_timeout", self._client.ack_timeout)
        try:
            ack = await self._client.wait_for_ack(client_order_id, ack_timeout)
        except ExecutionTimeoutError:
            logger.warning("Timed out waiting for acknowledgement of %s", client_order_id)
            return TradeExecutionResponse(order={})
        if not ack.get("accepted", False):
            logger.warning("Order %s rejected: %s", client_order_id, ack.get("reason"))
            return TradeExecutionResponse(order={})
        fill_timeout = self._resolve_timeout(config, "fill_timeout", self._client.fill_timeout)
        try:
            fill = await self._client.wait_for_fill(client_order_id, fill_timeout)
        except ExecutionTimeoutError:
            logger.warning("Timed out waiting for fill of %s", client_order_id)
            return TradeExecutionResponse(order={})
        if not fill.get("success", False):
            logger.warning("Order %s failed: %s", client_order_id, fill.get("error"))
            return TradeExecutionResponse(order=fill.get("order", {}))
        return TradeExecutionResponse(order=fill.get("order", {}))
