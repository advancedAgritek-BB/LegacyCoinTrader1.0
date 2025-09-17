from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Mapping, MutableMapping, Optional

import httpx
import pandas as pd

from libs.services.interfaces import (
    ExchangeRequest,
    ExchangeResponse,
    ExecutionService,
    ServiceContainer,
    TradeExecutionRequest,
    TradeExecutionResponse,
)

from crypto_bot.services.adapters.market_data import MarketDataAdapter
from crypto_bot.services.adapters.monitoring import MonitoringAdapter
from crypto_bot.services.adapters.portfolio import PortfolioAdapter
from crypto_bot.services.adapters.strategy import StrategyAdapter
from crypto_bot.services.adapters.token_discovery import TokenDiscoveryAdapter

from crypto_bot.services.adapters.execution import ExecutionApiClient, ExecutionTimeoutError

logger = logging.getLogger(__name__)


_DEFAULT_EXECUTION_BASE_URL = os.getenv(
    "EXECUTION_SERVICE_URL", "http://execution:8006/api/v1/execution"
)


class ExecutionGatewayClient(ExecutionService):
    """HTTP client that forwards trade execution to the execution service."""

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
        async_client: Optional[httpx.AsyncClient] = None,
        sync_client: Optional[httpx.Client] = None,
    ) -> None:
        self._api = ExecutionApiClient(
            base_url=base_url or _DEFAULT_EXECUTION_BASE_URL,
            service_token=service_token,
            signing_key=signing_key,
            timeout=timeout,
            ack_timeout=ack_timeout,
            fill_timeout=fill_timeout,
            retries=retries,
            retry_backoff=retry_backoff,
            client=async_client,
            sync_client=sync_client,
        )

    async def aclose(self) -> None:
        """Release underlying HTTP resources."""

        await self._api.aclose()

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
        except (TypeError, ValueError):
            logger.warning("Invalid %s override: %r", key, value)
            return default

    def create_exchange(self, request: ExchangeRequest) -> ExchangeResponse:
        metadata = self._api.ensure_exchange(request.config)
        return ExchangeResponse(exchange=metadata, ws_client=None)

    async def execute_trade(self, request: TradeExecutionRequest) -> TradeExecutionResponse:
        config: MutableMapping[str, Any] = dict(request.config or {})
        client_order_id = self._generate_client_order_id(config)
        payload: Dict[str, Any] = {
            "symbol": request.symbol,
            "side": request.side,
            "amount": request.amount,
            "client_order_id": client_order_id,
            "dry_run": request.dry_run,
            "use_websocket": request.use_websocket,
            "score": request.score,
            "config": config,
            "metadata": {"source": "trading-engine"},
        }
        submission = await self._api.submit_order(payload)
        client_order_id = submission.get("client_order_id", client_order_id)
        ack_timeout = self._resolve_timeout(config, "ack_timeout", self._api.ack_timeout)
        try:
            ack = await self._api.wait_for_ack(client_order_id, ack_timeout)
        except ExecutionTimeoutError:
            logger.warning("Timed out waiting for acknowledgement for %s", client_order_id)
            return TradeExecutionResponse(order={})
        if not ack.get("accepted", False):
            logger.warning("Order %s rejected: %s", client_order_id, ack.get("reason"))
            return TradeExecutionResponse(order={})
        fill_timeout = self._resolve_timeout(config, "fill_timeout", self._api.fill_timeout)
        try:
            fill = await self._api.wait_for_fill(client_order_id, fill_timeout)
        except ExecutionTimeoutError:
            logger.warning("Timed out waiting for fill for %s", client_order_id)
            return TradeExecutionResponse(order={})
        if not fill.get("success", False):
            logger.warning("Order %s failed: %s", client_order_id, fill.get("error"))
            return TradeExecutionResponse(order=fill.get("order", {}))
        return TradeExecutionResponse(order=fill.get("order", {}))


class RiskServiceClient:
    """HTTP client facade for the risk management microservice."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        config: Optional[Mapping[str, Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        resolved_base = base_url or os.getenv("RISK_SERVICE_URL", "http://risk:8010/api/v1")
        resolved_timeout = (
            timeout if timeout is not None else float(os.getenv("RISK_SERVICE_TIMEOUT", "10"))
        )
        self._client = client or httpx.AsyncClient(base_url=resolved_base, timeout=resolved_timeout)
        self._owns_client = client is None
        self._config = dict(config or {})

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _serialise_dataframe(df: Optional[pd.DataFrame]) -> Optional[list[dict[str, Any]]]:
        if df is None or df.empty:
            return None
        try:
            trimmed = df.tail(250).reset_index()
        except Exception:  # pragma: no cover - fallback for unusual structures
            trimmed = df.tail(250)
        try:
            return json.loads(trimmed.to_json(orient="records", date_format="iso"))
        except TypeError:  # pragma: no cover - defensive conversion
            records: list[dict[str, Any]] = []
            for row in trimmed.to_dict(orient="records"):
                serialised: dict[str, Any] = {}
                for key, value in row.items():
                    if hasattr(value, "isoformat"):
                        serialised[key] = value.isoformat()
                    elif isinstance(value, (float, int, str, bool)) or value is None:
                        serialised[key] = value
                    else:
                        serialised[key] = str(value)
                records.append(serialised)
            return records

    async def allow_trade(
        self, df: pd.DataFrame, strategy: Optional[str]
    ) -> tuple[bool, str]:
        payload = {
            "strategy": strategy,
            "config": self._config,
            "market_data": self._serialise_dataframe(df),
        }
        response = await self._client.post("/risk/allow", json=payload)
        response.raise_for_status()
        data = response.json() if response.content else {}
        allowed = bool(data.get("allowed", True))
        reason = data.get("reason", "")
        return allowed, str(reason) if isinstance(reason, str) else ""

    async def position_size(
        self,
        confidence: float,
        balance: float,
        *,
        df: Optional[pd.DataFrame] = None,
        atr: Optional[float] = None,
        price: Optional[float] = None,
    ) -> float:
        payload = {
            "confidence": confidence,
            "balance": balance,
            "atr": atr,
            "price": price,
            "config": self._config,
            "market_data": self._serialise_dataframe(df),
        }
        response = await self._client.post("/risk/position-size", json=payload)
        response.raise_for_status()
        data = response.json() if response.content else {}
        return float(data.get("position_size", data.get("amount", 0.0)))

    async def can_allocate(self, strategy: str, amount: float, balance: float) -> bool:
        payload = {
            "strategy": strategy,
            "amount": amount,
            "balance": balance,
            "config": self._config,
        }
        response = await self._client.post("/risk/allocations/check", json=payload)
        response.raise_for_status()
        data = response.json() if response.content else {}
        return bool(data.get("allowed", data.get("can_allocate", True)))

    async def allocate_capital(self, strategy: str, amount: float) -> None:
        payload = {"strategy": strategy, "amount": amount, "config": self._config}
        response = await self._client.post("/risk/allocations/allocate", json=payload)
        response.raise_for_status()

    async def snapshot(self) -> Mapping[str, Any]:  # pragma: no cover - advisory helper
        response = await self._client.get("/risk/snapshot", params={"include_config": True})
        response.raise_for_status()
        payload = response.json() if response.content else {}
        if isinstance(payload, Mapping):
            return payload
        return {}


class PaperWalletServiceClient:
    """HTTP client for the paper wallet orchestration service."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        config: Optional[Mapping[str, Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        resolved_base = base_url or os.getenv("PAPER_WALLET_SERVICE_URL", "http://paper-wallet:8011/api/v1")
        resolved_timeout = (
            timeout if timeout is not None else float(os.getenv("PAPER_WALLET_SERVICE_TIMEOUT", "10"))
        )
        self._client = client or httpx.AsyncClient(base_url=resolved_base, timeout=resolved_timeout)
        self._owns_client = client is None
        self._config = dict(config or {})
        self.balance: float = 0.0
        self.positions: Mapping[str, Any] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self, method: str, path: str, payload: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        data = dict(payload or {})
        if self._config and "config" not in data:
            data["config"] = self._config
        if method.upper() == "GET":
            response = await self._client.request(method, path, params=data or None)
        else:
            response = await self._client.request(method, path, json=data if data else None)
        response.raise_for_status()
        body = response.json() if response.content else {}
        return body if isinstance(body, Mapping) else {}

    async def refresh_state(self) -> None:
        payload = await self._request("GET", "/wallet/state")
        balance = payload.get("balance")
        positions = payload.get("positions")
        try:
            self.balance = float(balance)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            self.balance = 0.0
        if isinstance(positions, Mapping):
            self.positions = dict(positions)
        else:
            self.positions = {}

    async def buy(self, symbol: str, amount: float, price: float) -> bool:
        payload = {"symbol": symbol, "amount": amount, "price": price, "side": "buy"}
        data = await self._request("POST", "/wallet/buy", payload)
        success = bool(data.get("success", data.get("purchased", False)))
        if success:
            await self.refresh_state()
        return success

    async def sell(self, symbol: str, amount: float, price: float) -> bool:
        payload = {"symbol": symbol, "amount": amount, "price": price, "side": "sell"}
        data = await self._request("POST", "/wallet/sell", payload)
        success = bool(data.get("success", data.get("sold", False)))
        if success:
            await self.refresh_state()
        return success


class PositionGuardServiceClient:
    """HTTP client facade for open position guard orchestration."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        config: Optional[Mapping[str, Any]] = None,
        max_open_trades: Optional[int] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        resolved_base = base_url or os.getenv("POSITION_GUARD_SERVICE_URL", "http://position-guard:8012/api/v1")
        resolved_timeout = (
            timeout if timeout is not None else float(os.getenv("POSITION_GUARD_SERVICE_TIMEOUT", "10"))
        )
        self._client = client or httpx.AsyncClient(base_url=resolved_base, timeout=resolved_timeout)
        self._owns_client = client is None
        self._config = dict(config or {})
        self._max_open_trades = max_open_trades

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self, method: str, path: str, payload: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        data = dict(payload or {})
        if self._config and "config" not in data:
            data["config"] = self._config
        if self._max_open_trades is not None:
            data.setdefault("max_open_trades", self._max_open_trades)
        if method.upper() == "GET":
            response = await self._client.request(method, path, params=data or None)
        else:
            response = await self._client.request(method, path, json=data if data else None)
        response.raise_for_status()
        body = response.json() if response.content else {}
        return body if isinstance(body, Mapping) else {}

    async def can_open(self, positions: Mapping[str, Any]) -> bool:
        data = await self._request("POST", "/guard/can-open", {"positions": dict(positions)})
        return bool(data.get("allowed", data.get("can_open", True)))


def build_service_container() -> ServiceContainer:
    return ServiceContainer(
        market_data=MarketDataAdapter(),
        strategy=StrategyAdapter(),
        portfolio=PortfolioAdapter(),
        execution=ExecutionGatewayClient(),
        token_discovery=TokenDiscoveryAdapter(),
        monitoring=MonitoringAdapter(),
    )


def build_risk_client(config: Mapping[str, Any]) -> Optional[RiskServiceClient]:
    if not config:
        return None
    try:
        return RiskServiceClient(config=config)
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("Risk manager client initialisation failed", exc_info=True)
        return None


def build_paper_wallet_client(config: Mapping[str, Any]) -> Optional[PaperWalletServiceClient]:
    mode = str(config.get("execution_mode", "dry_run")).lower()
    if mode != "dry_run":
        return None
    wallet_cfg = dict(config.get("paper_wallet", {}))
    initial_balance = wallet_cfg.get("initial_balance")
    if initial_balance is None:
        initial_balance = config.get("risk", {}).get("starting_balance", 0.0)
    try:
        wallet_cfg.setdefault("initial_balance", initial_balance)
        wallet_cfg.setdefault("allow_short", bool(wallet_cfg.get("allow_short", True)))
        wallet_cfg.setdefault(
            "max_open_trades",
            int(config.get("max_open_trades") or wallet_cfg.get("max_open_trades", 5)),
        )
        return PaperWalletServiceClient(config=wallet_cfg)
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("Paper wallet client initialisation failed", exc_info=True)
        return None


def build_position_guard_client(config: Mapping[str, Any]) -> Optional[PositionGuardServiceClient]:
    max_trades = config.get("max_open_trades") or config.get("paper_wallet", {}).get(
        "max_open_trades", 5
    )
    try:
        guard_config = config.get("position_guard", {})
        return PositionGuardServiceClient(
            config=guard_config,
            max_open_trades=int(max_trades) if max_trades is not None else None,
        )
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("Position guard client initialisation failed", exc_info=True)
        return None
