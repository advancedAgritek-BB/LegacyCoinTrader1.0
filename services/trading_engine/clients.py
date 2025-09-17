"""Asynchronous service clients for the trading engine."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import httpx
import pandas as pd

from libs.models import OpenPositionGuard, PaperWallet
from libs.risk import RiskManager
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
from crypto_bot.services.adapters.strategy import StrategyAdapter
from crypto_bot.services.adapters.token_discovery import TokenDiscoveryAdapter
from crypto_bot.services.adapters.portfolio import PortfolioAdapter

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # ExecutionService implementation
    # ------------------------------------------------------------------
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


class AsyncRiskServiceClient:
    """Asynchronous wrapper around the core risk manager implementation."""

    def __init__(self, manager: RiskManager) -> None:
        self._manager = manager

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AsyncRiskServiceClient":
        manager = RiskManager.from_config(config)
        return cls(manager)

    async def allow_trade(self, df: pd.DataFrame, strategy: Optional[str]) -> Tuple[bool, str]:
        return await asyncio.to_thread(self._manager.allow_trade, df, strategy)

    async def position_size(
        self,
        confidence: float,
        balance: float,
        *,
        df: Optional[pd.DataFrame] = None,
        atr: Optional[float] = None,
        price: Optional[float] = None,
    ) -> float:
        return float(
            await asyncio.to_thread(
                self._manager.position_size,
                confidence,
                balance,
                df,
                None,
                atr,
                price,
            )
        )

    async def can_allocate(self, strategy: str, amount: float, balance: float) -> bool:
        if not hasattr(self._manager, "can_allocate"):
            return True
        return bool(
            await asyncio.to_thread(self._manager.can_allocate, strategy, amount, balance)
        )

    async def allocate_capital(self, strategy: str, amount: float) -> None:
        if not hasattr(self._manager, "allocate_capital"):
            return
        await asyncio.to_thread(self._manager.allocate_capital, strategy, amount)

    async def snapshot(self) -> Mapping[str, Any]:
        state = getattr(self._manager, "config", None)
        if state is None:
            return {}
        return dict(state.__dict__)


class AsyncPaperWalletClient:
    """Minimal asynchronous wrapper around :class:`PaperWallet`."""

    def __init__(self, wallet: PaperWallet) -> None:
        self._wallet = wallet

    @property
    def balance(self) -> float:
        return float(self._wallet.balance)

    @property
    def positions(self) -> Mapping[str, Any]:
        return self._wallet.positions

    async def buy(self, symbol: str, amount: float, price: float) -> bool:
        return bool(await asyncio.to_thread(self._wallet.buy, symbol, amount, price))

    async def sell(self, symbol: str, amount: float, price: float) -> bool:
        return bool(await asyncio.to_thread(self._wallet.sell, symbol, amount, price))


class AsyncPositionGuardClient:
    """Asynchronous facade for :class:`OpenPositionGuard`."""

    def __init__(self, guard: OpenPositionGuard) -> None:
        self._guard = guard

    async def can_open(self, positions: Mapping[str, Any]) -> bool:
        return bool(await asyncio.to_thread(self._guard.can_open, positions))


def build_service_container() -> ServiceContainer:
    """Instantiate service clients used by the trading engine."""

    return ServiceContainer(
        market_data=MarketDataAdapter(),
        strategy=StrategyAdapter(),
        portfolio=PortfolioAdapter(),
        execution=ExecutionGatewayClient(),
        token_discovery=TokenDiscoveryAdapter(),
        monitoring=MonitoringAdapter(),
    )


def build_risk_client(config: Mapping[str, Any]) -> Optional[AsyncRiskServiceClient]:
    if not config:
        return None
    try:
        return AsyncRiskServiceClient.from_config(config)
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("Risk manager client initialisation failed", exc_info=True)
        return None


def build_paper_wallet_client(config: Mapping[str, Any]) -> Optional[AsyncPaperWalletClient]:
    mode = str(config.get("execution_mode", "dry_run")).lower()
    if mode != "dry_run":
        return None
    wallet_cfg = config.get("paper_wallet", {})
    initial_balance = wallet_cfg.get("initial_balance")
    if initial_balance is None:
        initial_balance = config.get("risk", {}).get("starting_balance", 0.0)
    try:
        wallet = PaperWallet(
            float(initial_balance or 0.0),
            max_open_trades=int(wallet_cfg.get("max_open_trades", 5)),
            allow_short=bool(wallet_cfg.get("allow_short", True)),
        )
        return AsyncPaperWalletClient(wallet)
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("Paper wallet client initialisation failed", exc_info=True)
        return None


def build_position_guard_client(config: Mapping[str, Any]) -> Optional[AsyncPositionGuardClient]:
    max_trades = config.get("max_open_trades") or config.get("paper_wallet", {}).get(
        "max_open_trades", 5
    )
    try:
        guard = OpenPositionGuard(int(max_trades))
        return AsyncPositionGuardClient(guard)
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("Position guard client initialisation failed", exc_info=True)
        return None
