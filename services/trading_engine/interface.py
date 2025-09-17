from __future__ import annotations

"""Adaptor that bridges the trading engine service with the shared interface."""

import asyncio
import copy
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from crypto_bot.phase_runner import BotContext

from services.interface_layer.cycle import CycleExecutionResult, TradingCycleInterface

from .phases import DEFAULT_PHASES

logger = logging.getLogger(__name__)


class CycleContext:
    """Adapter that exposes :class:`BotContext` through the service interface."""

    __slots__ = {"_ctx", "metadata", "timing", "state"}

    def __init__(
        self,
        bot_context: BotContext,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        object.__setattr__(self, "_ctx", bot_context)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "timing", {})
        object.__setattr__(self, "state", state if state is not None else {})

    def __getattr__(self, item: str) -> Any:
        return getattr(self._ctx, item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self.__slots__:
            object.__setattr__(self, key, value)
        else:
            setattr(self._ctx, key, value)

    def __delattr__(self, item: str) -> None:  # pragma: no cover - defensive
        if item in self.__slots__:
            raise AttributeError(f"Cannot delete attribute '{item}'")
        delattr(self._ctx, item)

    def perform_memory_maintenance(self) -> Dict[str, Any]:
        return self._ctx.perform_memory_maintenance()


class TradingContextBuilder:
    """Factory that manages a long-lived :class:`BotContext`."""

    def __init__(
        self,
        *,
        services: Any,
        config: Mapping[str, Any],
        exchange: Any = None,
        ws_client: Any = None,
        risk_manager: Any = None,
        notifier: Any = None,
        paper_wallet: Any = None,
        position_guard: Any = None,
        trade_manager: Any = None,
    ) -> None:
        self._services = services
        self._config = copy.deepcopy(dict(config))
        self._exchange = exchange
        self._ws_client = ws_client
        self._risk_manager = risk_manager
        self._notifier = notifier
        self._paper_wallet = paper_wallet
        self._position_guard = position_guard
        self._trade_manager = trade_manager
        self._bot_context: Optional[BotContext] = None
        self._state: Dict[str, Any] = {"cycles": 0}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def __call__(self, metadata: Optional[Mapping[str, Any]] = None) -> CycleContext:
        ctx = self._ensure_context()
        self._state["cycles"] = int(self._state.get("cycles", 0)) + 1
        cycle_meta = dict(metadata or {})
        cycle_meta.setdefault("cycle_id", self._state["cycles"])
        cycle_meta.setdefault("config_version", self._config.get("config_version"))
        return CycleContext(ctx, metadata=cycle_meta, state=self._state)

    @property
    def context(self) -> BotContext:
        return self._ensure_context()

    @property
    def services(self) -> Any:
        return self._services

    @property
    def exchange(self) -> Any:
        return self._exchange

    @property
    def ws_client(self) -> Any:
        return self._ws_client

    async def shutdown(self) -> None:
        """Close any resources owned by the builder."""

        async def _call(func: Callable[[], Any], *, use_thread: bool = False) -> None:
            try:
                if use_thread:
                    await asyncio.to_thread(func)
                    return
                result = func()
                if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                    await result  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("Resource cleanup failed", exc_info=True)

        if self._services:
            market_data = getattr(self._services, "market_data", None)
            if market_data and hasattr(market_data, "close"):
                await _call(market_data.close)
            strategy = getattr(self._services, "strategy", None)
            if strategy and hasattr(strategy, "aclose"):
                await _call(strategy.aclose)

        if self._ws_client:
            if hasattr(self._ws_client, "close_async"):
                await _call(self._ws_client.close_async)
            elif hasattr(self._ws_client, "close"):
                close_func = self._ws_client.close
                await _call(close_func, use_thread=not asyncio.iscoroutinefunction(close_func))

        if self._exchange and hasattr(self._exchange, "close"):
            close_func = self._exchange.close
            await _call(close_func, use_thread=not asyncio.iscoroutinefunction(close_func))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_context(self) -> BotContext:
        if self._bot_context is not None:
            return self._bot_context

        positions: Dict[str, Any] = {}
        if self._trade_manager is not None:
            try:
                for pos in self._trade_manager.get_all_positions():  # type: ignore[attr-defined]
                    symbol = getattr(pos, "symbol", None) or getattr(pos, "id", None)
                    if not symbol:
                        continue
                    entry_price = getattr(pos, "entry_price", None) or getattr(pos, "price", 0.0)
                    amount = getattr(pos, "total_amount", None) or getattr(pos, "amount", 0.0)
                    positions[symbol] = {
                        "symbol": symbol,
                        "side": getattr(pos, "side", "long"),
                        "entry_price": float(entry_price or 0.0),
                        "amount": float(amount or 0.0),
                    }
            except Exception:  # pragma: no cover - defensive
                logger.debug("Unable to prime positions from trade manager", exc_info=True)

        config = copy.deepcopy(dict(self._config))
        ctx = BotContext(
            positions=positions,
            df_cache=defaultdict(dict),
            regime_cache=defaultdict(dict),
            config=config,
            exchange=self._exchange,
            ws_client=self._ws_client,
            risk_manager=self._risk_manager,
            notifier=self._notifier,
            paper_wallet=self._paper_wallet,
            position_guard=self._position_guard,
            trade_manager=self._trade_manager,
            services=self._services,
        )

        starting_balance = (
            float(
                config.get("risk", {}).get("starting_balance")
                or config.get("paper_wallet", {}).get("initial_balance")
                or 0.0
            )
        )
        ctx.balance = starting_balance
        ctx.use_trade_manager_as_source = bool(config.get("use_trade_manager_as_source"))
        self._bot_context = ctx
        return ctx


class TradingEngineInterface:
    """High-level orchestration entry point for trading cycles."""

    def __init__(
        self,
        phases: Optional[Iterable] = None,
        *,
        clock: Optional[Callable[[], datetime]] = None,
        timer: Optional[Callable[[], float]] = None,
        services: Any = None,
        config: Optional[Mapping[str, Any]] = None,
        exchange: Any = None,
        ws_client: Any = None,
        risk_manager: Any = None,
        notifier: Any = None,
        paper_wallet: Any = None,
        position_guard: Any = None,
        trade_manager: Any = None,
    ) -> None:
        self._phases = list(phases or DEFAULT_PHASES)
        self._runner = TradingCycleInterface(self._phases, clock=clock, timer=timer)
        self._context_factory = TradingContextBuilder(
            services=services,
            config=config or {},
            exchange=exchange,
            ws_client=ws_client,
            risk_manager=risk_manager,
            notifier=notifier,
            paper_wallet=paper_wallet,
            position_guard=position_guard,
            trade_manager=trade_manager,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def context(self) -> BotContext:
        return self._context_factory.context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_phases(self, phases: Iterable) -> None:
        self._phases = list(phases)
        self._runner.set_phases(self._phases)

    async def run_cycle(self, metadata: Optional[Dict[str, Any]] = None) -> CycleExecutionResult:
        """Execute a single trading cycle and return timing information."""

        context = self._context_factory(metadata)
        result = await self._runner.run_cycle(context)
        context.timing = result.timings
        result.metadata.update(context.metadata)
        logger.debug("Trading cycle completed with timings: %s", result.timings)
        return result

    async def shutdown(self) -> None:
        await self._context_factory.shutdown()


__all__ = ["TradingEngineInterface", "CycleContext", "TradingContextBuilder"]
