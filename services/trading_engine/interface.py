"""Adaptor that bridges the trading engine service with the shared interface."""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

from crypto_bot.open_position_guard import OpenPositionGuard
from crypto_bot.phase_runner import BotContext
from crypto_bot.risk.risk_manager import RiskManager
from crypto_bot.services.interfaces import ServiceContainer
from services.interface_layer.cycle import CycleExecutionResult, TradingCycleInterface

from .phases import PRODUCTION_PHASES

logger = logging.getLogger(__name__)


class CycleContext:
    """Runtime context passed to trading phases."""

    __slots__ = ("bot", "services", "redis", "solana_feed", "metadata", "timing")

    _INTERNAL_FIELDS = {"bot", "services", "redis", "solana_feed", "metadata", "timing"}

    def __init__(
        self,
        *,
        bot: BotContext,
        services: ServiceContainer,
        redis: Optional[Any],
        solana_feed: Optional[Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        object.__setattr__(self, "bot", bot)
        object.__setattr__(self, "services", services)
        object.__setattr__(self, "redis", redis)
        object.__setattr__(self, "solana_feed", solana_feed)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "timing", {})

        # Ensure BotContext sees the same service container and optional feeds
        bot.services = services
        bot.solana_feed = solana_feed

    def __getattr__(self, item: str) -> Any:
        return getattr(self.bot, item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self._INTERNAL_FIELDS:
            object.__setattr__(self, key, value)
        else:
            setattr(self.bot, key, value)

    def perform_memory_maintenance(self) -> Dict[str, Any]:  # pragma: no cover - passthrough
        if hasattr(self.bot, "perform_memory_maintenance"):
            return self.bot.perform_memory_maintenance()
        return {}


class TradingEngineInterface:
    """High-level orchestration entry point for trading cycles."""

    def __init__(
        self,
        service_container: ServiceContainer,
        *,
        phases: Optional[Iterable] = None,
        clock: Optional[Callable[[], datetime]] = None,
        timer: Optional[Callable[[], float]] = None,
        redis_client: Optional[Any] = None,
        solana_feed: Optional[Any] = None,
        base_config: Optional[Mapping[str, Any]] = None,
        config_loader: Optional[Callable[[], Mapping[str, Any]]] = None,
    ) -> None:
        self._services = service_container
        self._redis = redis_client
        self._solana_feed = solana_feed
        self._config_loader = config_loader
        self._base_config: Dict[str, Any] = copy.deepcopy(base_config or {})

        self._phases = list(phases or PRODUCTION_PHASES)
        self._runner = TradingCycleInterface(self._phases, clock=clock, timer=timer)

    def set_phases(self, phases: Iterable) -> None:
        self._phases = list(phases)
        self._runner.set_phases(self._phases)

    async def run_cycle(self, metadata: Optional[Dict[str, Any]] = None) -> CycleExecutionResult:
        """Execute a single trading cycle and return timing information."""

        incoming_meta = dict(metadata or {})
        context = await self._build_context(incoming_meta)
        context.metadata.update(incoming_meta)

        result = await self._runner.run_cycle(context)
        context.timing = result.timings

        merged_metadata = dict(incoming_meta)
        merged_metadata.update(context.metadata)
        result.metadata.update(merged_metadata)

        logger.debug("Trading cycle completed with timings: %s", result.timings)
        return result

    async def _build_context(self, metadata: Mapping[str, Any]) -> CycleContext:
        config = self._load_config()
        positions = self._load_positions()
        trade_manager = self._load_trade_manager()
        position_guard = self._build_position_guard(config, positions)
        risk_manager = self._build_risk_manager(config)
        balance = self._resolve_initial_balance(config)

        df_cache: MutableMapping[str, MutableMapping[str, Any]] = defaultdict(dict)
        regime_cache: MutableMapping[str, MutableMapping[str, Any]] = defaultdict(dict)

        bot = BotContext(
            positions=positions,
            df_cache=df_cache,
            regime_cache=regime_cache,
            config=config,
            exchange=None,
            ws_client=None,
            risk_manager=risk_manager,
            notifier=None,
            paper_wallet=None,
            position_guard=position_guard,
            trade_manager=trade_manager,
            services=self._services,
        )
        bot.balance = balance

        return CycleContext(
            bot=bot,
            services=self._services,
            redis=self._redis,
            solana_feed=self._solana_feed,
            metadata=metadata,
        )

    def _load_config(self) -> Dict[str, Any]:
        if self._config_loader is not None:
            try:
                loaded = self._config_loader()
                self._base_config = copy.deepcopy(dict(loaded))
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to load trading configuration, using previous snapshot")
        return copy.deepcopy(self._base_config)

    def _load_positions(self) -> Dict[str, Any]:
        portfolio = getattr(self._services, "portfolio", None)
        if portfolio is None:
            return {}
        try:
            items = portfolio.list_positions()
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Unable to load positions from portfolio service", exc_info=True)
            return {}
        positions: Dict[str, Any] = {}
        for entry in items:
            symbol = getattr(entry, "symbol", None)
            if symbol:
                positions[str(symbol)] = entry
        return positions

    def _load_trade_manager(self) -> Any:
        portfolio = getattr(self._services, "portfolio", None)
        if portfolio is None:
            return None
        try:
            return portfolio.get_trade_manager()
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Trade manager unavailable", exc_info=True)
            return None

    def _build_position_guard(
        self,
        config: Mapping[str, Any],
        positions: Mapping[str, Any],
    ) -> OpenPositionGuard:
        max_open = config.get("max_open_trades")
        risk_cfg = config.get("risk", {}) or {}
        if max_open is None and isinstance(risk_cfg, Mapping):
            max_open = risk_cfg.get("max_positions")
        try:
            value = int(max_open) if max_open is not None else max(1, len(positions) + 5)
        except (TypeError, ValueError):  # pragma: no cover - defensive logging
            value = max(1, len(positions) + 5)
        return OpenPositionGuard(value)

    def _build_risk_manager(self, config: Mapping[str, Any]) -> RiskManager:
        base: Dict[str, Any] = {
            "max_drawdown": 1.0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
        }
        raw = config.get("risk", {}) or {}
        if isinstance(raw, Mapping):
            base.update(raw)
        return RiskManager.from_config(base)

    def _resolve_initial_balance(self, config: Mapping[str, Any]) -> float:
        risk_cfg = config.get("risk", {}) or {}
        candidates = [
            risk_cfg.get("starting_balance"),
            risk_cfg.get("initial_balance"),
            config.get("starting_balance"),
            config.get("initial_balance"),
        ]
        for value in candidates:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive logging
                continue
        return 0.0


__all__ = ["TradingEngineInterface", "CycleContext"]
