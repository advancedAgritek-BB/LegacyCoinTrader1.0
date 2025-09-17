from __future__ import annotations

"""Adaptor that bridges the trading engine service with the shared interface."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional

from services.interface_layer.cycle import CycleExecutionResult, TradingCycleInterface

from .phases import DEFAULT_PHASES
from .liquidation import LiquidationHelper, LiquidationReport

logger = logging.getLogger(__name__)


@dataclass
class CycleContext:
    """Minimal context passed to cycle phases."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    memory_manager: Optional[Any] = None

    def perform_memory_maintenance(self) -> Dict[str, Any]:  # pragma: no cover - simple stub
        return {}


class TradingEngineInterface:
    """High-level orchestration entry point for trading cycles."""

    def __init__(
        self,
        phases: Optional[Iterable] = None,
        *,
        clock: Optional[Callable[[], datetime]] = None,
        timer: Optional[Callable[[], float]] = None,
        liquidation_helper: Optional[LiquidationHelper] = None,
    ) -> None:
        self._phases = list(phases or DEFAULT_PHASES)
        self._runner = TradingCycleInterface(self._phases, clock=clock, timer=timer)
        self._liquidation_helper = liquidation_helper or LiquidationHelper()

    def set_phases(self, phases: Iterable) -> None:
        self._phases = list(phases)
        self._runner.set_phases(self._phases)

    async def run_cycle(self, metadata: Optional[Dict[str, Any]] = None) -> CycleExecutionResult:
        """Execute a single trading cycle and return timing information."""

        context = CycleContext(metadata=dict(metadata or {}))
        result = await self._runner.run_cycle(context)
        context.timing = result.timings
        logger.debug("Trading cycle completed with timings: %s", result.timings)
        return result

    async def liquidate_positions(self) -> LiquidationReport:
        """Trigger emergency liquidation via the configured helper."""

        return await self._liquidation_helper.close_all_positions()


__all__ = ["TradingEngineInterface", "CycleContext", "LiquidationReport"]
