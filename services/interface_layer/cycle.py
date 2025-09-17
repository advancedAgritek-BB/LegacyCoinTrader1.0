from __future__ import annotations

"""Shared trading cycle orchestration primitives.

This module provides a small interface layer that can be consumed by any
service that needs to orchestrate trading cycles.  It encapsulates the core
loop that was previously implemented directly in ``crypto_bot.phase_runner``
and exposes a clean abstraction that returns structured results.
"""

import inspect
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


PhaseCallable = Callable[[Any], Awaitable[None]]


@dataclass
class CycleExecutionResult:
    """Represents the outcome of a trading cycle execution."""

    timings: Dict[str, float] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Return the total execution time for the cycle in seconds."""

        if not self.completed_at or not self.started_at:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()


class TradingCycleInterface:
    """Execute a sequence of asynchronous phases and capture timings."""

    def __init__(self, phases: Optional[Iterable[PhaseCallable]] = None) -> None:
        self._phases: List[PhaseCallable] = list(phases or [])

    @property
    def phases(self) -> List[PhaseCallable]:
        """Return the configured phases."""

        return list(self._phases)

    def set_phases(self, phases: Iterable[PhaseCallable]) -> None:
        """Replace the configured phases with ``phases``."""

        self._phases = list(phases)

    async def run_cycle(
        self,
        ctx: Any,
        phases: Optional[Iterable[PhaseCallable]] = None,
    ) -> CycleExecutionResult:
        """Execute ``phases`` using ``ctx`` and return execution timings."""

        phase_sequence = list(phases or self._phases)
        if not phase_sequence:
            raise ValueError("No phases configured for trading cycle execution")

        result = CycleExecutionResult()
        result.started_at = datetime.now(timezone.utc)
        overall_start = time.perf_counter()

        for phase in phase_sequence:
            phase_name = getattr(phase, "__name__", "unknown_phase")
            start_time = time.perf_counter()

            memory_manager = getattr(ctx, "memory_manager", None)
            monitor = getattr(memory_manager, "memory_monitoring", None)

            try:
                if monitor:
                    with monitor(f"phase_{phase_name}"):
                        await phase(ctx)
                else:
                    await phase(ctx)
            except Exception:
                logger.exception("Trading cycle phase %s failed", phase_name)
                raise
            finally:
                result.timings[phase_name] = time.perf_counter() - start_time

            maintenance_result: Optional[Dict[str, Any]] = None
            if hasattr(ctx, "perform_memory_maintenance"):
                try:
                    maintenance = ctx.perform_memory_maintenance()
                    if inspect.isawaitable(maintenance):
                        maintenance = await maintenance  # type: ignore[assignment]
                    if isinstance(maintenance, dict):
                        maintenance_result = maintenance
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug("Memory maintenance failed after %s", phase_name, exc_info=True)

            if maintenance_result and maintenance_result.get("memory_pressure"):
                logger.warning("Memory pressure detected during %s", phase_name)

        result.completed_at = datetime.now(timezone.utc)
        result.metadata["duration_seconds"] = time.perf_counter() - overall_start
        return result


__all__ = ["TradingCycleInterface", "CycleExecutionResult", "PhaseCallable"]
