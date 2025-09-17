from __future__ import annotations

"""Async scheduler for orchestrating trading cycles."""

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from services.interface_layer.cycle import CycleExecutionResult

from .interface import LiquidationReport, TradingEngineInterface
from .redis_state import CycleState, RedisCycleStateStore

logger = logging.getLogger(__name__)


class CycleScheduler:
    """Manage periodic execution of trading cycles."""

    def __init__(
        self,
        interface: TradingEngineInterface,
        state_store: RedisCycleStateStore,
        default_interval: int,
    ) -> None:
        self._interface = interface
        self._state_store = state_store
        self._interval = max(default_interval, 1)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

    async def start(
        self,
        interval_seconds: Optional[int] = None,
        immediate: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CycleState:
        async with self._lock:
            interval = int(interval_seconds or self._interval)
            self._interval = max(interval, 1)

            state = await self._state_store.load_state()
            state.running = True
            state.interval_seconds = self._interval
            state.metadata = dict(metadata or {})
            now = datetime.now(timezone.utc)
            state.next_run_at = now if immediate else state.predict_next_run() or now
            await self._state_store.save_state(state)

            self._running = True
            if self._task and not self._task.done():
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            self._task = asyncio.create_task(self._run_loop(immediate))
            logger.info("Trading cycle scheduler started with interval %s", self._interval)
            return state

    async def stop(self) -> CycleState:
        async with self._lock:
            self._running = False
            if self._task:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
                self._task = None
            state = await self._state_store.update(running=False, next_run_at=None)
            logger.info("Trading cycle scheduler stopped")
            return state

    async def run_once(self, metadata: Optional[Dict[str, Any]] = None) -> CycleExecutionResult:
        metadata = dict(metadata or {})
        started_at = datetime.now(timezone.utc)
        await self._state_store.mark_cycle_start(started_at)
        try:
            result = await self._interface.run_cycle(metadata=metadata)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Trading cycle execution failed")
            await self._state_store.mark_cycle_failure(
                str(exc), datetime.now(timezone.utc), metadata=metadata
            )
            raise
        completed_at = result.completed_at or datetime.now(timezone.utc)
        state = await self._state_store.mark_cycle_complete(
            result, completed_at, metadata=metadata
        )
        if state.running:
            await self._state_store.update(next_run_at=state.predict_next_run())
        return result

    async def get_state(self) -> CycleState:
        return await self._state_store.load_state()

    async def liquidate_positions(self) -> LiquidationReport:
        """Delegate liquidation to the trading engine interface."""

        return await self._interface.liquidate_positions()

    async def shutdown(self) -> None:
        try:
            await self.stop()
        finally:
            self._task = None

    async def _run_loop(self, immediate: bool) -> None:
        try:
            if immediate:
                await self.run_once()
            while self._running:
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                await self.run_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - background loop safety
            logger.exception("Trading cycle loop crashed: %s", exc)
        finally:
            self._task = None


__all__ = ["CycleScheduler"]
