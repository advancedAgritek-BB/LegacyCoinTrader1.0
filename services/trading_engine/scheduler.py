from __future__ import annotations

"""Async scheduler for orchestrating trading cycles."""

import asyncio
import contextlib
import inspect
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional, cast

from services.common.tenant import TenantContext, TenantLimitError
from services.interface_layer.cycle import CycleExecutionResult

from .interface import TradingEngineInterface
from .redis_state import CycleState, RedisCycleStateStore

logger = logging.getLogger(__name__)


class _TenantSchedule:
    __slots__ = ("interface", "state_store", "interval", "task", "running", "lock")

    def __init__(
        self,
        interface: TradingEngineInterface,
        state_store: RedisCycleStateStore,
        interval: int,
    ) -> None:
        self.interface = interface
        self.state_store = state_store
        self.interval = max(interval, 1)
        self.task: Optional[asyncio.Task[Any]] = None
        self.running = False
        self.lock = asyncio.Lock()


class CycleScheduler:
    """Manage periodic execution of trading cycles on a per-tenant basis."""

    def __init__(
        self,
        interface_factory: Callable[
            [TenantContext], TradingEngineInterface | Awaitable[TradingEngineInterface]
        ],
        state_store: RedisCycleStateStore,
        default_interval: int,
    ) -> None:
        self._interface_factory = interface_factory
        self._state_store = state_store
        self._default_interval = max(default_interval, 1)
        self._tenants: Dict[str, _TenantSchedule] = {}

    async def _build_interface(self, tenant: TenantContext) -> TradingEngineInterface:
        candidate = self._interface_factory(tenant)
        if inspect.isawaitable(candidate):
            candidate = await candidate  # type: ignore[assignment]
        return cast(TradingEngineInterface, candidate)

    async def _ensure_schedule(self, tenant: TenantContext) -> _TenantSchedule:
        schedule = self._tenants.get(tenant.tenant_id)
        if schedule is None:
            interface = await self._build_interface(tenant)
            store = self._state_store.for_tenant(tenant.tenant_id)
            await store.ensure_defaults(self._default_interval)
            schedule = _TenantSchedule(interface, store, self._default_interval)
            self._tenants[tenant.tenant_id] = schedule
        return schedule

    async def start(
        self,
        tenant: TenantContext,
        *,
        interval_seconds: Optional[int] = None,
        immediate: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        risk_allocation: Optional[float] = None,
    ) -> CycleState:
        schedule = await self._ensure_schedule(tenant)
        async with schedule.lock:
            active_cycles = 1 if schedule.running else 0
            tenant.risk_policy.validate_cycle_start(active_cycles)
            schedule.interval = max(
                int(interval_seconds or schedule.interval or self._default_interval), 1
            )
            state = await schedule.state_store.load_state()
            state.running = True
            state.interval_seconds = schedule.interval
            state.metadata = tenant.enrich_metadata(metadata)
            allocation_value = tenant.risk_policy.validate_allocation(risk_allocation)
            state.metadata["risk_allocation"] = allocation_value
            state.tenant_id = tenant.tenant_id
            now = datetime.now(timezone.utc)
            state.next_run_at = now if immediate else state.predict_next_run() or now
            await schedule.state_store.save_state(state)
            schedule.running = True
            if schedule.task and not schedule.task.done():
                schedule.task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await schedule.task
            schedule.task = asyncio.create_task(self._run_loop(tenant, schedule))
        if immediate:
            await self._execute_cycle(schedule, tenant, dict(state.metadata))
        logger.info(
            "Trading cycle scheduler started for tenant %s with interval %s",
            tenant.tenant_id,
            schedule.interval,
        )
        return await schedule.state_store.load_state()

    async def stop(self, tenant: TenantContext) -> CycleState:
        schedule = await self._ensure_schedule(tenant)
        async with schedule.lock:
            schedule.running = False
            if schedule.task:
                schedule.task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await schedule.task
                schedule.task = None
            state = await schedule.state_store.load_state()
            state.running = False
            state.next_run_at = None
            state.metadata = tenant.enrich_metadata({}, base=state.metadata)
            state.metadata["risk_allocation"] = 0.0
            state.tenant_id = tenant.tenant_id
            await schedule.state_store.save_state(state)
        logger.info("Trading cycle scheduler stopped for tenant %s", tenant.tenant_id)
        return await schedule.state_store.load_state()

    async def run_once(
        self,
        tenant: TenantContext,
        metadata: Optional[Dict[str, Any]] = None,
        risk_allocation: Optional[float] = None,
    ) -> CycleExecutionResult:
        schedule = await self._ensure_schedule(tenant)
        async with schedule.lock:
            state = await schedule.state_store.load_state()
            current_allocation = float(state.metadata.get("risk_allocation", 0.0) or 0.0)
            additional = float(risk_allocation or 0.0)
            total_allocation = tenant.risk_policy.validate_additional_allocation(
                current_allocation, additional
            )
            merged_metadata = tenant.enrich_metadata(metadata, base=state.metadata)
            merged_metadata["risk_allocation"] = total_allocation
            state.metadata = merged_metadata
            state.tenant_id = tenant.tenant_id
            await schedule.state_store.save_state(state)
            metadata_payload = dict(state.metadata)
        return await self._execute_cycle(schedule, tenant, metadata_payload)

    async def get_state(self, tenant: TenantContext) -> CycleState:
        schedule = self._tenants.get(tenant.tenant_id)
        if schedule is not None:
            return await schedule.state_store.load_state()
        store = self._state_store.for_tenant(tenant.tenant_id)
        return await store.load_state()

    async def shutdown(self) -> None:
        schedules = list(self._tenants.items())
        self._tenants.clear()
        for tenant_id, schedule in schedules:
            schedule.running = False
            if schedule.task:
                schedule.task.cancel()
            if schedule.task:
                with contextlib.suppress(asyncio.CancelledError):
                    await schedule.task
            try:
                await schedule.interface.shutdown()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("Failed to shut down interface for tenant %s", tenant_id, exc_info=True)

    async def _execute_cycle(
        self,
        schedule: _TenantSchedule,
        tenant: TenantContext,
        metadata: Dict[str, Any],
    ) -> CycleExecutionResult:
        started_at = datetime.now(timezone.utc)
        await schedule.state_store.mark_cycle_start(started_at)
        try:
            result = await schedule.interface.run_cycle(metadata=metadata)
        except TenantLimitError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Trading cycle execution failed for tenant %s", tenant.tenant_id)
            await schedule.state_store.mark_cycle_failure(
                str(exc), datetime.now(timezone.utc), metadata=metadata
            )
            raise
        completed_at = result.completed_at or datetime.now(timezone.utc)
        state = await schedule.state_store.mark_cycle_complete(
            result, completed_at, metadata=metadata
        )
        if schedule.running:
            await schedule.state_store.update(next_run_at=state.predict_next_run())
        return result

    async def _run_loop(self, tenant: TenantContext, schedule: _TenantSchedule) -> None:
        try:
            while schedule.running:
                await asyncio.sleep(schedule.interval)
                if not schedule.running:
                    break
                state = await schedule.state_store.load_state()
                metadata = tenant.enrich_metadata({}, base=state.metadata)
                await self._execute_cycle(schedule, tenant, metadata)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - background loop safety
            logger.exception(
                "Trading cycle loop crashed for tenant %s: %s", tenant.tenant_id, exc
            )
        finally:
            schedule.task = None


__all__ = ["CycleScheduler"]
