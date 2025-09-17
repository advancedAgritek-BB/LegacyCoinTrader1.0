from __future__ import annotations

"""Pydantic models exposed by the trading engine API."""

from services.trading_engine.contracts import (
    TradingCycleCommand,
    TradingCycleRunResponse,
    TradingCycleState,
)

from .redis_state import CycleState


class StartCycleRequest(TradingCycleCommand):
    """Re-export of the trading engine command contract for FastAPI compatibility."""


class RunCycleResponse(TradingCycleRunResponse):
    """Alias of the shared contract used in the HTTP handlers."""


class CycleStateResponse(TradingCycleState):
    """Response object exposing the scheduler state as JSON."""

    @classmethod
    def from_state(cls, state: CycleState) -> "CycleStateResponse":
        return cls(
            running=state.running,
            interval_seconds=state.interval_seconds,
            next_run_at=state.next_run_at,
            last_run_started_at=state.last_run_started_at,
            last_run_completed_at=state.last_run_completed_at,
            last_timings=state.last_timings,
            last_error=state.last_error,
            metadata=state.metadata,
        )


__all__ = ["StartCycleRequest", "RunCycleResponse", "CycleStateResponse"]
