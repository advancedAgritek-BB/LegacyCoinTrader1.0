from __future__ import annotations

"""Pydantic models exposed by the trading engine API."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .redis_state import CycleState


class StartCycleRequest(BaseModel):
    interval_seconds: Optional[int] = Field(default=None, ge=1)
    immediate: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunCycleResponse(BaseModel):
    status: str
    timings: Dict[str, float]
    started_at: datetime
    completed_at: datetime
    duration: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CycleStateResponse(BaseModel):
    running: bool
    interval_seconds: int
    next_run_at: Optional[datetime]
    last_run_started_at: Optional[datetime]
    last_run_completed_at: Optional[datetime]
    last_timings: Dict[str, float] = Field(default_factory=dict)
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
