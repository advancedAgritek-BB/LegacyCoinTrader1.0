from __future__ import annotations

"""Pydantic models exposed by the trading engine API."""

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from .redis_state import CycleState
from .liquidation import LiquidationReport


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


class PositionLiquidationResponse(BaseModel):
    symbol: str
    side: str
    amount: float
    status: str
    error: Optional[str] = None


class LiquidationResponse(BaseModel):
    status: Literal["completed", "partial", "failed"]
    total_positions: int
    closed_positions: int
    failed_positions: int
    positions: list[PositionLiquidationResponse] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @classmethod
    def from_report(cls, report: LiquidationReport) -> "LiquidationResponse":
        return cls(
            status=report.status,
            total_positions=report.total_positions,
            closed_positions=report.closed_positions,
            failed_positions=report.failed_positions,
            positions=[
                PositionLiquidationResponse(
                    symbol=position.symbol,
                    side=position.side,
                    amount=position.amount,
                    status=position.status,
                    error=position.error,
                )
                for position in report.positions
            ],
            errors=list(report.errors),
        )


__all__.extend(["LiquidationResponse", "PositionLiquidationResponse"])
