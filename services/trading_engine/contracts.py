from __future__ import annotations

"""Contracts describing the Trading Engine service interfaces."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from services.common.contracts import EventEnvelope, GrpcMethodDescriptor, HttpEndpoint


class TradingCycleCommand(BaseModel):
    """Command payload used by REST and gRPC endpoints to control the cycle runner."""

    interval_seconds: Optional[int] = Field(default=None, ge=1)
    immediate: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TradingCycleRunResponse(BaseModel):
    status: str
    timings: Dict[str, float] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime
    duration: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TradingCycleState(BaseModel):
    running: bool
    interval_seconds: int
    next_run_at: Optional[datetime]
    last_run_started_at: Optional[datetime]
    last_run_completed_at: Optional[datetime]
    last_timings: Dict[str, float] = Field(default_factory=dict)
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TradingCycleEventPayload(BaseModel):
    status: str
    interval_seconds: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timings: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TradingCycleEvent(EventEnvelope):
    """Event describing the outcome of a trading cycle action."""

    event_type: str = Field(default="trading-engine.cycle", const=True)
    payload: TradingCycleEventPayload


class EmptyMessage(BaseModel):
    """Placeholder type used for gRPC unary calls without payloads."""


@runtime_checkable
class TradingEngineService(Protocol):
    """gRPC contract for the trading engine service."""

    async def StartCycle(self, request: TradingCycleCommand) -> TradingCycleState:  # noqa: N802
        ...

    async def StopCycle(self, request: EmptyMessage) -> TradingCycleState:  # noqa: N802
        ...

    async def RunCycle(self, request: TradingCycleCommand) -> TradingCycleRunResponse:  # noqa: N802
        ...

    async def GetCycleState(self, request: EmptyMessage) -> TradingCycleState:  # noqa: N802
        ...


HTTP_CONTRACT: List[HttpEndpoint] = [
    HttpEndpoint(
        method="POST",
        path="/cycles/start",
        summary="Start the orchestrated trading cycle scheduler",
        request_model="services.trading_engine.contracts.TradingCycleCommand",
        response_model="services.trading_engine.contracts.TradingCycleState",
    ),
    HttpEndpoint(
        method="POST",
        path="/cycles/stop",
        summary="Stop the active trading cycle",
        request_model="services.trading_engine.contracts.EmptyMessage",
        response_model="services.trading_engine.contracts.TradingCycleState",
    ),
    HttpEndpoint(
        method="GET",
        path="/cycles/status",
        summary="Retrieve the current state of the cycle",
        response_model="services.trading_engine.contracts.TradingCycleState",
    ),
    HttpEndpoint(
        method="POST",
        path="/cycles/run",
        summary="Trigger a single trading cycle execution",
        request_model="services.trading_engine.contracts.TradingCycleCommand",
        response_model="services.trading_engine.contracts.TradingCycleRunResponse",
    ),
]


GRPC_CONTRACT: List[GrpcMethodDescriptor] = [
    GrpcMethodDescriptor(
        name="StartCycle",
        request="services.trading_engine.contracts.TradingCycleCommand",
        response="services.trading_engine.contracts.TradingCycleState",
    ),
    GrpcMethodDescriptor(
        name="StopCycle",
        request="services.trading_engine.contracts.EmptyMessage",
        response="services.trading_engine.contracts.TradingCycleState",
    ),
    GrpcMethodDescriptor(
        name="RunCycle",
        request="services.trading_engine.contracts.TradingCycleCommand",
        response="services.trading_engine.contracts.TradingCycleRunResponse",
    ),
    GrpcMethodDescriptor(
        name="GetCycleState",
        request="services.trading_engine.contracts.EmptyMessage",
        response="services.trading_engine.contracts.TradingCycleState",
    ),
]


__all__ = [
    "EmptyMessage",
    "GRPC_CONTRACT",
    "HTTP_CONTRACT",
    "TradingCycleCommand",
    "TradingCycleEvent",
    "TradingCycleEventPayload",
    "TradingCycleRunResponse",
    "TradingCycleState",
    "TradingEngineService",
]
