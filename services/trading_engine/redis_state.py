from __future__ import annotations

"""Redis-backed persistence for trading cycle state."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from redis.asyncio import Redis

from services.interface_layer.cycle import CycleExecutionResult


@dataclass
class CycleState:
    """Lightweight representation of scheduler state."""

    running: bool = False
    interval_seconds: int = 60
    next_run_at: Optional[datetime] = None
    last_run_started_at: Optional[datetime] = None
    last_run_completed_at: Optional[datetime] = None
    last_timings: Dict[str, float] = field(default_factory=dict)
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _dt_to_iso(value: Optional[datetime]) -> Optional[str]:
        if value is None:
            return None
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _iso_to_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        return datetime.fromisoformat(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "interval_seconds": self.interval_seconds,
            "next_run_at": self._dt_to_iso(self.next_run_at),
            "last_run_started_at": self._dt_to_iso(self.last_run_started_at),
            "last_run_completed_at": self._dt_to_iso(self.last_run_completed_at),
            "last_timings": self.last_timings,
            "last_error": self.last_error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleState":
        return cls(
            running=bool(data.get("running", False)),
            interval_seconds=int(data.get("interval_seconds", 60)),
            next_run_at=cls._iso_to_dt(data.get("next_run_at")),
            last_run_started_at=cls._iso_to_dt(data.get("last_run_started_at")),
            last_run_completed_at=cls._iso_to_dt(data.get("last_run_completed_at")),
            last_timings=dict(data.get("last_timings", {})),
            last_error=data.get("last_error"),
            metadata=dict(data.get("metadata", {})),
        )

    def predict_next_run(self) -> Optional[datetime]:
        if not self.running or self.interval_seconds <= 0:
            return None
        reference = self.last_run_completed_at or datetime.now(timezone.utc)
        return reference + timedelta(seconds=self.interval_seconds)


class RedisCycleStateStore:
    """Persist ``CycleState`` using Redis."""

    def __init__(self, client: Redis, key_prefix: str = "trading_engine") -> None:
        self._client = client
        self._state_key = f"{key_prefix}:state"

    async def load_state(self) -> CycleState:
        raw = await self._client.get(self._state_key)
        if not raw:
            return CycleState()
        payload = json.loads(raw)
        return CycleState.from_dict(payload)

    async def save_state(self, state: CycleState) -> CycleState:
        await self._client.set(self._state_key, json.dumps(state.to_dict()))
        return state

    async def ensure_defaults(self, interval_seconds: int) -> CycleState:
        state = await self.load_state()
        if state.interval_seconds != interval_seconds and not state.running:
            state.interval_seconds = interval_seconds
        if state.next_run_at is None and state.running:
            state.next_run_at = state.predict_next_run()
        await self.save_state(state)
        return state

    async def update(self, **fields: Any) -> CycleState:
        state = await self.load_state()
        for key, value in fields.items():
            if hasattr(state, key):
                setattr(state, key, value)
        await self.save_state(state)
        return state

    async def mark_cycle_start(self, started_at: datetime) -> CycleState:
        return await self.update(last_run_started_at=started_at, last_error=None)

    async def mark_cycle_complete(
        self,
        result: CycleExecutionResult,
        completed_at: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CycleState:
        state = await self.load_state()
        state.last_timings = result.timings
        state.last_run_completed_at = completed_at
        state.last_error = None
        state.last_run_started_at = state.last_run_started_at or completed_at
        state.next_run_at = state.predict_next_run()
        if metadata is not None:
            state.metadata = dict(metadata)
        await self.save_state(state)
        return state

    async def mark_cycle_failure(
        self,
        error: str,
        completed_at: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CycleState:
        update_fields: Dict[str, Any] = {"last_error": error, "last_run_completed_at": completed_at}
        if metadata is not None:
            update_fields["metadata"] = dict(metadata)
        state = await self.update(**update_fields)
        return state


__all__ = ["CycleState", "RedisCycleStateStore"]
