from __future__ import annotations

"""Default phases executed by the trading engine service."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .interface import CycleContext


async def prepare_cycle(context: "CycleContext") -> None:
    """Record that a new cycle is being prepared."""

    context.metadata.setdefault("events", []).append("prepare")
    await asyncio.sleep(0)
    logger.debug("Prepared trading cycle with metadata: %s", context.metadata)


async def orchestrate_cycle(context: "CycleContext") -> None:
    """Simulate orchestration work for the trading cycle."""

    context.metadata["cycle_started_at"] = datetime.now(timezone.utc).isoformat()
    await asyncio.sleep(0)
    logger.debug("Orchestrated trading cycle")


async def finalize_cycle(context: "CycleContext") -> None:
    """Finalize the trading cycle and mark completion time."""

    context.metadata["cycle_completed_at"] = datetime.now(timezone.utc).isoformat()
    await asyncio.sleep(0)
    logger.debug("Finalized trading cycle")


DEFAULT_PHASES = [prepare_cycle, orchestrate_cycle, finalize_cycle]


__all__ = [
    "prepare_cycle",
    "orchestrate_cycle",
    "finalize_cycle",
    "DEFAULT_PHASES",
]
