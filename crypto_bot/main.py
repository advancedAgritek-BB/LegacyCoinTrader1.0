"""Entry point that wires dependencies and delegates to the orchestrator layer."""
from __future__ import annotations

import asyncio
import random
from datetime import datetime
from typing import Any, Callable, Optional

from crypto_bot.balance_management import get_paper_wallet_status
from crypto_bot.orchestration import (
    MemoryManager,
    _main_impl as orchestrator_main_impl,
)
from crypto_bot.risk import RiskManager, build_risk_manager
from crypto_bot.runtime_signals import cleanup_pid_file
from crypto_bot.services.interfaces import ExchangeRequest, ServiceContainer
from crypto_bot.startup_utils import create_service_container, load_config
from crypto_bot.utils.telemetry import telemetry
from crypto_bot.utils.telegram import TelegramNotifier

__all__ = [
    "MemoryManager",
    "cleanup_pid_file",
    "get_paper_wallet_status",
    "bootstrap",
    "main",
    "run_bot",
    "_main_impl",
]


async def bootstrap(
    *,
    config: dict | None = None,
    services: ServiceContainer | None = None,
    exchange: object | None = None,
    ws_client: Any | None = None,
    risk_manager: RiskManager | None = None,
    telemetry_client: Any = telemetry,
    clock: Optional[Callable[[], datetime]] = None,
    timer: Optional[Callable[[], float]] = None,
    rng: Optional[random.Random] = None,
    numpy_rng: Optional[Any] = None,
) -> TelegramNotifier:
    """Prepare dependencies and delegate to :func:`orchestrator_main_impl`."""

    config = config or load_config()
    services = services or create_service_container()
    if exchange is None or ws_client is None:
        exchange_resp = services.execution.create_exchange(
            ExchangeRequest(config=config)
        )
        exchange = exchange_resp.exchange
        ws_client = exchange_resp.ws_client
    if risk_manager is None:
        volume_ratio = 0.01 if config.get("testing_mode") else 1.0
        risk_manager = build_risk_manager(config, volume_ratio)
    return await orchestrator_main_impl(
        config=config,
        services=services,
        exchange=exchange,
        ws_client=ws_client,
        risk_manager=risk_manager,
        telemetry_client=telemetry_client,
        clock=clock,
        timer=timer,
        rng=rng,
        numpy_rng=numpy_rng,
    )


async def main() -> None:
    """Bootstrap dependencies and run the trading bot."""

    await bootstrap()


if __name__ == "__main__":  # pragma: no cover - manual execution
    asyncio.run(main())


async def _main_impl(
    *,
    clock: Optional[Callable[[], datetime]] = None,
    timer: Optional[Callable[[], float]] = None,
    rng: Optional[random.Random] = None,
    numpy_rng: Optional[Any] = None,
) -> TelegramNotifier:
    """Compatibility wrapper that delegates to the orchestrator implementation."""

    return await bootstrap(
        clock=clock,
        timer=timer,
        rng=rng,
        numpy_rng=numpy_rng,
    )


async def run_bot(
    *,
    clock: Optional[Callable[[], datetime]] = None,
    timer: Optional[Callable[[], float]] = None,
    rng: Optional[random.Random] = None,
    numpy_rng: Optional[Any] = None,
) -> TelegramNotifier:
    """Maintain compatibility with legacy entry points."""

    return await bootstrap(
        clock=clock,
        timer=timer,
        rng=rng,
        numpy_rng=numpy_rng,
    )
