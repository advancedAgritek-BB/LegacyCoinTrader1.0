#!/usr/bin/env python3
"""Dedicated entrypoint for running the pump sniper system in isolation."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path
from typing import Any, Dict

import yaml

from crypto_bot.solana.pump_sniper_integration import (
    start_pump_sniper_system,
    stop_pump_sniper_system,
)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return data


async def _run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    main_config = _load_config(config_path)

    started = await start_pump_sniper_system(
        main_config,
        dry_run=not args.live,
        paper_wallet=None,
    )
    if not started:
        raise SystemExit("pump sniper system failed to start")

    stop_event = asyncio.Event()

    def _handle_signal(*_ignored) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:  # pragma: no cover - Windows compatibility
            signal.signal(sig, lambda *_: stop_event.set())

    await stop_event.wait()
    await stop_pump_sniper_system()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pump sniper runtime in isolation")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the base bot configuration (default: config.yaml)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run with live trading enabled (requires dedicated wallet configuration)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:  # pragma: no cover - redundant safeguard
        pass


if __name__ == "__main__":
    main()

