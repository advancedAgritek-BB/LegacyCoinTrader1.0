"""Signal handling helpers for the trading bot runtime."""
from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Optional

logger = logging.getLogger("bot")


def check_existing_instance(bot_pid_file: Path) -> bool:
    """Check if another bot instance is already running."""
    if not bot_pid_file.exists():
        return False
    try:
        with open(bot_pid_file, "r", encoding="utf-8") as file:
            pid_str = file.read().strip()
        if not pid_str:
            return False
        pid = int(pid_str)
        os.kill(pid, 0)
        return True
    except (ValueError, OSError):
        return False


def write_pid_file(bot_pid_file: Path) -> None:
    """Write current process PID to file."""
    with open(bot_pid_file, "w", encoding="utf-8") as file:
        file.write(str(os.getpid()))


def cleanup_pid_file(bot_pid_file: Path) -> None:
    """Remove PID file on shutdown."""
    try:
        if bot_pid_file.exists():
            bot_pid_file.unlink()
    except Exception:  # pragma: no cover - best effort cleanup
        logger.debug("Failed to remove pid file %s", bot_pid_file)


def install_signal_handlers(bot_pid_file: Path) -> None:
    """Install SIGINT/SIGTERM handlers that clean up the PID file."""

    def _handler(signum: int, _frame: Optional[FrameType]) -> None:
        logger.info("Received signal %d, shutting down...", signum)
        cleanup_pid_file(bot_pid_file)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
