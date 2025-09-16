"""Utility helpers for configuring strategy loggers."""

from __future__ import annotations

import logging
from pathlib import Path

from .logger import LOG_DIR, setup_logger


_STRATEGY_LOG_DIR = LOG_DIR / "strategies"


def _sanitise_name(name: str) -> str:
    """Return a filesystem-friendly representation of ``name``."""

    return "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip()
    ) or "strategy"


def setup_strategy_logger(name: str) -> logging.Logger:
    """Return a logger configured for use inside strategy modules.

    The logger writes to ``crypto_bot/logs/strategies/<name>.log`` and is
    configured with standard stream and file handlers via
    :func:`crypto_bot.utils.logger.setup_logger`.
    """

    safe_name = _sanitise_name(name)
    log_path = Path(_STRATEGY_LOG_DIR) / f"{safe_name}.log"
    logger = setup_logger(f"strategy.{name}", log_path)
    logger.propagate = False
    return logger


__all__ = ["setup_strategy_logger"]
