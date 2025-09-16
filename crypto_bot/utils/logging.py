"""Utilities for configuring strategy specific logging."""

from __future__ import annotations

import logging
from pathlib import Path

from crypto_bot.utils.logger import LOG_DIR, setup_logger


STRATEGY_LOG_DIR = LOG_DIR / "strategies"


def _format_log_filename(name: str) -> Path:
    """Return the log file path for ``name`` within :data:`STRATEGY_LOG_DIR`."""

    safe_name = name.replace("/", "_").replace(".", "_")
    return STRATEGY_LOG_DIR / f"{safe_name}.log"


def setup_strategy_logger(name: str, *, to_console: bool = False) -> logging.Logger:
    """Return a logger configured for strategy modules.

    Parameters
    ----------
    name:
        Logical name of the strategy. The name is used both for the logger
        hierarchy (``strategy.<name>``) and the log file name. Dots and slashes
        are replaced with underscores when creating the filename so calling the
        helper with ``__name__`` is safe.
    to_console:
        When ``True`` the logger is also configured with a ``StreamHandler``.

    The helper reuses :func:`crypto_bot.utils.logger.setup_logger` to avoid
    duplicating handler configuration while ensuring all strategies emit logs
    under a dedicated directory.
    """

    log_file = _format_log_filename(name)
    logger = setup_logger(f"strategy.{name}", log_file, to_console=to_console)
    logger.propagate = False
    return logger


__all__ = ["setup_strategy_logger"]
