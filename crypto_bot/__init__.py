"""Lightweight package initializer.

Avoid importing heavy subpackages at import time to keep unit tests fast
and prevent optional dependency issues during collection. Modules should
import what they need directly rather than relying on package side effects.
"""

from typing import Any, Optional

LOG_DIR: Any = None


def _setup_logger_stub(
    name: str,
    log_file: Optional[str] = None,
    to_console: bool = True,
) -> Any:
    class _Dummy:
        def info(self, *a: Any, **k: Any) -> None:
            pass

        def warning(self, *a: Any, **k: Any) -> None:
            pass

        def error(self, *a: Any, **k: Any) -> None:
            pass

    return _Dummy()


# Default to stub; will be replaced if real logger import succeeds
setup_logger = _setup_logger_stub


# Only import basic utilities that don't have circular dependencies
try:
    from .utils import logger as _logger
    LOG_DIR = _logger.LOG_DIR
    setup_logger = _logger.setup_logger
except Exception:  # pragma: no cover - allow package import without logger
    pass


# Provide lightweight accessors for subpackages used by tests
# without importing them eagerly. This allows patching like
# `crypto_bot.backtest.enhanced_backtester` to work.

def __getattr__(name: str) -> Any:
    if name == "backtest":
        import importlib
        return importlib.import_module("crypto_bot.backtest")
    if name == "strategy_router":
        import importlib
        return importlib.import_module("crypto_bot.strategy_router")
    if name == "solana":
        import importlib
        return importlib.import_module("crypto_bot.solana")
    raise AttributeError(name)


__all__ = [
    name for name in [
        "LOG_DIR",
        "setup_logger",
        "backtest",
        "strategy_router",
        "solana",
    ]
]
