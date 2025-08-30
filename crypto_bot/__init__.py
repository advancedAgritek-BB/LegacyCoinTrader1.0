"""Lightweight package initializer.

Avoid importing heavy subpackages at import time to keep unit tests fast
and prevent optional dependency issues during collection. Modules should
import what they need directly rather than relying on package side effects.
"""

# Only import basic utilities that don't have circular dependencies
try:
    from .utils.logger import LOG_DIR, setup_logger
except Exception:  # pragma: no cover - allow package import without logger in minimal env
    LOG_DIR = None
    def setup_logger(*_a, **_k):  # type: ignore
        class _Dummy:
            def info(self, *a, **k):
                pass
            def warning(self, *a, **k):
                pass
            def error(self, *a, **k):
                pass
        return _Dummy()
__all__ = [name for name in ["LOG_DIR", "setup_logger"] if name in globals()]
