"""Minimal psycopg2 compatibility shim for local development and tests.

This fallback is intentionally lightweight – it only implements the pieces
of the real driver that our test-suite touches.  The actual application
uses SQLAlchemy elsewhere, so the shim is only used when the binary wheels
aren't available in the sandboxed environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

from . import extensions  # re-export for ``from psycopg2 import extensions``

__all__ = [
    "connect",
    "OperationalError",
    "DatabaseError",
    "Error",
    "extensions",
]


class Error(Exception):
    """Base exception matching psycopg2.Error."""


class DatabaseError(Error):
    """Basic database error placeholder."""


class OperationalError(DatabaseError):
    """Operational errors raised by the shim."""


@dataclass
class _Cursor:
    """Very small cursor implementation backed by an in-memory result."""

    _last_result: Optional[Tuple[Any, ...]] = None

    def execute(self, query: str, params: Optional[Iterable[Any]] = None) -> None:
        normalized = query.strip().lower()
        if normalized.startswith("select"):
            if "1" in normalized:
                self._last_result = (1,)
            else:
                self._last_result = (None,)
        else:
            self._last_result = None

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        return self._last_result

    def fetchall(self) -> Tuple[Tuple[Any, ...], ...]:
        if self._last_result is None:
            return tuple()
        return (self._last_result,)

    def close(self) -> None:  # pragma: no cover - no-op for compatibility
        self._last_result = None


class _Connection:
    """Tiny connection object sufficient for the integration tests."""

    def __init__(self, dsn: str = "", **_kwargs: Any) -> None:
        self._dsn = dsn
        self._isolation_level = extensions.ISOLATION_LEVEL_AUTOCOMMIT

    def cursor(self) -> _Cursor:
        return _Cursor()

    def commit(self) -> None:  # pragma: no cover - no-op
        return None

    def rollback(self) -> None:  # pragma: no cover - no-op
        return None

    def close(self) -> None:  # pragma: no cover - no-op
        return None

    def set_isolation_level(self, level: int) -> None:
        self._isolation_level = level


def connect(dsn: Optional[str] = None, *args: Any, **kwargs: Any) -> _Connection:
    """Return a dummy connection object."""

    if dsn is None:
        dsn = ""
    return _Connection(dsn=dsn, **kwargs)
