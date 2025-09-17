"""Trading engine service package."""

from __future__ import annotations

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy import
    if name == "app":
        from .app import app as application

        return application
    raise AttributeError(name)
