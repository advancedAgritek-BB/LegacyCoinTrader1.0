"""Backwards compatible logging helpers import.

This module preserves the historical ``services.monitoring.logging`` import
path while delegating to ``logging_utils`` where the implementation now
resides. Several services – including the API gateway – still import from the
older location, so keeping this thin shim prevents runtime import errors
without forcing widespread code churn.
"""

from __future__ import annotations

from .logging_utils import configure_logging, OpenSearchLogHandler

__all__ = ["configure_logging", "OpenSearchLogHandler"]

