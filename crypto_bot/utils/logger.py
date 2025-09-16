from __future__ import annotations

import contextvars
import json
import logging
import logging.config
import uuid
import weakref
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Union

# Default directory for all log files used across the project
LOG_DIR = Path(__file__).resolve().parents[2] / "crypto_bot" / "logs"

_CORRELATION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)

_LOGGING_CONFIGURED = False
_ADDITIONAL_HANDLERS: list[logging.Handler] = []
_CONFIGURED_LOGGERS: weakref.WeakSet[logging.Logger] = weakref.WeakSet()

_DEFAULT_LOG_RECORD_KEYS = frozenset(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__.keys()
) | {"message"}


def _generate_correlation_id() -> str:
    """Generate a new correlation identifier."""

    return uuid.uuid4().hex


def get_correlation_id() -> str:
    """Return the current correlation ID, creating one if necessary."""

    correlation_id = _CORRELATION_ID.get()
    if correlation_id is None:
        correlation_id = _generate_correlation_id()
        _CORRELATION_ID.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Bind a correlation ID to the current context and return it."""

    correlation_id = correlation_id or _generate_correlation_id()
    _CORRELATION_ID.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""

    _CORRELATION_ID.set(None)


@contextmanager
def correlation_id_context(correlation_id: Optional[str] = None) -> Iterator[str]:
    """Context manager that temporarily sets a correlation ID."""

    token = _CORRELATION_ID.set(correlation_id or _generate_correlation_id())
    try:
        yield _CORRELATION_ID.get() or ""
    finally:
        _CORRELATION_ID.reset(token)


def _serialize(value: object) -> object:
    """Safely serialise values for JSON output."""

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


class JsonFormatter(logging.Formatter):
    """Formatter that emits structured JSON log records."""

    def __init__(self, *, ensure_ascii: bool = False) -> None:
        super().__init__()
        self.ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        timestamp = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        log_record = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_record["stack"] = record.stack_info

        for key, value in record.__dict__.items():
            if key in _DEFAULT_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            log_record[key] = _serialize(value)

        return json.dumps(log_record, ensure_ascii=self.ensure_ascii)


def _configure_logging() -> None:
    """Initialise the global logging configuration if required."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": JsonFormatter,
                }
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {
                "handlers": ["stdout"],
                "level": "INFO",
            },
        }
    )

    _LOGGING_CONFIGURED = True


def _handler_matches_file(handler: logging.Handler, path: Path) -> bool:
    return isinstance(handler, logging.FileHandler) and Path(
        getattr(handler, "baseFilename", "")
    ) == path


def register_centralized_handler(
    handler: logging.Handler, *, set_json_formatter: bool = True
) -> None:
    """Attach an additional handler for centralised log sinks.

    Parameters
    ----------
    handler:
        The handler instance to register (e.g. HTTPHandler, SysLogHandler).
    set_json_formatter:
        Apply :class:`JsonFormatter` when the handler has no formatter defined.
    """

    _configure_logging()

    if set_json_formatter and handler.formatter is None:
        handler.setFormatter(JsonFormatter())

    if handler not in _ADDITIONAL_HANDLERS:
        _ADDITIONAL_HANDLERS.append(handler)

    root_logger = logging.getLogger()
    if handler not in root_logger.handlers:
        root_logger.addHandler(handler)

    for configured_logger in list(_CONFIGURED_LOGGERS):
        if handler not in configured_logger.handlers:
            configured_logger.addHandler(handler)


def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    to_console: bool = True,
) -> logging.Logger:
    """Return a JSON logger configured for stdout and optional sinks."""

    _configure_logging()

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = to_console

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(_handler_matches_file(h, log_path) for h in logger.handlers):
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)

    for handler in _ADDITIONAL_HANDLERS:
        if handler not in logger.handlers:
            logger.addHandler(handler)

    _CONFIGURED_LOGGERS.add(logger)

    return logger


__all__ = [
    "LOG_DIR",
    "JsonFormatter",
    "setup_logger",
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "correlation_id_context",
    "register_centralized_handler",
]
