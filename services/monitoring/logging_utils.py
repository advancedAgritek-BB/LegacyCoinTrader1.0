"""Logging helpers for shipping structured logs to central sinks."""

from __future__ import annotations

import json
import logging
from typing import Optional

from crypto_bot.utils.logger import register_centralized_handler, setup_logger

from .config import MonitoringSettings, OpenSearchSettings

try:  # pragma: no cover - optional dependency guard
    from opensearchpy import OpenSearch
except Exception:  # pragma: no cover - library may not be installed in tests
    OpenSearch = None  # type: ignore


class OpenSearchLogHandler(logging.Handler):
    """Logging handler that forwards structured logs into OpenSearch."""

    def __init__(self, settings: OpenSearchSettings) -> None:
        super().__init__()
        self.settings = settings
        self._client: Optional[OpenSearch] = None
        self._failed = False

    def _ensure_client(self) -> Optional[OpenSearch]:
        if self._failed or not self.settings.enabled:
            return None
        if self._client is not None:
            return self._client
        if OpenSearch is None:
            self._failed = True
            return None

        hosts = [{
            "host": self.settings.host,
            "port": self.settings.port,
            "use_ssl": self.settings.use_ssl,
        }]
        auth = None
        if self.settings.username and self.settings.password:
            auth = (self.settings.username, self.settings.password)

        try:
            self._client = OpenSearch(  # type: ignore[call-arg]
                hosts=hosts,
                http_auth=auth,
                verify_certs=self.settings.verify_certs,
                timeout=self.settings.timeout,
            )
        except Exception:  # pragma: no cover - network failure is non-critical in tests
            self._failed = True
            self._client = None
        return self._client

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - network side effect
        client = self._ensure_client()
        if client is None:
            return
        try:
            formatted = self.format(record)
            try:
                document = json.loads(formatted)
            except json.JSONDecodeError:
                document = {"message": formatted}
            client.index(index=self.settings.index, document=document)
        except Exception:
            self._failed = True


def configure_logging(settings: MonitoringSettings) -> logging.Logger:
    """Configure structured logging and optional log shipping."""

    # Ensure service_name and log_level are strings (handle FieldInfo issue)
    service_name = getattr(settings, 'service_name', 'legacycoin-service')
    if hasattr(service_name, 'default'):
        service_name = service_name.default
    
    log_level = getattr(settings, 'log_level', 'INFO')
    if hasattr(log_level, 'default'):
        log_level = log_level.default

    logger = setup_logger(service_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Handle FieldInfo for opensearch settings
    opensearch_enabled = False
    try:
        opensearch_enabled = getattr(settings.opensearch, 'enabled', False)
        if hasattr(opensearch_enabled, 'default'):
            opensearch_enabled = opensearch_enabled.default
    except AttributeError:
        opensearch_enabled = False

    if opensearch_enabled:
        handler = OpenSearchLogHandler(settings.opensearch)
        register_centralized_handler(handler)

    return logger


__all__ = ["OpenSearchLogHandler", "configure_logging"]
