"""Exchange connectivity utilities for the execution service."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from typing import Any, Mapping

from libs.execution import get_exchange

from .models import ExchangeCredentials, ExchangeSession
from .nonce import NonceManager
from .secrets import SecretLoader


@contextlib.contextmanager
def _temporary_env(values: Mapping[str, str]):
    """Temporarily set environment variables while creating an exchange."""
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, prior in previous.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _hash_config(config: Mapping[str, Any]) -> str:
    """Return a deterministic hash for ``config``."""

    def _normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, (list, tuple)):
            return [_normalize(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    normalized = _normalize(config)
    payload = json.dumps(normalized, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class ExchangeFactory:
    """Factory that creates and memoises exchange sessions."""

    def __init__(self, secret_loader: SecretLoader) -> None:
        self._secret_loader = secret_loader

    def create_session(
        self,
        config: Mapping[str, Any],
        credentials: ExchangeCredentials | None,
        nonce_manager: NonceManager,
    ) -> ExchangeSession:
        env_values = self._secret_loader.to_env_mapping(credentials)
        with _temporary_env(env_values):
            exchange, ws_client = get_exchange(dict(config))
        if hasattr(exchange, "nonce"):
            exchange.nonce = nonce_manager.next_nonce  # type: ignore[assignment]
        return ExchangeSession(exchange=exchange, ws_client=ws_client, config_hash=_hash_config(config))
