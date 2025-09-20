"""Utilities for encapsulating Solana wallet configuration for isolated runtimes."""

from __future__ import annotations

import json
import logging
import os
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union


logger = logging.getLogger(__name__)


PrivateKeyLike = Union[str, Sequence[int]]


def _ensure_iterable_secret(raw: PrivateKeyLike) -> Sequence[int]:
    """Normalize ``raw`` into an iterable of integers representing the secret key."""

    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError("private key list cannot be empty")
        return list(int(v) for v in raw)

    raw_str = str(raw).strip()
    if not raw_str:
        raise ValueError("private key value is empty")

    if raw_str.startswith("[") and raw_str.endswith("]"):
        try:
            data = json.loads(raw_str)
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid data
            raise ValueError("private key JSON is invalid") from exc
        if not isinstance(data, list) or not data:
            raise ValueError("private key JSON must be a non-empty list")
        return [int(v) for v in data]

    try:
        decoded = base64.b64decode(raw_str, validate=False)
    except Exception as exc:  # pragma: no cover - invalid base64
        raise ValueError("private key must be JSON list or base64 string") from exc

    if not decoded:
        raise ValueError("decoded private key is empty")
    return list(decoded)


def normalize_private_key(raw: PrivateKeyLike) -> str:
    """Return the private key as a JSON encoded list of integers."""

    iterable = _ensure_iterable_secret(raw)
    return json.dumps(list(iterable))


def decode_private_key_bytes(raw: PrivateKeyLike) -> bytes:
    """Return the private key as bytes suitable for ``Keypair`` helpers."""

    iterable = _ensure_iterable_secret(raw)
    return bytes(iterable)


@dataclass
class WalletContext:
    """Snapshot of wallet-related configuration for isolated Solana runtimes."""

    name: str = "pump_sniper"
    public_key: Optional[str] = None
    private_key_env: Optional[str] = None
    private_key_path: Optional[str] = None
    rpc_url: Optional[str] = None
    jito_key_env: Optional[str] = None
    allow_main_balance_fallback: bool = False
    rate_limit_per_second: Optional[float] = None

    _private_key_cache: Optional[str] = field(default=None, init=False, repr=False)
    _jito_key_cache: Optional[str] = field(default=None, init=False, repr=False)

    def resolve_private_key(self, refresh: bool = False) -> Optional[str]:
        """Return the private key as JSON if configured, caching results."""

        if not refresh and self._private_key_cache:
            return self._private_key_cache

        raw: Optional[str] = None
        if self.private_key_env:
            raw = os.getenv(self.private_key_env)
        if not raw and self.private_key_path:
            try:
                raw = Path(self.private_key_path).read_text().strip()
            except FileNotFoundError:
                logger.error(
                    "Wallet %s private key file missing: %s",
                    self.name,
                    self.private_key_path,
                )
                return None
            except Exception as exc:  # pragma: no cover - filesystem errors
                logger.error(
                    "Wallet %s failed to read private key file %s: %s",
                    self.name,
                    self.private_key_path,
                    exc,
                )
                return None

        if not raw:
            return None

        try:
            normalized = normalize_private_key(raw)
        except ValueError as exc:
            logger.error("Wallet %s private key invalid: %s", self.name, exc)
            return None

        self._private_key_cache = normalized
        return normalized

    def resolve_jito_key(self, refresh: bool = False) -> Optional[str]:
        """Return the configured Jito bundle key if present."""

        if not refresh and self._jito_key_cache:
            return self._jito_key_cache

        if not self.jito_key_env:
            return None

        value = os.getenv(self.jito_key_env)
        if value:
            self._jito_key_cache = value.strip()
        return self._jito_key_cache

    def execution_override(self) -> Dict[str, Any]:
        """Return keyword arguments for execution helpers using this wallet."""

        override: Dict[str, Any] = {}
        private_key = self.resolve_private_key()
        if private_key:
            override["private_key"] = private_key
        if self.public_key:
            override["public_key"] = self.public_key
        if self.rpc_url:
            override["rpc_url"] = self.rpc_url
        jito_key = self.resolve_jito_key()
        if jito_key:
            override["jito_key"] = jito_key
        return override

    @property
    def is_configured(self) -> bool:
        """Return ``True`` when both public and private keys are resolved."""

        return bool(self.public_key and self.resolve_private_key())


def load_wallet_context(config: Mapping[str, Any]) -> WalletContext:
    """Build a :class:`WalletContext` from ``config`` data."""

    params: MutableMapping[str, Any] = dict(config)
    context = WalletContext(
        name=str(params.get("name", "pump_sniper")),
        public_key=params.get("public_key") or params.get("address"),
        private_key_env=params.get("private_key_env"),
        private_key_path=params.get("private_key_path"),
        rpc_url=params.get("rpc_url"),
        jito_key_env=params.get("jito_key_env"),
        allow_main_balance_fallback=bool(params.get("allow_main_balance_fallback", False)),
        rate_limit_per_second=params.get("rate_limit_per_second"),
    )

    if not context.is_configured:
        logger.info(
            "Wallet context %s incomplete: public key or private key not fully configured",
            context.name,
        )

    return context

