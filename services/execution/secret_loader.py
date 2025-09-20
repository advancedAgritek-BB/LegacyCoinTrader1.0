"""Helpers for securely loading secrets used by the execution service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from services.common.secret_manager import SecretRetrievalError, resolve_secret

from .config import CredentialsConfig
from .models import ExchangeCredentials, SecretRef

try:  # pragma: no cover - optional dependency
    import hvac  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hvac = None


class SecretNotFoundError(RuntimeError):
    """Raised when a requested secret cannot be resolved."""


class SecretLoader:
    """Resolve secret references backed by Vault or Kubernetes."""

    def __init__(
        self,
        *,
        kubernetes_base: Union[str, Path, None] = None,
        vault_addr: Optional[str] = None,
        vault_token: Optional[str] = None,
    ) -> None:
        self._k8s_base = Path(kubernetes_base or "/var/run/secrets")
        self._vault_client = None
        if hvac and vault_addr and vault_token:
            client = hvac.Client(url=vault_addr, token=vault_token)
            if client.is_authenticated():  # pragma: no branch - hvac handles auth
                self._vault_client = client

    def load_secret(self, ref: SecretRef | None) -> Optional[str]:
        """Resolve ``ref`` returning the secret value or ``None``."""
        if ref is None:
            return None
        source = (ref.source or "env").lower()
        if source == "literal":
            return ref.name
        if source == "env":
            value = os.getenv(ref.name)
            if value:
                return value
            try:
                return resolve_secret(ref.name)
            except SecretRetrievalError as exc:
                raise SecretNotFoundError(
                    f"Environment variable {ref.name!r} not set and secrets manager lookup failed"
                ) from exc
        if source in {"file", "kubernetes"}:
            base = self._k8s_base if source == "kubernetes" else Path("/")
            path = Path(ref.name)
            if not path.is_absolute():
                path = base / path
            if ref.key:
                path = path / ref.key
            try:
                return path.read_text().strip()
            except FileNotFoundError as exc:  # pragma: no cover - filesystem specific
                raise SecretNotFoundError(str(exc)) from exc
        if source in {"manager", "secrets_manager", "secretmanager"}:
            key_name = ref.key or ref.name
            try:
                return resolve_secret(key_name, vault_path=ref.name if ref.key else None)
            except SecretRetrievalError as exc:
                raise SecretNotFoundError(
                    f"Secrets manager reference {key_name!r} could not be resolved"
                ) from exc
        if source == "vault":
            if not ref.key:
                raise SecretNotFoundError("Vault references require a key")
            try:
                return resolve_secret(ref.key, vault_path=ref.name)
            except SecretRetrievalError:
                pass
            if not self._vault_client:
                raise SecretNotFoundError("Vault client not configured or hvac unavailable")
            secret = self._vault_client.secrets.kv.v2.read_secret_version(path=ref.name)
            data = secret.get("data", {}).get("data", {})
            if ref.key not in data:
                raise SecretNotFoundError(
                    f"Vault secret {ref.name!r} missing key {ref.key!r}"
                )
            return str(data[ref.key])
        raise SecretNotFoundError(f"Unknown secret source: {ref.source!r}")

    def load_credentials(self, cfg: CredentialsConfig | None) -> Optional[ExchangeCredentials]:
        """Load exchange credentials as configured."""
        if cfg is None:
            return None
        api_key = self.load_secret(cfg.api_key)
        api_secret = self.load_secret(cfg.api_secret)
        passphrase = self.load_secret(cfg.passphrase)
        ws_token = self.load_secret(cfg.ws_token)
        api_token = self.load_secret(cfg.api_token)
        return ExchangeCredentials(
            api_key=api_key or "",
            api_secret=api_secret or "",
            passphrase=passphrase,
            ws_token=ws_token,
            api_token=api_token,
        )

    def to_env_mapping(self, creds: ExchangeCredentials | None) -> dict[str, str]:
        """Return environment overrides for ``ccxt`` exchange creation."""
        if creds is None:
            return {}
        env: dict[str, str] = {
            "API_KEY": creds.api_key,
            "API_SECRET": creds.api_secret,
        }
        if creds.passphrase:
            env["API_PASSPHRASE"] = creds.passphrase
        if creds.ws_token:
            env["KRAKEN_WS_TOKEN"] = creds.ws_token
        if creds.api_token:
            env["KRAKEN_API_TOKEN"] = creds.api_token
        return env

    @staticmethod
    def from_json(path: Union[str, Path]) -> "SecretLoader":
        """Create loader configured via a JSON manifest (utility for tests)."""
        data = json.loads(Path(path).read_text())
        return SecretLoader(
            kubernetes_base=data.get("kubernetes_base"),
            vault_addr=data.get("vault", {}).get("addr"),
            vault_token=data.get("vault", {}).get("token"),
        )
