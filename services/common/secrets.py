"""Shared helpers for retrieving secrets from external providers."""

from __future__ import annotations

import base64
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


class SecretRetrievalError(RuntimeError):
    """Raised when a secret cannot be resolved."""


class SecretProvider(Protocol):
    """Protocol for secret providers."""

    def get_secret(self, key: str, *, path: Optional[str] = None) -> Optional[str]:
        """Return the secret value for *key* if available."""


class HashicorpVaultSecretProvider:
    """Load secrets from a Hashicorp Vault KV store."""

    def __init__(
        self,
        address: str,
        token: str,
        secret_path: str,
        *,
        timeout: float = 5.0,
        verify: Optional[bool] = None,
    ) -> None:
        if not address or not token:
            raise ValueError("Hashicorp Vault provider requires address and token")

        self.address = address.rstrip("/")
        self.token = token
        self.secret_path = secret_path.strip("/")
        self.timeout = timeout
        self.verify = verify
        self._cache: Dict[str, Optional[str]] = {}

    def get_secret(self, key: str, *, path: Optional[str] = None) -> Optional[str]:
        import requests

        secret_path = (path or self.secret_path).strip("/")
        if not secret_path:
            raise SecretRetrievalError("Vault secret path cannot be empty")

        cache_key = f"{secret_path}:{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{self.address}/v1/{secret_path}"
        headers = {"X-Vault-Token": self.token}
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify if self.verify is not None else True,
            )
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - runtime specific network errors
            raise SecretRetrievalError(f"Vault request failed for path '{secret_path}': {exc}") from exc

        try:
            payload = response.json()
        except Exception as exc:
            raise SecretRetrievalError(
                f"Vault response for path '{secret_path}' is not valid JSON: {exc}"
            ) from exc

        data: Any = payload.get("data", {})
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise SecretRetrievalError(
                f"Vault response for path '{secret_path}' does not contain secret data"
            )

        raw_value = data.get(key)
        if raw_value is None:
            self._cache[cache_key] = None
            return None

        value = str(raw_value)
        self._cache[cache_key] = value
        return value


class AwsSecretsManagerProvider:
    """Load secrets from AWS Secrets Manager."""

    def __init__(self, secret_name: str, region: str, profile: Optional[str] = None) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - boto3 is optional
            raise SecretRetrievalError("boto3 is required for AWS Secrets Manager support") from exc

        session_kwargs: Dict[str, str] = {}
        if profile:
            session_kwargs["profile_name"] = profile

        session = boto3.session.Session(**session_kwargs)
        self.client = session.client("secretsmanager", region_name=region)
        self.secret_name = secret_name
        self._cache: Dict[str, str] = {}

    def _load_secret_bundle(self) -> None:
        if self._cache:
            return

        response = self.client.get_secret_value(SecretId=self.secret_name)
        payload: Dict[str, Any] = {}

        secret_string = response.get("SecretString")
        if secret_string:
            try:
                payload = json.loads(secret_string)
            except json.JSONDecodeError:
                payload = {self.secret_name: secret_string}

        secret_binary = response.get("SecretBinary")
        if secret_binary and not payload:
            try:
                decoded = base64.b64decode(secret_binary).decode("utf-8")
                payload = json.loads(decoded)
            except Exception as exc:  # pragma: no cover - depends on runtime data
                raise SecretRetrievalError(
                    f"Failed to decode binary secret '{self.secret_name}': {exc}"
                ) from exc

        for key, value in payload.items():
            if value is not None:
                self._cache[str(key)] = str(value)

    def get_secret(self, key: str, *, path: Optional[str] = None) -> Optional[str]:
        del path  # Secrets Manager does not use per-key paths
        self._load_secret_bundle()
        return self._cache.get(key)


class SecretManager:
    """Resolve secrets from the environment or an optional external provider."""

    def __init__(self, provider: Optional[SecretProvider] = None) -> None:
        self.provider = provider or self._build_provider()

    def _build_provider(self) -> Optional[SecretProvider]:
        provider_name = (os.getenv("SECRETS_PROVIDER", "").strip().lower())
        if provider_name in {"hashicorp", "vault"}:
            address = os.getenv("VAULT_ADDR", "").strip()
            token = os.getenv("VAULT_TOKEN", "").strip()
            secret_path = os.getenv("VAULT_SECRET_PATH", "secret/data/frontend").strip()
            verify_env = os.getenv("VAULT_VERIFY", "true").strip().lower()
            verify: Optional[bool]
            if verify_env in {"false", "0", "no"}:
                verify = False
            elif verify_env in {"true", "1", "yes"}:
                verify = True
            else:
                verify = None

            timeout_env = os.getenv("VAULT_TIMEOUT", "5.0").strip()
            try:
                timeout = float(timeout_env)
            except ValueError:
                timeout = 5.0

            if not address or not token:
                logger.warning(
                    "Hashicorp Vault provider selected but VAULT_ADDR or VAULT_TOKEN is missing."
                )
                return None

            try:
                return HashicorpVaultSecretProvider(
                    address,
                    token,
                    secret_path,
                    timeout=timeout,
                    verify=verify,
                )
            except Exception as exc:  # pragma: no cover - configuration errors
                logger.warning("Failed to initialise Hashicorp Vault provider: %s", exc)
                return None

        if provider_name in {"aws", "aws_secrets_manager", "secretsmanager"}:
            secret_name = os.getenv("AWS_SECRET_NAME") or os.getenv("AWS_SECRETS_MANAGER_SECRET_NAME")
            region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            profile = os.getenv("AWS_PROFILE")
            if not secret_name or not region:
                logger.warning(
                    "AWS Secrets Manager provider selected but AWS_SECRET_NAME or AWS_REGION is missing."
                )
                return None
            try:
                return AwsSecretsManagerProvider(secret_name, region, profile=profile)
            except SecretRetrievalError as exc:  # pragma: no cover - depends on environment
                logger.warning("Failed to initialise AWS Secrets Manager provider: %s", exc)
                return None

        if provider_name:
            logger.warning(
                "Unknown secrets provider '%s' configured; falling back to environment",
                provider_name,
            )
        return None

    def resolve(
        self,
        key: str,
        *,
        default: Optional[str] = None,
        env_keys: Optional[Sequence[str]] = None,
        vault_path: Optional[str] = None,
    ) -> str:
        """Resolve *key* from the environment, provider, or *default*."""

        candidate_env_keys = list(env_keys or [])
        candidate_env_keys.append(key)

        seen: set[str] = set()
        for env_key in candidate_env_keys:
            for candidate in {env_key, env_key.upper()}:
                if candidate in seen:
                    continue
                seen.add(candidate)
                value = os.getenv(candidate)
                if value:
                    logger.debug("Resolved secret '%s' from environment variable '%s'", key, candidate)
                    return value

        path_env_key = f"{key}_VAULT_PATH"
        resolved_path = vault_path or os.getenv(path_env_key) or os.getenv(path_env_key.upper())

        if self.provider:
            try:
                secret_value = self.provider.get_secret(key, path=resolved_path)
            except SecretRetrievalError as exc:
                logger.warning("Failed to load secret '%s' from provider: %s", key, exc)
            else:
                if secret_value is not None:
                    logger.debug("Resolved secret '%s' using provider", key)
                    return secret_value

        if default is not None:
            logger.debug("Using default value for secret '%s'", key)
            return default

        raise SecretRetrievalError(
            f"Secret '{key}' could not be resolved from the environment or the configured provider"
        )


@lru_cache
def get_secret_manager() -> SecretManager:
    """Return a cached :class:`SecretManager` instance."""

    return SecretManager()


def resolve_secret(
    key: str,
    *,
    default: Optional[str] = None,
    env_keys: Optional[Sequence[str]] = None,
    vault_path: Optional[str] = None,
) -> str:
    """Convenience wrapper for :class:`SecretManager`."""

    manager = get_secret_manager()
    return manager.resolve(key, default=default, env_keys=env_keys, vault_path=vault_path)


__all__ = [
    "AwsSecretsManagerProvider",
    "HashicorpVaultSecretProvider",
    "SecretManager",
    "SecretProvider",
    "SecretRetrievalError",
    "get_secret_manager",
    "resolve_secret",
]
