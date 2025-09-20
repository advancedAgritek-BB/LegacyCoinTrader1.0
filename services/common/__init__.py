"""Shared service utilities for LegacyCoinTrader microservices."""

from pathlib import Path

try:  # Attempt to populate env vars for local execution without failing in minimal envs.
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore


def _load_default_environment() -> None:
    """Load `.env` files when available without overriding existing values."""

    if load_dotenv is None:
        return

    # Honour the standard `.env` resolution order and explicitly include project fallbacks.
    load_dotenv(override=False)

    project_root = Path(__file__).resolve().parents[2]
    for candidate in (project_root / ".env", project_root / "crypto_bot" / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


_load_default_environment()

from .secret_manager import (  # noqa: E402  - import after environment setup
    AwsSecretsManagerProvider,
    HashicorpVaultSecretProvider,
    SecretManager,
    SecretProvider,
    SecretRetrievalError,
    resolve_secret,
)

__all__ = [
    "AwsSecretsManagerProvider",
    "HashicorpVaultSecretProvider",
    "SecretManager",
    "SecretProvider",
    "SecretRetrievalError",
    "resolve_secret",
]
