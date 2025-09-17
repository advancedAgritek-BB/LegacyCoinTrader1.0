"""Shared service utilities for LegacyCoinTrader microservices."""

from .secrets import (
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
