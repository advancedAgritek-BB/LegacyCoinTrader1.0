"""Security utilities for credential handling."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from typing import Optional

DEFAULT_HASH_ITERATIONS = 320_000
_SALT_BYTES = 24


@dataclass(slots=True)
class HashedSecret:
    """Container describing a hashed secret."""

    hash: str
    salt: str
    iterations: int = DEFAULT_HASH_ITERATIONS


def _encode_bytes(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode_to_bytes(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value or "").encode("ascii") + padding.encode("ascii"))


def generate_salt(length: int = _SALT_BYTES) -> str:
    """Return a random salt encoded as URL safe base64."""

    return _encode_bytes(os.urandom(length))


def hash_secret(
    secret: str,
    *,
    salt: Optional[str] = None,
    iterations: int = DEFAULT_HASH_ITERATIONS,
) -> HashedSecret:
    """Derive a PBKDF2-HMAC hash for the provided secret."""

    if not secret:
        raise ValueError("Secret value must be provided")

    salt_value = salt or generate_salt()
    salt_bytes = _decode_to_bytes(salt_value)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        salt_bytes,
        iterations,
    )
    return HashedSecret(hash=_encode_bytes(derived), salt=salt_value, iterations=iterations)


def verify_secret(secret: str, hashed: HashedSecret) -> bool:
    """Check if *secret* matches the stored hash."""

    if not secret:
        return False
    candidate = hash_secret(secret, salt=hashed.salt, iterations=hashed.iterations)
    return hmac.compare_digest(candidate.hash, hashed.hash)


def generate_token(length: int = 32) -> str:
    """Return a cryptographically secure random token."""

    # ``token_urlsafe`` returns roughly 1.3 characters per byte. The ``length``
    # parameter therefore controls the resulting entropy without forcing
    # callers to reason about byte lengths.
    return secrets.token_urlsafe(length)


__all__ = [
    "DEFAULT_HASH_ITERATIONS",
    "HashedSecret",
    "generate_salt",
    "generate_token",
    "hash_secret",
    "verify_secret",
]
