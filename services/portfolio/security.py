from __future__ import annotations

import base64
import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_HASH_ITERATIONS = 210_000
_SALT_BYTES = 16


@dataclass
class HashedSecret:
    """Container for a salted and iterated PBKDF2 hash."""

    hash: str
    salt: str
    iterations: int = DEFAULT_HASH_ITERATIONS


def _encode_bytes(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode_to_bytes(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value or "").encode("ascii") + padding.encode("ascii"))


def generate_salt(length: int = _SALT_BYTES) -> str:
    """Return a random salt encoded as URL-safe base64."""

    return _encode_bytes(os.urandom(length))


def hash_secret(
    secret: str,
    *,
    salt: Optional[str] = None,
    iterations: int = DEFAULT_HASH_ITERATIONS,
) -> HashedSecret:
    """Derive a PBKDF2-HMAC hash for the provided secret."""

    if not secret:
        raise ValueError("Secret value must be a non-empty string")

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
    """Check whether *secret* matches the stored :class:`HashedSecret`."""

    if not secret:
        return False
    candidate = hash_secret(secret, salt=hashed.salt, iterations=hashed.iterations)
    return hmac.compare_digest(candidate.hash, hashed.hash)


__all__ = [
    "DEFAULT_HASH_ITERATIONS",
    "HashedSecret",
    "generate_salt",
    "hash_secret",
    "verify_secret",
]
