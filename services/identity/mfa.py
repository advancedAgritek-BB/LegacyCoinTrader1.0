"""Pluggable multi-factor authentication helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import struct
import time
from typing import Protocol

from .models import UserModel


class MfaChallengeError(RuntimeError):
    """Raised when MFA validation fails."""


class MfaProvider(Protocol):
    """Interface that custom MFA providers must implement."""

    def is_required(self, user: UserModel) -> bool:  # pragma: no cover - protocol definition
        ...

    def verify(self, user: UserModel, code: str) -> bool:  # pragma: no cover - protocol definition
        ...

    def challenge(self, user: UserModel) -> None:  # pragma: no cover - protocol definition
        ...


class NullMfaProvider:
    """MFA provider used when MFA is disabled."""

    def is_required(self, user: UserModel) -> bool:
        return False

    def verify(self, user: UserModel, code: str) -> bool:
        del user, code
        return True

    def challenge(self, user: UserModel) -> None:
        del user


class TotpMfaProvider:
    """Simple TOTP-based MFA provider compatible with most authenticators."""

    def __init__(self, *, digits: int = 6, interval: int = 30, window: int = 1) -> None:
        self.digits = digits
        self.interval = interval
        self.window = window

    def is_required(self, user: UserModel) -> bool:
        return bool(user.mfa_enforced and user.mfa_secret)

    def verify(self, user: UserModel, code: str) -> bool:
        if not self.is_required(user):
            return True
        if not code:
            return False
        expected = self._totp_codes(user.mfa_secret or "")
        return code in expected

    def challenge(self, user: UserModel) -> None:
        # A real implementation could send push notifications or email. The
        # hook is intentionally left blank but documented.
        del user

    def _totp_codes(self, secret: str) -> set[str]:
        key = _decode_base32(secret)
        codes: set[str] = set()
        timestamp = int(time.time() // self.interval)
        for offset in range(-self.window, self.window + 1):
            counter = timestamp + offset
            counter_bytes = struct.pack(">Q", counter)
            digest = hmac.new(key, counter_bytes, hashlib.sha1).digest()
            pos = digest[-1] & 0x0F
            truncated = digest[pos : pos + 4]
            code_int = int.from_bytes(truncated, "big") & 0x7FFFFFFF
            code = str(code_int % (10**self.digits)).zfill(self.digits)
            codes.add(code)
        return codes


def _decode_base32(value: str) -> bytes:
    padding = "=" * (-len(value) % 8)
    return base64.b32decode((value + padding).upper())


__all__ = [
    "MfaChallengeError",
    "MfaProvider",
    "NullMfaProvider",
    "TotpMfaProvider",
]
