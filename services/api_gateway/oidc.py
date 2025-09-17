from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import httpx
import jwt
from jwt import InvalidTokenError

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OidcConfiguration:
    """Static configuration for an OIDC issuer."""

    issuer: str
    jwks_url: str
    audience: Optional[str] = None
    cache_ttl: int = 300


class OidcValidationError(Exception):
    """Raised when an OIDC token cannot be validated."""


class OidcValidator:
    """Helper for validating OIDC tokens via cached JWKS metadata."""

    def __init__(self, config: OidcConfiguration, http_client: httpx.AsyncClient) -> None:
        self.config = config
        self.http_client = http_client
        self._jwks_cache: Dict[str, str] = {}
        self._jwks_loaded_at: float = 0.0

    async def decode(self, token: str) -> Dict[str, object]:
        """Decode a JWT using keys advertised by the OIDC issuer."""

        unverified_header = jwt.get_unverified_header(token)
        algorithm = unverified_header.get("alg", "RS256")
        kid = unverified_header.get("kid")

        jwks_map = await self._load_jwks()
        key_data = None
        if kid and kid in jwks_map:
            key_data = jwks_map[kid]
        elif jwks_map:
            key_data = next(iter(jwks_map.values()))

        if not key_data:
            raise OidcValidationError("No signing keys available for token verification")

        try:
            algorithm_impl = jwt.algorithms.get_default_algorithms()[algorithm]
        except KeyError as exc:  # pragma: no cover - defensive path
            raise OidcValidationError(f"Unsupported signing algorithm: {algorithm}") from exc

        try:
            public_key = algorithm_impl.from_jwk(key_data)
        except Exception as exc:  # pragma: no cover - unexpected JWK formats
            raise OidcValidationError("Failed to construct public key from JWKS") from exc

        options = {"verify_aud": bool(self.config.audience)}
        try:
            claims = jwt.decode(
                token,
                public_key,
                algorithms=[algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options=options,
            )
        except InvalidTokenError as exc:
            raise OidcValidationError(str(exc)) from exc

        return claims

    async def _load_jwks(self) -> Dict[str, str]:
        now = time.monotonic()
        if self._jwks_cache and now - self._jwks_loaded_at < self.config.cache_ttl:
            return self._jwks_cache

        try:
            response = await self.http_client.get(self.config.jwks_url, timeout=5)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.error("Failed to retrieve JWKS from %s: %s", self.config.jwks_url, exc)
            raise OidcValidationError("Unable to retrieve OIDC signing keys") from exc

        data = response.json()
        keys = data.get("keys") or []
        jwks_map: Dict[str, str] = {}
        for idx, key in enumerate(keys):
            if not isinstance(key, dict):
                continue
            kid = key.get("kid") or f"key-{idx}"
            jwks_map[kid] = json.dumps(key)

        if not jwks_map:
            LOGGER.error("OIDC JWKS response did not include any keys")
            raise OidcValidationError("OIDC JWKS response missing keys")

        self._jwks_cache = jwks_map
        self._jwks_loaded_at = now
        return jwks_map


__all__ = [
    "OidcConfiguration",
    "OidcValidationError",
    "OidcValidator",
]
