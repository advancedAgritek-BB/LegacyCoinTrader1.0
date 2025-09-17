"""Helper utilities for validating identity-issued tokens."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import jwt
from jwt import InvalidTokenError, PyJWKClient

DEFAULT_TENANT_HEADER = "X-Tenant-ID"


@dataclass(slots=True)
class TokenClaims:
    """Normalised view over JWT/OIDC claims."""

    subject: str
    tenant: Optional[str]
    roles: List[str]
    scopes: List[str]
    expires_at: datetime
    issued_at: datetime
    raw_claims: Dict[str, object]

    @property
    def is_expired(self) -> bool:
        return self.expires_at < datetime.now(timezone.utc)


class IdentityTokenValidator:
    """Validate JWT tokens issued by the identity service using JWKS."""

    def __init__(
        self,
        jwks_url: str,
        *,
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self.jwks_client = PyJWKClient(jwks_url, cache_keys=True, lifespan=cache_ttl_seconds)
        self.issuer = issuer
        self.audience = audience

    def validate(self, token: str) -> TokenClaims:
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=[signing_key.algorithm],
                audience=self.audience,
                issuer=self.issuer,
                options={
                    "verify_aud": self.audience is not None,
                    "verify_iss": self.issuer is not None,
                },
            )
        except InvalidTokenError as exc:
            raise InvalidTokenError(f"Identity token validation failed: {exc}") from exc
        return self._claims_from_payload(payload)

    def _claims_from_payload(self, payload: Dict[str, object]) -> TokenClaims:
        scopes_raw = payload.get("scope") or payload.get("scopes") or []
        if isinstance(scopes_raw, str):
            scopes = [scope for scope in scopes_raw.split() if scope]
        else:
            scopes = [str(scope) for scope in scopes_raw]
        roles_raw = payload.get("roles") or []
        if isinstance(roles_raw, str):
            roles = [role for role in roles_raw.split() if role]
        else:
            roles = [str(role) for role in roles_raw]
        issued_at = datetime.fromtimestamp(int(payload.get("iat", 0)), tz=timezone.utc)
        expires_at = datetime.fromtimestamp(int(payload.get("exp", 0)), tz=timezone.utc)
        return TokenClaims(
            subject=str(payload.get("sub")),
            tenant=str(payload.get("tenant") or payload.get("tid")) if payload.get("tenant") or payload.get("tid") else None,
            roles=roles,
            scopes=scopes,
            issued_at=issued_at,
            expires_at=expires_at,
            raw_claims=dict(payload),
        )


class IdentityIntrospectionClient:
    """Client for the identity service token introspection endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        service_token: str,
        tenant_header: str = DEFAULT_TENANT_HEADER,
        default_tenant: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_token = service_token
        self.tenant_header = tenant_header
        self.default_tenant = default_tenant
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def introspect(self, token: str, *, tenant: Optional[str] = None) -> Dict[str, object]:
        headers = {"Content-Type": "application/json", DEFAULT_TENANT_HEADER: tenant or self.default_tenant or ""}
        if self.tenant_header:
            headers[self.tenant_header] = tenant or self.default_tenant or ""
        headers["x-service-token"] = self.service_token
        response = self._client.post("/oauth/introspect", json={"token": token}, headers=headers)
        response.raise_for_status()
        return response.json()

    def __enter__(self) -> "IdentityIntrospectionClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "DEFAULT_TENANT_HEADER",
    "IdentityIntrospectionClient",
    "IdentityTokenValidator",
    "TokenClaims",
]
