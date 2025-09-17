from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import jwt
from fastapi import HTTPException, Request, status
from jwt import InvalidTokenError

from services.identity.auth import IdentityTokenValidator

from .config import GatewaySettings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TokenPayload:
    """Represents authentication context for the current request."""

    token_type: str
    subject: str
    scopes: List[str]
    tenant_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    raw_token: Optional[str] = None
    service_name: Optional[str] = None
    claims: Optional[Dict[str, object]] = None
    client_host: Optional[str] = None

    @property
    def rate_limit_key(self) -> str:
        tenant_prefix = f"tenant:{self.tenant_id}:" if self.tenant_id else ""
        if self.token_type == "jwt":
            return f"{tenant_prefix}user:{self.subject}"
        if self.token_type == "service" and self.service_name:
            return f"{tenant_prefix}service:{self.service_name}"
        if self.client_host:
            return f"{tenant_prefix}ip:{self.client_host}"
        return f"{tenant_prefix}anonymous" if tenant_prefix else "anonymous"


class AuthManager:
    """Authentication helper for validating incoming requests."""

    def __init__(self, settings: GatewaySettings):
        self.settings = settings
        self._service_tokens = {
            token: name for name, token in settings.service_tokens.items() if token
        }

        self._token_validator: Optional[IdentityTokenValidator] = None
        if settings.identity_jwks_url:
            try:
                self._token_validator = IdentityTokenValidator(
                    settings.identity_jwks_url,
                    issuer=settings.identity_issuer,
                    audience=settings.identity_audience,
                    cache_ttl_seconds=settings.identity_jwks_cache_seconds,
                )
            except Exception as exc:  # pragma: no cover - defensive guardrail
                LOGGER.warning("Failed to initialise identity token validator: %s", exc)
        else:
            LOGGER.warning(
                "IDENTITY_JWKS_URL not configured; falling back to local JWT validation"
            )

        if not self.settings.service_tokens:
            LOGGER.warning(
                "No service tokens configured. Downstream communication will not be secured."
            )

    async def authenticate_request(
        self, request: Request, allowed_modes: Iterable[str]
    ) -> TokenPayload:
        """Validate the request against the configured authentication mechanisms."""

        if not self.settings.require_authentication:
            client_host = request.client.host if request.client else None
            return TokenPayload(
                token_type="anonymous",
                subject="anonymous",
                scopes=[],
                roles=[],
                client_host=client_host,
            )

        modes = list(allowed_modes)

        if "service" in modes:
            token_payload = self._validate_service_token(request)
            if token_payload:
                return token_payload

        if "jwt" in modes:
            token_payload = self._validate_jwt_token(request)
            if token_payload:
                return token_payload

        if "anonymous" in modes:
            client_host = request.client.host if request.client else None
            return TokenPayload(
                token_type="anonymous",
                subject="anonymous",
                scopes=[],
                roles=[],
                client_host=client_host,
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authentication credentials",
        )

    def _validate_service_token(self, request: Request) -> Optional[TokenPayload]:
        header_keys = ["x-service-token", "x-internal-token", "x-api-gateway-token"]
        for header in header_keys:
            token = request.headers.get(header)
            if token:
                service_name = self._service_tokens.get(token)
                if not service_name:
                    LOGGER.warning("Rejected request with invalid service token")
                    break
                scopes = ["internal", f"service:{service_name}"]
                roles = [f"service:{service_name}"]
                return TokenPayload(
                    token_type="service",
                    subject=service_name,
                    scopes=scopes,
                    roles=roles,
                    service_name=service_name,
                    raw_token=token,
                    client_host=request.client.host if request.client else None,
                )
        return None

    def _validate_jwt_token(self, request: Request) -> Optional[TokenPayload]:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None

        try:
            scheme, token = auth_header.split(" ", 1)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format",
            )

        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unsupported authorization scheme",
            )

        if self._token_validator:
            try:
                claims = self._token_validator.validate(token)
            except InvalidTokenError as exc:
                LOGGER.warning("JWT validation failed: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                ) from exc
            scopes = list(claims.scopes)
            self._ensure_tenant_scope(claims.tenant, scopes)
            return TokenPayload(
                token_type="jwt",
                subject=claims.subject or "user",
                scopes=scopes,
                tenant_id=claims.tenant,
                roles=list(claims.roles),
                raw_token=token,
                claims=claims.raw_claims,
                client_host=request.client.host if request.client else None,
            )

        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret,
                algorithms=[self.settings.jwt_algorithm],
                audience=self.settings.jwt_audience,
                options={"verify_aud": bool(self.settings.jwt_audience)},
            )
        except InvalidTokenError as exc:
            LOGGER.warning("JWT validation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )

        scopes = payload.get("scopes") or payload.get("scope") or []
        if isinstance(scopes, str):
            scopes = [scope for scope in scopes.split() if scope]

        subject = str(payload.get("sub") or payload.get("user_id") or "user")
        tenant = payload.get("tenant") or payload.get("tid")
        tenant_id = str(tenant) if tenant else None
        roles_raw = payload.get("roles") or payload.get("role") or []
        if isinstance(roles_raw, str):
            roles = [role for role in roles_raw.split() if role]
        else:
            roles = [str(role) for role in roles_raw]
        scopes_list = list(scopes)
        self._ensure_tenant_scope(tenant_id, scopes_list)
        return TokenPayload(
            token_type="jwt",
            subject=subject,
            scopes=scopes_list,
            tenant_id=tenant_id,
            roles=roles,
            raw_token=token,
            claims=payload,
            client_host=request.client.host if request.client else None,
        )

    @staticmethod
    def _ensure_tenant_scope(tenant: Optional[str], scopes: List[str]) -> None:
        if not tenant:
            return
        normalised = {scope.lower() for scope in scopes}
        if f"tenant:{tenant}".lower() in normalised:
            return
        if "tenant:*" in normalised or "tenant:all" in normalised:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token lacks required tenant scope",
        )

