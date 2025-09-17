from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import jwt
from fastapi import HTTPException, Request, status
from jwt import InvalidTokenError

from .config import GatewaySettings
from .oidc import OidcValidationError, OidcValidator

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TokenPayload:
    """Represents authentication context for the current request."""

    token_type: str
    subject: str
    scopes: List[str]
    raw_token: Optional[str] = None
    service_name: Optional[str] = None
    claims: Optional[Dict[str, Any]] = None
    client_host: Optional[str] = None
    tenant_id: Optional[str] = None
    tenant_slug: Optional[str] = None
    tenant_scopes: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    tenant_plan: Optional[str] = None

    @property
    def rate_limit_key(self) -> str:
        if self.tenant_id and self.token_type in {"jwt", "oidc"}:
            return f"tenant:{self.tenant_id}:user:{self.subject}"
        if self.token_type in {"jwt", "oidc"}:
            return f"user:{self.subject}"
        if self.token_type == "service" and self.service_name:
            return f"service:{self.service_name}"
        if self.client_host:
            return f"ip:{self.client_host}"
        return "anonymous"

    @property
    def tenant_rate_limit_key(self) -> Optional[str]:
        if self.tenant_id:
            return f"tenant:{self.tenant_id}"
        return None


class AuthManager:
    """Authentication helper for validating incoming requests."""

    def __init__(
        self, settings: GatewaySettings, *, oidc_validator: Optional[OidcValidator] = None
    ) -> None:
        self.settings = settings
        self.oidc_validator = oidc_validator
        self._service_tokens = {
            token: name for name, token in settings.service_tokens.items() if token
        }

        if not self.settings.service_tokens:
            LOGGER.warning(
                "No service tokens configured. Downstream communication will not be secured."
            )
        if "oidc" in {mode for route in settings.service_routes.values() for mode in route.allowed_auth_modes} and not self.oidc_validator:
            LOGGER.warning(
                "OIDC authentication requested by route configuration but no validator was provided."
            )

    async def authenticate_request(
        self, request: Request, allowed_modes: Iterable[str]
    ) -> TokenPayload:
        """Validate the request against the configured authentication mechanisms."""

        if not self.settings.require_authentication:
            client_host = request.client.host if request.client else None
            return TokenPayload(
                token_type="anonymous", subject="anonymous", scopes=[], client_host=client_host
            )

        modes = list(allowed_modes)

        if "service" in modes:
            token_payload = self._validate_service_token(request)
            if token_payload:
                return token_payload

        if "oidc" in modes:
            token_payload = await self._validate_oidc_token(request)
            if token_payload:
                return token_payload

        if "jwt" in modes:
            token_payload = self._validate_jwt_token(request)
            if token_payload:
                return token_payload

        if "anonymous" in modes:
            client_host = request.client.host if request.client else None
            return TokenPayload(
                token_type="anonymous", subject="anonymous", scopes=[], client_host=client_host
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
                    service_name=service_name,
                    raw_token=token,
                    client_host=request.client.host if request.client else None,
                    roles=roles,
                )
        return None

    def _extract_bearer_token(
        self, request: Request, *, required: bool = False
    ) -> Optional[str]:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            if required:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing Authorization header",
                )
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
        return token.strip() or None

    @staticmethod
    def _normalize_scope_claim(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part for part in re.split(r"[\s,]+", value) if part]
        if isinstance(value, (list, tuple, set)):
            return [str(part) for part in value if str(part)]
        return []

    async def _validate_oidc_token(self, request: Request) -> Optional[TokenPayload]:
        if not self.oidc_validator:
            return None

        token = self._extract_bearer_token(request)
        if not token:
            return None

        try:
            claims = await self.oidc_validator.decode(token)
        except OidcValidationError as exc:
            LOGGER.warning("OIDC validation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            ) from exc

        scopes = self._normalize_scope_claim(
            claims.get("scopes") or claims.get("scope")
        )
        tenant_scopes = self._normalize_scope_claim(claims.get("tenant_scopes"))
        roles = self._normalize_scope_claim(claims.get("roles") or scopes)

        subject = str(claims.get("sub") or claims.get("user_id") or "user")
        tenant_id = claims.get("tenant_id") or claims.get("tenant")
        tenant_slug = claims.get("tenant_slug") or None
        tenant_plan = claims.get("tenant_plan") or None

        return TokenPayload(
            token_type="oidc",
            subject=subject,
            scopes=scopes,
            raw_token=token,
            claims=claims,
            client_host=request.client.host if request.client else None,
            tenant_id=str(tenant_id) if tenant_id else None,
            tenant_slug=str(tenant_slug) if tenant_slug else None,
            tenant_scopes=tenant_scopes or scopes,
            roles=roles or scopes,
            tenant_plan=str(tenant_plan) if tenant_plan else None,
        )

    def _validate_jwt_token(self, request: Request) -> Optional[TokenPayload]:
        token = self._extract_bearer_token(request)
        if not token:
            return None

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

        scopes = self._normalize_scope_claim(payload.get("scopes") or payload.get("scope"))
        tenant_scopes = self._normalize_scope_claim(payload.get("tenant_scopes"))
        roles = self._normalize_scope_claim(payload.get("roles")) or scopes
        subject = str(payload.get("sub") or payload.get("user_id") or "user")
        tenant_id = payload.get("tenant_id") or payload.get("tenant")
        tenant_slug = payload.get("tenant_slug") or None
        tenant_plan = payload.get("tenant_plan") or None

        return TokenPayload(
            token_type="jwt",
            subject=subject,
            scopes=scopes,
            raw_token=token,
            claims=payload,
            client_host=request.client.host if request.client else None,
            tenant_id=str(tenant_id) if tenant_id else None,
            tenant_slug=str(tenant_slug) if tenant_slug else None,
            tenant_scopes=tenant_scopes or scopes,
            roles=roles,
            tenant_plan=str(tenant_plan) if tenant_plan else None,
        )

