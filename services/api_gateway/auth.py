from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import jwt
from fastapi import HTTPException, Request, status
from jwt import InvalidTokenError

from .config import GatewaySettings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TokenPayload:
    """Represents authentication context for the current request."""

    token_type: str
    subject: str
    scopes: List[str]
    raw_token: Optional[str] = None
    service_name: Optional[str] = None
    claims: Optional[Dict[str, object]] = None
    client_host: Optional[str] = None

    @property
    def rate_limit_key(self) -> str:
        if self.token_type == "jwt":
            return f"user:{self.subject}"
        if self.token_type == "service" and self.service_name:
            return f"service:{self.service_name}"
        if self.client_host:
            return f"ip:{self.client_host}"
        return "anonymous"


class AuthManager:
    """Authentication helper for validating incoming requests."""

    def __init__(self, settings: GatewaySettings):
        self.settings = settings
        self._service_tokens = {
            token: name for name, token in settings.service_tokens.items() if token
        }

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
                token_type="anonymous", subject="anonymous", scopes=[], client_host=client_host
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
                return TokenPayload(
                    token_type="service",
                    subject=service_name,
                    scopes=scopes,
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
        return TokenPayload(
            token_type="jwt",
            subject=subject,
            scopes=list(scopes),
            raw_token=token,
            claims=payload,
            client_host=request.client.host if request.client else None,
        )

