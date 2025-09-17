from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import jwt

from services.portfolio.config import PortfolioConfig
from services.portfolio.identity import (
    ApiKeyRotationRequiredError,
    ApiKeyValidationError,
    InactiveAccountError,
    InvalidCredentialsError,
    PasswordExpiredError,
    PortfolioIdentityService,
    UserIdentity,
)

from .config import GatewaySettings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IssuedToken:
    """Represents a JWT issued by the identity service."""

    access_token: str
    expires_at: datetime
    username: str
    roles: List[str]
    password_expires_at: Optional[datetime]


class IdentityService:
    """Bridge between the API gateway and the portfolio identity store."""

    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings
        self.identity_store = PortfolioIdentityService(PortfolioConfig.from_env())

    # ------------------------------------------------------------------
    # Token issuance
    # ------------------------------------------------------------------
    def issue_token(self, username: str, password: str) -> IssuedToken:
        """Validate credentials and mint a JWT for downstream requests."""

        identity = self.identity_store.authenticate_user(username, password)
        issued_at = datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(seconds=max(60, self.settings.token_ttl_seconds))

        roles = identity.roles or ["user"]
        payload = {
            "sub": identity.username,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
            "scopes": roles,
            "roles": roles,
            "iss": self.settings.token_issuer,
        }
        if self.settings.jwt_audience:
            payload["aud"] = self.settings.jwt_audience

        token = jwt.encode(
            payload,
            self.settings.jwt_secret,
            algorithm=self.settings.jwt_algorithm,
        )

        LOGGER.info("Issued access token for %s", identity.username)
        password_expires_at = (
            identity.password_expires_at.replace(tzinfo=timezone.utc)
            if identity.password_expires_at
            else None
        )
        return IssuedToken(
            access_token=token,
            expires_at=expires_at,
            username=identity.username,
            roles=roles,
            password_expires_at=password_expires_at,
        )

    # ------------------------------------------------------------------
    # Credential maintenance
    # ------------------------------------------------------------------
    def rotate_password(
        self,
        username: str,
        current_password: str,
        new_password: str,
    ) -> UserIdentity:
        return self.identity_store.rotate_password(username, current_password, new_password)

    def validate_api_key(self, api_key: str) -> UserIdentity:
        return self.identity_store.validate_api_key(api_key)

    def rotate_api_key(
        self, username: str, *, new_api_key: Optional[str] = None
    ) -> tuple[str, UserIdentity]:
        result = self.identity_store.rotate_api_key(username, new_api_key=new_api_key)
        return result.api_key, result.user

    def list_users(self) -> List[UserIdentity]:
        return self.identity_store.list_users()


__all__ = [
    "ApiKeyRotationRequiredError",
    "ApiKeyValidationError",
    "IdentityService",
    "InactiveAccountError",
    "InvalidCredentialsError",
    "IssuedToken",
    "PasswordExpiredError",
]
