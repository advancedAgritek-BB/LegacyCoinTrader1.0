"""Async client for interacting with the identity microservice."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin

import httpx

from .config import GatewaySettings

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions shared with FastAPI handlers
# ---------------------------------------------------------------------------


class IdentityServiceError(RuntimeError):
    """Base class for identity service errors."""


class InvalidCredentialsError(IdentityServiceError):
    """Raised when the identity service reports invalid credentials."""


class InactiveAccountError(IdentityServiceError):
    """Raised when an account is disabled."""


class PasswordExpiredError(IdentityServiceError):
    """Raised when a password rotation is required."""


class ApiKeyValidationError(IdentityServiceError):
    """Raised when an API key could not be validated."""


class ApiKeyRotationRequiredError(ApiKeyValidationError):
    """Raised when an API key is past its rotation deadline."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IssuedToken:
    """Represents a JWT pair issued by the identity service."""

    access_token: str
    refresh_token: Optional[str]
    issued_at: datetime
    expires_at: datetime
    username: str
    roles: List[str]
    password_expires_at: Optional[datetime]


@dataclass(slots=True)
class UserIdentity:
    """Simplified user metadata returned by the identity service."""

    username: str
    roles: List[str]
    tenant: Optional[str]
    password_rotated_at: Optional[datetime]
    password_expires_at: Optional[datetime]
    api_key_last_rotated_at: Optional[datetime]
    api_key_expires_at: Optional[datetime]


# ---------------------------------------------------------------------------
# Identity client
# ---------------------------------------------------------------------------


class IdentityService:
    """HTTP client that proxies calls to the identity microservice."""

    def __init__(
        self,
        settings: GatewaySettings,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.settings = settings
        self._base_url = settings.identity_service_url.rstrip("/")
        self._tenant = settings.identity_default_tenant
        self._tenant_header = settings.identity_tenant_header
        self._service_token = settings.identity_service_token
        self._service_token_header = settings.identity_service_token_header
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=settings.identity_request_timeout)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Token issuance and refresh
    # ------------------------------------------------------------------
    async def issue_token(self, username: str, password: str) -> IssuedToken:
        payload = {
            "grant_type": "password",
            "username": username,
            "password": password,
        }
        response = await self._client.post(
            self._url("/oauth/token"),
            json=payload,
            headers=self._headers(),
        )
        if response.status_code == 401:
            raise InvalidCredentialsError("Invalid username or password")
        if response.status_code == 403:
            detail = response.json().get("detail") if response.headers.get("Content-Type", "").startswith("application/json") else response.text
            if detail and "expired" in detail.lower():
                raise PasswordExpiredError(detail)
            raise InactiveAccountError(detail or "Account disabled")
        response.raise_for_status()
        data = response.json()
        return self._parse_issued_token(data)

    async def refresh_token(self, refresh_token: str) -> IssuedToken:
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        response = await self._client.post(
            self._url("/oauth/token"),
            json=payload,
            headers=self._headers(),
        )
        if response.status_code == 401:
            raise InvalidCredentialsError("Refresh token invalid or expired")
        response.raise_for_status()
        data = response.json()
        return self._parse_issued_token(data)

    # ------------------------------------------------------------------
    # Credential maintenance
    # ------------------------------------------------------------------
    async def rotate_password(
        self,
        username: str,
        current_password: str,
        new_password: str,
    ) -> UserIdentity:
        payload = {
            "username": username,
            "current_password": current_password,
            "new_password": new_password,
        }
        response = await self._client.post(
            self._url("/credentials/password/rotate"),
            json=payload,
            headers=self._headers(),
        )
        if response.status_code == 401:
            raise InvalidCredentialsError("Current password is incorrect")
        if response.status_code == 403:
            raise InactiveAccountError(response.json().get("detail", "Account disabled"))
        response.raise_for_status()
        return self._parse_user_identity(response.json())

    async def validate_api_key(self, api_key: str) -> UserIdentity:
        response = await self._client.post(
            self._url("/credentials/api-key/validate"),
            json={"api_key": api_key},
            headers=self._headers(),
        )
        if response.status_code == 403:
            raise ApiKeyRotationRequiredError(response.json().get("detail", "Rotation required"))
        if response.status_code == 401:
            raise ApiKeyValidationError(response.json().get("detail", "Invalid API key"))
        response.raise_for_status()
        return self._parse_user_identity(response.json())

    async def rotate_api_key(
        self, username: str, *, new_api_key: Optional[str] = None
    ) -> tuple[str, UserIdentity]:
        payload = {"username": username}
        if new_api_key:
            payload["new_api_key"] = new_api_key
        response = await self._client.post(
            self._url("/credentials/api-key/rotate"),
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        api_key = data.get("api_key") or ""
        user = self._parse_user_identity(data)
        return api_key, user

    async def list_users(self) -> List[UserIdentity]:
        headers = self._headers(include_service_token=True)
        response = await self._client.get(self._url("/scim/v2/Users"), headers=headers)
        if response.status_code == 401:
            raise IdentityServiceError("Service token required to list users")
        response.raise_for_status()
        payload = response.json()
        resources = payload.get("Resources", [])
        identities: List[UserIdentity] = []
        for resource in resources:
            identities.append(
                UserIdentity(
                    username=resource.get("userName", ""),
                    roles=[role.get("value") for role in resource.get("roles", []) if role.get("value")],
                    tenant=self._tenant,
                    password_rotated_at=None,
                    password_expires_at=None,
                    api_key_last_rotated_at=None,
                    api_key_expires_at=None,
                )
            )
        return identities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _headers(self, *, include_service_token: bool = False) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._tenant:
            headers[self._tenant_header] = self._tenant
        if self._service_token and (include_service_token or self._service_token_header):
            headers[self._service_token_header] = self._service_token
        return headers

    def _url(self, path: str) -> str:
        return urljoin(f"{self._base_url}/", path.lstrip("/"))

    def _parse_issued_token(self, payload: dict) -> IssuedToken:
        issued_at = self._parse_datetime(payload.get("issued_at"))
        expires_at = self._parse_datetime(payload.get("expires_at"))
        password_expires_at = self._parse_datetime(payload.get("password_expires_at"))
        roles = payload.get("roles") or []
        if isinstance(roles, str):
            roles = [role for role in roles.split() if role]
        return IssuedToken(
            access_token=payload.get("access_token", ""),
            refresh_token=payload.get("refresh_token"),
            issued_at=issued_at,
            expires_at=expires_at,
            username=payload.get("username", ""),
            roles=list(roles),
            password_expires_at=password_expires_at,
        )

    def _parse_user_identity(self, payload: dict) -> UserIdentity:
        return UserIdentity(
            username=payload.get("username", ""),
            roles=[str(role) for role in payload.get("roles", [])],
            tenant=payload.get("tenant") or self._tenant,
            password_rotated_at=self._parse_datetime(payload.get("password_rotated_at")),
            password_expires_at=self._parse_datetime(payload.get("password_expires_at")),
            api_key_last_rotated_at=self._parse_datetime(payload.get("api_key_last_rotated_at")),
            api_key_expires_at=self._parse_datetime(payload.get("api_key_expires_at")),
        )

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            LOGGER.debug("Failed to parse datetime value '%s'", value)
            return None


__all__ = [
    "ApiKeyRotationRequiredError",
    "ApiKeyValidationError",
    "IdentityService",
    "IssuedToken",
    "InactiveAccountError",
    "InvalidCredentialsError",
    "PasswordExpiredError",
    "UserIdentity",
]
