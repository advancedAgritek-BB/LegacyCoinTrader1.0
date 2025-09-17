from __future__ import annotations

"""Data contracts exposed by the API gateway service."""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, constr

from services.common.contracts import EventEnvelope, HttpEndpoint


class TokenRequest(BaseModel):
    """Incoming credentials for token issuance."""

    username: constr(min_length=1, strip_whitespace=True)
    password: constr(min_length=1)


class TokenResponse(BaseModel):
    """JWT payload returned to authenticated clients."""

    access_token: str
    token_type: str = Field(default="bearer")
    expires_at: datetime
    username: str
    roles: List[str]
    password_expires_at: Optional[datetime] = None


class PasswordRotationRequest(BaseModel):
    username: constr(min_length=1, strip_whitespace=True)
    current_password: constr(min_length=1)
    new_password: constr(min_length=8)


class PasswordRotationResponse(BaseModel):
    username: str
    roles: List[str]
    password_rotated_at: datetime
    password_expires_at: Optional[datetime] = None


class ApiKeyValidationRequest(BaseModel):
    api_key: constr(min_length=1)


class ApiKeyValidationResponse(BaseModel):
    username: str
    roles: List[str]
    api_key_last_rotated_at: Optional[datetime] = None


class AuthenticationPayload(BaseModel):
    """Structured payload emitted when an authentication action occurs."""

    username: str
    successful: bool
    roles: List[str] = Field(default_factory=list)
    subject: str = Field(default="user")
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuthenticationEvent(EventEnvelope):
    """Event published whenever a token or API key action is performed."""

    event_type: str = Field(default="api-gateway.authentication", const=True)
    payload: AuthenticationPayload


HTTP_CONTRACT: List[HttpEndpoint] = [
    HttpEndpoint(
        method="POST",
        path="/auth/token",
        summary="Issue access tokens for REST/gRPC clients",
        request_model="services.api_gateway.contracts.TokenRequest",
        response_model="services.api_gateway.contracts.TokenResponse",
    ),
    HttpEndpoint(
        method="POST",
        path="/auth/password/rotate",
        summary="Rotate user passwords",
        request_model="services.api_gateway.contracts.PasswordRotationRequest",
        response_model="services.api_gateway.contracts.PasswordRotationResponse",
    ),
    HttpEndpoint(
        method="POST",
        path="/auth/api-key/validate",
        summary="Validate an API key issued by the gateway",
        request_model="services.api_gateway.contracts.ApiKeyValidationRequest",
        response_model="services.api_gateway.contracts.ApiKeyValidationResponse",
    ),
]


__all__ = [
    "ApiKeyValidationRequest",
    "ApiKeyValidationResponse",
    "AuthenticationEvent",
    "AuthenticationPayload",
    "HTTP_CONTRACT",
    "PasswordRotationRequest",
    "PasswordRotationResponse",
    "TokenRequest",
    "TokenResponse",
]
