"""Pydantic schemas exposed by the identity service."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TokenPair(BaseModel):
    """Response returned when issuing or refreshing tokens."""

    access_token: str = Field(description="Bearer token used for API access")
    refresh_token: str = Field(description="Opaque token used to obtain new access tokens")
    token_type: str = Field(default="Bearer")
    expires_in: int = Field(description="Lifetime of the access token in seconds")
    expires_at: datetime = Field(description="Timestamp when the access token expires")
    issued_at: datetime = Field(description="Timestamp when the token pair was minted")
    scope: List[str] = Field(default_factory=list, description="Scopes granted to the caller")
    roles: List[str] = Field(default_factory=list, description="Roles granted to the caller")
    username: str = Field(description="Username associated with the token")
    tenant: str = Field(description="Tenant identifier for the subject")
    password_expires_at: Optional[datetime] = None


class IntrospectionResponse(BaseModel):
    """Response payload for token introspection."""

    active: bool
    username: Optional[str] = None
    subject: Optional[str] = None
    tenant: Optional[str] = None
    scope: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    token_type: Optional[str] = None
    client_id: Optional[str] = None
    claims: Dict[str, object] = Field(default_factory=dict)


class CredentialRotationRequest(BaseModel):
    """Payload for rotating a password or API key."""

    username: str
    current_secret: Optional[str] = Field(default=None)
    new_secret: Optional[str] = Field(default=None)


class CredentialRotationResponse(BaseModel):
    """Metadata returned after rotating credentials."""

    username: str
    roles: List[str]
    rotated_at: datetime
    expires_at: Optional[datetime]
    tenant: str


class ApiKeyValidationResponse(BaseModel):
    """Response returned when validating an API key."""

    username: str
    roles: List[str]
    tenant: str
    api_key_last_rotated_at: Optional[datetime]
    api_key_expires_at: Optional[datetime]


class ScimMeta(BaseModel):
    """Subset of SCIM metadata returned with user resources."""

    resourceType: str = Field(default="User")
    created: datetime
    lastModified: datetime
    location: Optional[str] = None


class ScimName(BaseModel):
    """SCIM name representation."""

    givenName: Optional[str] = None
    familyName: Optional[str] = None
    formatted: Optional[str] = None


class ScimEmail(BaseModel):
    """SCIM email representation."""

    value: str
    primary: bool = True
    type: str = "work"


class ScimUser(BaseModel):
    """Simplified SCIM user representation."""

    model_config = ConfigDict(extra="allow")

    id: str
    externalId: Optional[str] = None
    userName: str
    name: Optional[ScimName] = None
    active: bool = True
    emails: List[ScimEmail] = Field(default_factory=list)
    displayName: Optional[str] = None
    roles: List[Dict[str, str]] = Field(default_factory=list)
    meta: Optional[ScimMeta] = None


__all__ = [
    "ApiKeyValidationResponse",
    "CredentialRotationRequest",
    "CredentialRotationResponse",
    "IntrospectionResponse",
    "ScimEmail",
    "ScimMeta",
    "ScimName",
    "ScimUser",
    "TokenPair",
]
