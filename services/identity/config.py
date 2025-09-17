"""Configuration helpers for the identity service."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IdentitySettings(BaseSettings):
    """Runtime configuration for the identity microservice."""

    database_url: str = Field(
        default="sqlite:///./identity.db",
        description="SQLAlchemy database URL",
    )
    access_token_ttl_seconds: int = Field(
        default=3600,
        description="Default lifetime for access tokens in seconds.",
    )
    refresh_token_ttl_seconds: int = Field(
        default=86_400,
        description="Lifetime for refresh tokens in seconds.",
    )
    token_algorithm: str = Field(
        default="RS256",
        description="JWT signing algorithm used for issued tokens.",
    )
    default_issuer: str = Field(
        default="https://identity.legacycointrader.local",
        description="Issuer identifier used for newly created tenants.",
    )
    default_tenant_slug: str = Field(
        default="primary",
        description="Slug used when a request omits explicit tenant information.",
    )
    service_token_header: str = Field(
        default="x-service-token",
        description="Header used to authenticate service-to-service requests.",
    )
    internal_service_token: Optional[str] = Field(
        default=None,
        description="Shared secret that downstream services must present when invoking privileged endpoints.",
    )
    allow_development_fallback_keys: bool = Field(
        default=True,
        description="Generate ephemeral signing keys when external secret stores are unavailable (development only).",
    )
    jwks_cache_seconds: int = Field(
        default=300,
        description="How long JWKS responses should be cached by clients.",
    )
    tenant_header_candidates: List[str] = Field(
        default_factory=lambda: ["x-tenant-id", "x-tenant", "x-realm"],
        description="Headers inspected to determine the active tenant for a request.",
    )
    scim_enabled: bool = Field(
        default=True,
        description="Whether the SCIM provisioning API should be exposed.",
    )
    model_config = SettingsConfigDict(
        env_prefix="IDENTITY_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def load_identity_settings() -> IdentitySettings:
    """Return cached identity settings."""

    return IdentitySettings()


__all__ = [
    "IdentitySettings",
    "load_identity_settings",
]
