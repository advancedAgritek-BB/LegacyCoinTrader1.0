"""Configuration helpers for the API gateway service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List

from services.portfolio.rbac import load_role_definitions

DEFAULT_JWT_ALGORITHM = "HS256"
DEFAULT_TOKEN_EXPIRATION_MINUTES = 60


def _load_cors_origins() -> List[str]:
    value = os.getenv("API_GATEWAY_CORS_ORIGINS", "")
    return [origin.strip() for origin in value.split(",") if origin.strip()]


def _load_role_definitions() -> Dict[str, List[str]]:
    try:
        return load_role_definitions()
    except ValueError as exc:
        raise ValueError(f"Invalid ROLE_DEFINITIONS value: {exc}") from exc


@dataclass
class GatewaySecurityConfig:
    jwt_secret: str = field(
        default_factory=lambda: os.getenv(
            "API_GATEWAY_JWT_SECRET",
            os.getenv("SESSION_SECRET_KEY", "legacycointrader-change-me"),
        )
    )
    jwt_algorithm: str = field(
        default_factory=lambda: os.getenv(
            "API_GATEWAY_JWT_ALGORITHM", DEFAULT_JWT_ALGORITHM
        )
    )
    access_token_expiration_minutes: int = field(
        default_factory=lambda: int(
            os.getenv(
                "API_GATEWAY_ACCESS_TOKEN_MINUTES",
                str(DEFAULT_TOKEN_EXPIRATION_MINUTES),
            )
        )
    )
    cors_origins: List[str] = field(default_factory=_load_cors_origins)
    role_definitions: Dict[str, List[str]] = field(default_factory=_load_role_definitions)

    @property
    def token_lifetime(self) -> timedelta:
        return timedelta(minutes=self.access_token_expiration_minutes)


@dataclass
class GatewayConfig:
    host: str = field(default_factory=lambda: os.getenv("API_GATEWAY_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_GATEWAY_PORT", "8000")))
    issuer: str = field(
        default_factory=lambda: os.getenv("API_GATEWAY_ISSUER", "legacycointrader-gateway")
    )
    audience: str = field(
        default_factory=lambda: os.getenv("API_GATEWAY_AUDIENCE", "legacycointrader-clients")
    )
    security: GatewaySecurityConfig = field(default_factory=GatewaySecurityConfig)


__all__ = ["GatewayConfig", "GatewaySecurityConfig"]
