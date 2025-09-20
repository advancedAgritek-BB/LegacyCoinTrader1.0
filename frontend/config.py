"""Secure configuration management for the LegacyCoinTrader frontend."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence

from pydantic import BaseModel, Field, PositiveInt, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from services.common.secret_manager import SecretRetrievalError, resolve_secret

logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]
DEFAULT_SCRIPT_SOURCES = ["'self'", "https://cdn.jsdelivr.net", "'unsafe-inline'"]
DEFAULT_STYLE_SOURCES = [
    "'self'",
    "https://fonts.googleapis.com",
    "https://cdn.jsdelivr.net",
    "https://cdnjs.cloudflare.com",
    "'unsafe-inline'",
]
DEFAULT_FONT_SOURCES = [
    "'self'",
    "https://fonts.googleapis.com",
    "https://fonts.gstatic.com",
    "https://cdnjs.cloudflare.com",
]
DEFAULT_IMG_SOURCES = ["'self'", "data:", "https:"]
DEFAULT_CONNECT_SOURCES = ["*", "http:", "https:", "wss:"]
SAFE_HTTP_METHODS = ("GET", "POST", "OPTIONS", "HEAD")
DISALLOWED_HTTP_METHODS = {"PUT", "DELETE", "PATCH"}


class SecuritySettings(BaseModel):
    """Security configuration and helper methods for the frontend."""

    cors_origins: list[str] = Field(default_factory=lambda: DEFAULT_CORS_ORIGINS.copy())
    csp_default_src: list[str] = Field(default_factory=lambda: ["'self'"])
    csp_script_src: list[str] = Field(default_factory=lambda: DEFAULT_SCRIPT_SOURCES.copy())
    csp_style_src: list[str] = Field(default_factory=lambda: DEFAULT_STYLE_SOURCES.copy())
    csp_img_src: list[str] = Field(default_factory=lambda: DEFAULT_IMG_SOURCES.copy())
    csp_connect_src: list[str] = Field(default_factory=lambda: DEFAULT_CONNECT_SOURCES.copy())
    csp_font_src: list[str] = Field(default_factory=lambda: DEFAULT_FONT_SOURCES.copy())
    rate_limit_requests: PositiveInt = Field(default=100)
    rate_limit_window: PositiveInt = Field(default=60)
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_storage_url: Optional[str] = Field(default=None)
    rate_limit_prefix: str = Field(default="frontend")
    session_secret_key: str = Field(default="")
    session_timeout: PositiveInt = Field(default=3600)
    api_key_header: str = Field(default="X-API-Key")
    allowed_methods: list[str] = Field(default_factory=lambda: list(SAFE_HTTP_METHODS[:3]))

    model_config = {
        "validate_default": True,
        "extra": "ignore",
    }

    @staticmethod
    def _parse_comma_separated(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
            return [item for item in items if item]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("Value must be a list or comma-separated string")

    @field_validator(
        "cors_origins",
        "csp_default_src",
        "csp_script_src",
        "csp_style_src",
        "csp_img_src",
        "csp_connect_src",
        "csp_font_src",
        mode="before",
    )
    def validate_sources(cls, value: Any) -> list[str]:
        return cls._parse_comma_separated(value)

    @field_validator("allowed_methods", mode="before")
    def validate_allowed_methods(cls, value: Any) -> list[str]:
        methods = cls._parse_comma_separated(value)
        if not methods:
            return list(SAFE_HTTP_METHODS[:3])
        unique_methods: list[str] = []
        for method in methods:
            method_upper = method.upper()
            if method_upper in DISALLOWED_HTTP_METHODS:
                raise ValueError(f"HTTP method {method_upper} is not allowed for security reasons")
            if method_upper not in SAFE_HTTP_METHODS:
                raise ValueError(f"Unsupported HTTP method '{method_upper}' in allowed methods")
            if method_upper not in unique_methods:
                unique_methods.append(method_upper)
        return unique_methods

    @field_validator("session_secret_key", mode="before")
    def resolve_session_secret(cls, value: Any) -> str:
        if isinstance(value, str) and value.strip():
            return value
        try:
            return resolve_secret(
                "SESSION_SECRET_KEY",
                env_keys=("SECURITY__SESSION_SECRET_KEY",),
            )
        except SecretRetrievalError as exc:
            raise ValueError(
                "SESSION_SECRET_KEY must be provided via the environment or configured secrets manager"
            ) from exc

    @field_validator("session_secret_key")
    def ensure_session_secret(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Session secret key cannot be empty")
        return value

    @field_validator("session_timeout", "rate_limit_window")
    def ensure_positive_duration(cls, value: PositiveInt) -> PositiveInt:
        if value <= 0:
            raise ValueError("Timeout values must be positive")
        return value

    @field_validator("api_key_header")
    def ensure_api_key_header(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("API key header must be a non-empty string")
        return value.strip()

    def get_csp_header(self) -> str:
        """Build the Content Security Policy header."""

        parts = [
            f"default-src {' '.join(self.csp_default_src)}",
            f"script-src {' '.join(self.csp_script_src)}",
            f"style-src {' '.join(self.csp_style_src)}",
            f"img-src {' '.join(self.csp_img_src)}",
            f"connect-src {' '.join(self.csp_connect_src)}",
            f"font-src {' '.join(self.csp_font_src)}",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        return "; ".join(parts)

    def get_cors_headers(self, origin: str) -> Dict[str, str]:
        """Generate CORS headers for a request origin."""

        headers = {
            "Access-Control-Allow-Methods": ", ".join(self.allowed_methods),
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
            "Access-Control-Max-Age": "86400",
        }

        if origin in self.cors_origins:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
        return headers


class AppSettings(BaseSettings):
    """Application configuration loaded from the environment or secret stores."""

    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    app_name: str = Field(default="LegacyCoinTrader")
    version: str = Field(default="2.0.0")
    host: str = Field(default="0.0.0.0")
    port: PositiveInt = Field(default=5000)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    database_url: str = Field(default="sqlite:///legacy_trader.db")
    redis_host: str = Field(default="localhost")
    redis_port: PositiveInt = Field(default=6379)
    redis_db: int = Field(default=0, ge=0)
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="./logs")

    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore", case_sensitive=False)

    @field_validator("environment")
    def normalise_environment(cls, value: str) -> str:
        allowed = {"development", "staging", "production", "test"}
        value_normalised = value.strip().lower()
        if value_normalised not in allowed:
            raise ValueError(
                f"Environment '{value}' is invalid. Expected one of: {', '.join(sorted(allowed))}"
            )
        return value_normalised

    @field_validator("port", "redis_port")
    def validate_port(cls, value: PositiveInt) -> PositiveInt:
        if not (0 < int(value) <= 65535):
            raise ValueError("Port numbers must be between 1 and 65535")
        return value

    @field_validator("database_url", mode="before")
    def resolve_database_url(cls, value: Any) -> str:
        if isinstance(value, str) and value.strip():
            return value
        return resolve_secret(
            "DATABASE_URL",
            default="sqlite:///legacy_trader.db",
            env_keys=("DATABASE_URL", "APP__DATABASE_URL"),
        )

    @field_validator("database_url")
    def ensure_database_url(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Database URL cannot be empty")
        return value

    @field_validator("log_level")
    def normalise_log_level(cls, value: str) -> str:
        return value.strip().upper() or "INFO"

    @field_validator("log_dir")
    def ensure_log_dir(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Log directory cannot be empty")
        return value.strip()

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        """Return a dictionary representation compatible with legacy callers."""

        if not args and not kwargs:
            return self.model_dump()
        return super().model_dump(*args, **kwargs)


@lru_cache
def get_settings() -> AppSettings:
    """Return a cached :class:`AppSettings` instance."""

    settings = AppSettings()
    logger.debug("Frontend settings loaded: environment=%s", settings.environment)
    return settings


def reload_settings() -> AppSettings:
    """Clear cached settings and reload them."""

    get_settings.cache_clear()
    return get_settings()


try:
    get_settings()
except ValidationError as exc:  # pragma: no cover - executed on import
    logger.error("Configuration validation error: %s", exc)
    raise


__all__ = [
    "AppSettings",
    "SecuritySettings",
    "get_settings",
    "reload_settings",
]
