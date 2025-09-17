"""Secure configuration management for the LegacyCoinTrader frontend."""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
from functools import lru_cache
from typing import Any, Dict, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, PositiveInt, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
]
DEFAULT_FONT_SOURCES = [
    "'self'",
    "https://fonts.googleapis.com",
    "https://fonts.gstatic.com",
    "https://cdnjs.cloudflare.com",
]
DEFAULT_IMG_SOURCES = ["'self'", "data:", "https:"]
DEFAULT_CONNECT_SOURCES = ["'self'", "https:", "wss:"]
SAFE_HTTP_METHODS = ("GET", "POST", "OPTIONS", "HEAD")
DISALLOWED_HTTP_METHODS = {"PUT", "DELETE", "PATCH"}


class SecretRetrievalError(RuntimeError):
    """Raised when a secret cannot be resolved."""


class SecretProvider(Protocol):
    """Protocol for secret providers."""

    def get_secret(self, key: str, *, path: Optional[str] = None) -> Optional[str]:
        """Return the secret value for *key* if available."""


class HashicorpVaultSecretProvider:
    """Load secrets from a Hashicorp Vault KV store."""

    def __init__(
        self,
        address: str,
        token: str,
        secret_path: str,
        *,
        timeout: float = 5.0,
        verify: Optional[bool] = None,
    ) -> None:
        if not address or not token:
            raise ValueError("Hashicorp Vault provider requires address and token")

        self.address = address.rstrip("/")
        self.token = token
        self.secret_path = secret_path.strip("/")
        self.timeout = timeout
        self.verify = verify
        self._cache: Dict[str, Optional[str]] = {}

    def get_secret(self, key: str, *, path: Optional[str] = None) -> Optional[str]:
        import requests

        secret_path = (path or self.secret_path).strip("/")
        if not secret_path:
            raise SecretRetrievalError("Vault secret path cannot be empty")

        cache_key = f"{secret_path}:{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{self.address}/v1/{secret_path}"
        headers = {"X-Vault-Token": self.token}
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify if self.verify is not None else True,
            )
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network and TLS errors are runtime specific
            raise SecretRetrievalError(f"Vault request failed for path '{secret_path}': {exc}") from exc

        try:
            payload = response.json()
        except Exception as exc:
            raise SecretRetrievalError(
                f"Vault response for path '{secret_path}' is not valid JSON: {exc}"
            ) from exc

        data: Any = payload.get("data", {})
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise SecretRetrievalError(
                f"Vault response for path '{secret_path}' does not contain secret data"
            )

        raw_value = data.get(key)
        if raw_value is None:
            self._cache[cache_key] = None
            return None

        value = str(raw_value)
        self._cache[cache_key] = value
        return value


class AwsSecretsManagerProvider:
    """Load secrets from AWS Secrets Manager."""

    def __init__(self, secret_name: str, region: str, profile: Optional[str] = None) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - boto3 is optional
            raise SecretRetrievalError("boto3 is required for AWS Secrets Manager support") from exc

        session_kwargs: Dict[str, str] = {}
        if profile:
            session_kwargs["profile_name"] = profile

        session = boto3.session.Session(**session_kwargs)
        self.client = session.client("secretsmanager", region_name=region)
        self.secret_name = secret_name
        self._cache: Dict[str, str] = {}

    def _load_secret_bundle(self) -> None:
        if self._cache:
            return

        response = self.client.get_secret_value(SecretId=self.secret_name)
        payload: Dict[str, Any] = {}

        secret_string = response.get("SecretString")
        if secret_string:
            try:
                payload = json.loads(secret_string)
            except json.JSONDecodeError:
                payload = {self.secret_name: secret_string}

        secret_binary = response.get("SecretBinary")
        if secret_binary and not payload:
            try:
                decoded = base64.b64decode(secret_binary).decode("utf-8")
                payload = json.loads(decoded)
            except Exception as exc:  # pragma: no cover - depends on runtime data
                raise SecretRetrievalError(
                    f"Failed to decode binary secret '{self.secret_name}': {exc}"
                ) from exc

        for key, value in payload.items():
            if value is not None:
                self._cache[str(key)] = str(value)

    def get_secret(self, key: str, *, path: Optional[str] = None) -> Optional[str]:
        del path  # Secrets Manager does not use per-key paths
        self._load_secret_bundle()
        return self._cache.get(key)


class SecretManager:
    """Resolve secrets from the environment or an optional external provider."""

    def __init__(self) -> None:
        self.provider = self._build_provider()

    def _build_provider(self) -> Optional[SecretProvider]:
        provider_name = (os.getenv("SECRETS_PROVIDER", "").strip().lower())
        if provider_name in {"hashicorp", "vault"}:
            address = os.getenv("VAULT_ADDR", "").strip()
            token = os.getenv("VAULT_TOKEN", "").strip()
            secret_path = os.getenv("VAULT_SECRET_PATH", "secret/data/frontend").strip()
            verify_env = os.getenv("VAULT_VERIFY", "true").strip().lower()
            verify: Optional[bool]
            if verify_env in {"false", "0", "no"}:
                verify = False
            elif verify_env in {"true", "1", "yes"}:
                verify = True
            else:
                verify = None

            timeout_env = os.getenv("VAULT_TIMEOUT", "5.0").strip()
            try:
                timeout = float(timeout_env)
            except ValueError:
                timeout = 5.0

            if not address or not token:
                logger.warning(
                    "Hashicorp Vault provider selected but VAULT_ADDR or VAULT_TOKEN is missing."
                )
                return None

            try:
                return HashicorpVaultSecretProvider(
                    address,
                    token,
                    secret_path,
                    timeout=timeout,
                    verify=verify,
                )
            except Exception as exc:  # pragma: no cover - configuration errors
                logger.warning("Failed to initialise Hashicorp Vault provider: %s", exc)
                return None

        if provider_name in {"aws", "aws_secrets_manager", "secretsmanager"}:
            secret_name = os.getenv("AWS_SECRET_NAME") or os.getenv("AWS_SECRETS_MANAGER_SECRET_NAME")
            region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            profile = os.getenv("AWS_PROFILE")
            if not secret_name or not region:
                logger.warning(
                    "AWS Secrets Manager provider selected but AWS_SECRET_NAME or AWS_REGION is missing."
                )
                return None

            try:
                return AwsSecretsManagerProvider(secret_name, region, profile=profile)
            except SecretRetrievalError as exc:  # pragma: no cover - depends on environment
                logger.warning("Failed to initialise AWS Secrets Manager provider: %s", exc)
                return None

        if provider_name:
            logger.warning("Unknown secrets provider '%s' configured; falling back to environment", provider_name)
        return None

    def resolve(
        self,
        key: str,
        *,
        default: Optional[str] = None,
        env_keys: Optional[Sequence[str]] = None,
        vault_path: Optional[str] = None,
    ) -> str:
        """Resolve *key* from the environment, provider, or *default*."""

        candidate_env_keys = list(env_keys or [])
        candidate_env_keys.append(key)

        seen: set[str] = set()
        for env_key in candidate_env_keys:
            for candidate in {env_key, env_key.upper()}:
                if candidate in seen:
                    continue
                seen.add(candidate)
                value = os.getenv(candidate)
                if value:
                    logger.debug("Resolved secret '%s' from environment variable '%s'", key, candidate)
                    return value

        path_env_key = f"{key}_VAULT_PATH"
        resolved_path = vault_path or os.getenv(path_env_key) or os.getenv(path_env_key.upper())

        if self.provider:
            try:
                secret_value = self.provider.get_secret(key, path=resolved_path)
            except SecretRetrievalError as exc:
                logger.warning("Failed to load secret '%s' from provider: %s", key, exc)
            else:
                if secret_value is not None:
                    logger.debug("Resolved secret '%s' using provider", key)
                    return secret_value

        if default is not None:
            logger.debug("Using default value for secret '%s'", key)
            return default

        raise SecretRetrievalError(
            f"Secret '{key}' could not be resolved from the environment or the configured provider"
        )


_secret_manager = SecretManager()


def resolve_secret(
    key: str,
    *,
    default: Optional[str] = None,
    env_keys: Optional[Sequence[str]] = None,
    vault_path: Optional[str] = None,
) -> str:
    """Convenience wrapper for :class:`SecretManager`."""

    return _secret_manager.resolve(key, default=default, env_keys=env_keys, vault_path=vault_path)


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
        return resolve_secret(
            "SESSION_SECRET_KEY",
            default=secrets.token_urlsafe(32),
            env_keys=("SECURITY__SESSION_SECRET_KEY",),
        )

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
