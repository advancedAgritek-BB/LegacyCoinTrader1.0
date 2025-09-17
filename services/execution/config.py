"""Configuration dataclasses for the execution microservice."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, MutableMapping, Optional

try:  # pragma: no cover - optional dependency for richer settings support
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    Field = None  # type: ignore[assignment]
    BaseSettings = None  # type: ignore[assignment]
    SettingsConfigDict = None  # type: ignore[assignment]

from .models import SecretRef


@dataclass(slots=True)
class TelegramConfig:
    """Settings controlling Telegram notifications."""

    enabled: bool = False
    token: Optional[SecretRef] = None
    chat_id: Optional[SecretRef] = None
    parse_mode: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | "TelegramConfig" | None) -> "TelegramConfig":
        if isinstance(data, cls):
            return data
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", data.get("token"))),
            token=SecretRef.from_value(data.get("token"), default_source="env"),
            chat_id=SecretRef.from_value(data.get("chat_id"), default_source="literal"),
            parse_mode=data.get("parse_mode"),
        )


@dataclass(slots=True)
class MonitoringConfig:
    """Settings controlling monitoring callbacks for order lifecycle events."""

    enabled: bool = False
    sink: str = "local"
    extra_tags: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | "MonitoringConfig" | None
    ) -> "MonitoringConfig":
        if isinstance(data, cls):
            return data
        data = data or {}
        tags = data.get("extra_tags") or {}
        if not isinstance(tags, Mapping):
            raise TypeError("extra_tags must be a mapping of string keys")
        return cls(
            enabled=bool(data.get("enabled")),
            sink=str(data.get("sink", "local")),
            extra_tags=dict(tags),
        )


@dataclass(slots=True)
class CredentialsConfig:
    """References describing where to load exchange credentials."""

    api_key: SecretRef
    api_secret: SecretRef
    passphrase: Optional[SecretRef] = None
    ws_token: Optional[SecretRef] = None
    api_token: Optional[SecretRef] = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | "CredentialsConfig" | None
    ) -> Optional["CredentialsConfig"]:
        if isinstance(data, cls):
            return data
        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise TypeError("credentials configuration must be a mapping")
        api_key = SecretRef.from_value(data.get("api_key"))
        api_secret = SecretRef.from_value(data.get("api_secret"))
        if api_key is None or api_secret is None:
            raise ValueError("api_key and api_secret must be provided for credentials")
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=SecretRef.from_value(data.get("passphrase")),
            ws_token=SecretRef.from_value(data.get("ws_token")),
            api_token=SecretRef.from_value(data.get("api_token")),
        )


@dataclass(slots=True)
class ExecutionServiceConfig:
    """Runtime configuration for :class:`services.execution.service.ExecutionService`."""

    exchange: Mapping[str, Any] = field(default_factory=dict)
    credentials: Optional[CredentialsConfig] = None
    dry_run: bool = True
    use_websocket: bool = False
    ack_topic: str = "execution.acks"
    fill_topic: str = "execution.fills"
    idempotency_ttl: float = 3600.0
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @classmethod
    def from_mapping(
        cls,
        base: Mapping[str, Any] | MutableMapping[str, Any] | "ExecutionServiceConfig" | None,
    ) -> "ExecutionServiceConfig":
        if isinstance(base, cls):
            return base
        config = dict(base or {})
        telegram_cfg = TelegramConfig.from_mapping(config.get("telegram"))
        monitoring_cfg = MonitoringConfig.from_mapping(config.get("monitoring"))
        credentials_cfg = CredentialsConfig.from_mapping(config.get("credentials"))
        dry_run = config.get("execution_mode", "dry_run") == "dry_run"
        return cls(
            exchange=config,
            credentials=credentials_cfg,
            dry_run=dry_run,
            use_websocket=bool(config.get("use_websocket", False)),
            ack_topic=str(config.get("ack_topic", "execution.acks")),
            fill_topic=str(config.get("fill_topic", "execution.fills")),
            idempotency_ttl=float(config.get("idempotency_ttl", 3600.0)),
            telegram=telegram_cfg,
            monitoring=monitoring_cfg,
        )


if BaseSettings is not None:  # pragma: no cover - exercised when dependency available

    class ExecutionApiSettings(BaseSettings):
        """Runtime configuration for the execution FastAPI layer."""

        model_config = SettingsConfigDict(env_prefix="EXECUTION_SERVICE_", env_file=None)

        host: str = Field(default="0.0.0.0")
        port: int = Field(default=8006)
        base_path: str = Field(default="/api/v1/execution")
        log_level: str = Field(default="INFO")

        service_token: str = Field(default="insecure-local-token-execution")
        signing_key: Optional[str] = Field(default=None)
        signature_ttl_seconds: int = Field(default=60, ge=1)

        ack_timeout: float = Field(default=15.0, ge=0.1)
        fill_timeout: float = Field(default=60.0, ge=0.1)


    @lru_cache
    def get_execution_api_settings() -> ExecutionApiSettings:
        """Return cached API settings for the execution service."""

        return ExecutionApiSettings()

else:  # pragma: no cover - minimal fallback without pydantic

    @dataclass(slots=True)
    class ExecutionApiSettings:  # type: ignore[no-redef]
        host: str = field(default_factory=lambda: os.getenv("EXECUTION_SERVICE_HOST", "0.0.0.0"))
        port: int = field(default_factory=lambda: int(os.getenv("EXECUTION_SERVICE_PORT", "8006")))
        base_path: str = field(default_factory=lambda: os.getenv("EXECUTION_SERVICE_BASE_PATH", "/api/v1/execution"))
        log_level: str = field(default_factory=lambda: os.getenv("EXECUTION_SERVICE_LOG_LEVEL", "INFO"))
        service_token: str = field(
            default_factory=lambda: os.getenv("EXECUTION_SERVICE_TOKEN", "insecure-local-token-execution")
        )
        signing_key: Optional[str] = field(default_factory=lambda: os.getenv("EXECUTION_SERVICE_SIGNING_KEY"))
        signature_ttl_seconds: int = field(
            default_factory=lambda: int(os.getenv("EXECUTION_SERVICE_SIGNATURE_TTL_SECONDS", "60"))
        )
        ack_timeout: float = field(default_factory=lambda: float(os.getenv("EXECUTION_SERVICE_ACK_TIMEOUT", "15.0")))
        fill_timeout: float = field(default_factory=lambda: float(os.getenv("EXECUTION_SERVICE_FILL_TIMEOUT", "60.0")))


    @lru_cache
    def get_execution_api_settings() -> ExecutionApiSettings:  # type: ignore[no-redef]
        return ExecutionApiSettings()


__all__ = [
    "CredentialsConfig",
    "ExecutionApiSettings",
    "ExecutionServiceConfig",
    "MonitoringConfig",
    "TelegramConfig",
    "get_execution_api_settings",
]
