"""
Modern Configuration Management System

This module provides a comprehensive configuration management system using Pydantic
with support for environment variables, validation, and type safety.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from pydantic.types import SecretStr


class Environment(str, Enum):
    """Application environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ExchangeName(str, Enum):
    """Supported exchange enumeration."""

    KRAKEN = "kraken"
    COINBASE = "coinbase"
    SOLANA = "solana"


class ExecutionMode(str, Enum):
    """Execution mode enumeration."""

    DRY_RUN = "dry_run"
    LIVE = "live"
    PAPER = "paper"


class LogLevel(str, Enum):
    """Logging level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""

    url: str = Field(..., env="DATABASE_URL")
    pool_size: int = Field(10, ge=1, le=100)
    max_overflow: int = Field(20, ge=0, le=100)
    pool_timeout: int = Field(30, ge=5, le=300)
    pool_recycle: int = Field(3600, ge=300, le=86400)

    class Config:
        env_prefix = "DB_"


class RedisConfig(BaseSettings):
    """Redis configuration settings."""

    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, ge=1024, le=65535, env="REDIS_PORT")
    db: int = Field(0, ge=0, le=15, env="REDIS_DB")
    password: Optional[SecretStr] = Field(None, env="REDIS_PASSWORD")
    ssl: bool = Field(False, env="REDIS_SSL")
    socket_timeout: int = Field(5, ge=1, le=30)
    socket_connect_timeout: int = Field(5, ge=1, le=30)
    socket_keepalive: bool = Field(True)
    socket_keepalive_options: Dict[str, Union[int, str]] = Field(
        default_factory=dict
    )
    health_check_interval: int = Field(30, ge=10, le=300)

    class Config:
        env_prefix = "REDIS_"


class ExchangeConfig(BaseSettings):
    """Exchange-specific configuration."""

    name: ExchangeName = Field(..., env="NAME")
    api_key: SecretStr = Field(..., env="API_KEY")
    api_secret: SecretStr = Field(..., env="API_SECRET")
    passphrase: Optional[SecretStr] = Field(None, env="API_PASSPHRASE")
    sandbox: bool = Field(False)
    timeout: int = Field(30, ge=5, le=300)
    rate_limit: bool = Field(True)
    retry_count: int = Field(3, ge=0, le=10)
    retry_delay: float = Field(1.0, ge=0.1, le=60.0)

    class Config:
        env_prefix = "EXCHANGE_"


class TelegramConfig(BaseSettings):
    """Telegram bot configuration."""

    token: SecretStr = Field(..., env="TELEGRAM_TOKEN")
    chat_id: Optional[int] = Field(None, env="TELEGRAM_CHAT_ID")
    enabled: bool = Field(True)
    balance_updates: bool = Field(False)
    status_updates: bool = Field(True)
    trade_updates: bool = Field(True)

    class Config:
        env_prefix = "TELEGRAM_"


class SolanaConfig(BaseSettings):
    """Solana-specific configuration."""

    rpc_url: str = Field(
        "https://api.mainnet.solana.com", env="SOLANA_RPC_URL"
    )
    private_key: Optional[SecretStr] = Field(None, env="SOLANA_PRIVATE_KEY")
    wallet_address: Optional[str] = Field(None, env="SOLANA_WALLET_ADDRESS")
    helius_key: Optional[SecretStr] = Field(None, env="HELIUS_KEY")
    commitment: str = Field(
        "confirmed", pattern=r"^(processed|confirmed|finalized)$"
    )

    class Config:
        env_prefix = "SOLANA_"


class SecurityConfig(BaseSettings):
    """Security configuration settings."""

    jwt_secret_key: SecretStr = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256")
    jwt_expiration_hours: int = Field(24, ge=1, le=168)
    bcrypt_rounds: int = Field(12, ge=10, le=16)

    # API Security
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8000"]
    )
    cors_credentials: bool = Field(True)
    cors_methods: List[str] = Field(
        ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    cors_headers: List[str] = Field(["*"])

    # Rate Limiting
    rate_limit_requests: int = Field(100, ge=10, le=1000)
    rate_limit_window_seconds: int = Field(60, ge=10, le=3600)

    class Config:
        env_prefix = "SECURITY_"


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""

    enabled: bool = Field(True)
    log_level: LogLevel = Field(LogLevel.INFO)
    structured_logging: bool = Field(True)

    # Sentry Configuration
    sentry_dsn: Optional[SecretStr] = Field(None, env="SENTRY_DSN")
    sentry_environment: str = Field("development")

    # Metrics
    metrics_enabled: bool = Field(True)
    metrics_port: int = Field(8001, ge=1024, le=65535)

    # Health Checks
    health_check_enabled: bool = Field(True)
    health_check_interval: int = Field(30, ge=10, le=300)

    class Config:
        env_prefix = "MONITORING_"


class TradingConfig(BaseSettings):
    """Trading-specific configuration."""

    execution_mode: ExecutionMode = Field(ExecutionMode.DRY_RUN)
    max_open_positions: int = Field(10, ge=1, le=100)
    max_risk_per_trade: float = Field(0.05, ge=0.001, le=0.5)
    max_total_risk: float = Field(0.2, ge=0.01, le=1.0)

    # Position Sizing
    position_size_pct: float = Field(0.1, ge=0.001, le=1.0)
    min_position_size_usd: float = Field(10.0, ge=1.0, le=10000.0)

    # Risk Management
    stop_loss_pct: float = Field(0.02, ge=0.001, le=0.1)
    take_profit_pct: float = Field(0.04, ge=0.001, le=0.2)
    trailing_stop_pct: float = Field(0.01, ge=0.001, le=0.05)

    # Timeframes
    timeframes: List[str] = Field(["1h", "4h", "1d"])
    default_timeframe: str = Field("1h")

    # Strategy Configuration
    strategy_allocation: Dict[str, float] = Field(default_factory=dict)
    regime_enabled: bool = Field(True)
    sentiment_enabled: bool = Field(True)

    class Config:
        env_prefix = "TRADING_"


class AppConfig(BaseSettings):
    """
    Main application configuration.

    This class aggregates all configuration settings and provides validation
    and type safety across the entire application.
    """

    # Application Metadata
    app_name: Literal["LegacyCoinTrader"] = "LegacyCoinTrader"
    version: Literal["2.0.0"] = "2.0.0"
    environment: Environment = Field(Environment.DEVELOPMENT)
    debug: bool = Field(False)

    # Paths
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent
    )
    log_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent
        / "logs"
    )
    config_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent
        / "configs"
    )

    # Core Services
    database: DatabaseConfig
    redis: RedisConfig
    exchange: Optional[ExchangeConfig] = None
    telegram: Optional[TelegramConfig] = None
    solana: Optional[SolanaConfig] = None

    # Advanced Configuration
    security: SecurityConfig
    monitoring: MonitoringConfig
    trading: TradingConfig

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("log_dir", "config_dir")
    @classmethod
    def ensure_path_exists(cls, v):
        """Ensure that path directories exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @model_validator(mode="after")
    def validate_configuration(self):
        """Validate the complete configuration."""
        environment = self.environment
        execution_mode = self.trading.execution_mode if self.trading else None

        # Production validations
        if environment == Environment.PRODUCTION:
            if not self.telegram:
                raise ValueError(
                    "Telegram configuration is required in production"
                )

            if execution_mode == ExecutionMode.LIVE:
                # Additional production validations
                if not self.solana:
                    raise ValueError(
                        "Solana configuration is required for live trading"
                    )

                # Validate API credentials are not default values
                if self.exchange and hasattr(self.exchange, "api_key"):
                    api_key = getattr(self.exchange, "api_key", None)
                    if api_key and api_key.get_secret_value() in [
                        "",
                        "your_api_key_here",
                    ]:
                        raise ValueError(
                            "Valid API key required in production"
                        )

        return self

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled."""
        return self.trading.execution_mode == ExecutionMode.LIVE

    def get_log_file_path(self, component: str) -> Path:
        """Get the log file path for a specific component."""
        return self.log_dir / f"{component}.log"

    def get_config_file_path(self, filename: str) -> Path:
        """Get the configuration file path."""
        return self.config_dir / filename


# Global configuration instance
settings: Optional[AppConfig] = None


def get_settings() -> AppConfig:
    """
    Get the global application configuration.

    Returns:
        AppConfig: The application configuration instance.

    Raises:
        RuntimeError: If configuration has not been initialized.
    """
    global settings
    if settings is None:
        raise RuntimeError(
            "Configuration not initialized. Call init_config() first."
        )
    return settings


def init_config(env_file: Optional[str] = None) -> AppConfig:
    """
    Initialize the application configuration.

    Args:
        env_file: Optional path to environment file.

    Returns:
        AppConfig: The initialized configuration.
    """
    global settings

    # Allow overriding the env file
    config_kwargs = {}
    if env_file:
        config_kwargs["env_file"] = env_file

    # Try to create exchange config if environment variables are set
    try:
        exchange_config = ExchangeConfig()
        config_kwargs["exchange"] = exchange_config
    except Exception:
        # Exchange config not available, will be None
        pass

    settings = AppConfig(**config_kwargs)
    return settings


def reload_config() -> AppConfig:
    """
    Reload the configuration from environment variables.

    Returns:
        AppConfig: The reloaded configuration.
    """
    global settings
    if settings is None:
        raise RuntimeError(
            "Configuration not initialized. Call init_config() first."
        )

    # Create new instance with current environment
    settings = AppConfig()
    return settings


# Export commonly used configuration components
__all__ = [
    "AppConfig",
    "DatabaseConfig",
    "RedisConfig",
    "ExchangeConfig",
    "TelegramConfig",
    "SolanaConfig",
    "SecurityConfig",
    "MonitoringConfig",
    "TradingConfig",
    "Environment",
    "ExchangeName",
    "ExecutionMode",
    "LogLevel",
    "get_settings",
    "init_config",
    "reload_config",
]
