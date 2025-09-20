"""Configuration for the token discovery microservice."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the token discovery service."""

    app_name: str = Field(default="Token Discovery Service")
    log_level: str = Field(default="INFO")

    # Redis configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_use_ssl: bool = Field(default=False)
    redis_channel_tokens: str = Field(
        default="trading_engine.token_candidates",
        description="Channel for publishing discovered token batches.",
    )
    redis_channel_opportunities: str = Field(
        default="trading_engine.token_opportunities",
        description="Channel for publishing scored opportunity payloads.",
    )

    # Kafka configuration
    kafka_enabled: bool = Field(default=False)
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_topic_tokens: str = Field(default="token-discovery.tokens")
    kafka_topic_opportunities: str = Field(
        default="token-discovery.opportunities"
    )
    kafka_client_id: str = Field(default="token-discovery-service")

    # Background scanning configuration
    background_basic_interval: int = Field(
        default=300, description="Interval between basic scans in seconds."
    )
    background_enhanced_interval: int = Field(
        default=900,
        description="Interval between enhanced scans/opportunity publication in seconds.",
    )
    basic_scan_timeout_seconds: int = Field(
        default=15,
        ge=1,
        description="Max seconds to wait on a basic scan before falling back.",
    )
    publish_batch_size: int = Field(
        default=50, description="Maximum number of items to publish per batch."
    )

    # Solana scanner configuration
    solana_scanner_limit: int = Field(default=100)
    solana_min_volume_usd: float = Field(default=0.0)
    solana_gecko_search: bool = Field(default=True)
    helius_key: str = Field(default="")
    raydium_api_key: str = Field(default="")
    pump_fun_api_key: str = Field(default="")

    # Enhanced scanner configuration
    enable_enhanced_scanner: bool = Field(default=True)
    enhanced_min_score: float = Field(default=0.25)
    enhanced_limit: int = Field(default=100)

    # Centralised exchange (CEX) scanner configuration
    enable_cex_scanner: bool = Field(
        default=True,
        description="Enable background discovery for CEX listings.",
    )
    cex_exchange: str = Field(
        default="kraken",
        description="Primary exchange to monitor for new listings.",
    )
    cex_scanner_limit: int = Field(
        default=200,
        ge=1,
        le=500,
        description="Maximum number of newly discovered CEX pairs per scan.",
    )
    background_cex_interval: int = Field(
        default=900,
        ge=0,
        description="Interval between CEX discovery scans in seconds (0 disables loop).",
    )
    cex_state_file: str = Field(
        default="crypto_bot/logs/cex_scanner_state.json",
        description="Path used to persist seen CEX listings between runs.",
    )

    model_config = SettingsConfigDict(
        env_prefix="TOKEN_DISCOVERY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def redis_dsn(self) -> str:
        """Return the configured Redis DSN."""

        protocol = "rediss" if self.redis_use_ssl else "redis"
        return f"{protocol}://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache()
def get_settings() -> Settings:
    """Return cached :class:`Settings` instance."""

    return Settings()


__all__ = ["Settings", "get_settings"]
