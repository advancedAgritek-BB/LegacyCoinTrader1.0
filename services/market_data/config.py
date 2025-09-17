"""Runtime configuration for the market data service."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration values sourced from environment variables."""

    model_config = SettingsConfigDict(env_prefix="MARKET_DATA_", env_file=None)

    app_name: str = Field(default="market-data-service")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8002)

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_use_ssl: bool = Field(default=False)
    redis_password: Optional[str] = Field(default=None)

    cache_ttl_seconds: int = Field(default=900, ge=0)

    ohlcv_channel: str = Field(default="market-data:ohlcv")
    regime_channel: str = Field(default="market-data:regime")
    order_book_channel: str = Field(default="market-data:order-book")
    symbols_channel: str = Field(default="market-data:symbols")

    websocket_rate_limit_per_minute: int = Field(default=12, ge=1)
    websocket_min_interval_seconds: float = Field(default=15.0, ge=0.1)

    log_level: str = Field(default="INFO")

    def redis_dsn(self) -> str:
        protocol = "rediss" if self.redis_use_ssl else "redis"
        if self.redis_password:
            return f"{protocol}://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"{protocol}://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    """Return cached service configuration."""

    return Settings()


__all__ = ["Settings", "get_settings"]
