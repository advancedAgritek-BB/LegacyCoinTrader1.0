"""Configuration for the strategy engine service."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration values."""

    model_config = SettingsConfigDict(env_prefix="STRATEGY_ENGINE_", env_file=None)

    app_name: str = Field(default="strategy-engine-service")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8004)

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_use_ssl: bool = Field(default=False)

    cache_prefix: str = Field(default="strategy_engine:results")
    evaluation_cache_ttl: int = Field(default=120, ge=0)
    market_data_channel: str = Field(default="market-data-events")
    model_key_prefix: str = Field(default="strategy_engine:models")

    log_level: str = Field(default="INFO")

    def redis_dsn(self) -> str:
        protocol = "rediss" if self.redis_use_ssl else "redis"
        return f"{protocol}://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()


__all__ = ["Settings", "get_settings"]
