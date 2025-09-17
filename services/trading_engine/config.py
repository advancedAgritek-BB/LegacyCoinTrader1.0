from __future__ import annotations

"""Runtime configuration for the trading engine service."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="TRADING_ENGINE_", env_file=None)

    app_name: str = Field(default="trading-engine-service")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_use_ssl: bool = Field(default=False)

    default_cycle_interval: int = Field(default=60, ge=1)
    state_key_prefix: str = Field(default="trading_engine")

    log_level: str = Field(default="INFO")
    cycle_timeout_seconds: Optional[int] = Field(default=900, ge=1)

    def redis_dsn(self) -> str:
        protocol = "rediss" if self.redis_use_ssl else "redis"
        return f"{protocol}://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""

    return Settings()


__all__ = ["Settings", "get_settings"]
