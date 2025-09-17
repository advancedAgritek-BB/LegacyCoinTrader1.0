"""Configuration helpers for the portfolio service."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_DATABASE_URL = "postgresql+psycopg2://portfolio:portfolio@localhost:5432/portfolio"
DEFAULT_REST_PORT = 8003
DEFAULT_GRPC_PORT = 50053
DEFAULT_PASSWORD_ROTATION_DAYS = 90
DEFAULT_API_KEY_ROTATION_DAYS = 180


@dataclass
class PortfolioConfig:
    """Runtime configuration for the portfolio service."""

    database_url: str = DEFAULT_DATABASE_URL
    rest_host: str = "0.0.0.0"
    rest_port: int = DEFAULT_REST_PORT
    grpc_host: str = "0.0.0.0"
    grpc_port: int = DEFAULT_GRPC_PORT
    password_rotation_days: int = DEFAULT_PASSWORD_ROTATION_DAYS
    api_key_rotation_days: int = DEFAULT_API_KEY_ROTATION_DAYS

    @classmethod
    def from_env(cls) -> "PortfolioConfig":
        """Create a configuration object from environment variables."""

        rotation_days = _positive_int_from_env(
            "PORTFOLIO_PASSWORD_ROTATION_DAYS", DEFAULT_PASSWORD_ROTATION_DAYS
        )
        api_key_rotation = _positive_int_from_env(
            "PORTFOLIO_API_KEY_ROTATION_DAYS", DEFAULT_API_KEY_ROTATION_DAYS
        )

        return cls(
            database_url=os.getenv("PORTFOLIO_DATABASE_URL", DEFAULT_DATABASE_URL),
            rest_host=os.getenv("PORTFOLIO_REST_HOST", "0.0.0.0"),
            rest_port=int(os.getenv("PORTFOLIO_REST_PORT", DEFAULT_REST_PORT)),
            grpc_host=os.getenv("PORTFOLIO_GRPC_HOST", "0.0.0.0"),
            grpc_port=int(os.getenv("PORTFOLIO_GRPC_PORT", DEFAULT_GRPC_PORT)),
            password_rotation_days=rotation_days,
            api_key_rotation_days=api_key_rotation,
        )


def get_service_base_url() -> str:
    """Return the base URL for REST clients."""

    host = os.getenv("PORTFOLIO_SERVICE_HOST", "localhost")
    port = os.getenv("PORTFOLIO_SERVICE_PORT", str(DEFAULT_REST_PORT))
    scheme = os.getenv("PORTFOLIO_SERVICE_SCHEME", "http")
    return f"{scheme}://{host}:{port}"


def _positive_int_from_env(key: str, default: int) -> int:
    """Return a positive integer from the environment or *default*."""

    value = os.getenv(key)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def get_grpc_target() -> str:
    """Return the host:port target for gRPC clients."""

    host = os.getenv("PORTFOLIO_GRPC_SERVICE_HOST", "localhost")
    port = os.getenv("PORTFOLIO_GRPC_SERVICE_PORT", str(DEFAULT_GRPC_PORT))
    return f"{host}:{port}"
