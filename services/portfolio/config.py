"""Configuration helpers for the portfolio service."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_DATABASE_URL = "postgresql+psycopg2://portfolio:portfolio@localhost:5432/portfolio"
DEFAULT_REST_PORT = 8003
DEFAULT_GRPC_PORT = 50053


@dataclass
class PortfolioConfig:
    """Runtime configuration for the portfolio service."""

    database_url: str = DEFAULT_DATABASE_URL
    rest_host: str = "0.0.0.0"
    rest_port: int = DEFAULT_REST_PORT
    grpc_host: str = "0.0.0.0"
    grpc_port: int = DEFAULT_GRPC_PORT

    @classmethod
    def from_env(cls) -> "PortfolioConfig":
        """Create a configuration object from environment variables."""

        return cls(
            database_url=os.getenv("PORTFOLIO_DATABASE_URL", DEFAULT_DATABASE_URL),
            rest_host=os.getenv("PORTFOLIO_REST_HOST", "0.0.0.0"),
            rest_port=int(os.getenv("PORTFOLIO_REST_PORT", DEFAULT_REST_PORT)),
            grpc_host=os.getenv("PORTFOLIO_GRPC_HOST", "0.0.0.0"),
            grpc_port=int(os.getenv("PORTFOLIO_GRPC_PORT", DEFAULT_GRPC_PORT)),
        )


def get_service_base_url() -> str:
    """Return the base URL for REST clients."""

    host = os.getenv("PORTFOLIO_SERVICE_HOST", "localhost")
    port = os.getenv("PORTFOLIO_SERVICE_PORT", str(DEFAULT_REST_PORT))
    scheme = os.getenv("PORTFOLIO_SERVICE_SCHEME", "http")
    return f"{scheme}://{host}:{port}"


def get_grpc_target() -> str:
    """Return the host:port target for gRPC clients."""

    host = os.getenv("PORTFOLIO_GRPC_SERVICE_HOST", "localhost")
    port = os.getenv("PORTFOLIO_GRPC_SERVICE_PORT", str(DEFAULT_GRPC_PORT))
    return f"{host}:{port}"
