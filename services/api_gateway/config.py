"""
Configuration management for API Gateway service.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    """Configuration for API Gateway."""

    port: int = Field(default=8000, description="Port for the API Gateway")
    host: str = Field(default="0.0.0.0", description="Host for the API Gateway")

    # Redis configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")

    # Service authentication
    service_auth_token: str = Field(
        default="default-service-token",
        description="Token for service-to-service authentication"
    )

    # Rate limiting
    rate_limit_requests: int = Field(
        default=1000,
        description="Max requests per window"
    )
    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds"
    )

    # Timeouts
    request_timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    connect_timeout: int = Field(
        default=10,
        description="Connection timeout in seconds"
    )

    # Service discovery
    service_discovery_enabled: bool = Field(
        default=True,
        description="Enable service discovery"
    )
    service_ttl: int = Field(
        default=30,
        description="Service registration TTL in seconds"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    # CORS
    cors_origins: list = Field(
        default=["*"],
        description="Allowed CORS origins"
    )

    # Authentication
    jwt_secret: str = Field(
        default="default-jwt-secret",
        description="JWT secret key"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm"
    )

    class Config:
        env_prefix = "GATEWAY_"


def load_config_from_env() -> GatewayConfig:
    """Load configuration from environment variables."""
    return GatewayConfig()


def load_config_from_file(config_path: str) -> GatewayConfig:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return GatewayConfig(**config_data.get('api_gateway', {}))
