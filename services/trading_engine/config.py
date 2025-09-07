"""
Configuration for Trading Engine service.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class TradingEngineConfig(BaseModel):
    """Configuration for Trading Engine service."""

    # Service configuration
    port: int = Field(default=8001, description="Port for the trading engine service")
    host: str = Field(default="0.0.0.0", description="Host for the trading engine service")

    # Redis configuration
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")

    # API Gateway configuration
    api_gateway_host: str = Field(default="localhost", description="API Gateway host")
    api_gateway_port: int = Field(default=8000, description="API Gateway port")

    # Service authentication
    service_auth_token: str = Field(
        default="legacy-coin-trader-service-token-2024",
        description="Token for service-to-service authentication"
    )

    # Trading configuration
    cycle_interval: int = Field(
        default=120,
        description="Interval between trading cycles in seconds"
    )
    batch_size: int = Field(
        default=25,
        description="Number of symbols to process per batch"
    )
    batch_delay: float = Field(
        default=1.0,
        description="Delay between batches in seconds"
    )

    # Default symbols
    default_symbols: List[str] = Field(
        default=[
            "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
            "LINK/USD", "UNI/USD", "AAVE/USD", "AVAX/USD", "MATIC/USD"
        ],
        description="Default trading symbols"
    )

    # Risk management
    max_risk_per_trade: float = Field(
        default=0.05,
        description="Maximum risk per trade as fraction of balance"
    )
    max_position_size_pct: float = Field(
        default=0.2,
        description="Maximum position size as fraction of balance"
    )

    # Market data
    default_timeframe: str = Field(
        default="1h",
        description="Default timeframe for market data"
    )

    # Performance monitoring
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    performance_metrics_interval: int = Field(
        default=60,
        description="Interval for performance metrics in seconds"
    )

    # Health checks
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    max_consecutive_failures: int = Field(
        default=3,
        description="Maximum consecutive failures before marking unhealthy"
    )

    class Config:
        env_prefix = "TRADING_ENGINE_"


def load_config_from_env() -> TradingEngineConfig:
    """Load configuration from environment variables."""
    return TradingEngineConfig()


def load_config_from_file(config_path: str) -> TradingEngineConfig:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return TradingEngineConfig(**config_data.get('trading_engine', {}))
