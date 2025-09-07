"""
Unit Tests for Configuration System

Comprehensive tests for the modern configuration management system,
including validation, environment variables, and error handling.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

import sys
from pathlib import Path

# Add the modern/src directory to Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.config import (
    AppConfig,
    DatabaseConfig,
    RedisConfig,
    ExchangeConfig,
    TradingConfig,
    Environment,
    ExecutionMode,
    ExchangeName,
    LogLevel,
    get_settings,
    init_config,
    reload_config
)


class TestDatabaseConfig:
    """Test Database configuration."""

    def test_valid_database_config(self):
        """Test valid database configuration."""
        config = DatabaseConfig(
            url="postgresql://user:pass@localhost:5432/db",
            pool_size=10,
            max_overflow=20,
            pool_timeout=30
        )
        assert config.url == "postgresql://user:pass@localhost:5432/db"
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.pool_timeout == 30

    def test_database_config_validation(self):
        """Test database configuration validation."""
        # Test invalid pool_size
        with pytest.raises(ValidationError):
            DatabaseConfig(url="postgresql://test", pool_size=0)

        # Test invalid max_overflow
        with pytest.raises(ValidationError):
            DatabaseConfig(url="postgresql://test", max_overflow=-1)

    def test_database_config_env_prefix(self):
        """Test database configuration environment prefix."""
        config = DatabaseConfig()
        assert config.__config__.env_prefix == "DB_"


class TestRedisConfig:
    """Test Redis configuration."""

    def test_valid_redis_config(self):
        """Test valid Redis configuration."""
        config = RedisConfig(
            host="redis-server",
            port=6380,
            db=5,
            password="secret",
            ssl=True,
            socket_timeout=10
        )
        assert config.host == "redis-server"
        assert config.port == 6380
        assert config.db == 5
        assert config.password.get_secret_value() == "secret"
        assert config.ssl is True
        assert config.socket_timeout == 10

    def test_redis_config_validation(self):
        """Test Redis configuration validation."""
        # Test invalid port
        with pytest.raises(ValidationError):
            RedisConfig(host="localhost", port=70000)

        # Test invalid db
        with pytest.raises(ValidationError):
            RedisConfig(host="localhost", db=16)

    def test_redis_config_env_prefix(self):
        """Test Redis configuration environment prefix."""
        config = RedisConfig()
        assert config.__config__.env_prefix == "REDIS_"


class TestExchangeConfig:
    """Test Exchange configuration."""

    def test_valid_exchange_config(self):
        """Test valid exchange configuration."""
        config = ExchangeConfig(
            name=ExchangeName.KRAKEN,
            api_key="test_key_123",
            api_secret="test_secret_456",
            passphrase="test_passphrase",
            sandbox=False,
            timeout=45,
            rate_limit=True,
            retry_count=5,
            retry_delay=2.5
        )
        assert config.name == ExchangeName.KRAKEN
        assert config.api_key.get_secret_value() == "test_key_123"
        assert config.api_secret.get_secret_value() == "test_secret_456"
        assert config.passphrase.get_secret_value() == "test_passphrase"
        assert config.sandbox is False
        assert config.timeout == 45
        assert config.rate_limit is True
        assert config.retry_count == 5
        assert config.retry_delay == 2.5

    def test_exchange_config_validation(self):
        """Test exchange configuration validation."""
        # Test missing required fields
        with pytest.raises(ValidationError):
            ExchangeConfig(name=ExchangeName.KRAKEN)

        # Test invalid timeout
        with pytest.raises(ValidationError):
            ExchangeConfig(
                name=ExchangeName.KRAKEN,
                api_key="test",
                api_secret="test",
                timeout=0
            )

    def test_exchange_config_env_prefix(self):
        """Test exchange configuration environment prefix."""
        config = ExchangeConfig(name=ExchangeName.KRAKEN, api_key="test", api_secret="test")
        assert config.__config__.env_prefix == "EXCHANGE_"


class TestTradingConfig:
    """Test Trading configuration."""

    def test_valid_trading_config(self):
        """Test valid trading configuration."""
        config = TradingConfig(
            execution_mode=ExecutionMode.LIVE,
            max_open_positions=15,
            max_risk_per_trade=0.08,
            max_total_risk=0.25,
            position_size_pct=0.15,
            min_position_size_usd=25.0,
            stop_loss_pct=0.025,
            take_profit_pct=0.055,
            trailing_stop_pct=0.015,
            timeframes=["5m", "1h", "4h"],
            default_timeframe="1h",
            strategy_allocation={"strategy1": 0.4, "strategy2": 0.6},
            regime_enabled=True,
            sentiment_enabled=False
        )
        assert config.execution_mode == ExecutionMode.LIVE
        assert config.max_open_positions == 15
        assert config.max_risk_per_trade == 0.08
        assert config.max_total_risk == 0.25
        assert config.position_size_pct == 0.15
        assert config.min_position_size_usd == 25.0
        assert config.stop_loss_pct == 0.025
        assert config.take_profit_pct == 0.055
        assert config.trailing_stop_pct == 0.015
        assert config.timeframes == ["5m", "1h", "4h"]
        assert config.default_timeframe == "1h"
        assert config.strategy_allocation == {"strategy1": 0.4, "strategy2": 0.6}
        assert config.regime_enabled is True
        assert config.sentiment_enabled is False

    def test_trading_config_validation(self):
        """Test trading configuration validation."""
        # Test invalid max_open_positions
        with pytest.raises(ValidationError):
            TradingConfig(execution_mode=ExecutionMode.DRY_RUN, max_open_positions=0)

        # Test invalid max_risk_per_trade
        with pytest.raises(ValidationError):
            TradingConfig(execution_mode=ExecutionMode.DRY_RUN, max_risk_per_trade=1.5)

        # Test invalid position_size_pct
        with pytest.raises(ValidationError):
            TradingConfig(execution_mode=ExecutionMode.DRY_RUN, position_size_pct=1.1)

    def test_trading_config_env_prefix(self):
        """Test trading configuration environment prefix."""
        config = TradingConfig()
        assert config.__config__.env_prefix == "TRADING_"


class TestAppConfig:
    """Test Application configuration."""

    def test_valid_app_config(self):
        """Test valid application configuration."""
        config = AppConfig(
            app_name="TestApp",
            version="1.0.0",
            environment=Environment.DEVELOPMENT,
            debug=True
        )
        assert config.app_name == "TestApp"
        assert config.version == "1.0.0"
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is True

    def test_app_config_properties(self):
        """Test application configuration properties."""
        config = AppConfig(environment=Environment.PRODUCTION)

        assert config.is_production is True
        assert config.is_development is False
        assert config.is_live_trading is False  # Default is DRY_RUN

        config.trading.execution_mode = ExecutionMode.LIVE
        assert config.is_live_trading is True

    def test_app_config_path_methods(self):
        """Test application configuration path methods."""
        config = AppConfig()

        log_path = config.get_log_file_path("test")
        assert isinstance(log_path, Path)
        assert "test.log" in str(log_path)

        config_path = config.get_config_file_path("test.yaml")
        assert isinstance(config_path, Path)
        assert "test.yaml" in str(config_path)

    @patch.dict(os.environ, {
        "APP_NAME": "TestApp",
        "VERSION": "2.0.0",
        "DEBUG": "true"
    })
    def test_app_config_from_env(self):
        """Test application configuration from environment variables."""
        config = AppConfig()
        assert config.app_name == "TestApp"
        assert config.version == "2.0.0"
        assert config.debug is True

    def test_app_config_validation_development(self):
        """Test application configuration validation in development."""
        # Should not raise validation errors in development
        config = AppConfig(environment=Environment.DEVELOPMENT)
        assert config.environment == Environment.DEVELOPMENT

    def test_app_config_validation_production_without_telegram(self):
        """Test application configuration validation in production without Telegram."""
        # Should raise validation error in production without Telegram
        with pytest.raises(ValidationError):
            AppConfig(environment=Environment.PRODUCTION)

    def test_app_config_validation_production_without_solana(self):
        """Test application configuration validation in production without Solana."""
        # Should raise validation error in production with live trading but no Solana
        with pytest.raises(ValidationError):
            AppConfig(
                environment=Environment.PRODUCTION,
                telegram=MagicMock(),
                trading=TradingConfig(execution_mode=ExecutionMode.LIVE)
            )

    def test_app_config_env_prefix(self):
        """Test application configuration environment file."""
        config = AppConfig()
        assert config.__config__.env_file == ".env"
        assert config.__config__.env_file_encoding == "utf-8"
        assert config.__config__.case_sensitive is False


class TestConfigManagement:
    """Test configuration management functions."""

    def test_get_settings_uninitialized(self):
        """Test get_settings when configuration is not initialized."""
        # Clear any existing settings
        import src.core.config as config_module
        config_module.settings = None

        with pytest.raises(RuntimeError, match="Configuration not initialized"):
            get_settings()

    def test_init_config(self):
        """Test configuration initialization."""
        config = init_config()
        assert isinstance(config, AppConfig)

        # Test that get_settings works after initialization
        retrieved_config = get_settings()
        assert retrieved_config is config

    def test_init_config_with_env_file(self, temp_dir):
        """Test configuration initialization with custom env file."""
        env_file = temp_dir / ".test.env"
        env_file.write_text("APP_NAME=CustomTestApp\nDEBUG=true\n")

        config = init_config(str(env_file))
        assert config.app_name == "CustomTestApp"
        assert config.debug is True

    def test_reload_config(self):
        """Test configuration reloading."""
        # Initialize config first
        original_config = init_config()
        original_name = original_config.app_name

        # Modify environment
        with patch.dict(os.environ, {"APP_NAME": "ReloadedApp"}):
            reloaded_config = reload_config()
            assert reloaded_config.app_name == "ReloadedApp"
            assert reloaded_config is not original_config

    def test_reload_config_uninitialized(self):
        """Test reload_config when configuration is not initialized."""
        import src.core.config as config_module
        config_module.settings = None

        with pytest.raises(RuntimeError, match="Configuration not initialized"):
            reload_config()


class TestEnumValidation:
    """Test enum validation."""

    def test_environment_enum(self):
        """Test Environment enum values."""
        assert Environment.DEVELOPMENT == "development"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"
        assert Environment.TESTING == "testing"

    def test_execution_mode_enum(self):
        """Test ExecutionMode enum values."""
        assert ExecutionMode.DRY_RUN == "dry_run"
        assert ExecutionMode.LIVE == "live"
        assert ExecutionMode.PAPER == "paper"

    def test_exchange_name_enum(self):
        """Test ExchangeName enum values."""
        assert ExchangeName.KRAKEN == "kraken"
        assert ExchangeName.COINBASE == "coinbase"
        assert ExchangeName.SOLANA == "solana"

    def test_log_level_enum(self):
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"


class TestPathValidation:
    """Test path validation."""

    def test_path_creation(self, temp_dir):
        """Test that paths are created correctly."""
        config = AppConfig(
            base_dir=temp_dir,
            log_dir=temp_dir / "logs",
            config_dir=temp_dir / "configs"
        )

        assert config.base_dir == temp_dir
        assert config.log_dir == temp_dir / "logs"
        assert config.config_dir == temp_dir / "configs"

        # Check that directories exist
        assert config.log_dir.exists()
        assert config.config_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
