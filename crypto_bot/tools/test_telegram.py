"""Pytest tests for Telegram bot functionality."""

import pytest
from typing import Dict, Any
from pathlib import Path
import yaml


@pytest.fixture
def token():
    """Mock Telegram bot token for testing."""
    return "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"


@pytest.fixture
def chat_id():
    """Mock Telegram chat ID for testing."""
    return "123456789"


@pytest.fixture
def config():
    """Mock configuration for testing."""
    return {
        'telegram': {
            'enabled': True,
            'token': '1234567890:ABCdefGHIjklMNOpqrsTUVwxyz',
            'chat_id': '123456789',
            'timeout_seconds': 30,
            'fail_silently': False
        }
    }


def test_config_values(config: Dict[str, Any]) -> None:
    """Test configuration values for common issues."""
    telegram = config.get('telegram', {})
    
    # Check required fields
    assert telegram.get('token'), "Missing Telegram token"
    assert telegram.get('chat_id'), "Missing Telegram chat ID"
    
    # Check token format
    token = telegram['token']
    assert token and len(token) >= 20, "Invalid Telegram token format"
    
    # Check chat ID format
    chat_id = str(telegram['chat_id'])
    assert chat_id.isdigit(), "Chat ID should be a numeric value"
    
    # Check optional settings
    timeout = telegram.get('timeout_seconds', 30)
    assert timeout >= 10, "timeout_seconds should be at least 10"


def test_config_file_exists():
    """Test that the configuration file exists."""
    config_path = Path("crypto_bot/config.yaml")
    assert config_path.exists(), f"Configuration file not found: {config_path}"


def test_config_file_parsable():
    """Test that the configuration file can be parsed."""
    config_path = Path("crypto_bot/config.yaml")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict), "Config should be a dictionary"
    except Exception as e:
        pytest.fail(f"Failed to parse config file: {e}")


def test_telegram_enabled_in_config(config: Dict[str, Any]):
    """Test that Telegram is enabled in configuration."""
    telegram = config.get('telegram', {})
    assert telegram.get('enabled', False), "Telegram should be enabled in config"


def test_token_format(token: str):
    """Test that the token has the correct format."""
    assert len(token) >= 20, "Token should be at least 20 characters"
    assert ':' in token, "Token should contain a colon separator"


def test_chat_id_format(chat_id: str):
    """Test that the chat ID has the correct format."""
    assert chat_id.isdigit(), "Chat ID should be numeric"
    assert len(chat_id) > 0, "Chat ID should not be empty"
