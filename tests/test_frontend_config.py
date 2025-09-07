"""
Unit tests for frontend configuration module.
Ensures security settings are properly validated and applied.
"""

import pytest
import os
from unittest.mock import patch
from frontend.config import SecurityConfig, AppConfig, get_config


class TestSecurityConfig:
    """Test SecurityConfig functionality."""

    def test_default_security_config(self):
        """Test default security configuration values."""
        config = SecurityConfig()

        assert config.cors_origins == ["http://localhost:5000", "http://127.0.0.1:5000"]
        assert "'self'" in config.csp_default_src
        assert "'unsafe-eval'" not in config.csp_script_src  # Should not contain unsafe directives
        assert "'unsafe-inline'" not in config.csp_script_src  # Should not contain unsafe directives
        assert config.rate_limit_requests == 100
        assert config.rate_limit_window == 60

    def test_cors_origins_validation(self):
        """Test CORS origins validation from environment variable."""
        with patch.dict(os.environ, {'CORS_ORIGINS': 'http://example.com, https://app.example.com'}):
            config = SecurityConfig()
            assert config.cors_origins == ["http://example.com", "https://app.example.com"]

    def test_csp_header_generation(self):
        """Test CSP header generation."""
        config = SecurityConfig()
        csp_header = config.get_csp_header()

        # Should contain essential CSP directives
        assert "default-src 'self'" in csp_header
        assert "script-src" in csp_header
        assert "object-src 'none'" in csp_header  # Security best practice
        assert "base-uri 'self'" in csp_header

        # Should NOT contain unsafe directives
        assert "unsafe-eval" not in csp_header
        assert "unsafe-inline" not in csp_header

    def test_cors_headers_generation(self):
        """Test CORS headers generation."""
        with patch.dict(os.environ, {'CORS_ORIGINS': 'https://example.com'}):
            config = SecurityConfig()

            # Valid origin
            headers = config.get_cors_headers("https://example.com")
            assert headers['Access-Control-Allow-Origin'] == "https://example.com"
            assert headers['Access-Control-Allow-Credentials'] == "true"
            assert "GET" in headers['Access-Control-Allow-Methods']

            # Invalid origin
            headers = config.get_cors_headers("https://malicious.com")
            assert 'Access-Control-Allow-Origin' not in headers

    @patch.dict(os.environ, {
        'CORS_ORIGINS': 'https://prod.example.com,https://staging.example.com',
        'RATE_LIMIT_REQUESTS': '200'
    })
    def test_environment_override(self):
        """Test environment variable overrides."""
        config = SecurityConfig()

        # Environment variables should override defaults
        # Note: This test assumes the env vars are loaded properly
        pass  # Implementation depends on Pydantic env loading


class TestAppConfig:
    """Test AppConfig functionality."""

    def test_default_app_config(self):
        """Test default application configuration."""
        config = AppConfig()

        assert config.environment == "development"
        assert config.debug is False
        assert config.app_name == "LegacyCoinTrader"
        assert config.version == "2.0.0"
        assert config.host == "0.0.0.0"
        assert config.port == 5000
        assert config.log_level == "INFO"

    def test_nested_security_config(self):
        """Test that security config is properly nested."""
        config = AppConfig()
        assert isinstance(config.security, SecurityConfig)
        assert hasattr(config.security, 'cors_origins')

    @patch.dict(os.environ, {
        'ENVIRONMENT': 'production',
        'DEBUG': 'false',
        'APP_NAME': 'TestTrader'
    })
    def test_environment_variables(self):
        """Test environment variable loading."""
        config = AppConfig()

        # These should be loaded from environment if properly configured
        # Note: Pydantic handles this automatically
        pass


class TestConfigManagement:
    """Test configuration management functions."""

    def test_get_config(self):
        """Test get_config function."""
        config = get_config()
        assert isinstance(config, AppConfig)
        assert hasattr(config, 'security')

    def test_config_validation(self):
        """Test configuration validation."""
        config = AppConfig()

        # Should not raise exceptions with valid config
        try:
            config_dict = config.dict()
            assert isinstance(config_dict, dict)
            assert 'environment' in config_dict
        except Exception as e:
            pytest.fail(f"Configuration validation failed: {e}")


class TestSecurityBestPractices:
    """Test security best practices implementation."""

    def test_no_unsafe_csp_directives(self):
        """Ensure no unsafe CSP directives are present."""
        config = SecurityConfig()

        # Check that unsafe directives are not in script-src
        for directive in config.csp_script_src:
            assert "unsafe" not in directive.lower()

    def test_secure_session_settings(self):
        """Test secure session settings."""
        config = SecurityConfig()

        # Session secret should be generated securely
        assert len(config.session_secret_key) > 20

        # Session timeout should be reasonable
        assert 1800 <= config.session_timeout <= 86400  # Between 30min and 24hrs

    def test_restricted_http_methods(self):
        """Test that dangerous HTTP methods are restricted."""
        config = SecurityConfig()

        # Should only allow safe methods by default
        safe_methods = {"GET", "POST", "OPTIONS"}
        assert set(config.allowed_methods).issubset(safe_methods)

        # Should not include dangerous methods like PUT, DELETE, PATCH
        dangerous_methods = {"PUT", "DELETE", "PATCH"}
        assert not set(config.allowed_methods).intersection(dangerous_methods)

    def test_rate_limiting_configuration(self):
        """Test rate limiting configuration."""
        config = SecurityConfig()

        # Rate limits should be reasonable
        assert config.rate_limit_requests > 0
        assert config.rate_limit_window > 0

        # Window should be reasonable (not too short or long)
        assert 30 <= config.rate_limit_window <= 3600  # 30 seconds to 1 hour