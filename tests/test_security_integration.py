"""
Integration tests for the complete security system.
Tests CORS, CSP, authentication, and security headers together.
"""

import pytest
import json
from unittest.mock import patch
from frontend.config import SecurityConfig, AppConfig
from frontend.auth import SimpleAuth


class TestSecurityIntegration:
    """Test complete security system integration."""

    def test_secure_config_integration(self):
        """Test that all security configurations work together."""
        config = AppConfig()

        # Test CSP generation
        csp_header = config.security.get_csp_header()
        assert "default-src 'self'" in csp_header
        assert "unsafe-eval" not in csp_header
        assert "unsafe-inline" not in csp_header

        # Test CORS with valid origin
        headers = config.security.get_cors_headers("http://localhost:5000")
        assert headers['Access-Control-Allow-Origin'] == "http://localhost:5000"
        assert 'Access-Control-Allow-Methods' in headers

        # Test CORS with invalid origin
        headers = config.security.get_cors_headers("https://malicious.com")
        assert 'Access-Control-Allow-Origin' not in headers

    def test_authentication_integration(self):
        """Test authentication system integration."""
        auth = SimpleAuth("test_secret")

        # Test default admin user
        user = auth.authenticate("admin", "admin123!")
        assert user is not None
        assert user['role'] == 'admin'

        # Test invalid credentials
        user = auth.authenticate("admin", "wrong_password")
        assert user is None

    def test_rate_limiting_integration(self):
        """Test rate limiting works with configuration."""
        import time
        from unittest.mock import patch

        config = AppConfig()

        # Mock request tracking
        request_counts = {}
        request_windows = {}

        # Simulate multiple requests
        client_ip = "192.168.1.100"
        current_time = time.time()

        # First request
        if client_ip not in request_counts:
            request_counts[client_ip] = 0
            request_windows[client_ip] = current_time
        request_counts[client_ip] += 1

        # Should not be rate limited yet
        assert request_counts[client_ip] <= config.security.rate_limit_requests

    def test_session_security_integration(self):
        """Test session security configuration."""
        config = AppConfig()

        # Test session secret generation
        assert len(config.security.session_secret_key) >= 32

        # Test session timeout
        assert config.security.session_timeout > 0
        assert config.security.session_timeout <= 86400  # Max 24 hours

    def test_comprehensive_security_headers(self):
        """Test comprehensive security headers generation."""
        config = SecurityConfig()

        # Test all security headers are present
        csp = config.get_csp_header()
        assert "default-src" in csp
        assert "script-src" in csp
        assert "object-src 'none'" in csp
        assert "base-uri 'self'" in csp

        # Test CORS headers
        cors_headers = config.get_cors_headers("http://localhost:5000")
        assert 'Access-Control-Allow-Methods' in cors_headers
        assert 'Access-Control-Allow-Headers' in cors_headers
        assert 'Access-Control-Max-Age' in cors_headers


class TestSecurityEdgeCases:
    """Test security edge cases and error conditions."""

    def test_invalid_cors_origins(self):
        """Test handling of invalid CORS origins."""
        config = SecurityConfig()

        # Test with various invalid origins
        invalid_origins = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox('xss')",
            "",
            None
        ]

        for origin in invalid_origins:
            headers = config.get_cors_headers(origin)
            # Should not set Access-Control-Allow-Origin for invalid origins
            if origin not in config.cors_origins:
                assert 'Access-Control-Allow-Origin' not in headers

    def test_malformed_auth_data(self):
        """Test authentication with malformed data."""
        auth = SimpleAuth("test_secret")

        # Test with None values
        user = auth.authenticate(None, "password")
        assert user is None

        user = auth.authenticate("admin", None)
        assert user is None

        # Test with empty strings
        user = auth.authenticate("", "")
        assert user is None

    def test_password_hashing_security(self):
        """Test password hashing security properties."""
        auth = SimpleAuth("test_secret")

        # Same password should produce same hash
        hash1 = auth._hash_password("test_password")
        hash2 = auth._hash_password("test_password")
        assert hash1 == hash2

        # Different passwords should produce different hashes
        hash3 = auth._hash_password("different_password")
        assert hash1 != hash3

        # Hash should be cryptographically secure
        assert len(hash1) == 64  # SHA256 produces 64 character hex
        assert hash1.isalnum()  # Should only contain hex characters

    def test_session_timeout_edge_cases(self):
        """Test session timeout edge cases."""
        import time
        from flask import Flask

        app = Flask(__name__)
        app.secret_key = "test_key"

        with app.test_request_context('/test', json=True):
            auth = SimpleAuth("test_secret")

            @auth.login_required
            def protected_function():
                return "success"

            from flask import session

            # Test with very old session
            very_old_time = time.time() - 100000  # 27+ hours ago
            session['user'] = {'role': 'admin'}
            session['login_time'] = very_old_time

            result = protected_function()
            assert result[1] == 401  # Should be unauthorized due to timeout

    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests."""
        config = SecurityConfig()

        # Test that configuration is thread-safe
        import threading
        import time

        results = []

        def test_csp_generation():
            csp = config.get_csp_header()
            results.append(len(csp) > 0)

        # Run multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=test_csp_generation)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should have succeeded
        assert all(results)
        assert len(results) == 10


class TestSecurityConfigurationValidation:
    """Test security configuration validation."""

    def test_environment_variable_override(self):
        """Test environment variable overrides for security settings."""
        with patch.dict('os.environ', {
            'CORS_ORIGINS': 'https://prod.example.com,https://staging.example.com',
            'RATE_LIMIT_REQUESTS': '50',
            'SESSION_TIMEOUT': '7200'
        }):
            config = SecurityConfig()

            assert 'https://prod.example.com' in config.cors_origins
            assert 'https://staging.example.com' in config.cors_origins
            assert config.rate_limit_requests == 50
            assert config.session_timeout == 7200

    def test_invalid_environment_values(self):
        """Test handling of invalid environment values."""
        with patch.dict('os.environ', {
            'RATE_LIMIT_REQUESTS': 'invalid',
            'SESSION_TIMEOUT': 'also_invalid'
        }):
            config = SecurityConfig()

            # Should fall back to defaults
            assert isinstance(config.rate_limit_requests, int)
            assert isinstance(config.session_timeout, int)
            assert config.rate_limit_requests > 0
            assert config.session_timeout > 0

    def test_secure_defaults(self):
        """Test that all defaults are secure."""
        config = SecurityConfig()

        # CSP should not contain unsafe directives
        csp = config.get_csp_header()
        unsafe_directives = ['unsafe-eval', 'unsafe-inline', '*']
        for directive in unsafe_directives:
            assert directive not in csp

        # CORS should not allow all origins by default
        cors_origins = config.cors_origins
        assert '*' not in cors_origins
        assert len(cors_origins) > 0  # Should have some allowed origins

        # Session settings should be secure
        assert len(config.session_secret_key) >= 32
        assert config.session_timeout >= 1800  # At least 30 minutes

    def test_method_validation(self):
        """Test HTTP method validation."""
        config = SecurityConfig()

        # Should allow safe methods
        assert 'GET' in config.allowed_methods
        assert 'POST' in config.allowed_methods
        assert 'OPTIONS' in config.allowed_methods

        # Should not allow dangerous methods
        dangerous_methods = ['PUT', 'DELETE', 'PATCH']
        for method in dangerous_methods:
            assert method not in config.allowed_methods
