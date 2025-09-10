"""
Unit tests for frontend authentication system.
Ensures authentication and authorization work correctly.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from frontend.auth import SimpleAuth, get_auth, login_required, admin_required, api_key_required


class TestSimpleAuth:
    """Test SimpleAuth functionality."""

    def test_auth_initialization(self):
        """Test authentication system initialization."""
        auth = SimpleAuth("test_secret")
        assert auth.secret_key == "test_secret"
        assert isinstance(auth.users, dict)
        assert "admin" in auth.users  # Default admin user should be created

    def test_password_hashing(self):
        """Test password hashing."""
        auth = SimpleAuth("test_secret")
        hash1 = auth._hash_password("password123")
        hash2 = auth._hash_password("password123")
        hash3 = auth._hash_password("different_password")

        assert hash1 == hash2  # Same password should produce same hash
        assert hash1 != hash3  # Different passwords should produce different hashes
        assert len(hash1) == 64  # SHA256 produces 64 character hex string

    def test_user_authentication(self):
        """Test user authentication."""
        auth = SimpleAuth("test_secret")

        # Test valid credentials
        user = auth.authenticate("admin", "admin123!")
        assert user is not None
        assert user['username'] == "admin"
        assert user['role'] == "admin"

        # Test invalid credentials
        user = auth.authenticate("admin", "wrong_password")
        assert user is None

        # Test non-existent user
        user = auth.authenticate("nonexistent", "password")
        assert user is None

    def test_disabled_user(self):
        """Test authentication with disabled user."""
        auth = SimpleAuth("test_secret")

        # Disable admin user
        auth.users['admin']['enabled'] = False

        # Should not authenticate
        user = auth.authenticate("admin", "admin123!")
        assert user is None

    def test_custom_users_file(self):
        """Test loading users from custom file."""
        # Create temporary users file
        users_data = {
            "testuser": {
                "username": "testuser",
                "password_hash": "hashed_password",
                "role": "user",
                "enabled": True,
                "created_at": 1234567890
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(users_data, f)
            temp_file = f.name

        try:
            auth = SimpleAuth("test_secret", temp_file)
            assert "testuser" in auth.users
            assert auth.users["testuser"]["role"] == "user"
        finally:
            os.unlink(temp_file)


class TestAuthDecorators:
    """Test authentication decorators."""

    def test_login_required_decorator(self):
        """Test login_required decorator."""
        from flask import Flask
        app = Flask(__name__)
        app.secret_key = "test_secret_key"

        # Add a login route for testing
        @app.route('/login')
        def login():
            return "login page"

        with app.test_request_context('/protected'):
            auth = SimpleAuth("test_secret")

            @auth.login_required
            def protected_function():
                return "success"

            # Clear session for this test
            from flask import session
            session.clear()

            # Should redirect to login page
            result = protected_function()
            assert result.status_code == 302  # Redirect status

    def test_admin_required_decorator(self):
        """Test admin_required decorator."""
        from flask import Flask
        app = Flask(__name__)
        app.secret_key = "test_secret_key"

        with app.test_request_context('/admin', json=True):
            auth = SimpleAuth("test_secret")

            @auth.admin_required
            def admin_function():
                return "admin_success"

            # First test: no user in session (should return 401)
            from flask import session
            session.clear()

            result = admin_function()
            assert result[1] == 401  # Unauthorized

            # Second test: user with non-admin role (should return 403)
            import time
            session['user'] = {'role': 'user'}
            session['login_time'] = time.time()  # Set login time to pass timeout check

            result = admin_function()
            assert result[1] == 403  # Forbidden

    def test_api_key_required_decorator(self):
        """Test API key required decorator."""
        from flask import Flask
        app = Flask(__name__)

        with app.test_request_context('/api/test', headers={"X-API-Key": "test_key"}):
            auth = SimpleAuth("test_secret")

            @auth.api_key_required
            def api_function():
                return "api_success"

            # Valid API key - should succeed
            result = api_function()
            assert result == "api_success"

        # Test without API key
        with app.test_request_context('/api/test'):
            @auth.api_key_required
            def api_function_no_key():
                return "should_not_reach_here"

            result = api_function_no_key()
            assert result[1] == 401


class TestGlobalAuth:
    """Test global authentication functions."""

    def test_get_auth(self):
        """Test get_auth function."""
        with patch('frontend.config.get_config') as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.security.session_secret_key = "test_key"
            mock_config.return_value = mock_config_instance

            auth = get_auth()
            assert isinstance(auth, SimpleAuth)
            assert len(auth.secret_key) > 0  # Should have a secret key

    def test_convenience_decorators(self):
        """Test convenience decorator functions."""
        with patch('frontend.auth.get_auth') as mock_get_auth:
            mock_auth = MagicMock()
            mock_get_auth.return_value = mock_auth

            # Test that decorators are called
            func = MagicMock()
            decorated = login_required(func)
            assert decorated is not None

            decorated = admin_required(func)
            assert decorated is not None

            decorated = api_key_required(func)
            assert decorated is not None


class TestSecurityBestPractices:
    """Test security best practices in authentication."""

    def test_session_timeout(self):
        """Test session timeout functionality."""
        import time
        from flask import Flask
        app = Flask(__name__)
        app.secret_key = "test_secret_key"

        with app.test_request_context('/protected', json=True):
            auth = SimpleAuth("test_secret")

            @auth.login_required
            def protected_function():
                return "success"

            # Set session with old login time
            from flask import session
            session['user'] = {'role': 'admin'}
            session['login_time'] = time.time() - 3700  # More than 1 hour ago

            # Should return session expired
            result = protected_function()
            assert result[1] == 401

    def test_secure_password_hashing(self):
        """Test that password hashing uses secure methods."""
        auth = SimpleAuth("test_secret")

        # Hash should be different from plain password
        password = "test_password"
        hashed = auth._hash_password(password)

        assert hashed != password
        assert len(hashed) >= 32  # At least 128 bits of entropy

        # Hash should use HMAC
        assert hashed.isalnum()  # Should be hex characters only

    def test_default_admin_warning(self):
        """Test that default admin credentials generate warning."""
        import tempfile
        import os

        # Create temporary directory for users file
        with tempfile.TemporaryDirectory() as temp_dir:
            users_file = os.path.join(temp_dir, "users.json")

            with patch('builtins.print') as mock_print:
                auth = SimpleAuth("test_secret", users_file)

                # Should print warning about default credentials
                mock_print.assert_called()
                # Check that at least one print call contains expected content
                print_calls = [str(call[0][0]) for call in mock_print.call_args_list]
                warning_found = any("admin123!" in call for call in print_calls)
                assert warning_found, f"Expected warning about admin123! not found in: {print_calls}"
