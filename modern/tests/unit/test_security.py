"""
Unit Tests for Security Infrastructure

Comprehensive tests for all security components including encryption,
secret management, JWT, password hashing, and audit logging.
"""

import pytest
import json
import base64
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

import sys
from pathlib import Path

# Add the modern/src directory to Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.security import (
    EncryptionService,
    SecretManager,
    JWTService,
    PasswordService,
    APIKeyService,
    SecurityAuditLogger
)
from core.config import AppConfig, Environment, SecurityConfig


class TestEncryptionService:
    """Test EncryptionService."""

    def test_encryption_service_initialization(self):
        """Test encryption service initialization."""
        service = EncryptionService()

        assert service.key is not None
        assert len(service.key) == 44  # Fernet key length
        assert isinstance(service.key, bytes)

    def test_encryption_service_from_password(self):
        """Test encryption service creation from password."""
        service = EncryptionService.from_password("test_password")

        assert service.key is not None
        assert len(service.key) == 44

        # Same password should produce same key
        service2 = EncryptionService.from_password("test_password")
        assert service.key == service2.key

    def test_string_encryption_decryption(self):
        """Test string encryption and decryption."""
        service = EncryptionService()
        original_text = "Hello, World!"

        encrypted = service.encrypt(original_text)
        decrypted = service.decrypt(encrypted)

        assert decrypted == original_text
        assert encrypted != original_text

    def test_dict_encryption_decryption(self):
        """Test dictionary encryption and decryption."""
        service = EncryptionService()
        original_data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        encrypted = service.encrypt(original_data)
        decrypted = service.decrypt(encrypted)

        assert decrypted == original_data
        assert isinstance(decrypted, dict)

    def test_bytes_encryption_decryption(self):
        """Test bytes encryption and decryption."""
        service = EncryptionService()
        original_bytes = b"Hello, World!"

        encrypted = service.encrypt(original_bytes)
        decrypted = service.decrypt(encrypted)

        assert decrypted == original_bytes.decode('utf-8')

    def test_key_rotation(self):
        """Test encryption key rotation."""
        service = EncryptionService()
        original_key = service.key

        new_service = service.rotate_key()
        new_key = new_service.key

        assert new_key != original_key
        assert len(new_key) == 44

    def test_invalid_encryption_key(self):
        """Test invalid encryption key handling."""
        with pytest.raises(ValueError):
            # Try to decrypt invalid data
            service = EncryptionService()
            service.decrypt("invalid_encrypted_data")

    def test_different_keys_produce_different_results(self):
        """Test that different keys produce different encryption results."""
        service1 = EncryptionService()
        service2 = EncryptionService()

        original_text = "Test message"
        encrypted1 = service1.encrypt(original_text)
        encrypted2 = service2.encrypt(original_text)

        assert encrypted1 != encrypted2

        # Each service can only decrypt its own data
        assert service1.decrypt(encrypted1) == original_text
        assert service2.decrypt(encrypted2) == original_text

        # Cross-decryption should fail
        with pytest.raises(ValueError):
            service1.decrypt(encrypted2)
        with pytest.raises(ValueError):
            service2.decrypt(encrypted1)


class TestSecretManager:
    """Test SecretManager."""

    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test configuration."""
        config = AppConfig(
            config_dir=tmp_path / "config",
            security=SecurityConfig(jwt_secret_key="test_jwt_secret")
        )
        return config

    @pytest.fixture
    def encryption_service(self):
        """Create test encryption service."""
        return EncryptionService.from_password("test_password")

    @pytest.fixture
    def secret_manager(self, test_config, encryption_service):
        """Create test secret manager."""
        return SecretManager(test_config, encryption_service)

    def test_secret_manager_initialization(self, secret_manager):
        """Test secret manager initialization."""
        assert secret_manager.config is not None
        assert secret_manager.encryption is not None
        assert isinstance(secret_manager._secrets, dict)

    def test_set_and_get_secret(self, secret_manager):
        """Test setting and getting secrets."""
        # Set a secret
        secret_manager.set_secret("api_key", "secret_value", "api")

        # Get the secret
        retrieved = secret_manager.get_secret("api_key", "api")
        assert retrieved == "secret_value"

    def test_get_nonexistent_secret(self, secret_manager):
        """Test getting non-existent secret."""
        result = secret_manager.get_secret("nonexistent", "api")
        assert result is None

    def test_delete_secret(self, secret_manager):
        """Test deleting secrets."""
        # Set and verify secret exists
        secret_manager.set_secret("temp_secret", "value", "temp")
        assert secret_manager.get_secret("temp_secret", "temp") == "value"

        # Delete secret
        result = secret_manager.delete_secret("temp_secret", "temp")
        assert result is True

        # Verify secret is gone
        assert secret_manager.get_secret("temp_secret", "temp") is None

    def test_delete_nonexistent_secret(self, secret_manager):
        """Test deleting non-existent secret."""
        result = secret_manager.delete_secret("nonexistent", "temp")
        assert result is False

    def test_list_secrets(self, secret_manager):
        """Test listing secrets."""
        # Set some secrets
        secret_manager.set_secret("key1", "value1", "cat1")
        secret_manager.set_secret("key2", "value2", "cat1")
        secret_manager.set_secret("key3", "value3", "cat2")

        # List all secrets
        all_secrets = secret_manager.list_secrets()
        assert "cat1" in all_secrets
        assert "cat2" in all_secrets
        assert len(all_secrets["cat1"]) == 2
        assert len(all_secrets["cat2"]) == 1

        # List secrets from specific category
        cat1_secrets = secret_manager.list_secrets("cat1")
        assert "key1" in cat1_secrets["cat1"]
        assert "key2" in cat1_secrets["cat1"]
        assert "cat2" not in cat1_secrets

    def test_persistence(self, secret_manager, tmp_path):
        """Test secret persistence to file."""
        # Set a secret
        secret_manager.set_secret("persistent", "value", "test")

        # Create new secret manager (simulating restart)
        new_manager = SecretManager(secret_manager.config, secret_manager.encryption)

        # Verify secret was loaded
        retrieved = new_manager.get_secret("persistent", "test")
        assert retrieved == "value"

    def test_key_rotation(self, secret_manager):
        """Test encryption key rotation."""
        # Set a secret
        secret_manager.set_secret("test_key", "test_value", "test")

        # Rotate key
        secret_manager.rotate_encryption_key()

        # Verify secret is still accessible
        retrieved = secret_manager.get_secret("test_key", "test")
        assert retrieved == "test_value"


class TestJWTService:
    """Test JWTService."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return AppConfig(
            security=SecurityConfig(
                jwt_secret_key="test_jwt_secret_key_12345",
                jwt_algorithm="HS256",
                jwt_expiration_hours=1
            )
        )

    @pytest.fixture
    def jwt_service(self, test_config):
        """Create test JWT service."""
        return JWTService(test_config)

    def test_jwt_service_initialization(self, jwt_service, test_config):
        """Test JWT service initialization."""
        assert jwt_service.secret_key == "test_jwt_secret_key_12345"
        assert jwt_service.algorithm == "HS256"
        assert jwt_service.expiration_hours == 1

    def test_create_and_verify_token(self, jwt_service):
        """Test JWT token creation and verification."""
        data = {"user_id": "123", "role": "admin"}

        # Create token
        token = jwt_service.create_token(data)
        assert isinstance(token, str)
        assert len(token) > 0

        # Verify token
        decoded = jwt_service.verify_token(token)
        assert decoded is not None
        assert decoded["user_id"] == "123"
        assert decoded["role"] == "admin"
        assert "exp" in decoded

    def test_verify_expired_token(self, jwt_service):
        """Test verification of expired token."""
        data = {"user_id": "123"}

        # Create token that expires immediately
        jwt_service.expiration_hours = 0
        token = jwt_service.create_token(data)

        # Verify token (should fail due to expiration)
        decoded = jwt_service.verify_token(token)
        assert decoded is None

    def test_verify_invalid_token(self, jwt_service):
        """Test verification of invalid token."""
        invalid_token = "invalid.jwt.token"

        decoded = jwt_service.verify_token(invalid_token)
        assert decoded is None

    def test_refresh_token_workflow(self, jwt_service):
        """Test refresh token workflow."""
        data = {"user_id": "123", "role": "user"}

        # Create refresh token
        refresh_token = jwt_service.create_refresh_token(data)
        assert isinstance(refresh_token, str)

        # Use refresh token to get new access token
        new_access_token = jwt_service.refresh_access_token(refresh_token)
        assert new_access_token is not None
        assert isinstance(new_access_token, str)

        # Verify new access token
        decoded = jwt_service.verify_token(new_access_token)
        assert decoded is not None
        assert decoded["user_id"] == "123"
        assert decoded["role"] == "user"

    def test_refresh_with_invalid_token(self, jwt_service):
        """Test refresh token with invalid token."""
        invalid_refresh_token = "invalid.refresh.token"

        new_token = jwt_service.refresh_access_token(invalid_refresh_token)
        assert new_token is None


class TestPasswordService:
    """Test PasswordService."""

    @pytest.fixture
    def password_service(self):
        """Create test password service."""
        return PasswordService()

    def test_password_hashing(self, password_service):
        """Test password hashing."""
        password = "my_secure_password"

        # Hash password
        hashed = password_service.hash_password(password)
        assert hashed != password
        assert len(hashed) > 0

        # Verify password
        is_valid = password_service.verify_password(password, hashed)
        assert is_valid is True

    def test_password_verification_wrong_password(self, password_service):
        """Test password verification with wrong password."""
        password = "correct_password"
        wrong_password = "wrong_password"

        hashed = password_service.hash_password(password)

        is_valid = password_service.verify_password(wrong_password, hashed)
        assert is_valid is False

    def test_password_hash_consistency(self, password_service):
        """Test that same password produces different hashes (due to salt)."""
        password = "test_password"

        hash1 = password_service.hash_password(password)
        hash2 = password_service.hash_password(password)

        # Hashes should be different due to unique salt
        assert hash1 != hash2

        # But both should verify the same password
        assert password_service.verify_password(password, hash1)
        assert password_service.verify_password(password, hash2)

    def test_get_password_hash_info(self, password_service):
        """Test password hash information retrieval."""
        password = "test_password"
        hashed = password_service.hash_password(password)

        info = password_service.get_password_hash_info(hashed)

        assert "scheme" in info
        assert "deprecated" in info
        assert info["scheme"].name == "bcrypt"


class TestAPIKeyService:
    """Test APIKeyService."""

    @pytest.fixture
    def api_key_service(self):
        """Create test API key service."""
        return APIKeyService("test_secret_key")

    def test_generate_api_key(self, api_key_service):
        """Test API key generation."""
        keys = api_key_service.generate_api_key("test")

        assert "key_id" in keys
        assert "secret" in keys
        assert keys["key_id"].startswith("test_")
        assert len(keys["secret"]) == 64  # 32 bytes hex

    def test_signature_creation_and_validation(self, api_key_service):
        """Test signature creation and validation."""
        keys = api_key_service.generate_api_key()
        message = "GET /api/trades"
        secret = keys["secret"]

        # Create signature
        signature = api_key_service.create_signature(secret, message)
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex length

        # Validate signature
        is_valid = api_key_service.validate_request(
            keys["key_id"], secret, message, signature
        )
        assert is_valid is True

    def test_invalid_signature_validation(self, api_key_service):
        """Test validation with invalid signature."""
        keys = api_key_service.generate_api_key()
        message = "GET /api/trades"
        invalid_signature = "invalid_signature"

        is_valid = api_key_service.validate_request(
            keys["key_id"], keys["secret"], message, invalid_signature
        )
        assert is_valid is False

    def test_different_message_invalidates_signature(self, api_key_service):
        """Test that different message invalidates signature."""
        keys = api_key_service.generate_api_key()
        original_message = "GET /api/trades"
        different_message = "POST /api/trades"

        # Create signature for original message
        signature = api_key_service.create_signature(keys["secret"], original_message)

        # Try to validate with different message
        is_valid = api_key_service.validate_request(
            keys["key_id"], keys["secret"], different_message, signature
        )
        assert is_valid is False


class TestSecurityAuditLogger:
    """Test SecurityAuditLogger."""

    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test configuration."""
        return AppConfig(
            log_dir=tmp_path / "logs",
            environment=Environment.TESTING
        )

    @pytest.fixture
    def audit_logger(self, test_config):
        """Create test security audit logger."""
        return SecurityAuditLogger(test_config)

    def test_audit_logger_initialization(self, audit_logger, test_config):
        """Test audit logger initialization."""
        assert audit_logger.config == test_config
        assert audit_logger.logger is not None

    @patch('logging.FileHandler')
    def test_log_auth_event(self, mock_file_handler, audit_logger):
        """Test authentication event logging."""
        mock_handler = MagicMock()
        mock_file_handler.return_value = mock_handler

        audit_logger.log_auth_event(
            "login_success",
            user_id="user123",
            ip_address="192.168.1.1",
            details={"user_agent": "test_agent"}
        )

        # Verify logger was called
        assert audit_logger.logger.info.called

    @patch('logging.FileHandler')
    def test_log_security_event(self, mock_file_handler, audit_logger):
        """Test security event logging."""
        mock_handler = MagicMock()
        mock_file_handler.return_value = mock_handler

        audit_logger.log_security_event(
            "unauthorized_access",
            "high",
            {"endpoint": "/api/admin", "ip": "192.168.1.1"}
        )

        # Verify logger was called
        assert audit_logger.logger.error.called

    @patch('logging.FileHandler')
    def test_log_api_access(self, mock_file_handler, audit_logger):
        """Test API access logging."""
        mock_handler = MagicMock()
        mock_file_handler.return_value = mock_handler

        audit_logger.log_api_access(
            "/api/trades",
            "GET",
            200,
            user_id="user123",
            ip_address="192.168.1.1"
        )

        # Verify logger was called
        assert audit_logger.logger.info.called


class TestSecurityIntegration:
    """Test security components integration."""

    @pytest.fixture
    def test_config(self, tmp_path):
        """Create comprehensive test configuration."""
        return AppConfig(
            config_dir=tmp_path / "config",
            log_dir=tmp_path / "logs",
            security=SecurityConfig(
                jwt_secret_key="integration_test_secret_12345",
                jwt_algorithm="HS256",
                jwt_expiration_hours=2
            ),
            environment=Environment.TESTING
        )

    def test_encryption_secret_manager_integration(self, test_config):
        """Test encryption service and secret manager integration."""
        from ...src.core.security import init_security_services

        # Initialize security services
        init_security_services(test_config)

        # Get services
        encryption = test_config.security.jwt_secret_key.get_secret_value()
        secret_manager = test_config.config_dir / "secrets.enc"

        # Test secret storage and retrieval
        from ...src.core.security import get_secret_manager
        manager = get_secret_manager()

        # Store a secret
        manager.set_secret("integration_test", "secret_value", "test")

        # Retrieve the secret
        retrieved = manager.get_secret("integration_test", "test")
        assert retrieved == "secret_value"

    def test_jwt_password_integration(self, test_config):
        """Test JWT and password service integration."""
        from ...src.core.security import init_security_services

        # Initialize security services
        init_security_services(test_config)

        from ...src.core.security import get_jwt_service, get_password_service

        jwt_service = get_jwt_service()
        password_service = get_password_service()

        # Test password hashing
        password = "integration_test_password"
        hashed = password_service.hash_password(password)

        # Create JWT with password hash
        token_data = {"password_hash": hashed}
        token = jwt_service.create_token(token_data)

        # Verify token
        decoded = jwt_service.verify_token(token)
        assert decoded is not None
        assert decoded["password_hash"] == hashed

        # Verify password
        is_valid = password_service.verify_password(password, decoded["password_hash"])
        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
