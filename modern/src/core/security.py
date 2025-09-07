"""
Security Infrastructure

Enterprise-grade security infrastructure with proper secret management,
encryption, authentication, and authorization.
"""

import os
import hashlib
import hmac
import secrets
import base64
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from pydantic import BaseModel, SecretStr, validator
import jwt
from passlib.context import CryptContext

from .config import AppConfig, get_settings


logger = logging.getLogger(__name__)


class EncryptionService:
    """
    Encryption service for sensitive data.

    Provides symmetric encryption using Fernet (AES 128) for data at rest
    and in transit.
    """

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryption service.

        Args:
            key: Encryption key. If None, generates a new key.
        """
        if key is None:
            key = Fernet.generate_key()
        self._fernet = Fernet(key)
        self._key = key

    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> 'EncryptionService':
        """
        Create encryption service from password using PBKDF2.

        Args:
            password: Password to derive key from.
            salt: Salt for key derivation. If None, uses default salt.

        Returns:
            EncryptionService: Configured encryption service.
        """
        if salt is None:
            salt = b'legacy_coin_trader_salt'

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return cls(key)

    @property
    def key(self) -> bytes:
        """Get the encryption key."""
        return self._key

    def encrypt(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """
        Encrypt data.

        Args:
            data: Data to encrypt (string, bytes, or dict).

        Returns:
            str: Base64-encoded encrypted data.
        """
        if isinstance(data, dict):
            data = json.dumps(data, default=str)
        elif isinstance(data, str):
            data = data.encode('utf-8')

        encrypted = self._fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')

    def decrypt(self, encrypted_data: str) -> Union[str, Dict[str, Any]]:
        """
        Decrypt data.

        Args:
            encrypted_data: Base64-encoded encrypted data.

        Returns:
            Union[str, Dict[str, Any]]: Decrypted data.

        Raises:
            ValueError: If decryption fails.
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
            decrypted_str = decrypted_bytes.decode('utf-8')

            # Try to parse as JSON
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data") from e

    def rotate_key(self, new_key: Optional[bytes] = None) -> 'EncryptionService':
        """
        Rotate encryption key.

        Args:
            new_key: New encryption key. If None, generates a new key.

        Returns:
            EncryptionService: New encryption service with rotated key.
        """
        if new_key is None:
            new_key = Fernet.generate_key()

        return EncryptionService(new_key)


class SecretManager:
    """
    Secret management service.

    Handles secure storage and retrieval of sensitive configuration data
    with encryption and access controls.
    """

    def __init__(self, config: AppConfig, encryption_service: Optional[EncryptionService] = None):
        """
        Initialize secret manager.

        Args:
            config: Application configuration.
            encryption_service: Encryption service. If None, creates default.
        """
        self.config = config
        self.encryption = encryption_service or EncryptionService()
        self._secrets: Dict[str, Dict[str, Any]] = {}
        self._secrets_file = config.config_dir / "secrets.enc"

        # Load existing secrets if file exists
        if self._secrets_file.exists():
            self._load_secrets()

    def _load_secrets(self) -> None:
        """Load encrypted secrets from file."""
        try:
            with open(self._secrets_file, 'r') as f:
                encrypted_data = f.read().strip()

            if encrypted_data:
                self._secrets = self.encryption.decrypt(encrypted_data)

        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            self._secrets = {}

    def _save_secrets(self) -> None:
        """Save encrypted secrets to file."""
        try:
            encrypted_data = self.encryption.encrypt(self._secrets)

            with open(self._secrets_file, 'w') as f:
                f.write(encrypted_data)

            # Set restrictive permissions
            self._secrets_file.chmod(0o600)

        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise

    def set_secret(self, key: str, value: Any, category: str = "general") -> None:
        """
        Store a secret securely.

        Args:
            key: Secret key.
            value: Secret value.
            category: Secret category for organization.
        """
        if category not in self._secrets:
            self._secrets[category] = {}

        self._secrets[category][key] = {
            "value": value,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        self._save_secrets()
        logger.info(f"Secret '{key}' in category '{category}' updated")

    def get_secret(self, key: str, category: str = "general") -> Optional[Any]:
        """
        Retrieve a secret.

        Args:
            key: Secret key.
            category: Secret category.

        Returns:
            Optional[Any]: Secret value if found, None otherwise.
        """
        try:
            return self._secrets.get(category, {}).get(key, {}).get("value")
        except Exception as e:
            logger.error(f"Failed to retrieve secret '{key}': {e}")
            return None

    def delete_secret(self, key: str, category: str = "general") -> bool:
        """
        Delete a secret.

        Args:
            key: Secret key.
            category: Secret category.

        Returns:
            bool: True if deleted, False if not found.
        """
        if category in self._secrets and key in self._secrets[category]:
            del self._secrets[category][key]
            self._save_secrets()
            logger.info(f"Secret '{key}' in category '{category}' deleted")
            return True
        return False

    def list_secrets(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        List all secrets (without values for security).

        Args:
            category: Specific category to list. If None, lists all.

        Returns:
            Dict[str, Any]: Secret metadata.
        """
        if category:
            secrets = self._secrets.get(category, {})
        else:
            secrets = self._secrets

        # Return metadata without actual values
        result = {}
        for cat, cat_secrets in secrets.items():
            result[cat] = {}
            for key, data in cat_secrets.items():
                result[cat][key] = {
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at")
                }

        return result

    def rotate_encryption_key(self, new_key: Optional[bytes] = None) -> None:
        """
        Rotate the encryption key for all secrets.

        Args:
            new_key: New encryption key. If None, generates a new key.
        """
        old_encryption = self.encryption
        self.encryption = self.encryption.rotate_key(new_key)

        # Re-encrypt all secrets with new key
        self._save_secrets()

        logger.info("Encryption key rotated successfully")


class JWTService:
    """
    JWT (JSON Web Token) service for authentication and authorization.

    Provides token generation, validation, and refresh capabilities.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize JWT service.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.secret_key = config.security.jwt_secret_key.get_secret_value()
        self.algorithm = config.security.jwt_algorithm
        self.expiration_hours = config.security.jwt_expiration_hours

    def create_token(self, data: Dict[str, Any]) -> str:
        """
        Create a JWT token.

        Args:
            data: Data to encode in the token.

        Returns:
            str: JWT token.
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=self.expiration_hours)
        to_encode.update({"exp": expire})

        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token to verify.

        Returns:
            Optional[Dict[str, Any]]: Decoded token data if valid, None otherwise.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """
        Create a refresh token with longer expiration.

        Args:
            data: Data to encode in the refresh token.

        Returns:
            str: Refresh token.
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=30)  # 30 days for refresh
        to_encode.update({"exp": expire, "type": "refresh"})

        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return token

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Create new access token from refresh token.

        Args:
            refresh_token: Refresh token.

        Returns:
            Optional[str]: New access token if refresh token is valid.
        """
        payload = self.verify_token(refresh_token)
        if payload and payload.get("type") == "refresh":
            # Remove refresh-specific fields
            token_data = {k: v for k, v in payload.items() if k not in ["exp", "type"]}
            return self.create_token(token_data)
        return None


class PasswordService:
    """
    Password hashing and verification service.

    Uses bcrypt for secure password hashing.
    """

    def __init__(self):
        """Initialize password service with bcrypt context."""
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12
        )

    def hash_password(self, password: str) -> str:
        """
        Hash a password.

        Args:
            password: Plain text password.

        Returns:
            str: Hashed password.
        """
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password.
            hashed_password: Hashed password.

        Returns:
            bool: True if password matches hash.
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash_info(self, hashed_password: str) -> Dict[str, Any]:
        """
        Get information about a password hash.

        Args:
            hashed_password: Hashed password.

        Returns:
            Dict[str, Any]: Hash information.
        """
        return {
            "scheme": self.pwd_context.identify(hashed_password),
            "deprecated": self.pwd_context.deprecated.hashes(hashed_password),
            "rounds": self.pwd_context.rounds if hasattr(self.pwd_context, 'rounds') else None
        }


class APIKeyService:
    """
    API key generation and validation service.

    Provides secure API key generation and HMAC validation.
    """

    def __init__(self, secret_key: str):
        """
        Initialize API key service.

        Args:
            secret_key: Secret key for HMAC signing.
        """
        self.secret_key = secret_key.encode('utf-8')

    def generate_api_key(self, prefix: str = "lct") -> Dict[str, str]:
        """
        Generate a new API key pair.

        Args:
            prefix: Key prefix.

        Returns:
            Dict[str, str]: Dictionary with 'key_id' and 'secret'.
        """
        key_id = f"{prefix}_{secrets.token_hex(16)}"
        secret = secrets.token_hex(32)

        return {
            "key_id": key_id,
            "secret": secret
        }

    def validate_request(self, key_id: str, secret: str, message: str, signature: str) -> bool:
        """
        Validate an API request using HMAC.

        Args:
            key_id: API key ID.
            secret: API key secret.
            message: Request message to validate.
            signature: HMAC signature.

        Returns:
            bool: True if signature is valid.
        """
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected_signature, signature)

    def create_signature(self, secret: str, message: str) -> str:
        """
        Create HMAC signature for a message.

        Args:
            secret: API key secret.
            message: Message to sign.

        Returns:
            str: HMAC signature.
        """
        signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature


class SecurityAuditLogger:
    """
    Security audit logging service.

    Logs security-related events for compliance and monitoring.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize security audit logger.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)

        # Create audit log file handler
        audit_log_path = config.get_log_file_path("security_audit.log")
        handler = logging.FileHandler(audit_log_path)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def log_auth_event(self, event_type: str, user_id: Optional[str] = None,
                      ip_address: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log authentication event.

        Args:
            event_type: Type of authentication event.
            user_id: User ID if applicable.
            ip_address: IP address of the request.
            details: Additional event details.
        """
        log_data = {
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }

        self.logger.info(f"AUTH_EVENT: {json.dumps(log_data)}")

    def log_security_event(self, event_type: str, severity: str,
                          details: Dict[str, Any]) -> None:
        """
        Log security event.

        Args:
            event_type: Type of security event.
            severity: Event severity (low, medium, high, critical).
            details: Event details.
        """
        log_data = {
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }

        if severity in ["high", "critical"]:
            self.logger.error(f"SECURITY_EVENT: {json.dumps(log_data)}")
        else:
            self.logger.warning(f"SECURITY_EVENT: {json.dumps(log_data)}")

    def log_api_access(self, endpoint: str, method: str, status_code: int,
                      user_id: Optional[str] = None, ip_address: Optional[str] = None) -> None:
        """
        Log API access event.

        Args:
            endpoint: API endpoint accessed.
            method: HTTP method used.
            status_code: HTTP status code returned.
            user_id: User ID if authenticated.
            ip_address: IP address of the request.
        """
        log_data = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.logger.info(f"API_ACCESS: {json.dumps(log_data)}")


# Global instances
_encryption_service: Optional[EncryptionService] = None
_secret_manager: Optional[SecretManager] = None
_jwt_service: Optional[JWTService] = None
_password_service: Optional[PasswordService] = None
_api_key_service: Optional[APIKeyService] = None
_audit_logger: Optional[SecurityAuditLogger] = None


def init_security_services(config: AppConfig) -> None:
    """
    Initialize all security services.

    Args:
        config: Application configuration.
    """
    global _encryption_service, _secret_manager, _jwt_service, _password_service, _api_key_service, _audit_logger

    # Initialize encryption service
    master_key = os.getenv("MASTER_ENCRYPTION_KEY")
    if master_key:
        _encryption_service = EncryptionService.from_password(master_key)
    else:
        _encryption_service = EncryptionService()

    # Initialize secret manager
    _secret_manager = SecretManager(config, _encryption_service)

    # Initialize JWT service
    _jwt_service = JWTService(config)

    # Initialize password service
    _password_service = PasswordService()

    # Initialize API key service
    jwt_secret = config.security.jwt_secret_key.get_secret_value()
    _api_key_service = APIKeyService(jwt_secret)

    # Initialize audit logger
    _audit_logger = SecurityAuditLogger(config)


def get_encryption_service() -> EncryptionService:
    """Get the global encryption service instance."""
    if _encryption_service is None:
        raise RuntimeError("Security services not initialized. Call init_security_services() first.")
    return _encryption_service


def get_secret_manager() -> SecretManager:
    """Get the global secret manager instance."""
    if _secret_manager is None:
        raise RuntimeError("Security services not initialized. Call init_security_services() first.")
    return _secret_manager


def get_jwt_service() -> JWTService:
    """Get the global JWT service instance."""
    if _jwt_service is None:
        raise RuntimeError("Security services not initialized. Call init_security_services() first.")
    return _jwt_service


def get_password_service() -> PasswordService:
    """Get the global password service instance."""
    if _password_service is None:
        raise RuntimeError("Security services not initialized. Call init_security_services() first.")
    return _password_service


def get_api_key_service() -> APIKeyService:
    """Get the global API key service instance."""
    if _api_key_service is None:
        raise RuntimeError("Security services not initialized. Call init_security_services() first.")
    return _api_key_service


def get_audit_logger() -> SecurityAuditLogger:
    """Get the global security audit logger instance."""
    if _audit_logger is None:
        raise RuntimeError("Security services not initialized. Call init_security_services() first.")
    return _audit_logger


# Export all security components
__all__ = [
    "EncryptionService",
    "SecretManager",
    "JWTService",
    "PasswordService",
    "APIKeyService",
    "SecurityAuditLogger",
    "init_security_services",
    "get_encryption_service",
    "get_secret_manager",
    "get_jwt_service",
    "get_password_service",
    "get_api_key_service",
    "get_audit_logger",
]
