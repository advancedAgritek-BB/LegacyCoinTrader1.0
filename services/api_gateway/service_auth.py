"""Service-to-service authentication and token management."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import redis.asyncio as redis

from .config import ServiceTokenConfig


class ServiceTokenManager:
    """Manages service-to-service authentication tokens."""

    def __init__(
        self,
        config: ServiceTokenConfig,
        redis_client: Optional[redis.Redis] = None,
        secret_key: Optional[str] = None
    ):
        self.config = config
        self.redis = redis_client
        self.secret_key = secret_key or os.getenv("SERVICE_AUTH_SECRET", "default-service-secret")
        self.token_prefix = "service_token:"

    async def generate_service_token(self, service_name: str) -> str:
        """Generate a new token for a service."""
        if service_name not in self.config.allowed_services:
            raise ValueError(f"Service '{service_name}' is not in allowed services list")

        if self.redis:
            # Generate a random token when Redis is available
            token = secrets.token_hex(self.config.token_length // 2)

            # Create a hash of the token for storage
            token_hash = hashlib.sha256(token.encode()).hexdigest()

            # Store in Redis with expiration
            key = f"{self.token_prefix}{service_name}"
            await self.redis.setex(
                key,
                timedelta(days=self.config.token_rotation_days),
                token_hash
            )
        else:
            # Use deterministic token when Redis is unavailable
            # This ensures tokens can be validated in fallback mode
            token = hashlib.sha256(
                f"{service_name}:{self.secret_key}".encode()
            ).hexdigest()[:self.config.token_length]

        return token

    async def validate_service_token(self, service_name: str, token: str) -> bool:
        """Validate a service token."""
        if service_name not in self.config.allowed_services:
            return False

        if not self.redis:
            # Fallback validation without Redis
            return self._validate_token_fallback(service_name, token)

        key = f"{self.token_prefix}{service_name}"
        stored_hash = await self.redis.get(key)

        if not stored_hash:
            return False

        # Compare token hash with stored hash
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return hmac.compare_digest(token_hash, stored_hash.decode())

    def _validate_token_fallback(self, service_name: str, token: str) -> bool:
        """Fallback validation when Redis is not available."""
        # Create a deterministic token based on service name and secret
        expected_token = hashlib.sha256(
            f"{service_name}:{self.secret_key}".encode()
        ).hexdigest()[:self.config.token_length]

        return hmac.compare_digest(token, expected_token)

    async def rotate_service_token(self, service_name: str) -> str:
        """Rotate (regenerate) a service token."""
        return await self.generate_service_token(service_name)

    async def revoke_service_token(self, service_name: str) -> bool:
        """Revoke a service token."""
        if not self.redis:
            return False

        key = f"{self.token_prefix}{service_name}"
        return bool(await self.redis.delete(key))

    async def list_active_services(self) -> List[str]:
        """List services with active tokens."""
        if not self.redis:
            return self.config.allowed_services.copy()

        keys = await self.redis.keys(f"{self.token_prefix}*")
        active_services = []

        for key in keys:
            service_name = key.decode().replace(self.token_prefix, "")
            if service_name in self.config.allowed_services:
                active_services.append(service_name)

        return active_services

    async def get_token_expiry(self, service_name: str) -> Optional[datetime]:
        """Get token expiry time for a service."""
        if not self.redis:
            return None

        key = f"{self.token_prefix}{service_name}"
        ttl = await self.redis.ttl(key)

        if ttl == -1:  # Key exists but no expiry
            return None
        elif ttl == -2:  # Key doesn't exist
            return None
        else:
            return datetime.utcnow() + timedelta(seconds=ttl)

    def create_service_header(self, service_name: str, token: str) -> Dict[str, str]:
        """Create HTTP headers for service authentication."""
        return {
            "X-Service-Name": service_name,
            "X-Service-Token": token,
            "X-Request-Timestamp": str(int(datetime.utcnow().timestamp())),
        }

    async def authenticate_service_request(
        self,
        service_name: str,
        token: str,
        timestamp: Optional[str] = None
    ) -> bool:
        """Authenticate a service request with optional timestamp validation."""
        # Validate service name
        if service_name not in self.config.allowed_services:
            return False

        # Validate token
        if not await self.validate_service_token(service_name, token):
            return False

        # Validate timestamp if provided (prevent replay attacks)
        if timestamp:
            try:
                request_time = datetime.fromtimestamp(int(timestamp))
                now = datetime.utcnow()

                # Allow 5-minute window for timestamp validation
                if abs((now - request_time).total_seconds()) > 300:
                    return False
            except (ValueError, OverflowError):
                return False

        return True
