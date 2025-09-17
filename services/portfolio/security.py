"""Identity and credential management for LegacyCoinTrader."""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from passlib.context import CryptContext
from sqlalchemy import select

from .config import DEFAULT_API_KEY_SECRET, PortfolioConfig
from .database import get_session
from .models import UserAccountModel

logger = logging.getLogger(__name__)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@dataclass(slots=True)
class AuthenticatedUser:
    """Representation of an authenticated account returned to callers."""

    id: int
    username: str
    role: str
    is_active: bool
    password_rotated_at: Optional[datetime]
    password_expires_at: Optional[datetime]
    api_key_hash: Optional[str]
    last_login_at: Optional[datetime]
    last_failed_login_at: Optional[datetime]
    failed_login_attempts: int
    created_at: datetime
    updated_at: datetime


class IdentityError(RuntimeError):
    """Base class for identity and credential errors."""


class AuthenticationError(IdentityError):
    """Raised when credentials are invalid."""


class InactiveUserError(AuthenticationError):
    """Raised when an account is disabled."""


class PasswordExpiredError(AuthenticationError):
    """Raised when a credential has exceeded the configured rotation window."""


class IdentityService:
    """Service responsible for credential lifecycle management."""

    def __init__(
        self,
        config: Optional[PortfolioConfig] = None,
        *,
        password_max_age_days: Optional[int] = None,
    ) -> None:
        self.config = config or PortfolioConfig.from_env()
        self.password_max_age_days = (
            password_max_age_days
            if password_max_age_days is not None
            else self.config.password_rotation_days
        )
        self._api_key_secret = self.config.api_key_secret.encode()

        if self.config.api_key_secret == DEFAULT_API_KEY_SECRET:
            logger.warning(
                "PORTFOLIO_API_KEY_SECRET is using the insecure default value. "
                "Override it in production deployments."
            )
        if self.password_max_age_days <= 0:
            logger.warning(
                "Password rotation is effectively disabled. "
                "Set PORTFOLIO_PASSWORD_ROTATION_DAYS to a positive integer to enforce it."
            )

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------
    @staticmethod
    def hash_password(password: str) -> str:
        """Return a strong hash for the supplied password."""

        if not password:
            raise ValueError("Password must not be empty")
        return _pwd_context.hash(password)

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a plaintext password against the stored hash."""

        if not password or not hashed:
            return False
        try:
            return _pwd_context.verify(password, hashed)
        except Exception:
            return False

    def hash_api_key(self, api_key: str) -> str:
        """Create a deterministic HMAC for an API key."""

        if not api_key:
            raise ValueError("API key must not be empty")
        return hmac.new(self._api_key_secret, api_key.encode(), "sha256").hexdigest()

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _to_user(self, model: UserAccountModel) -> AuthenticatedUser:
        return AuthenticatedUser(
            id=model.id,
            username=model.username,
            role=model.role,
            is_active=model.is_active,
            password_rotated_at=model.password_rotated_at,
            password_expires_at=model.password_expires_at,
            api_key_hash=model.api_key_hash,
            last_login_at=model.last_login_at,
            last_failed_login_at=model.last_failed_login_at,
            failed_login_attempts=model.failed_login_attempts,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def get_user(self, username: str) -> Optional[AuthenticatedUser]:
        """Retrieve a user by username without performing authentication."""

        if not username:
            return None

        with get_session(self.config) as session:
            stmt = select(UserAccountModel).where(UserAccountModel.username == username)
            model = session.execute(stmt).scalar_one_or_none()
            if model is None:
                return None
            session.expunge(model)
            return self._to_user(model)

    def list_users(self) -> List[AuthenticatedUser]:
        """Return every known user in the credential store."""

        with get_session(self.config) as session:
            stmt = select(UserAccountModel).order_by(UserAccountModel.username)
            models = session.execute(stmt).scalars().all()
            return [self._to_user(model) for model in models]

    # ------------------------------------------------------------------
    # Authentication flows
    # ------------------------------------------------------------------
    def authenticate_user(self, username: str, password: str) -> AuthenticatedUser:
        """Authenticate a user via username/password credentials."""

        if not username or not password:
            raise AuthenticationError("Invalid credentials")

        with get_session(self.config) as session:
            stmt = select(UserAccountModel).where(UserAccountModel.username == username)
            model = session.execute(stmt).scalar_one_or_none()
            if model is None:
                raise AuthenticationError("Invalid credentials")
            if not model.is_active:
                model.failed_login_attempts += 1
                model.last_failed_login_at = datetime.utcnow()
                session.add(model)
                raise InactiveUserError("Account is disabled")
            if not self.verify_password(password, model.password_hash):
                model.failed_login_attempts += 1
                model.last_failed_login_at = datetime.utcnow()
                session.add(model)
                raise AuthenticationError("Invalid credentials")
            if self._password_expired(model):
                raise PasswordExpiredError("Password rotation window exceeded")

            model.failed_login_attempts = 0
            model.last_failed_login_at = None
            model.last_login_at = datetime.utcnow()
            session.add(model)
            session.flush()
            return self._to_user(model)

    def authenticate_api_key(self, api_key: str) -> AuthenticatedUser:
        """Authenticate a service call via API key."""

        hashed = self.hash_api_key(api_key)
        with get_session(self.config) as session:
            stmt = select(UserAccountModel).where(UserAccountModel.api_key_hash == hashed)
            model = session.execute(stmt).scalar_one_or_none()
            if model is None:
                raise AuthenticationError("Invalid API key")
            if not model.is_active:
                raise InactiveUserError("Account is disabled")
            if self._password_expired(model):
                raise PasswordExpiredError("Password rotation window exceeded")
            model.last_login_at = datetime.utcnow()
            session.add(model)
            session.flush()
            return self._to_user(model)

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------
    def create_user(
        self,
        username: str,
        password: str,
        role: str,
        *,
        api_key: Optional[str] = None,
        is_active: bool = True,
    ) -> AuthenticatedUser:
        """Provision a new user with optional API key."""

        now = datetime.utcnow()
        expiry = self._compute_expiry(now)
        password_hash = self.hash_password(password)
        api_key_hash = self.hash_api_key(api_key) if api_key else None

        with get_session(self.config) as session:
            existing = (
                session.execute(
                    select(UserAccountModel).where(UserAccountModel.username == username)
                )
                .scalars()
                .first()
            )
            if existing is not None:
                raise IdentityError(f"User '{username}' already exists")

            if api_key_hash:
                clash = (
                    session.execute(
                        select(UserAccountModel).where(
                            UserAccountModel.api_key_hash == api_key_hash
                        )
                    )
                    .scalars()
                    .first()
                )
                if clash is not None:
                    raise IdentityError("API key already in use")

            model = UserAccountModel(
                username=username,
                password_hash=password_hash,
                role=role,
                is_active=is_active,
                password_rotated_at=now,
                password_expires_at=expiry,
                api_key_hash=api_key_hash,
            )
            session.add(model)
            session.flush()
            return self._to_user(model)

    def rotate_password(
        self,
        username: str,
        new_password: str,
        *,
        api_key: Optional[str] = None,
        remove_api_key: bool = False,
    ) -> AuthenticatedUser:
        """Rotate the password (and optionally API key) for a user."""

        if not new_password:
            raise ValueError("New password must not be empty")

        password_hash = self.hash_password(new_password)
        now = datetime.utcnow()
        expiry = self._compute_expiry(now)
        should_update_api_key = api_key is not None or remove_api_key
        requested_api_key_hash = self.hash_api_key(api_key) if api_key else None

        with get_session(self.config) as session:
            stmt = select(UserAccountModel).where(UserAccountModel.username == username)
            model = session.execute(stmt).scalar_one_or_none()
            if model is None:
                raise IdentityError(f"User '{username}' does not exist")

            if should_update_api_key and requested_api_key_hash:
                clash = (
                    session.execute(
                        select(UserAccountModel)
                        .where(UserAccountModel.api_key_hash == requested_api_key_hash)
                        .where(UserAccountModel.id != model.id)
                    )
                    .scalars()
                    .first()
                )
                if clash is not None:
                    raise IdentityError("API key already in use")

            model.password_hash = password_hash
            model.password_rotated_at = now
            model.password_expires_at = expiry
            if should_update_api_key:
                model.api_key_hash = requested_api_key_hash
            session.add(model)
            session.flush()
            return self._to_user(model)

    def set_api_key(self, username: str, api_key: Optional[str]) -> AuthenticatedUser:
        """Assign or revoke an API key for a user."""

        with get_session(self.config) as session:
            stmt = select(UserAccountModel).where(UserAccountModel.username == username)
            model = session.execute(stmt).scalar_one_or_none()
            if model is None:
                raise IdentityError(f"User '{username}' does not exist")

            if api_key:
                api_key_hash = self.hash_api_key(api_key)
                clash = (
                    session.execute(
                        select(UserAccountModel)
                        .where(UserAccountModel.api_key_hash == api_key_hash)
                        .where(UserAccountModel.id != model.id)
                    )
                    .scalars()
                    .first()
                )
                if clash is not None:
                    raise IdentityError("API key already in use")
                model.api_key_hash = api_key_hash
            else:
                model.api_key_hash = None
            session.add(model)
            session.flush()
            return self._to_user(model)

    def set_active(self, username: str, is_active: bool) -> AuthenticatedUser:
        """Enable or disable a user account."""

        with get_session(self.config) as session:
            stmt = select(UserAccountModel).where(UserAccountModel.username == username)
            model = session.execute(stmt).scalar_one_or_none()
            if model is None:
                raise IdentityError(f"User '{username}' does not exist")
            model.is_active = is_active
            session.add(model)
            session.flush()
            return self._to_user(model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _password_expired(self, model: UserAccountModel) -> bool:
        if self.password_max_age_days <= 0:
            return False
        rotated_at = model.password_rotated_at
        if rotated_at is None:
            return True
        expiry = model.password_expires_at or self._compute_expiry(rotated_at)
        if expiry is None:
            return False
        return datetime.utcnow() >= expiry

    def _compute_expiry(self, rotated_at: datetime) -> Optional[datetime]:
        if self.password_max_age_days <= 0:
            return None
        return rotated_at + timedelta(days=self.password_max_age_days)


__all__ = [
    "AuthenticatedUser",
    "AuthenticationError",
    "IdentityError",
    "IdentityService",
    "InactiveUserError",
    "PasswordExpiredError",
]
