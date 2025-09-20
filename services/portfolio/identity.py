from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from services.common.secret_manager import SecretRetrievalError, resolve_secret

from .config import PortfolioConfig
from .database import Base, get_engine, get_session
from .models import UserAccountModel
from .security import DEFAULT_HASH_ITERATIONS, HashedSecret, hash_secret, verify_secret


class IdentityError(RuntimeError):
    """Base class for identity-related errors."""


class InvalidCredentialsError(IdentityError):
    """Raised when a username/password combination is invalid."""


class InactiveAccountError(IdentityError):
    """Raised when a user account is disabled."""


class PasswordExpiredError(IdentityError):
    """Raised when a user must rotate an expired password."""


class UserAlreadyExistsError(IdentityError):
    """Raised when attempting to create a duplicate user."""


class ApiKeyValidationError(IdentityError):
    """Raised when an API key is missing or invalid."""


class ApiKeyRotationRequiredError(ApiKeyValidationError):
    """Raised when an API key has passed its rotation deadline."""


@dataclass
class UserIdentity:
    """Summary of a user's identity and credential metadata."""

    id: int
    username: str
    roles: List[str]
    is_active: bool
    password_rotated_at: datetime
    password_expires_at: Optional[datetime]
    api_key_last_rotated_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]


@dataclass
class ApiKeyRotationResult:
    """Returned when a new API key is issued for a user."""

    user: UserIdentity
    api_key: str


class PortfolioIdentityService:
    """Manage user credentials stored in the portfolio database."""

    def __init__(self, config: Optional[PortfolioConfig] = None) -> None:
        self.config = config or PortfolioConfig.from_env()
        self._ensure_schema()
        self.password_rotation_days = max(1, int(self.config.password_rotation_days))
        self.api_key_rotation_days = max(1, int(self.config.api_key_rotation_days))

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------
    def create_user(
        self,
        username: str,
        password: str,
        *,
        roles: Optional[Iterable[str]] = None,
        is_active: bool = True,
        api_key: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> UserIdentity:
        """Provision a new user with the supplied credentials."""

        password_secret = hash_secret(password)
        api_key_secret = hash_secret(api_key) if api_key else None
        role_list = self._normalise_roles(roles)
        now = self._utcnow()

        def _create(sess: Session) -> UserIdentity:
            if self._find_user_model(sess, username):
                raise UserAlreadyExistsError(f"User '{username}' already exists")

            model = UserAccountModel(
                username=username,
                password_hash=password_secret.hash,
                password_salt=password_secret.salt,
                password_iterations=password_secret.iterations,
                roles=role_list,
                is_active=is_active,
                password_rotated_at=now,
                password_expires_at=self._compute_password_expiry(now),
                created_at=now,
                updated_at=now,
            )

            if api_key_secret:
                model.api_key_hash = api_key_secret.hash
                model.api_key_salt = api_key_secret.salt
                model.api_key_iterations = api_key_secret.iterations
                model.api_key_last_rotated_at = now

            sess.add(model)
            try:
                sess.flush()
            except IntegrityError as exc:  # pragma: no cover - DB specific
                raise UserAlreadyExistsError(f"User '{username}' already exists") from exc
            sess.refresh(model)
            return self._to_identity(model)

        if session is not None:
            return _create(session)

        with get_session(self.config) as sess:
            return _create(sess)

    def authenticate_user(self, username: str, password: str) -> UserIdentity:
        """Validate credentials and return the associated identity."""

        with get_session(self.config) as sess:
            model = self._require_user(sess, username)
            if not model.is_active:
                raise InactiveAccountError(f"User '{username}' is disabled")

            self._ensure_password_not_expired(model)

            if not verify_secret(password, self._password_secret(model)):
                raise InvalidCredentialsError("Invalid username or password")

            model.last_login_at = self._utcnow()
            model.updated_at = model.last_login_at
            sess.add(model)
            sess.flush()
            sess.refresh(model)
            return self._to_identity(model)

    def rotate_password(
        self,
        username: str,
        current_password: str,
        new_password: str,
    ) -> UserIdentity:
        """Rotate a user's password after verifying their current credentials."""

        with get_session(self.config) as sess:
            model = self._require_user(sess, username)
            if not model.is_active:
                raise InactiveAccountError(f"User '{username}' is disabled")

            if not verify_secret(current_password, self._password_secret(model)):
                raise InvalidCredentialsError("Current password is incorrect")

            now = self._utcnow()
            new_secret = hash_secret(new_password)
            model.password_hash = new_secret.hash
            model.password_salt = new_secret.salt
            model.password_iterations = new_secret.iterations
            model.password_rotated_at = now
            model.password_expires_at = self._compute_password_expiry(now)
            model.updated_at = now
            sess.add(model)
            sess.flush()
            sess.refresh(model)
            return self._to_identity(model)

    def validate_api_key(self, api_key: str) -> UserIdentity:
        """Validate an API key and return the associated user."""

        if not api_key:
            raise ApiKeyValidationError("API key cannot be empty")

        with get_session(self.config) as sess:
            stmt = select(UserAccountModel).where(UserAccountModel.api_key_hash.isnot(None))
            for model in sess.scalars(stmt):
                if not model.api_key_hash or not model.api_key_salt or not model.api_key_iterations:
                    continue
                if verify_secret(api_key, self._api_key_secret(model)):
                    if not model.is_active:
                        raise InactiveAccountError(f"User '{model.username}' is disabled")
                    if self._api_key_rotation_due(model):
                        raise ApiKeyRotationRequiredError(
                            f"API key for '{model.username}' must be rotated"
                        )
                    return self._to_identity(model)

        raise ApiKeyValidationError("Invalid API key")

    def rotate_api_key(
        self,
        username: str,
        *,
        new_api_key: Optional[str] = None,
    ) -> ApiKeyRotationResult:
        """Issue and persist a new API key for *username*."""

        with get_session(self.config) as sess:
            model = self._require_user(sess, username)
            if not model.is_active:
                raise InactiveAccountError(f"User '{username}' is disabled")
            api_key = self._resolve_new_api_key(new_api_key)
            secret = hash_secret(api_key)
            now = self._utcnow()
            model.api_key_hash = secret.hash
            model.api_key_salt = secret.salt
            model.api_key_iterations = secret.iterations
            model.api_key_last_rotated_at = now
            model.updated_at = now
            sess.add(model)
            sess.flush()
            sess.refresh(model)
            return ApiKeyRotationResult(user=self._to_identity(model), api_key=api_key)

    def list_users(self) -> List[UserIdentity]:
        """Return a snapshot of all provisioned users."""

        with get_session(self.config) as sess:
            stmt = select(UserAccountModel).order_by(UserAccountModel.username)
            return [self._to_identity(model) for model in sess.scalars(stmt)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        engine = get_engine(self.config)
        Base.metadata.create_all(engine)

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.utcnow()

    def _compute_password_expiry(self, rotated_at: datetime) -> datetime:
        return rotated_at + timedelta(days=self.password_rotation_days)

    def _api_key_rotation_due(self, model: UserAccountModel) -> bool:
        if not model.api_key_last_rotated_at:
            return False
        due_at = model.api_key_last_rotated_at + timedelta(days=self.api_key_rotation_days)
        return self._utcnow() >= due_at

    def _resolve_new_api_key(self, candidate: Optional[str]) -> str:
        if candidate and candidate.strip():
            return candidate
        try:
            resolved = resolve_secret(
                "PORTFOLIO_NEW_API_KEY",
                env_keys=("PORTFOLIO_ROTATION_API_KEY",),
            )
        except SecretRetrievalError as exc:
            raise ApiKeyValidationError(
                "A new API key must be supplied via the secrets manager or provided explicitly"
            ) from exc
        if not resolved or not resolved.strip():
            raise ApiKeyValidationError("Resolved API key value is empty")
        return resolved

    @staticmethod
    def _password_secret(model: UserAccountModel) -> HashedSecret:
        return HashedSecret(
            hash=model.password_hash,
            salt=model.password_salt,
            iterations=model.password_iterations or DEFAULT_HASH_ITERATIONS,
        )

    @staticmethod
    def _api_key_secret(model: UserAccountModel) -> HashedSecret:
        return HashedSecret(
            hash=model.api_key_hash or "",
            salt=model.api_key_salt or "",
            iterations=model.api_key_iterations or DEFAULT_HASH_ITERATIONS,
        )

    @staticmethod
    def _find_user_model(session: Session, username: str) -> Optional[UserAccountModel]:
        stmt = select(UserAccountModel).where(UserAccountModel.username == username)
        return session.scalars(stmt).first()

    def _require_user(self, session: Session, username: str) -> UserAccountModel:
        model = self._find_user_model(session, username)
        if not model:
            raise InvalidCredentialsError("Invalid username or password")
        return model

    def _ensure_password_not_expired(self, model: UserAccountModel) -> None:
        if model.password_expires_at and model.password_expires_at <= self._utcnow():
            raise PasswordExpiredError(
                f"Password for '{model.username}' expired on {model.password_expires_at.isoformat()}"
            )

    @staticmethod
    def _to_identity(model: UserAccountModel) -> UserIdentity:
        roles = model.roles or ["user"]
        if isinstance(roles, str):
            roles = [role.strip() for role in roles.split(",") if role.strip()]
        return UserIdentity(
            id=model.id,
            username=model.username,
            roles=list(roles) if isinstance(roles, list) else roles,
            is_active=model.is_active,
            password_rotated_at=model.password_rotated_at,
            password_expires_at=model.password_expires_at,
            api_key_last_rotated_at=model.api_key_last_rotated_at,
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_login_at=model.last_login_at,
        )

    @staticmethod
    def _normalise_roles(roles: Optional[Iterable[str]]) -> List[str]:
        if not roles:
            return ["user"]
        normalised: List[str] = []
        for role in roles:
            role_name = str(role).strip().lower()
            if not role_name:
                continue
            if role_name not in normalised:
                normalised.append(role_name)
        return normalised or ["user"]


__all__ = [
    "ApiKeyRotationRequiredError",
    "ApiKeyRotationResult",
    "ApiKeyValidationError",
    "InactiveAccountError",
    "InvalidCredentialsError",
    "PasswordExpiredError",
    "PortfolioIdentityService",
    "UserAlreadyExistsError",
    "UserIdentity",
]
