"""SQLAlchemy models backing the identity service."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class CredentialType(str, enum.Enum):
    """Supported credential categories."""

    PASSWORD = "password"
    API_KEY = "api_key"
    SERVICE = "service"
    REFRESH_TOKEN = "refresh_token"


class TenantModel(Base):
    """Tenant metadata used for multi-tenant isolation."""

    __tablename__ = "identity_tenants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    issuer: Mapped[str] = mapped_column(String(256), nullable=False)
    key_id: Mapped[str] = mapped_column(String(128), nullable=False)
    secret_provider: Mapped[str] = mapped_column(String(32), default="vault", nullable=False)
    secret_reference: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    roles: Mapped[list["RoleModel"]] = relationship(
        "RoleModel",
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    users: Mapped[list["UserModel"]] = relationship(
        "UserModel",
        back_populates="tenant",
        cascade="all, delete-orphan",
    )


class RoleModel(Base):
    """Role definitions scoped to a tenant."""

    __tablename__ = "identity_roles"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_identity_role_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("identity_tenants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    tenant: Mapped[TenantModel] = relationship("TenantModel", back_populates="roles")
    users: Mapped[list["UserModel"]] = relationship(
        "UserModel",
        secondary="identity_user_roles",
        back_populates="roles",
    )


class UserModel(Base):
    """User accounts for both human and service identities."""

    __tablename__ = "identity_users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "username", name="uq_identity_user_username"),
        UniqueConstraint("tenant_id", "email", name="uq_identity_user_email"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("identity_tenants.id", ondelete="CASCADE"), nullable=False
    )
    external_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    username: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_service_account: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    mfa_enforced: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    mfa_recovery_codes: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    password_rotated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    password_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    api_key_last_rotated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    api_key_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    tenant: Mapped[TenantModel] = relationship("TenantModel", back_populates="users")
    roles: Mapped[list[RoleModel]] = relationship(
        "RoleModel",
        secondary="identity_user_roles",
        back_populates="users",
        lazy="joined",
    )
    credentials: Mapped[list["CredentialModel"]] = relationship(
        "CredentialModel",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    refresh_tokens: Mapped[list["RefreshTokenModel"]] = relationship(
        "RefreshTokenModel",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class UserRoleLinkModel(Base):
    """Association table linking users to roles."""

    __tablename__ = "identity_user_roles"
    __table_args__ = (
        UniqueConstraint("user_id", "role_id", name="uq_identity_user_role"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("identity_users.id", ondelete="CASCADE"), nullable=False
    )
    role_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("identity_roles.id", ondelete="CASCADE"), nullable=False
    )
    assigned_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class CredentialModel(Base):
    """Secrets associated with a user."""

    __tablename__ = "identity_credentials"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "user_id",
            "credential_type",
            "version",
            name="uq_identity_credential_version",
        ),
        Index(
            "ix_identity_credential_active",
            "user_id",
            "credential_type",
            "is_active",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("identity_users.id", ondelete="CASCADE"), nullable=False
    )
    credential_type: Mapped[CredentialType] = mapped_column(
        SAEnum(CredentialType), nullable=False, index=True
    )
    secret_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    secret_salt: Mapped[str] = mapped_column(String(256), nullable=False)
    secret_iterations: Mapped[int] = mapped_column(Integer, default=210_000, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    rotation_interval_days: Mapped[int] = mapped_column(Integer, default=90, nullable=False)
    rotated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    external_secret_reference: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped[UserModel] = relationship("UserModel", back_populates="credentials")


class RefreshTokenModel(Base):
    """Persisted refresh tokens for session renewal."""

    __tablename__ = "identity_refresh_tokens"
    __table_args__ = (
        UniqueConstraint("tenant_id", "token_id", name="uq_identity_refresh_token"),
        Index("ix_identity_refresh_user", "user_id", "revoked_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("identity_users.id", ondelete="CASCADE"), nullable=False
    )
    token_id: Mapped[str] = mapped_column(String(128), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    token_salt: Mapped[str] = mapped_column(String(256), nullable=False)
    token_iterations: Mapped[int] = mapped_column(Integer, default=210_000, nullable=False)
    client_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    user: Mapped[UserModel] = relationship("UserModel", back_populates="refresh_tokens")


__all__ = [
    "CredentialModel",
    "CredentialType",
    "RefreshTokenModel",
    "RoleModel",
    "TenantModel",
    "UserModel",
    "UserRoleLinkModel",
]
