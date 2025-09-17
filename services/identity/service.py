"""Core business logic for the identity microservice."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence

import jwt
from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from .config import IdentitySettings, load_identity_settings
from .database import Base, get_engine, get_session
from .mfa import MfaProvider, NullMfaProvider
from .models import (
    CredentialModel,
    CredentialType,
    RefreshTokenModel,
    RoleModel,
    TenantModel,
    UserModel,
)
from .schemas import IntrospectionResponse, ScimUser, TokenPair
from .scim import ScimUpdate, apply_scim_update, build_scim_user, parse_scim_payload
from .security import HashedSecret, generate_token, hash_secret, verify_secret
from .tokens import TokenSigner


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IdentityError(RuntimeError):
    """Base class for identity related exceptions."""


class InvalidCredentialsError(IdentityError):
    """Raised when credentials cannot be validated."""


class InactiveAccountError(IdentityError):
    """Raised when an account has been disabled."""


class PasswordExpiredError(IdentityError):
    """Raised when a password must be rotated before continuing."""


class ApiKeyValidationError(IdentityError):
    """Raised when an API key does not match any user."""


class ApiKeyRotationRequiredError(ApiKeyValidationError):
    """Raised when an API key has exceeded its rotation deadline."""


class RefreshTokenError(IdentityError):
    """Raised when refresh token validation fails."""


class ScimOperationError(IdentityError):
    """Raised when SCIM provisioning fails."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IdentityUser:
    """Serializable representation of a user."""

    id: int
    tenant: str
    username: str
    roles: List[str]
    is_active: bool
    password_rotated_at: Optional[datetime]
    password_expires_at: Optional[datetime]
    api_key_last_rotated_at: Optional[datetime]
    api_key_expires_at: Optional[datetime]


# ---------------------------------------------------------------------------
# Identity service implementation
# ---------------------------------------------------------------------------


class IdentityService:
    """Identity and credential management service."""

    def __init__(
        self,
        settings: Optional[IdentitySettings] = None,
        *,
        mfa_provider: Optional[MfaProvider] = None,
    ) -> None:
        self.settings = settings or load_identity_settings()
        self.mfa_provider = mfa_provider or NullMfaProvider()
        self.token_signer = TokenSigner(self.settings)
        self._initialise_schema()
        self._ensure_default_tenant()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _initialise_schema(self) -> None:
        engine = get_engine(self.settings)
        Base.metadata.create_all(engine)

    def _ensure_default_tenant(self) -> None:
        with get_session(self.settings) as session:
            stmt = select(TenantModel).where(TenantModel.slug == self.settings.default_tenant_slug)
            tenant = session.scalar(stmt)
            if tenant:
                return
            issuer = self._build_tenant_issuer(self.settings.default_tenant_slug)
            secret_reference = f"tenants/{self.settings.default_tenant_slug}/identity"
            tenant = TenantModel(
                slug=self.settings.default_tenant_slug,
                name="Default Tenant",
                issuer=issuer,
                key_id=f"{self.settings.default_tenant_slug}-default",
                secret_reference=secret_reference,
                secret_provider="vault",
                metadata_json={"token_algorithm": self.settings.token_algorithm},
            )
            session.add(tenant)

    def _build_tenant_issuer(self, slug: str) -> str:
        base = self.settings.default_issuer.rstrip("/")
        return f"{base}/{slug}"

    # ------------------------------------------------------------------
    # Tenant helpers
    # ------------------------------------------------------------------
    def _resolve_tenant(self, session: Session, slug: Optional[str]) -> TenantModel:
        tenant_slug = slug or self.settings.default_tenant_slug
        stmt = select(TenantModel).where(TenantModel.slug == tenant_slug)
        tenant = session.scalar(stmt)
        if not tenant:
            raise IdentityError(f"Unknown tenant '{tenant_slug}'")
        if not tenant.is_active:
            raise IdentityError(f"Tenant '{tenant_slug}' is disabled")
        return tenant

    # ------------------------------------------------------------------
    # Token issuance and validation
    # ------------------------------------------------------------------
    def issue_token(
        self,
        tenant_slug: Optional[str],
        username: str,
        password: str,
        *,
        scope: Optional[Sequence[str]] = None,
        requested_scopes: Optional[str] = None,
        mfa_code: Optional[str] = None,
    ) -> TokenPair:
        scopes = self._normalise_scopes(scope, requested_scopes)
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant(session, tenant_slug)
            user = self._require_user(session, tenant, username)
            self._validate_user_password(user, password)
            if self.mfa_provider.is_required(user):
                if not self.mfa_provider.verify(user, mfa_code or ""):
                    raise InvalidCredentialsError("Multi-factor challenge failed")
            token_pair = self._mint_tokens(session, tenant, user, scopes)
            return token_pair

    def refresh_token(
        self,
        tenant_slug: Optional[str],
        refresh_token: str,
        *,
        scope: Optional[Sequence[str]] = None,
    ) -> TokenPair:
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant_from_token(session, tenant_slug, refresh_token)
            claims = self.token_signer.verify_token(tenant, refresh_token)
            if claims.get("type") != "refresh":
                raise RefreshTokenError("Provided token is not a refresh token")
            jti = str(claims.get("jti")) if claims.get("jti") else None
            if not jti:
                raise RefreshTokenError("Refresh token missing identifier")
            stmt = (
                select(RefreshTokenModel)
                .where(
                    and_(
                        RefreshTokenModel.tenant_id == tenant.id,
                        RefreshTokenModel.token_id == jti,
                        RefreshTokenModel.revoked_at.is_(None),
                    )
                )
                .options(joinedload(RefreshTokenModel.user).joinedload(UserModel.roles))
            )
            model = session.scalar(stmt)
            if not model:
                raise RefreshTokenError("Refresh token has been revoked or is unknown")
            if model.expires_at < self._utcnow():
                model.revoked_at = self._utcnow()
                session.add(model)
                raise RefreshTokenError("Refresh token has expired")
            if not verify_secret(refresh_token, self._hashed_refresh_secret(model)):
                model.revoked_at = self._utcnow()
                session.add(model)
                raise RefreshTokenError("Refresh token validation failed")
            user = model.user
            if not user.is_active:
                raise InactiveAccountError("Account disabled")
            scopes = self._normalise_scopes(scope, None)
            model.revoked_at = self._utcnow()
            session.add(model)
            return self._mint_tokens(session, tenant, user, scopes)

    def introspect_token(
        self,
        tenant_slug: Optional[str],
        token: str,
    ) -> IntrospectionResponse:
        with get_session(self.settings) as session:
            try:
                tenant = self._resolve_tenant_from_token(session, tenant_slug, token)
            except IdentityError:
                return IntrospectionResponse(active=False)
            try:
                claims = self.token_signer.verify_token(tenant, token)
            except Exception:
                return IntrospectionResponse(active=False)

            roles = self._extract_roles_from_claims(claims)
            scope = self._extract_scopes_from_claims(claims)
            issued = datetime.fromtimestamp(int(claims.get("iat", 0)), tz=timezone.utc)
            expires = datetime.fromtimestamp(int(claims.get("exp", 0)), tz=timezone.utc)
            return IntrospectionResponse(
                active=True,
                username=str(claims.get("sub")),
                subject=str(claims.get("sub")),
                tenant=tenant.slug,
                scope=scope,
                roles=roles,
                issued_at=issued,
                expires_at=expires,
                token_type=str(claims.get("type", "access")),
                client_id=str(claims.get("client_id")) if claims.get("client_id") else None,
                claims=claims,
            )

    # ------------------------------------------------------------------
    # Credential management
    # ------------------------------------------------------------------
    def rotate_password(
        self,
        tenant_slug: Optional[str],
        username: str,
        current_password: str,
        new_password: str,
    ) -> IdentityUser:
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant(session, tenant_slug)
            user = self._require_user(session, tenant, username)
            self._validate_user_password(user, current_password)
            credential = self._rotate_secret(
                session,
                user,
                CredentialType.PASSWORD,
                new_password,
            )
            user.password_rotated_at = credential.rotated_at
            user.password_expires_at = credential.expires_at
            session.add(user)
            session.flush()
            return self._to_identity_user(user)

    def rotate_api_key(
        self,
        tenant_slug: Optional[str],
        username: str,
        *,
        new_api_key: Optional[str] = None,
    ) -> tuple[str, IdentityUser]:
        api_key = new_api_key or generate_token(48)
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant(session, tenant_slug)
            user = self._require_user(session, tenant, username)
            credential = self._rotate_secret(
                session,
                user,
                CredentialType.API_KEY,
                api_key,
                rotation_interval_days=180,
            )
            user.api_key_last_rotated_at = credential.rotated_at
            user.api_key_expires_at = credential.expires_at
            session.add(user)
            session.flush()
            return api_key, self._to_identity_user(user)

    def validate_api_key(
        self,
        tenant_slug: Optional[str],
        api_key: str,
    ) -> IdentityUser:
        if not api_key:
            raise ApiKeyValidationError("API key cannot be empty")
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant(session, tenant_slug)
            stmt = (
                select(CredentialModel)
                .join(UserModel)
                .where(
                    and_(
                        CredentialModel.tenant_id == tenant.id,
                        CredentialModel.credential_type == CredentialType.API_KEY,
                        CredentialModel.is_active.is_(True),
                    )
                )
                .options(
                    joinedload(CredentialModel.user).joinedload(UserModel.roles),
                    joinedload(CredentialModel.user).joinedload(UserModel.tenant),
                )
            )
            for credential in session.scalars(stmt):
                if verify_secret(api_key, self._hashed_secret(credential)):
                    if credential.expires_at and credential.expires_at < self._utcnow():
                        raise ApiKeyRotationRequiredError("API key has expired and must be rotated")
                    user = credential.user
                    if not user.is_active:
                        raise InactiveAccountError("Account disabled")
                    return self._to_identity_user(user)
        raise ApiKeyValidationError("Invalid API key")

    def list_users(self, tenant_slug: Optional[str]) -> List[IdentityUser]:
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant(session, tenant_slug)
            stmt = (
                select(UserModel)
                .where(UserModel.tenant_id == tenant.id)
                .options(joinedload(UserModel.roles), joinedload(UserModel.tenant))
            )
            return [self._to_identity_user(user) for user in session.scalars(stmt)]

    # ------------------------------------------------------------------
    # SCIM provisioning
    # ------------------------------------------------------------------
    def scim_list_users(self, tenant_slug: Optional[str], base_url: str) -> List[ScimUser]:
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant(session, tenant_slug)
            stmt = (
                select(UserModel)
                .where(UserModel.tenant_id == tenant.id)
                .options(joinedload(UserModel.roles), joinedload(UserModel.tenant))
            )
            return [build_scim_user(user, base_url) for user in session.scalars(stmt)]

    def scim_get_user(self, tenant_slug: Optional[str], user_id: int, base_url: str) -> ScimUser:
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant(session, tenant_slug)
            user = self._get_user_by_id(session, tenant, user_id)
            if not user:
                raise ScimOperationError("User not found")
            return build_scim_user(user, base_url)

    def scim_create_user(
        self,
        tenant_slug: Optional[str],
        payload: dict,
        base_url: str,
    ) -> ScimUser:
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant(session, tenant_slug)
            update = parse_scim_payload(payload)
            password = payload.get("password")
            if not password:
                raise ScimOperationError("SCIM payload must include 'password'")
            user = UserModel(
                tenant_id=tenant.id,
                username=update.username or payload.get("userName"),
                display_name=update.display_name,
                email=update.email,
                is_active=update.active if update.active is not None else True,
                attributes=update.attributes or {},
                external_id=payload.get("externalId"),
            )
            session.add(user)
            session.flush()
            self._rotate_secret(session, user, CredentialType.PASSWORD, password)
            self._assign_roles(session, tenant, user, update.role_names or [])
            session.flush()
            user.tenant = tenant
            return build_scim_user(user, base_url)

    def scim_replace_user(
        self,
        tenant_slug: Optional[str],
        user_id: int,
        payload: dict,
        base_url: str,
    ) -> ScimUser:
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant(session, tenant_slug)
            user = self._get_user_by_id(session, tenant, user_id)
            if not user:
                raise ScimOperationError("User not found")
            update = parse_scim_payload(payload)
            apply_scim_update(user, update, self._ensure_roles(session, tenant, update.role_names or []))
            session.add(user)
            session.flush()
            if password := payload.get("password"):
                self._rotate_secret(session, user, CredentialType.PASSWORD, password)
            return build_scim_user(user, base_url)

    def scim_delete_user(self, tenant_slug: Optional[str], user_id: int) -> None:
        with get_session(self.settings) as session:
            tenant = self._resolve_tenant(session, tenant_slug)
            user = self._get_user_by_id(session, tenant, user_id)
            if not user:
                return
            session.delete(user)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _mint_tokens(
        self,
        session: Session,
        tenant: TenantModel,
        user: UserModel,
        scopes: Sequence[str],
    ) -> TokenPair:
        roles = [role.name for role in user.roles]
        access_token, issued_at, expires_at, _ = self.token_signer.mint_access_token(
            tenant,
            user.username,
            list(scopes),
            roles,
            self.settings.access_token_ttl_seconds,
            additional_claims={
                "type": "access",
                "password_expires_at": int(user.password_expires_at.timestamp())
                if user.password_expires_at
                else None,
            },
        )
        refresh_token, refresh_issued, refresh_expires, refresh_jti = self.token_signer.mint_refresh_token(
            tenant,
            user.username,
            self.settings.refresh_token_ttl_seconds,
        )
        hashed_refresh = hash_secret(refresh_token)
        refresh_model = RefreshTokenModel(
            tenant_id=tenant.id,
            user_id=user.id,
            token_id=refresh_jti,
            token_hash=hashed_refresh.hash,
            token_salt=hashed_refresh.salt,
            token_iterations=hashed_refresh.iterations,
            issued_at=refresh_issued,
            expires_at=refresh_expires,
        )
        session.add(refresh_model)
        user.last_login_at = self._utcnow()
        session.add(user)
        session.flush()
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.settings.access_token_ttl_seconds,
            expires_at=expires_at,
            issued_at=issued_at,
            scope=list(scopes),
            roles=roles,
            username=user.username,
            tenant=tenant.slug,
            password_expires_at=user.password_expires_at,
        )

    def _rotate_secret(
        self,
        session: Session,
        user: UserModel,
        credential_type: CredentialType,
        secret: str,
        *,
        rotation_interval_days: int = 90,
    ) -> CredentialModel:
        hashed = hash_secret(secret)
        self._deactivate_credentials(session, user, credential_type)
        version_stmt = select(func.max(CredentialModel.version)).where(
            and_(
                CredentialModel.user_id == user.id,
                CredentialModel.credential_type == credential_type,
            )
        )
        latest_version = session.scalar(version_stmt) or 0
        rotated_at = self._utcnow()
        expires_at = rotated_at + timedelta(days=rotation_interval_days)
        credential = CredentialModel(
            tenant_id=user.tenant_id,
            user_id=user.id,
            credential_type=credential_type,
            secret_hash=hashed.hash,
            secret_salt=hashed.salt,
            secret_iterations=hashed.iterations,
            version=latest_version + 1,
            rotation_interval_days=rotation_interval_days,
            rotated_at=rotated_at,
            expires_at=expires_at,
            is_active=True,
        )
        session.add(credential)
        session.flush()
        return credential

    def _deactivate_credentials(
        self,
        session: Session,
        user: UserModel,
        credential_type: CredentialType,
    ) -> None:
        stmt = (
            select(CredentialModel)
            .where(
                and_(
                    CredentialModel.user_id == user.id,
                    CredentialModel.credential_type == credential_type,
                    CredentialModel.is_active.is_(True),
                )
            )
        )
        for credential in session.scalars(stmt):
            credential.is_active = False
            session.add(credential)

    def _assign_roles(
        self,
        session: Session,
        tenant: TenantModel,
        user: UserModel,
        role_names: Iterable[str],
    ) -> None:
        roles = self._ensure_roles(session, tenant, role_names)
        user.roles = roles
        session.add(user)

    def _ensure_roles(
        self,
        session: Session,
        tenant: TenantModel,
        role_names: Iterable[str],
    ) -> List[RoleModel]:
        existing_stmt = (
            select(RoleModel)
            .where(
                and_(
                    RoleModel.tenant_id == tenant.id,
                    RoleModel.name.in_(list(role_names) or [""]),
                )
            )
        )
        existing = {role.name: role for role in session.scalars(existing_stmt)}
        roles: List[RoleModel] = []
        for name in role_names:
            if not name:
                continue
            role = existing.get(name)
            if not role:
                role = RoleModel(tenant_id=tenant.id, name=name, description=f"SCIM role {name}")
                session.add(role)
                try:
                    session.flush()
                except IntegrityError:
                    session.rollback()
                    stmt = (
                        select(RoleModel)
                        .where(
                            and_(
                                RoleModel.tenant_id == tenant.id,
                                RoleModel.name == name,
                            )
                        )
                    )
                    role = session.scalar(stmt)
                    if not role:
                        raise
            roles.append(role)
        return roles

    def _require_user(self, session: Session, tenant: TenantModel, username: str) -> UserModel:
        stmt = (
            select(UserModel)
            .where(
                and_(UserModel.tenant_id == tenant.id, UserModel.username == username)
            )
            .options(joinedload(UserModel.roles), joinedload(UserModel.tenant))
        )
        user = session.scalar(stmt)
        if not user:
            raise InvalidCredentialsError("Invalid username or password")
        if not user.is_active:
            raise InactiveAccountError("Account disabled")
        return user

    def _get_user_by_id(
        self, session: Session, tenant: TenantModel, user_id: int
    ) -> Optional[UserModel]:
        stmt = (
            select(UserModel)
            .where(and_(UserModel.tenant_id == tenant.id, UserModel.id == user_id))
            .options(joinedload(UserModel.roles), joinedload(UserModel.tenant))
        )
        return session.scalar(stmt)

    def _validate_user_password(self, user: UserModel, password: str) -> None:
        credential = self._active_credential(user, CredentialType.PASSWORD)
        if not credential or not verify_secret(password, self._hashed_secret(credential)):
            raise InvalidCredentialsError("Invalid username or password")
        if credential.expires_at and credential.expires_at < self._utcnow():
            raise PasswordExpiredError("Password expired; rotation required")

    def _active_credential(
        self, user: UserModel, credential_type: CredentialType
    ) -> Optional[CredentialModel]:
        for credential in sorted(
            (cred for cred in user.credentials if cred.credential_type == credential_type and cred.is_active),
            key=lambda cred: cred.version,
            reverse=True,
        ):
            return credential
        return None

    def _resolve_tenant_from_token(
        self,
        session: Session,
        tenant_slug: Optional[str],
        token: str,
    ) -> TenantModel:
        if tenant_slug:
            return self._resolve_tenant(session, tenant_slug)
        try:
            claims = jwt.decode(token, options={"verify_signature": False})
        except Exception as exc:
            raise IdentityError("Unable to determine tenant from token") from exc
        slug = claims.get("tenant") or claims.get("tid")
        return self._resolve_tenant(session, slug)

    def _to_identity_user(self, user: UserModel) -> IdentityUser:
        roles = [role.name for role in user.roles]
        return IdentityUser(
            id=user.id,
            tenant=user.tenant.slug if user.tenant else "",
            username=user.username,
            roles=roles,
            is_active=user.is_active,
            password_rotated_at=user.password_rotated_at,
            password_expires_at=user.password_expires_at,
            api_key_last_rotated_at=user.api_key_last_rotated_at,
            api_key_expires_at=user.api_key_expires_at,
        )

    def _hashed_secret(self, credential: CredentialModel) -> HashedSecret:
        return HashedSecret(
            hash=credential.secret_hash,
            salt=credential.secret_salt,
            iterations=credential.secret_iterations,
        )

    def _hashed_refresh_secret(self, model: RefreshTokenModel) -> HashedSecret:
        return HashedSecret(
            hash=model.token_hash,
            salt=model.token_salt,
            iterations=model.token_iterations,
        )

    def _normalise_scopes(
        self,
        scope_list: Optional[Sequence[str]],
        scope_string: Optional[str],
    ) -> List[str]:
        scopes = set(scope_list or [])
        if scope_string:
            scopes.update(scope for scope in scope_string.split() if scope)
        scopes.add("openid")
        scopes.add("profile")
        return sorted(scopes)

    def _extract_roles_from_claims(self, claims: dict) -> List[str]:
        roles = claims.get("roles")
        if isinstance(roles, list):
            return [str(role) for role in roles]
        if isinstance(roles, str):
            return [role for role in roles.split() if role]
        return []

    def _extract_scopes_from_claims(self, claims: dict) -> List[str]:
        scope = claims.get("scope")
        if isinstance(scope, str):
            return [part for part in scope.split() if part]
        if isinstance(scope, list):
            return [str(part) for part in scope]
        return []

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)


__all__ = [
    "ApiKeyRotationRequiredError",
    "ApiKeyValidationError",
    "IdentityError",
    "IdentityService",
    "IdentityUser",
    "InactiveAccountError",
    "InvalidCredentialsError",
    "PasswordExpiredError",
    "RefreshTokenError",
    "ScimOperationError",
]
