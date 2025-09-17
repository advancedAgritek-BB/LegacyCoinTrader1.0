"""Token management helpers for the identity service."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from services.common.secrets import SecretRetrievalError, resolve_secret

from .config import IdentitySettings
from .models import TenantModel


@dataclass(slots=True)
class TenantSigningKey:
    """Represents signing material for a tenant."""

    tenant_slug: str
    key_id: str
    algorithm: str
    private_key_pem: Optional[str] = None
    public_key_pem: Optional[str] = None
    shared_secret: Optional[str] = None
    last_loaded_at: datetime = datetime.now(timezone.utc)

    def as_public_jwk(self) -> Optional[Dict[str, str]]:
        """Return the JWK representation of the signing key if possible."""

        if self.algorithm.upper().startswith("HS"):
            # Symmetric keys are not exposed via JWKS for security reasons.
            return None
        if not self.public_key_pem:
            return None
        public_key = serialization.load_pem_public_key(self.public_key_pem.encode("utf-8"))
        assert isinstance(public_key, rsa.RSAPublicKey)
        numbers = public_key.public_numbers()
        return {
            "kty": "RSA",
            "use": "sig",
            "alg": self.algorithm,
            "kid": self.key_id,
            "n": _int_to_base64(numbers.n),
            "e": _int_to_base64(numbers.e),
        }


def _int_to_base64(value: int) -> str:
    byte_length = (value.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(value.to_bytes(byte_length, "big")).rstrip(b"=").decode("ascii")


def _generate_jti() -> str:
    from uuid import uuid4

    return uuid4().hex


class TokenSigner:
    """Create and validate JWT tokens for tenants."""

    def __init__(self, settings: IdentitySettings) -> None:
        self.settings = settings
        self._cache: Dict[str, TenantSigningKey] = {}

    # ------------------------------------------------------------------
    # Key loading helpers
    # ------------------------------------------------------------------
    def get_signing_key(self, tenant: TenantModel) -> TenantSigningKey:
        cached = self._cache.get(tenant.slug)
        if cached:
            return cached

        signing_material = self._load_signing_material(tenant)
        self._cache[tenant.slug] = signing_material
        return signing_material

    def _load_signing_material(self, tenant: TenantModel) -> TenantSigningKey:
        env_keys = [
            f"IDENTITY_{tenant.slug.upper()}_SIGNING_KEY",
            "IDENTITY_SIGNING_KEY",
        ]
        try:
            raw_secret = resolve_secret(
                "OIDC_SIGNING_KEY",
                env_keys=env_keys,
                vault_path=tenant.secret_reference,
            )
        except SecretRetrievalError:
            if not self.settings.allow_development_fallback_keys:
                raise
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            private_pem = private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ).decode("utf-8")
            public_pem = private_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("utf-8")
            signing_key = TenantSigningKey(
                tenant_slug=tenant.slug,
                key_id=tenant.key_id,
                algorithm=self.settings.token_algorithm,
                private_key_pem=private_pem,
                public_key_pem=public_pem,
            )
            return signing_key

        algorithm = tenant.metadata_json.get("token_algorithm") or self.settings.token_algorithm
        private_pem: Optional[str] = None
        public_pem: Optional[str] = None
        shared_secret: Optional[str] = None

        stripped = raw_secret.strip()
        if stripped.startswith("{"):
            data = json.loads(stripped)
            private_pem = data.get("private_key") or data.get("privateKey")
            public_pem = data.get("public_key") or data.get("publicKey")
            shared_secret = data.get("shared_secret") or data.get("secret")
            algorithm = data.get("algorithm") or algorithm
        else:
            if algorithm.upper().startswith("HS"):
                shared_secret = stripped
            else:
                private_pem = stripped

        signing_key = TenantSigningKey(
            tenant_slug=tenant.slug,
            key_id=tenant.key_id,
            algorithm=algorithm,
            private_key_pem=private_pem,
            public_key_pem=public_pem,
            shared_secret=shared_secret,
        )

        if signing_key.private_key_pem and not signing_key.public_key_pem:
            private_key = serialization.load_pem_private_key(
                signing_key.private_key_pem.encode("utf-8"),
                password=None,
            )
            assert isinstance(private_key, rsa.RSAPrivateKey)
            signing_key.public_key_pem = private_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("utf-8")
        return signing_key

    # ------------------------------------------------------------------
    # Token issuance
    # ------------------------------------------------------------------
    def mint_access_token(
        self,
        tenant: TenantModel,
        subject: str,
        scopes: list[str],
        roles: list[str],
        expires_in: int,
        *,
        audience: Optional[str] = None,
        additional_claims: Optional[Dict[str, object]] = None,
    ) -> tuple[str, datetime, datetime, str]:
        """Return an encoded JWT access token."""

        issued_at = datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(seconds=expires_in)
        jti = _generate_jti()
        payload: Dict[str, object] = {
            "sub": subject,
            "iss": tenant.issuer,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
            "scope": " ".join(scopes),
            "roles": roles,
            "tenant": tenant.slug,
            "jti": jti,
        }
        if audience:
            payload["aud"] = audience
        if additional_claims:
            payload.update(additional_claims)

        signing_key = self.get_signing_key(tenant)
        headers = {"kid": tenant.key_id, "typ": "JWT"}
        token = jwt.encode(
            payload,
            self._jwt_signing_value(signing_key),
            algorithm=signing_key.algorithm,
            headers=headers,
        )
        return token, issued_at, expires_at, jti

    def mint_refresh_token(
        self,
        tenant: TenantModel,
        subject: str,
        expires_in: int,
        *,
        client_id: Optional[str] = None,
    ) -> tuple[str, datetime, datetime, str]:
        issued_at = datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(seconds=expires_in)
        jti = _generate_jti()
        payload: Dict[str, object] = {
            "sub": subject,
            "iss": tenant.issuer,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
            "tenant": tenant.slug,
            "type": "refresh",
            "jti": jti,
        }
        if client_id:
            payload["client_id"] = client_id

        signing_key = self.get_signing_key(tenant)
        headers = {"kid": tenant.key_id, "typ": "JWT"}
        token = jwt.encode(
            payload,
            self._jwt_signing_value(signing_key),
            algorithm=signing_key.algorithm,
            headers=headers,
        )
        return token, issued_at, expires_at, jti

    def verify_token(
        self,
        tenant: TenantModel,
        token: str,
        *,
        audience: Optional[str] = None,
    ) -> Dict[str, object]:
        signing_key = self.get_signing_key(tenant)
        options = {"verify_aud": audience is not None}
        verified = jwt.decode(
            token,
            self._jwt_verification_value(signing_key),
            algorithms=[signing_key.algorithm],
            audience=audience,
            options=options,
        )
        return dict(verified)

    def _jwt_signing_value(self, signing_key: TenantSigningKey):
        if signing_key.shared_secret:
            return signing_key.shared_secret
        assert signing_key.private_key_pem, "No private key available for signing"
        return signing_key.private_key_pem

    def _jwt_verification_value(self, signing_key: TenantSigningKey):
        if signing_key.shared_secret:
            return signing_key.shared_secret
        assert signing_key.public_key_pem, "No public key available for verification"
        return signing_key.public_key_pem

    # ------------------------------------------------------------------
    # JWKS helpers
    # ------------------------------------------------------------------
    def build_jwks(self) -> Dict[str, object]:
        keys = [key.as_public_jwk() for key in self._cache.values()]
        filtered = [key for key in keys if key]
        return {"keys": filtered}


__all__ = ["TenantSigningKey", "TokenSigner"]
