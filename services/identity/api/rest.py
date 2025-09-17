"""FastAPI application exposing the identity service."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..config import IdentitySettings, load_identity_settings
from ..database import get_session
from ..models import TenantModel
from ..schemas import IntrospectionResponse
from ..service import (
    ApiKeyRotationRequiredError,
    ApiKeyValidationError,
    IdentityService,
    IdentityUser,
    InactiveAccountError,
    InvalidCredentialsError,
    PasswordExpiredError,
    RefreshTokenError,
    ScimOperationError,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


@lru_cache
def get_settings() -> IdentitySettings:
    return load_identity_settings()


@lru_cache
def get_identity_service() -> IdentityService:
    settings = get_settings()
    return IdentityService(settings)


def resolve_tenant(
    request: Request,
    settings: Annotated[IdentitySettings, Depends(get_settings)],
    tenant_header: Optional[str] = Header(default=None, alias="X-Tenant"),
) -> Optional[str]:
    candidates = list(settings.tenant_header_candidates)
    if tenant_header:
        return tenant_header
    for header in candidates:
        value = request.headers.get(header)
        if value:
            return value
    return settings.default_tenant_slug


def require_service_token(
    request: Request,
    settings: Annotated[IdentitySettings, Depends(get_settings)],
) -> None:
    token = request.headers.get(settings.service_token_header)
    if settings.internal_service_token and token != settings.internal_service_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service token")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class OAuthTokenRequest(BaseModel):
    grant_type: str = Field(..., alias="grant_type")
    username: Optional[str] = None
    password: Optional[str] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    mfa_code: Optional[str] = Field(default=None, alias="mfa_code")


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    expires_at: str
    issued_at: str
    scope: List[str]
    roles: List[str]
    username: str
    tenant: str
    password_expires_at: Optional[str] = None


class PasswordRotateRequest(BaseModel):
    username: str
    current_password: str
    new_password: str


class ApiKeyRotateRequest(BaseModel):
    username: str
    new_api_key: Optional[str] = None


class ApiKeyValidateRequest(BaseModel):
    api_key: str


class IntrospectionRequest(BaseModel):
    token: str


# ---------------------------------------------------------------------------
# Helper conversion functions
# ---------------------------------------------------------------------------


def _token_pair_to_response(token_pair) -> TokenResponse:
    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
        expires_in=token_pair.expires_in,
        expires_at=token_pair.expires_at.isoformat(),
        issued_at=token_pair.issued_at.isoformat(),
        scope=list(token_pair.scope),
        roles=list(token_pair.roles),
        username=token_pair.username,
        tenant=token_pair.tenant,
        password_expires_at=token_pair.password_expires_at.isoformat()
        if token_pair.password_expires_at
        else None,
    )


def _identity_user_to_payload(user: IdentityUser) -> dict:
    return {
        "username": user.username,
        "roles": user.roles,
        "password_rotated_at": user.password_rotated_at.isoformat() if user.password_rotated_at else None,
        "password_expires_at": user.password_expires_at.isoformat() if user.password_expires_at else None,
        "api_key_last_rotated_at": user.api_key_last_rotated_at.isoformat()
        if user.api_key_last_rotated_at
        else None,
        "api_key_expires_at": user.api_key_expires_at.isoformat() if user.api_key_expires_at else None,
        "tenant": user.tenant,
    }


# ---------------------------------------------------------------------------
# OAuth endpoints
# ---------------------------------------------------------------------------


@router.post("/oauth/token", response_model=TokenResponse)
def issue_token(
    payload: OAuthTokenRequest,
    tenant: Annotated[Optional[str], Depends(resolve_tenant)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
):
    try:
        if payload.grant_type == "password":
            token_pair = service.issue_token(
                tenant,
                payload.username or "",
                payload.password or "",
                requested_scopes=payload.scope,
                mfa_code=payload.mfa_code,
            )
        elif payload.grant_type == "refresh_token":
            if not payload.refresh_token:
                raise HTTPException(status_code=400, detail="refresh_token is required")
            token_pair = service.refresh_token(
                tenant,
                payload.refresh_token,
                scope=(payload.scope.split() if payload.scope else None),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported grant_type",
            )
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except InactiveAccountError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except PasswordExpiredError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except RefreshTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return _token_pair_to_response(token_pair)


@router.post("/oauth/introspect", response_model=IntrospectionResponse)
def introspect_token(
    payload: IntrospectionRequest,
    tenant: Annotated[Optional[str], Depends(resolve_tenant)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
    _: Annotated[None, Depends(require_service_token)] = None,
):
    return service.introspect_token(tenant, payload.token)


# ---------------------------------------------------------------------------
# Credential management endpoints
# ---------------------------------------------------------------------------


@router.post("/credentials/password/rotate")
def rotate_password(
    payload: PasswordRotateRequest,
    tenant: Annotated[Optional[str], Depends(resolve_tenant)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
):
    try:
        user = service.rotate_password(
            tenant,
            payload.username,
            payload.current_password,
            payload.new_password,
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except InactiveAccountError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return _identity_user_to_payload(user)


@router.post("/credentials/api-key/rotate")
def rotate_api_key(
    payload: ApiKeyRotateRequest,
    tenant: Annotated[Optional[str], Depends(resolve_tenant)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
):
    api_key, user = service.rotate_api_key(
        tenant,
        payload.username,
        new_api_key=payload.new_api_key,
    )
    response = _identity_user_to_payload(user)
    response["api_key"] = api_key
    return response


@router.post("/credentials/api-key/validate")
def validate_api_key(
    payload: ApiKeyValidateRequest,
    tenant: Annotated[Optional[str], Depends(resolve_tenant)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
):
    try:
        user = service.validate_api_key(tenant, payload.api_key)
    except ApiKeyRotationRequiredError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ApiKeyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return _identity_user_to_payload(user)


# ---------------------------------------------------------------------------
# SCIM endpoints
# ---------------------------------------------------------------------------


@router.get("/scim/v2/Users")
def scim_list_users(
    request: Request,
    tenant: Annotated[Optional[str], Depends(resolve_tenant)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
    _: Annotated[None, Depends(require_service_token)] = None,
):
    base_url = str(request.base_url).rstrip("/")
    users = service.scim_list_users(tenant, base_url)
    return {
        "Resources": [user.model_dump(mode="json") for user in users],
        "totalResults": len(users),
        "itemsPerPage": len(users),
        "startIndex": 1,
    }


@router.get("/scim/v2/Users/{user_id}")
def scim_get_user(
    user_id: int,
    request: Request,
    tenant: Annotated[Optional[str], Depends(resolve_tenant)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
    _: Annotated[None, Depends(require_service_token)] = None,
):
    base_url = str(request.base_url).rstrip("/")
    try:
        user = service.scim_get_user(tenant, user_id, base_url)
    except ScimOperationError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return JSONResponse(user.model_dump(mode="json"))


@router.post("/scim/v2/Users", status_code=status.HTTP_201_CREATED)
def scim_create_user(
    payload: dict,
    request: Request,
    tenant: Annotated[Optional[str], Depends(resolve_tenant)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
    _: Annotated[None, Depends(require_service_token)] = None,
):
    base_url = str(request.base_url).rstrip("/")
    try:
        user = service.scim_create_user(tenant, payload, base_url)
    except ScimOperationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return JSONResponse(user.model_dump(mode="json"), status_code=status.HTTP_201_CREATED)


@router.put("/scim/v2/Users/{user_id}")
def scim_replace_user(
    user_id: int,
    payload: dict,
    request: Request,
    tenant: Annotated[Optional[str], Depends(resolve_tenant)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
    _: Annotated[None, Depends(require_service_token)] = None,
):
    base_url = str(request.base_url).rstrip("/")
    try:
        user = service.scim_replace_user(tenant, user_id, payload, base_url)
    except ScimOperationError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return JSONResponse(user.model_dump(mode="json"))


@router.delete("/scim/v2/Users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def scim_delete_user(
    user_id: int,
    tenant: Annotated[Optional[str], Depends(resolve_tenant)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
    _: Annotated[None, Depends(require_service_token)] = None,
):
    service.scim_delete_user(tenant, user_id)
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content={})


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@router.get("/.well-known/openid-configuration")
def openid_configuration(
    request: Request,
    settings: Annotated[IdentitySettings, Depends(get_settings)],
    service: Annotated[IdentityService, Depends(get_identity_service)],
):
    base = str(request.base_url).rstrip("/")
    issuer = settings.default_issuer
    jwks_uri = f"{base}/.well-known/jwks.json"
    token_endpoint = f"{base}/oauth/token"
    introspection_endpoint = f"{base}/oauth/introspect"
    return {
        "issuer": issuer,
        "jwks_uri": jwks_uri,
        "token_endpoint": token_endpoint,
        "introspection_endpoint": introspection_endpoint,
        "scopes_supported": ["openid", "profile", "email", "offline_access"],
        "response_types_supported": ["token"],
        "grant_types_supported": ["password", "refresh_token"],
    }


@router.get("/.well-known/jwks.json")
def jwks(service: Annotated[IdentityService, Depends(get_identity_service)]):
    with get_session(service.settings) as session:
        tenants = session.scalars(select(TenantModel)).all()
        for tenant in tenants:
            service.token_signer.get_signing_key(tenant)
    keys = service.token_signer.build_jwks()
    return JSONResponse(keys)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="LegacyCoinTrader Identity Service", version="1.0.0")
    app.include_router(router)
    return app


app = create_app()
