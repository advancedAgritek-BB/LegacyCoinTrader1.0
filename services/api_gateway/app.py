"""FastAPI application providing gateway authentication services."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from services.portfolio.config import PortfolioConfig
from services.portfolio.security import (
    AuthenticationError,
    IdentityService,
    InactiveUserError,
    PasswordExpiredError,
)

from .config import GatewayConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="LegacyCoinTrader API Gateway", version="1.0.0")
config = GatewayConfig()
_portfolio_config = PortfolioConfig.from_env()
identity = IdentityService(_portfolio_config)

if config.security.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.security.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

bearer_scheme = HTTPBearer(auto_error=False)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str
    permissions: list[str]


class PrincipalResponse(BaseModel):
    username: str
    role: str
    permissions: list[str]
    issued_at: datetime
    expires_at: Optional[datetime] = None
    password_expires_at: Optional[datetime] = None


def _permissions_for_role(role: str) -> list[str]:
    permissions = config.security.role_definitions.get(role)
    if permissions is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="role_not_configured",
        )
    return permissions


def _issue_token(username: str, role: str) -> TokenResponse:
    permissions = _permissions_for_role(role)
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + config.security.token_lifetime
    payload = {
        "sub": username,
        "role": role,
        "permissions": permissions,
        "iss": config.issuer,
        "aud": config.audience,
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    token = jwt.encode(
        payload,
        config.security.jwt_secret,
        algorithm=config.security.jwt_algorithm,
    )
    return TokenResponse(
        access_token=token,
        expires_in=int(config.security.token_lifetime.total_seconds()),
        role=role,
        permissions=permissions,
    )


def _decode_token(credentials: HTTPAuthorizationCredentials) -> dict:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not_authenticated")
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            config.security.jwt_secret,
            algorithms=[config.security.jwt_algorithm],
            audience=config.audience,
            issuer=config.issuer,
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token_expired")
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid token presented: %s", exc)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_token")


def _principal_from_token(payload: dict) -> PrincipalResponse:
    username = payload.get("sub")
    role = payload.get("role")
    permissions = payload.get("permissions") or []
    if not username or not role:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_token")

    issued_at_ts = payload.get("iat")
    expires_at_ts = payload.get("exp")
    issued_at = datetime.fromtimestamp(issued_at_ts, tz=timezone.utc) if issued_at_ts else datetime.now(timezone.utc)
    expires_at = (
        datetime.fromtimestamp(expires_at_ts, tz=timezone.utc)
        if expires_at_ts
        else None
    )

    user = identity.get_user(username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user_not_found")

    permissions = _permissions_for_role(user.role)
    return PrincipalResponse(
        username=user.username,
        role=user.role,
        permissions=permissions,
        issued_at=issued_at,
        expires_at=expires_at,
        password_expires_at=user.password_expires_at,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/token", response_model=TokenResponse)
async def create_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> TokenResponse:
    try:
        user = identity.authenticate_user(form_data.username, form_data.password)
    except PasswordExpiredError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="password_expired")
    except InactiveUserError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="account_disabled")
    except AuthenticationError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_credentials")

    return _issue_token(user.username, user.role)


@app.get("/auth/verify", response_model=PrincipalResponse)
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> PrincipalResponse:
    payload = _decode_token(credentials)
    return _principal_from_token(payload)


class ApiKeyPayload(BaseModel):
    api_key: str


@app.post("/auth/api-key", response_model=PrincipalResponse)
async def verify_api_key(payload: ApiKeyPayload) -> PrincipalResponse:
    if not payload.api_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="api_key_required")

    try:
        user = identity.authenticate_api_key(payload.api_key)
    except PasswordExpiredError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="password_expired")
    except InactiveUserError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="account_disabled")
    except AuthenticationError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key")

    permissions = _permissions_for_role(user.role)
    issued_at = datetime.now(timezone.utc)
    return PrincipalResponse(
        username=user.username,
        role=user.role,
        permissions=permissions,
        issued_at=issued_at,
        password_expires_at=user.password_expires_at,
    )
