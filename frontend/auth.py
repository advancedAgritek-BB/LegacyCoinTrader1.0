from __future__ import annotations

"""Authentication helpers for the LegacyCoinTrader frontend."""

import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Optional

import httpx
from flask import jsonify, redirect, request, session, url_for

from frontend.gateway import build_gateway_url

LOGGER = logging.getLogger(__name__)


class IdentityAuth:
    """Authenticate users against the API gateway identity service."""

    def __init__(self, session_timeout: int, http_timeout: float = 5.0) -> None:
        self.session_timeout = session_timeout
        self.http_timeout = http_timeout
        self.token_url = build_gateway_url("/auth/token")
        self.api_key_validation_url = build_gateway_url("/auth/api-key/validate")

    # ------------------------------------------------------------------
    # Core authentication
    # ------------------------------------------------------------------
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Return a session payload when the provided credentials are valid."""

        if not username or not password:
            return None

        try:
            response = httpx.post(
                self.token_url,
                json={"username": username, "password": password},
                timeout=self.http_timeout,
            )
        except httpx.RequestError as exc:  # pragma: no cover - network failures are runtime specific
            LOGGER.error("Unable to reach identity service: %s", exc)
            return None

        if response.status_code != 200:
            LOGGER.info(
                "Login attempt failed for user '%s': %s",
                username,
                response.text,
            )
            return None

        data = response.json()
        access_token = data.get("access_token")
        expires_at = data.get("expires_at")
        roles = data.get("roles") or []
        if not isinstance(roles, list):
            roles = [str(roles)]

        if not access_token:
            LOGGER.error("Identity service response missing access token")
            return None

        user_payload = {
            "username": data.get("username", username),
            "roles": [str(role).lower() for role in roles],
            "access_token": access_token,
            "token_expires_at": expires_at,
            "password_expires_at": data.get("password_expires_at"),
            "login_time": time.time(),
        }
        return user_payload

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------
    def login_required(self, func):
        """Decorator ensuring that a valid session exists."""

        @wraps(func)
        def decorated(*args, **kwargs):
            if "user" not in session or not self._session_active():
                session.clear()
                if request.is_json or request.path.startswith("/api/"):
                    return jsonify({"error": "Authentication required"}), 401
                return redirect(url_for("login"))
            return func(*args, **kwargs)

        return decorated

    def admin_required(self, func):
        """Decorator ensuring the authenticated user has admin privileges."""

        @wraps(func)
        @self.login_required
        def decorated(*args, **kwargs):
            roles = session.get("user", {}).get("roles", [])
            if "admin" not in roles:
                if request.is_json or request.path.startswith("/api/"):
                    return jsonify({"error": "Admin access required"}), 403
                return "Admin access required", 403
            return func(*args, **kwargs)

        return decorated

    def api_key_required(self, func):
        """Decorator validating API keys against the identity service."""

        @wraps(func)
        def decorated(*args, **kwargs):
            from frontend.config import get_settings

            settings = get_settings()
            api_key = request.headers.get(settings.security.api_key_header)
            if not api_key:
                return jsonify({"error": "API key required"}), 401

            try:
                response = httpx.post(
                    self.api_key_validation_url,
                    json={"api_key": api_key},
                    timeout=self.http_timeout,
                )
            except httpx.RequestError as exc:  # pragma: no cover - network specific failures
                LOGGER.error("API key validation error: %s", exc)
                return jsonify({"error": "Identity service unavailable"}), 503

            if response.status_code not in (200, 201):
                message = "Invalid API key"
                try:
                    payload = response.json()
                    message = payload.get("detail") or payload.get("error") or message
                except ValueError:
                    pass
                status_code = 403 if response.status_code == 403 else 401
                return jsonify({"error": message}), status_code

            data = response.json()
            session["api_key_identity"] = {
                "username": data.get("username"),
                "roles": data.get("roles", []),
                "validated_at": time.time(),
            }
            return func(*args, **kwargs)

        return decorated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _session_active(self) -> bool:
        login_time = session.get("login_time")
        if not login_time:
            return False

        if time.time() - float(login_time) > self.session_timeout:
            LOGGER.debug("Session expired due to inactivity")
            return False

        expiry_iso = session.get("token_expires_at")
        if expiry_iso:
            expires_at = self._parse_iso_datetime(expiry_iso)
            if expires_at and datetime.now(timezone.utc) >= expires_at:
                LOGGER.debug("Session expired due to token expiry")
                return False
        return True

    @staticmethod
    def _parse_iso_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            text = str(value)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)


_auth_instance: Optional[IdentityAuth] = None


def get_auth() -> IdentityAuth:
    """Return a lazily instantiated :class:`IdentityAuth`."""

    global _auth_instance
    if _auth_instance is None:
        from frontend.config import get_settings

        settings = get_settings()
        _auth_instance = IdentityAuth(
            session_timeout=settings.security.session_timeout,
            http_timeout=5.0,
        )
    return _auth_instance


# Convenience decorators -----------------------------------------------------

def login_required(func):
    return get_auth().login_required(func)


def admin_required(func):
    return get_auth().admin_required(func)


def api_key_required(func):
    return get_auth().api_key_required(func)
