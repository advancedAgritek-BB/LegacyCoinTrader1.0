"""Authentication and authorization helpers for the LegacyCoinTrader UI."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Dict, Iterable, Optional, Sequence

from flask import g, jsonify, redirect, request, session, url_for

from services.portfolio.config import PortfolioConfig
from services.portfolio.security import (
    AuthenticationError,
    IdentityService,
    InactiveUserError,
    PasswordExpiredError,
)

logger = logging.getLogger(__name__)


class SimpleAuth:
    """Facade that authenticates against the portfolio credential store."""

    def __init__(self, app_config):
        self.config = app_config
        self.session_timeout = app_config.security.session_timeout
        self.role_definitions = app_config.security.role_definitions
        self._last_error: Optional[str] = None

        portfolio_config = PortfolioConfig.from_env()
        self.identity = IdentityService(
            portfolio_config,
            password_max_age_days=app_config.security.password_rotation_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user via username/password credentials."""

        self._last_error = None
        if not username or not password:
            self._last_error = "invalid_credentials"
            return None

        try:
            user = self.identity.authenticate_user(username, password)
        except PasswordExpiredError:
            self._last_error = "password_expired"
            logger.info("Password rotation required for user %s", username)
            return None
        except InactiveUserError:
            self._last_error = "account_disabled"
            logger.warning("Disabled user %s attempted authentication", username)
            return None
        except AuthenticationError:
            self._last_error = "invalid_credentials"
            logger.warning("Authentication failed for %s", username)
            return None
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Unexpected authentication error for %s", username)
            self._last_error = "authentication_error"
            return None

        if user.role not in self.role_definitions:
            logger.error("Role %s is not defined in the RBAC configuration", user.role)
            self._last_error = "role_not_configured"
            return None

        permissions = list(self.role_definitions.get(user.role, []))
        return {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "permissions": permissions,
            "login_time": time.time(),
            "password_expires_at": user.password_expires_at.isoformat()
            if user.password_expires_at
            else None,
        }

    def get_last_error(self) -> Optional[str]:
        """Return the last authentication error for UI feedback."""

        return self._last_error

    def login_required(self, func):
        """Decorator that enforces login and session timeout."""

        @wraps(func)
        def wrapped(*args, **kwargs):
            if "user" not in session:
                return self._unauthorized_response()

            login_time = session.get("login_time")
            current = time.time()
            if not login_time or current - login_time > self.session_timeout:
                session.clear()
                return self._session_expired_response()

            session["login_time"] = current
            return func(*args, **kwargs)

        return wrapped

    def role_required(self, roles: Iterable[str]):
        """Decorator enforcing that the user session owns one of ``roles``."""

        allowed = set(roles)

        def decorator(func):
            @wraps(func)
            @self.login_required
            def wrapped(*args, **kwargs):
                user = session.get("user", {})
                role = user.get("role")
                if role not in allowed:
                    if request.is_json or request.path.startswith("/api/"):
                        return jsonify({"error": "Insufficient privileges"}), 403
                    return "Insufficient privileges", 403
                return func(*args, **kwargs)

            return wrapped

        return decorator

    def admin_required(self, func):
        """Decorator restricting access to administrator roles."""

        return self.role_required({"admin"})(func)

    def api_key_required(
        self,
        func=None,
        *,
        roles: Optional[Sequence[str]] = None,
    ):
        """Decorator to enforce API key validation for service integrations."""

        required_roles = set(roles or [])

        def decorator(inner):
            @wraps(inner)
            def wrapped(*args, **kwargs):
                api_key = request.headers.get(self.config.security.api_key_header)
                if not api_key:
                    return jsonify({"error": "API key required"}), 401

                try:
                    user = self.identity.authenticate_api_key(api_key)
                except PasswordExpiredError:
                    return (
                        jsonify(
                            {
                                "error": "Password rotation required before API key can be used",
                            }
                        ),
                        403,
                    )
                except InactiveUserError:
                    return jsonify({"error": "Account disabled"}), 403
                except AuthenticationError:
                    return jsonify({"error": "Invalid API key"}), 401
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("API key validation failed unexpectedly")
                    return (
                        jsonify({"error": "Authentication service unavailable"}),
                        503,
                    )

                if required_roles and user.role not in required_roles:
                    return jsonify({"error": "Role not permitted for this endpoint"}), 403

                g.api_user = {"username": user.username, "role": user.role}
                return inner(*args, **kwargs)

            return wrapped

        if func is not None:
            return decorator(func)
        return decorator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _unauthorized_response(self):
        if request.is_json or request.path.startswith("/api/"):
            return jsonify({"error": "Authentication required"}), 401
        return redirect(url_for("login"))

    def _session_expired_response(self):
        if request.is_json or request.path.startswith("/api/"):
            return jsonify({"error": "Session expired"}), 401
        return redirect(url_for("login"))


# Global auth instance
_auth_instance: Optional[SimpleAuth] = None


def get_auth() -> SimpleAuth:
    """Return the global authentication helper."""

    global _auth_instance
    from frontend.config import get_config

    config = get_config()
    if _auth_instance is None:
        _auth_instance = SimpleAuth(config)
    return _auth_instance


def login_required(func):
    return get_auth().login_required(func)


def admin_required(func):
    return get_auth().admin_required(func)


def api_key_required(func=None, *, roles: Optional[Sequence[str]] = None):
    return get_auth().api_key_required(func, roles=roles)
