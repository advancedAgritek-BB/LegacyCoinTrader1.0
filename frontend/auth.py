"""
Simple Authentication System for LegacyCoinTrader Frontend
Provides basic authentication and authorization for the web interface.
"""

import hashlib
import hmac
import time
from typing import Optional, Dict, Any
from functools import wraps
from flask import request, session, jsonify, redirect, url_for


class SimpleAuth:
    """Simple authentication system for development and basic security."""

    def __init__(self, secret_key: str, users_file: str = None):
        self.secret_key = secret_key
        self.users_file = users_file or "./config/users.json"
        self.users = self._load_users()

    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Load users from file or create default admin user."""
        import os
        import json

        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

        # Create default admin user for development
        default_password = "admin123!"  # CHANGE THIS IN PRODUCTION
        default_user = {
            "username": "admin",
            "password_hash": self._hash_password(default_password),
            "role": "admin",
            "enabled": True,
            "created_at": time.time()
        }

        users = {"admin": default_user}

        # Save to file
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)

        print(f"WARNING: Default admin user created with password '{default_password}'")
        print(f"Please change the password and update {self.users_file}")

        return users

    def _hash_password(self, password: str) -> str:
        """Hash password with HMAC-SHA256."""
        return hmac.new(
            self.secret_key.encode(),
            password.encode(),
            hashlib.sha256
        ).hexdigest()

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials."""
        # Validate input parameters
        if not username or not password:
            return None

        user = self.users.get(username)
        if not user or not user.get('enabled', False):
            return None

        try:
            if self._hash_password(password) == user['password_hash']:
                return {
                    'username': username,
                    'role': user['role'],
                    'login_time': time.time()
                }
        except (AttributeError, TypeError):
            # Handle cases where password is None or not a string
            return None

        return None

    def login_required(self, f):
        """Decorator to require authentication for routes."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                if request.is_json or request.path.startswith('/api/'):
                    return jsonify({'error': 'Authentication required'}), 401
                else:
                    return redirect(url_for('login'))

            # Check session timeout
            login_time = session.get('login_time', 0)
            if time.time() - login_time > 3600:  # 1 hour timeout
                session.clear()
                if request.is_json or request.path.startswith('/api/'):
                    return jsonify({'error': 'Session expired'}), 401
                else:
                    return redirect(url_for('login'))

            return f(*args, **kwargs)
        return decorated_function

    def admin_required(self, f):
        """Decorator to require admin role."""
        @wraps(f)
        @self.login_required
        def decorated_function(*args, **kwargs):
            user = session.get('user', {})
            if user.get('role') != 'admin':
                if request.is_json or request.path.startswith('/api/'):
                    return jsonify({'error': 'Admin access required'}), 403
                else:
                    return "Admin access required", 403
            return f(*args, **kwargs)
        return decorated_function

    def api_key_required(self, f):
        """Decorator to require API key for API routes."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from frontend.config import get_config
            config = get_config()

            api_key = request.headers.get(config.security.api_key_header)
            if not api_key:
                return jsonify({'error': 'API key required'}), 401

            # In production, validate API key against database
            # For now, accept any non-empty key for development
            if not api_key.strip():
                return jsonify({'error': 'Invalid API key'}), 401

            return f(*args, **kwargs)
        return decorated_function


# Global auth instance
_auth_instance = None

def get_auth() -> SimpleAuth:
    """Get the global authentication instance."""
    global _auth_instance
    from frontend.config import get_config
    config = get_config()

    if _auth_instance is None:
        _auth_instance = SimpleAuth(config.security.session_secret_key)

    return _auth_instance


# Convenience functions
def login_required(f):
    """Decorator to require login."""
    return get_auth().login_required(f)

def admin_required(f):
    """Decorator to require admin role."""
    return get_auth().admin_required(f)

def api_key_required(f):
    """Decorator to require API key."""
    return get_auth().api_key_required(f)
