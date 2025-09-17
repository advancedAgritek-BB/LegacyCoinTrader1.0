"""
Secure Configuration Management for LegacyCoinTrader Frontend
Implements enterprise-grade security practices and configuration management.
"""

import os
import secrets
from typing import List, Optional, Dict, Any

from services.portfolio.rbac import load_role_definitions


class SecurityConfig:
    """Security configuration with enterprise-grade settings."""

    def __init__(self):
        # CORS Configuration - Load from environment
        cors_origins_str = os.getenv(
            "CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
        )
        self.cors_origins: List[str] = [
            origin.strip()
            for origin in cors_origins_str.split(",")
            if origin.strip()
        ]

        # CSP Configuration - Secure defaults with necessary allowances for functionality
        self.csp_default_src: List[str] = ["'self'"]
        # Allow inline scripts for functionality; allow trusted CDNs only
        self.csp_script_src: List[str] = [
            "'self'",
            "https://cdn.jsdelivr.net",
            "'unsafe-inline'",
        ]
        # Disallow inline styles; allow only trusted style CDNs
        self.csp_style_src: List[str] = [
            "'self'",
            "https://fonts.googleapis.com",
            "https://cdn.jsdelivr.net",
            "https://cdnjs.cloudflare.com",
        ]
        self.csp_img_src: List[str] = ["'self'", "data:", "https:"]
        self.csp_connect_src: List[str] = ["'self'", "https:", "wss:"]
        # Include CDNs used by Font Awesome for webfonts
        self.csp_font_src: List[str] = [
            "'self'",
            "https://fonts.googleapis.com",
            "https://fonts.gstatic.com",
            "https://cdnjs.cloudflare.com",
        ]

        # Rate Limiting
        try:
            self.rate_limit_requests: int = int(
                os.getenv("RATE_LIMIT_REQUESTS", "100")
            )
        except (ValueError, TypeError):
            self.rate_limit_requests: int = 100

        try:
            self.rate_limit_window: int = int(
                os.getenv("RATE_LIMIT_WINDOW", "60")
            )
        except (ValueError, TypeError):
            self.rate_limit_window: int = 60

        # Session Security
        self.session_secret_key: str = os.getenv(
            "SESSION_SECRET_KEY", secrets.token_urlsafe(32)
        )

        try:
            self.session_timeout: int = int(
                os.getenv("SESSION_TIMEOUT", "3600")
            )
        except (ValueError, TypeError):
            self.session_timeout: int = 3600

        try:
            self.password_rotation_days: int = int(
                os.getenv(
                    "PASSWORD_ROTATION_DAYS",
                    os.getenv("PORTFOLIO_PASSWORD_ROTATION_DAYS", "90"),
                )
            )
        except (ValueError, TypeError):
            self.password_rotation_days = 90

        # API Security
        self.api_key_header: str = os.getenv("API_KEY_HEADER", "X-API-Key")
        self.allowed_methods: List[str] = self._parse_allowed_methods()
        try:
            self.role_definitions = load_role_definitions()
        except ValueError as exc:
            raise ValueError(f"Invalid ROLE_DEFINITIONS configuration: {exc}") from exc

    def _parse_allowed_methods(self) -> List[str]:
        """Parse and validate allowed HTTP methods from environment."""
        methods_str = os.getenv("ALLOWED_METHODS", "GET,POST,OPTIONS")
        methods = [method.strip().upper() for method in methods_str.split(",")]

        # Security validation - only allow safe methods
        safe_methods = {"GET", "POST", "OPTIONS", "HEAD"}
        dangerous_methods = {"PUT", "DELETE", "PATCH"}

        for method in methods:
            if method in dangerous_methods:
                raise ValueError(
                    f"HTTP method {method} is not allowed for security reasons"
                )

        return [method for method in methods if method in safe_methods]

    def get_csp_header(self) -> str:
        """Generate Content Security Policy header."""
        csp_parts = [
            f"default-src {' '.join(self.csp_default_src)}",
            f"script-src {' '.join(self.csp_script_src)}",
            f"style-src {' '.join(self.csp_style_src)}",
            f"img-src {' '.join(self.csp_img_src)}",
            f"connect-src {' '.join(self.csp_connect_src)}",
            f"font-src {' '.join(self.csp_font_src)}",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        return "; ".join(csp_parts)

    def get_cors_headers(self, origin: str) -> Dict[str, str]:
        """Generate CORS headers for a request."""
        headers = {
            "Access-Control-Allow-Methods": ", ".join(self.allowed_methods),
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
            "Access-Control-Max-Age": "86400",  # 24 hours
        }

        # Only allow origins that are in our whitelist
        if origin in self.cors_origins:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"

        return headers


class AppConfig:
    """Main application configuration."""

    def __init__(self):
        # Environment
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.app_name: str = os.getenv("APP_NAME", "LegacyCoinTrader")
        self.version: str = os.getenv("VERSION", "2.0.0")

        # Server
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "5000"))

        # Security (nested config)
        self.security: SecurityConfig = SecurityConfig()

        # Database
        self.database_url: str = os.getenv(
            "DATABASE_URL", "sqlite:///legacy_trader.db"
        )

        # Redis
        self.redis_host: str = os.getenv("REDIS_HOST", "localhost")
        self.redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db: int = int(os.getenv("REDIS_DB", "0"))

        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_dir: str = os.getenv("LOG_DIR", "./logs")

    def dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for compatibility."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "app_name": self.app_name,
            "version": self.version,
            "host": self.host,
            "port": self.port,
            "security": {
                "cors_origins": self.security.cors_origins,
                "allowed_methods": self.security.allowed_methods,
                "rate_limit_requests": self.security.rate_limit_requests,
                "session_secret_key": self.security.session_secret_key,
                "password_rotation_days": self.security.password_rotation_days,
                "role_definitions": self.security.role_definitions,
            },
            "database_url": self.database_url,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
        }


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload configuration from environment."""
    global config
    config = AppConfig()


# Validate configuration on import
try:
    # Configuration is validated automatically by pydantic on instantiation
    _ = config.dict()  # This will raise an exception if validation fails
except Exception as e:
    print(f"Configuration validation error: {e}")
    raise
