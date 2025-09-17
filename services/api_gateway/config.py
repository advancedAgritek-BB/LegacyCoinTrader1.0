from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ServiceRouteConfig:
    """Configuration describing how to proxy requests for a downstream service."""

    name: str
    prefix: str
    url: str
    rate_limit_per_minute: int
    methods: Iterable[str] = field(
        default_factory=lambda: [
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
        ]
    )
    allowed_auth_modes: List[str] = field(
        default_factory=lambda: ["jwt", "service"]
    )
    service_token: Optional[str] = None
    health_endpoint: str = "/health"
    required_roles: List[str] = field(default_factory=list)

    def build_target_url(self, path_suffix: str) -> str:
        """Construct the target URL for the downstream service."""

        base = self.url.rstrip("/")
        if not path_suffix:
            return base
        cleaned = path_suffix.lstrip("/")
        return f"{base}/{cleaned}" if cleaned else base


@dataclass(slots=True)
class GatewaySettings:
    """Runtime configuration for the API gateway service."""

    host: str
    port: int
    log_level: str
    jwt_secret: str
    jwt_algorithm: str
    jwt_audience: Optional[str]
    require_authentication: bool
    service_tokens: Dict[str, str]
    default_rate_limit_per_minute: int
    rate_limit_window_seconds: int
    redis_host: str
    redis_port: int
    redis_db: int
    http_client_timeout: float
    allowed_origins: List[str]
    service_routes: Dict[str, ServiceRouteConfig]
    token_ttl_seconds: int
    token_issuer: str
    identity_service_url: str
    identity_request_timeout: float
    identity_default_tenant: Optional[str]
    identity_service_token: Optional[str]
    identity_service_token_header: str
    identity_tenant_header: str
    identity_jwks_url: Optional[str]
    identity_issuer: Optional[str]
    identity_audience: Optional[str]
    identity_jwks_cache_seconds: int

    @property
    def cors_origins(self) -> List[str]:
        if not self.allowed_origins:
            return ["*"]
        if len(self.allowed_origins) == 1 and self.allowed_origins[0] == "*":
            return ["*"]
        return self.allowed_origins


_SERVICE_PREFIX_OVERRIDES: Dict[str, str] = {
    "trading_engine": "/api/v1/trading",
    "market_data": "/api/v1/market-data",
    "portfolio": "/api/v1/portfolio",
    "strategy_engine": "/api/v1/strategy",
    "token_discovery": "/api/v1/token-discovery",
    "execution": "/api/v1/execution",
    "monitoring": "/api/v1/monitoring",
}

_SERVICE_ROLE_REQUIREMENTS: Dict[str, List[str]] = {
    "portfolio": ["portfolio"],
    "trading_engine": ["trading"],
    "market_data": ["market"],
    "strategy_engine": ["strategy"],
    "token_discovery": ["token"],
    "execution": ["execution"],
    "monitoring": ["monitoring"],
}


def _architecture_path() -> Path:
    return Path(__file__).resolve().parents[2] / "microservice_architecture.yaml"


def _load_architecture() -> Dict[str, Dict[str, object]]:
    path = _architecture_path()
    if not path.exists():
        LOGGER.warning("microservice_architecture.yaml not found at %s", path)
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
        services = data.get("services", {})
        return services
    except Exception as exc:  # pragma: no cover - defensive guardrail
        LOGGER.error("Failed to parse microservice architecture file: %s", exc)
        return {}


def _load_service_tokens() -> Dict[str, str]:
    """Load service tokens either from JSON or individual environment variables."""

    raw_tokens = os.getenv("GATEWAY_SERVICE_TOKENS")
    if raw_tokens:
        try:
            parsed = json.loads(raw_tokens)
            return {str(k): str(v) for k, v in parsed.items() if v}
        except json.JSONDecodeError as exc:
            LOGGER.warning("Invalid JSON provided via GATEWAY_SERVICE_TOKENS: %s", exc)

    tokens: Dict[str, str] = {}
    services = _load_architecture()
    for service_name, definition in services.items():
        if service_name == "api_gateway":
            continue
        if isinstance(definition, dict) and definition.get("type") == "infrastructure":
            continue
        env_key = f"GATEWAY_SERVICE_TOKEN_{service_name.upper()}".replace("-", "_")
        token = os.getenv(env_key)
        if token:
            tokens[service_name] = token
        else:
            # Provide an insecure default to simplify local development.
            # Production deployments should always override this value.
            tokens[service_name] = f"insecure-local-token-{service_name}"
    return tokens


def _build_service_url(service_name: str, port: int) -> str:
    host_env_key = f"{service_name.upper()}_HOST".replace("-", "_")
    port_env_key = f"{service_name.upper()}_PORT".replace("-", "_")

    host = os.getenv(host_env_key)
    if not host:
        host = service_name.replace("_", "-")
    target_port = int(os.getenv(port_env_key, str(port)))
    return f"http://{host}:{target_port}"


def _build_route_config(
    service_name: str,
    service_definition: Dict[str, object],
    service_tokens: Dict[str, str],
    default_limit: int,
) -> ServiceRouteConfig:
    prefix = _SERVICE_PREFIX_OVERRIDES.get(
        service_name, f"/api/v1/{service_name.replace('_', '-')}"
    )
    port = int(service_definition.get("port", 0) or 0)
    url = _build_service_url(service_name, port)
    limit_env = os.getenv(
        f"GATEWAY_RATE_LIMIT_{service_name.upper()}",
        os.getenv("GATEWAY_RATE_LIMIT_PER_MINUTE", str(default_limit)),
    )
    auth_modes = ["jwt", "service"]
    if service_name == "monitoring":
        auth_modes = ["service", "jwt"]

    return ServiceRouteConfig(
        name=service_name,
        prefix=prefix,
        url=url,
        rate_limit_per_minute=int(limit_env),
        allowed_auth_modes=auth_modes,
        service_token=service_tokens.get(service_name),
        required_roles=_SERVICE_ROLE_REQUIREMENTS.get(service_name, ["admin"]),
    )


def _load_service_routes(default_rate_limit: int) -> Dict[str, ServiceRouteConfig]:
    services = _load_architecture()
    service_tokens = _load_service_tokens()
    routes: Dict[str, ServiceRouteConfig] = {}
    for name, definition in services.items():
        if name == "api_gateway":
            continue
        if not isinstance(definition, dict):
            continue
        if definition.get("type") == "infrastructure":
            continue
        if not definition.get("port"):
            continue
        routes[name] = _build_route_config(name, definition, service_tokens, default_rate_limit)
    return routes


def load_gateway_settings() -> GatewaySettings:
    """Assemble gateway settings from environment variables and architecture file."""

    default_rate_limit = int(os.getenv("GATEWAY_RATE_LIMIT_REQUESTS", "60"))
    allowed_origins_env = os.getenv("GATEWAY_ALLOWED_ORIGINS", "*")
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
    if not allowed_origins:
        allowed_origins = ["*"]

    service_routes = _load_service_routes(default_rate_limit)
    resolved_tokens = {name: config.service_token or "" for name, config in service_routes.items()}

    identity_service_url = os.getenv("IDENTITY_SERVICE_URL", "http://identity:8006").rstrip("/")
    identity_jwks_url = os.getenv("IDENTITY_JWKS_URL") or f"{identity_service_url}/.well-known/jwks.json"
    identity_service_token = os.getenv("IDENTITY_SERVICE_TOKEN") or resolved_tokens.get("identity")
    identity_default_tenant = os.getenv("IDENTITY_DEFAULT_TENANT") or None
    identity_service_token_header = os.getenv("IDENTITY_SERVICE_TOKEN_HEADER", "x-service-token")
    identity_tenant_header = os.getenv("IDENTITY_TENANT_HEADER", "X-Tenant-ID")
    identity_issuer = os.getenv("IDENTITY_EXPECTED_ISSUER") or None
    identity_audience = os.getenv("IDENTITY_EXPECTED_AUDIENCE") or None
    identity_request_timeout = float(os.getenv("IDENTITY_REQUEST_TIMEOUT", "10"))
    identity_jwks_cache_seconds = int(os.getenv("IDENTITY_JWKS_CACHE_SECONDS", "300"))

    return GatewaySettings(
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("GATEWAY_PORT", "8000")),
        log_level=os.getenv("GATEWAY_LOG_LEVEL", "INFO"),
        jwt_secret=os.getenv("GATEWAY_JWT_SECRET", "change-me"),
        jwt_algorithm=os.getenv("GATEWAY_JWT_ALGORITHM", "HS256"),
        jwt_audience=os.getenv("GATEWAY_JWT_AUDIENCE", "") or None,
        require_authentication=os.getenv("GATEWAY_REQUIRE_AUTH", "1") != "0",
        service_tokens={name: token for name, token in resolved_tokens.items() if token},
        default_rate_limit_per_minute=default_rate_limit,
        rate_limit_window_seconds=int(os.getenv("GATEWAY_RATE_LIMIT_WINDOW", "60")),
        redis_host=os.getenv("GATEWAY_REDIS_HOST", "redis"),
        redis_port=int(os.getenv("GATEWAY_REDIS_PORT", "6379")),
        redis_db=int(os.getenv("GATEWAY_REDIS_DB", "0")),
        http_client_timeout=float(os.getenv("GATEWAY_HTTP_CLIENT_TIMEOUT", "30")),
        allowed_origins=allowed_origins,
        service_routes=service_routes,
        token_ttl_seconds=int(os.getenv("GATEWAY_TOKEN_TTL_SECONDS", "3600")),
        token_issuer=os.getenv("GATEWAY_TOKEN_ISSUER", "legacycointrader-gateway"),
        identity_service_url=identity_service_url,
        identity_request_timeout=identity_request_timeout,
        identity_default_tenant=identity_default_tenant,
        identity_service_token=identity_service_token,
        identity_service_token_header=identity_service_token_header,
        identity_tenant_header=identity_tenant_header,
        identity_jwks_url=identity_jwks_url,
        identity_issuer=identity_issuer,
        identity_audience=identity_audience,
        identity_jwks_cache_seconds=identity_jwks_cache_seconds,
    )

