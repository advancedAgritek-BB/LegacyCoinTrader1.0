from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

LOGGER = logging.getLogger(__name__)


_DEFAULT_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


@dataclass(slots=True)
class ServiceRouteConfig:
    """Configuration describing how to proxy requests for a downstream service."""

    name: str
    prefix: str
    url: str
    rate_limit_per_minute: int
    methods: Iterable[str] = field(default_factory=lambda: list(_DEFAULT_METHODS))
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


@dataclass
class TLSConfig:
    """TLS/SSL configuration for HTTPS support."""

    enabled: bool = False
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    ssl_version: str = "TLSv1.2"
    ciphers: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if TLS configuration is valid."""
        if not self.enabled:
            return True
        return bool(self.cert_file and self.key_file and Path(self.cert_file).exists() and Path(self.key_file).exists())


@dataclass
class ServiceTokenConfig:
    """Configuration for service-to-service authentication."""

    enabled: bool = True
    token_rotation_days: int = 30
    token_length: int = 64
    allowed_services: List[str] = field(default_factory=lambda: [
        "trading-engine",
        "portfolio",
        "market-data",
        "strategy-engine",
        "execution",
        "token-discovery",
        "monitoring"
    ])


@dataclass
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
    tenant_service_url: Optional[str]
    tenant_service_token: Optional[str]
    tenant_service_token_header: str
    tenant_service_timeout: float
    tenant_metadata_cache_seconds: int
    tenant_default_plan: str
    tenant_default_rate_limit_per_minute: int
    tenant_default_burst_limit: int
    tenant_default_burst_window_seconds: int
    kafka_bootstrap_servers: Optional[str]
    kafka_usage_topic: str
    audit_service_url: Optional[str]
    audit_service_token: Optional[str]
    audit_service_token_header: str
    audit_event_source: str
    audit_service_timeout: float
    tls: TLSConfig = field(default_factory=TLSConfig)
    service_auth: ServiceTokenConfig = field(default_factory=ServiceTokenConfig)

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
    "identity": "/api/v1/identity",
    "tenant_management": "/api/v1/tenants",
    "audit": "/api/v1/audit",
}

_SERVICE_AUTH_MODE_OVERRIDES: Dict[str, List[str]] = {
    "identity": ["anonymous", "jwt", "service"],
    "tenant_management": ["jwt", "service"],
    "audit": ["service", "jwt"],
    "monitoring": ["service", "jwt"],
}

_SERVICE_ROLE_REQUIREMENTS: Dict[str, List[str]] = {
    "portfolio": ["portfolio"],
    "trading_engine": ["trading"],
    "market_data": ["market"],
    "strategy_engine": ["strategy"],
    "token_discovery": ["token"],
    "execution": ["execution"],
    "monitoring": ["monitoring"],
    "identity": ["identity", "admin"],
    "tenant_management": ["tenant-admin", "admin"],
    "audit": ["compliance", "admin"],
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
    gateway_cfg = (
        service_definition.get("gateway")
        if isinstance(service_definition.get("gateway"), dict)
        else {}
    )

    prefix = str(
        gateway_cfg.get("prefix")
        or service_definition.get("prefix")
        or _SERVICE_PREFIX_OVERRIDES.get(
            service_name, f"/api/v1/{service_name.replace('_', '-')}"
        )
    )
    port = int(service_definition.get("port", 0) or 0)
    url = _build_service_url(service_name, port)
    configured_limit = gateway_cfg.get("rate_limit_per_minute")
    limit_default = default_limit
    if configured_limit is not None:
        try:
            limit_default = int(configured_limit)
        except (TypeError, ValueError):
            LOGGER.debug(
                "Invalid rate limit override for service %s: %s",
                service_name,
                configured_limit,
            )
    limit_env = os.getenv(
        f"GATEWAY_RATE_LIMIT_{service_name.upper()}",
        os.getenv("GATEWAY_RATE_LIMIT_PER_MINUTE", str(limit_default)),
    )

    methods_cfg = gateway_cfg.get("methods")
    if isinstance(methods_cfg, str):
        methods_list = [methods_cfg.upper()]
    elif isinstance(methods_cfg, (list, tuple, set)):
        methods_list = [str(method).upper() for method in methods_cfg if str(method)]
    else:
        methods_list = list(_DEFAULT_METHODS)

    modes_cfg = gateway_cfg.get("auth_modes")
    if isinstance(modes_cfg, str):
        configured_modes = [modes_cfg]
    elif isinstance(modes_cfg, (list, tuple, set)):
        configured_modes = [str(mode) for mode in modes_cfg]
    else:
        configured_modes = None
    auth_modes = [
        str(mode).lower()
        for mode in (
            configured_modes
            if configured_modes
            else _SERVICE_AUTH_MODE_OVERRIDES.get(service_name, ["jwt", "service"])
        )
        if str(mode)
    ]

    roles_cfg = gateway_cfg.get("required_roles")
    if isinstance(roles_cfg, str):
        required_roles = [roles_cfg]
    elif isinstance(roles_cfg, (list, tuple, set)):
        required_roles = [str(role) for role in roles_cfg if str(role)]
    else:
        required_roles = _SERVICE_ROLE_REQUIREMENTS.get(service_name, ["admin"])

    return ServiceRouteConfig(
        name=service_name,
        prefix=prefix,
        url=url,
        rate_limit_per_minute=int(limit_env),
        methods=methods_list,
        allowed_auth_modes=auth_modes,
        service_token=service_tokens.get(service_name),
        required_roles=required_roles,
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

<<<<<<< Updated upstream
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

    tenant_service_url = os.getenv("TENANT_SERVICE_URL", "http://tenant-management:8010")
    tenant_service_url = tenant_service_url.rstrip("/") if tenant_service_url else None
    tenant_service_token = (
        os.getenv("TENANT_SERVICE_TOKEN") or resolved_tokens.get("tenant_management")
    )
    tenant_service_token_header = os.getenv("TENANT_SERVICE_TOKEN_HEADER", "x-service-token")
    tenant_service_timeout = float(os.getenv("TENANT_SERVICE_TIMEOUT", "5"))
    tenant_metadata_cache_seconds = int(os.getenv("TENANT_METADATA_CACHE_SECONDS", "120"))
    tenant_default_plan = os.getenv("TENANT_DEFAULT_PLAN", "standard")
    tenant_default_rate_limit_per_minute = int(
        os.getenv("TENANT_DEFAULT_RATE_LIMIT", str(default_rate_limit))
    )
    tenant_default_burst_limit = int(
        os.getenv(
            "TENANT_DEFAULT_BURST_LIMIT",
            str(max(tenant_default_rate_limit_per_minute * 2, default_rate_limit)),
        )
    )
    tenant_default_burst_window_seconds = int(
        os.getenv("TENANT_DEFAULT_BURST_WINDOW", "10")
    )
    kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS") or None
    kafka_usage_topic = os.getenv("TENANT_USAGE_TOPIC", "tenant.usage")

    audit_service_url = os.getenv("AUDIT_SERVICE_URL", "http://audit:8012")
    audit_service_url = audit_service_url.rstrip("/") if audit_service_url else None
    audit_service_token = os.getenv("AUDIT_SERVICE_TOKEN") or resolved_tokens.get("audit")
    audit_service_token_header = os.getenv("AUDIT_SERVICE_TOKEN_HEADER", "x-service-token")
    audit_event_source = os.getenv("AUDIT_EVENT_SOURCE", "api-gateway")
    audit_service_timeout = float(os.getenv("AUDIT_SERVICE_TIMEOUT", "5"))

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
        tenant_service_url=tenant_service_url,
        tenant_service_token=tenant_service_token,
        tenant_service_token_header=tenant_service_token_header,
        tenant_service_timeout=tenant_service_timeout,
        tenant_metadata_cache_seconds=tenant_metadata_cache_seconds,
        tenant_default_plan=tenant_default_plan,
        tenant_default_rate_limit_per_minute=tenant_default_rate_limit_per_minute,
        tenant_default_burst_limit=tenant_default_burst_limit,
        tenant_default_burst_window_seconds=tenant_default_burst_window_seconds,
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        kafka_usage_topic=kafka_usage_topic,
        audit_service_url=audit_service_url,
        audit_service_token=audit_service_token,
        audit_service_token_header=audit_service_token_header,
        audit_event_source=audit_event_source,
        audit_service_timeout=audit_service_timeout,
    )

    environment = os.getenv("ENVIRONMENT", "development").strip().lower()
    allow_insecure_defaults = os.getenv("GATEWAY_ALLOW_INSECURE_DEFAULTS", "0").strip().lower() in {"1", "true", "yes"}

    if not allow_insecure_defaults and environment not in {"development", "dev", "test"}:
        if settings.jwt_secret == "change-me" or not settings.jwt_secret:
            raise RuntimeError(
                "GATEWAY_JWT_SECRET must be explicitly configured for non-development environments"
            )

        insecure_services = [
            name
            for name, route in service_routes.items()
            if (route.service_token or "").startswith("insecure-local-token-")
        ]
        if insecure_services:
            joined = ", ".join(sorted(insecure_services))
            raise RuntimeError(
                "Service tokens for the following services must be configured before deployment: "
                + joined
            )

    return settings
