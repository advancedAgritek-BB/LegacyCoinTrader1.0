"""Start the web dashboard and expose REST endpoints for the trading bot.

This module launches the Flask web server, manages the background trading
process and provides REST API routes used by the UI and tests.
"""

import os
import subprocess
import sys
import warnings
import json
import time
import yaml
import logging
import re
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import quote

from uuid import uuid4

from pydantic.fields import FieldInfo

from asgiref.wsgi import WsgiToAsgi
from flask_cors import CORS

try:
    from limits import parse as parse_rate_limit
    from limits.storage import storage_from_string
    from limits.strategies import FixedWindowRateLimiter
    _LIMITS_AVAILABLE = True
except Exception:  # pragma: no cover - limits is optional in constrained envs
    parse_rate_limit = None  # type: ignore[assignment]
    storage_from_string = None  # type: ignore[assignment]

    class FixedWindowRateLimiter:  # type: ignore[override]
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def hit(self, *_args, **_kwargs) -> bool:
            return True

    _LIMITS_AVAILABLE = False
from werkzeug.exceptions import HTTPException
try:
    from crypto_bot import log_reader
except ImportError:
    # Fallback if log_reader is not available
    log_reader = None
try:
    from crypto_bot.ml_signal_model import *
    ml_signal_model_available = True
except ImportError:
    ml_signal_model_available = False
try:
    import frontend.utils as utils
except ImportError:
    utils = None
try:
    from crypto_bot.utils.single_source_trade_manager import (
        get_single_source_trade_manager,
        create_frontend_subscriber,
        is_test_position
    )
except ImportError:
    is_test_position = None
try:
    from frontend.config import SecuritySettings, get_settings
    from frontend.auth import get_auth, login_required
except ImportError:
    SecuritySettings = None
    get_settings = None
    get_auth = None
    login_required = None
try:
    from crypto_bot.utils.price_fetcher import (
        get_current_price_for_symbol,
    )
except ImportError:
    get_current_price_for_symbol = None
try:
    from services.monitoring.config import get_monitoring_settings
    from services.monitoring.instrumentation import instrument_flask_app
    from services.monitoring.logging_compat import configure_logging
except ImportError:
    get_monitoring_settings = None
    instrument_flask_app = None
    configure_logging = None
try:
    from crypto_bot.config import load_config as load_bot_config, save_config, resolve_config_path
except ImportError:
    load_bot_config = None
    save_config = None
    resolve_config_path = None

try:
    from frontend.gateway import ApiGatewayError, get_gateway_json, post_gateway_json
except ImportError:
    ApiGatewayError = Exception
    get_gateway_json = None
    post_gateway_json = None

def safe_post_gateway_json(path, json=None, timeout=30.0):
    """Safe wrapper for post_gateway_json that handles None function."""
    if post_gateway_json is None:
        raise ApiGatewayError("API gateway not available - frontend.gateway module not found")
    return post_gateway_json(path, json=json, timeout=timeout)

# Global variable to track last reset time
_last_reset_time = None
try:
    from frontend.chart_scaling import compute_chart_coordinates
except ImportError:
    compute_chart_coordinates = None

# Suppress urllib3 OpenSSL warning
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
    category=UserWarning,
)

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None

try:
    from flask import (
        Flask,
        render_template,
        redirect,
        url_for,
        request,
        jsonify,
    )
except (
    Exception
):  # pragma: no cover - provide minimal shim for import-time tests

    class _Dummy:
        def __getattr__(self, _):
            return self

        def __call__(self, *a, **k):
            return None

    def _dummy_flask(*a, **k):
        return _Dummy()

    Flask = _dummy_flask  # type: ignore
    render_template = redirect = url_for = request = jsonify = _Dummy()

# Fix LOG_DIR path to point to the correct crypto_bot/logs directory
LOG_DIR = Path(__file__).resolve().parents[1] / "crypto_bot" / "logs"

app = Flask(__name__)

# Canonical execution mode mapping used across frontend endpoints
_MODE_ALIASES: Dict[str, str] = {
    "dry_run": "dry_run",
    "dry": "dry_run",
    "paper": "dry_run",
    "paper_trading": "dry_run",
    "simulation": "dry_run",
    "sim": "dry_run",
    "demo": "dry_run",
    "test": "dry_run",
    "live": "live",
    "production": "live",
    "real": "live",
}


def _canonicalize_execution_mode(mode: Optional[str]) -> Optional[str]:
    """Return canonical execution mode identifier or ``None`` for invalid values."""

    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if not normalized:
        return None
    alias = _MODE_ALIASES.get(normalized)
    if alias:
        return alias
    if normalized in {"dry_run", "live"}:
        return normalized
    return None


def _normalize_execution_mode(mode: Optional[str]) -> str:
    """Return canonical execution mode, defaulting to ``dry_run``."""

    canonical = _canonicalize_execution_mode(mode)
    if canonical:
        return canonical
    normalized = str(mode or "").strip().lower()
    return normalized or "dry_run"

# Configure CORS to allow requests from any origin (for development)
# In production, you should specify allowed origins explicitly
# Enable CORS for all routes
CORS(app)

# Expose ASGI-compatible wrapper for uvicorn/gunicorn workers
asgi_app = WsgiToAsgi(app)


class APIError(Exception):
    """Base exception for API errors that should return JSON responses."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details


def json_response(payload: Dict[str, Any], status_code: int = 200):
    """Return a JSON response with consistent headers."""

    return app.response_class(
        json.dumps(payload, default=str),
        status=status_code,
        mimetype="application/json",
    )


def json_error(
    message: str,
    *,
    status_code: int = 400,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """Return a standardized JSON error response."""

    payload: Dict[str, Any] = {
        "success": False,
        "error": {"message": message},
    }
    if error_code:
        payload["error"]["code"] = error_code
    if details:
        payload["error"]["details"] = details
    return json_response(payload, status_code)


def wants_json_response() -> bool:
    """Return True if the current request prefers a JSON response."""

    from flask import request

    if request.path.startswith("/api/"):
        return True
    accept = request.headers.get("Accept", "")
    return "application/json" in accept.lower()


def _standardize_error_response(response):
    """Ensure JSON error payloads follow the standard shape."""

    if (
        getattr(response, "mimetype", "")
        and response.mimetype.startswith("application/json")
        and response.status_code >= 400
    ):
        try:
            payload = json.loads(response.get_data(as_text=True) or "null")
        except (TypeError, ValueError):
            return response

        if (
            isinstance(payload, dict)
            and payload.get("success") is False
            and isinstance(payload.get("error"), dict)
            and payload["error"].get("message")
        ):
            return response

        message = None
        error_code = None
        details: Optional[Dict[str, Any]] = None

        if isinstance(payload, dict):
            if isinstance(payload.get("error"), dict) and payload["error"].get("message"):
                # Already partially standardised, include any missing fields
                message = payload["error"].get("message")
                error_code = payload["error"].get("code")
                remaining = {
                    key: value
                    for key, value in payload["error"].items()
                    if key not in {"message", "code"}
                }
                details = remaining or None
            else:
                message = payload.get("error") or payload.get("message")
                error_code = payload.get("code")
                remaining = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"error", "message", "code"}
                }
                details = remaining or None
        else:
            details = {"data": payload}

        standard_payload = {
            "success": False,
            "error": {"message": str(message or "Request failed")},
        }
        if error_code:
            standard_payload["error"]["code"] = error_code
        if details:
            standard_payload["error"]["details"] = details

        response.set_data(json.dumps(standard_payload, default=str))
        response.mimetype = "application/json"

    return response


# Lightweight healthcheck for container orchestration
@app.route("/health", methods=["GET"])  # Simple 200 OK health endpoint
def health():
    return json_response({"status": "ok"})

# Import secure configuration and authentication, login_required

# Get configuration and authentication instances
if get_settings:
    settings = get_settings()
    if isinstance(settings.security, FieldInfo):
        settings.security = SecuritySettings()
    elif not isinstance(settings.security, SecuritySettings):
        settings.security = SecuritySettings.model_validate(settings.security)
    config = settings  # Backwards compatibility for older references expecting ``config``
    auth = get_auth()
else:
    settings = None
    config = None
    auth = None

if get_monitoring_settings:
    monitoring_settings = get_monitoring_settings().for_service(
        "frontend",
        environment=settings.environment if settings else "development",
    )
else:
    monitoring_settings = None
if monitoring_settings:
    monitoring_settings.metrics.default_labels.setdefault("component", "frontend")
    if configure_logging:
        configure_logging(monitoring_settings)
    try:
        if instrument_flask_app:
            instrument_flask_app(app, settings=monitoring_settings)
    except RuntimeError as exc:  # pragma: no cover - instrumentation optional in some tests
        logging.getLogger(__name__).warning("Observability instrumentation disabled: %s", exc)

logger = logging.getLogger(__name__)

if not _LIMITS_AVAILABLE:  # pragma: no cover - executed only without limits
    logger.warning(
        "python-limits is not installed; API rate limiting is disabled in this environment"
    )

# Disable template caching for development - CRITICAL for template updates
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Set secure session configuration
if settings and settings.security:
    app.config["SECRET_KEY"] = settings.security.session_secret_key
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_PERMANENT"] = True
    app.config["PERMANENT_SESSION_LIFETIME"] = settings.security.session_timeout
    app.config["SESSION_COOKIE_SECURE"] = settings.environment == "production"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
else:
    # Fallback configuration
    app.config["SECRET_KEY"] = "fallback-secret-key"
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_PERMANENT"] = True

# Rate limiting (Redis-backed implementation)
_rate_limiter_enabled = (
    bool(getattr(settings.security, "rate_limit_enabled", True) if settings and settings.security else True)
    and _LIMITS_AVAILABLE
)
_rate_limiter: Optional[FixedWindowRateLimiter] = None
_rate_limit_item = None


def _get_rate_limit_identifier() -> str:
    """Return a unique identifier for the caller and route."""

    from flask import request

    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.remote_addr or "unknown"

    endpoint = request.endpoint or request.path
    prefix = getattr(settings.security, "rate_limit_prefix", "frontend") if settings and settings.security else "frontend"
    return f"{prefix}:{client_ip}:{endpoint}"


def is_rate_limited() -> bool:
    """Return True when the caller exceeded the configured rate limit."""

    global _rate_limiter

    if not _rate_limiter_enabled or _rate_limiter is None or _rate_limit_item is None:
        return False

    identifier = _get_rate_limit_identifier()
    try:
        allowed = _rate_limiter.hit(_rate_limit_item, identifier)
    except Exception as exc:  # pragma: no cover - backend failures are runtime specific
        logger.error("Rate limiter backend failure for %s: %s", identifier, exc)
        return False

    return not allowed


@app.before_request
def check_rate_limit():
    """Check rate limit before processing request."""

    if not _rate_limiter_enabled:
        return None

    if is_rate_limited():
        retry_after = None
        if _rate_limit_item is not None:
            try:
                retry_after = int(_rate_limit_item.get_expiry())
            except Exception:  # pragma: no cover - defensive
                retry_after = None

        response = json_error(
            "Rate limit exceeded",
            status_code=429,
            error_code="rate_limit_exceeded",
            details={"retry_after": retry_after} if retry_after else None,
        )
        if retry_after:
            response.headers["Retry-After"] = str(retry_after)
        return response

    return None


if settings and settings.security:
    _rate_limit_storage_url = (
        settings.security.rate_limit_storage_url
        or f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
    )
else:
    _rate_limit_storage_url = "memory://"

if _rate_limiter_enabled:
    try:
        storage = storage_from_string(_rate_limit_storage_url)
        _rate_limiter = FixedWindowRateLimiter(storage)
        if settings and settings.security:
            limit_expression = (
                f"{settings.security.rate_limit_requests}/{settings.security.rate_limit_window} seconds"
            )
        else:
            limit_expression = "100/minute"
        _rate_limit_item = parse_rate_limit(limit_expression)
        logger.debug(
            "Rate limiter configured with %s using backend %s",
            limit_expression,
            _rate_limit_storage_url,
        )
    except Exception as exc:  # pragma: no cover - backend availability depends on runtime
        logger.error(
            "Failed to configure Redis rate limiter using %s: %s",
            _rate_limit_storage_url,
            exc,
        )
        _rate_limiter_enabled = False


@app.errorhandler(APIError)
def handle_api_error(exc: APIError):
    """Return a standardized response for APIError exceptions."""

    return json_error(
        exc.message,
        status_code=exc.status_code,
        error_code=exc.error_code,
        details=exc.details,
    )


if ApiGatewayError:
    @app.errorhandler(ApiGatewayError)
    def handle_api_gateway_error(exc: ApiGatewayError):
        """Convert API gateway failures into standardized JSON errors."""

        return json_error(str(exc), status_code=502, error_code="gateway_error")


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    """Provide JSON responses for HTTP errors when appropriate."""

    if wants_json_response():
        description = exc.description or exc.name or "HTTP error"
        error_code = (exc.name or "http_error").lower().replace(" ", "_")
        return json_error(description, status_code=exc.code or 500, error_code=error_code)

    return exc


@app.errorhandler(Exception)
def handle_unexpected_exception(exc: Exception):
    """Catch-all exception handler that returns a JSON response when required."""

    if isinstance(exc, APIError):
        return handle_api_error(exc)
    if ApiGatewayError and isinstance(exc, ApiGatewayError):
        return handle_api_gateway_error(exc)
    if isinstance(exc, HTTPException):
        return handle_http_exception(exc)

    logger.exception("Unhandled exception while processing request", exc_info=exc)

    if wants_json_response():
        return json_error(
            "An internal server error occurred.",
            status_code=500,
            error_code="internal_server_error",
        )

    raise exc


PORTFOLIO_POSITIONS_PATH = "/api/v1/portfolio/state"
PORTFOLIO_STATE_PATH = "/api/v1/portfolio/state"
PORTFOLIO_PNL_PATH = "/api/v1/portfolio/pnl"
TRADING_ENGINE_STATE_PATH = "/trading-engine/cycles/status"
TRADING_ENGINE_START_PATH = "/trading-engine/cycles/start"
TRADING_ENGINE_STOP_PATH = "/trading-engine/cycles/stop"
TRADING_ENGINE_RUN_ONCE_PATH = "/trading-engine/cycles/run"
TRADING_ENGINE_RELOAD_CONFIG_PATH = "/trading-engine/config/reload"
MONITORING_METRICS_PATH = "/monitoring/metrics"
STRATEGY_PERFORMANCE_PATH = "/monitoring/strategy/performance"
STRATEGY_SCORES_PATH = "/monitoring/strategy/scores"

# Treat anything below this absolute quantity as dust to avoid ghost positions
POSITION_DUST_THRESHOLD = 1e-8


def _fallback_portfolio_state_from_trade_manager() -> Dict[str, Any]:
    """Construct a minimal portfolio state from the SingleSourceTradeManager."""

    try:
        trade_manager = get_single_source_trade_manager()
        positions_payload: List[Dict[str, Any]] = []
        for tm_position in trade_manager.get_all_positions():
            if not getattr(tm_position, "is_open", True):
                continue
            total_amount = float(getattr(tm_position, "total_amount", 0) or 0)
            if abs(total_amount) <= POSITION_DUST_THRESHOLD:
                continue

            entry_price = float(getattr(tm_position, "average_price", 0) or 0)
            side_raw = str(getattr(tm_position, "side", "")).strip().lower()
            side = "short" if side_raw in {"short", "sell"} else "long"

            entry_time = getattr(tm_position, "entry_time", None)
            last_update = getattr(tm_position, "last_update", None)

            positions_payload.append(
                {
                    "symbol": getattr(tm_position, "symbol", "").upper(),
                    "side": side,
                    "total_amount": total_amount,
                    "amount": total_amount,
                    "average_price": entry_price,
                    "entry_price": entry_price,
                    "entry_time": entry_time.isoformat() if hasattr(entry_time, "isoformat") else None,
                    "last_update": last_update.isoformat() if hasattr(last_update, "isoformat") else None,
                    "trades": [
                        {
                            "side": side,
                            "amount": total_amount,
                            "price": entry_price,
                        }
                    ],
                    "is_open": True,
                }
            )

        price_cache_entries: List[Dict[str, Any]] = []
        for symbol, price in (getattr(trade_manager, "price_cache", {}) or {}).items():
            try:
                float_price = float(price)
            except (TypeError, ValueError):
                continue
            price_cache_entries.append(
                {
                    "symbol": str(symbol).upper(),
                    "price": float_price,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        return {
            "positions": positions_payload,
            "price_cache": price_cache_entries,
            "closed_positions": [],
        }

    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to build fallback portfolio state: %s", exc)
        return {"positions": [], "price_cache": [], "closed_positions": []}


def fetch_positions_from_service(*, skip_enrichment: bool = False) -> list[dict[str, Any]]:
    """Retrieve open positions from the portfolio service via the API gateway.

    When ``skip_enrichment`` is true we return the normalised portfolio payload
    without attempting to backfill missing market prices. This avoids the
    additional market-data round-trips that are useful for dashboard refreshes
    but unnecessary (and sometimes slow) for server-side tasks that only need
    the basic position envelope such as manual order submission.

    If a reset was recently performed (within last 60 seconds), prefer local TradeManager state.
    """

    # Check if TradeManager state file indicates a reset (empty positions)
    # Try multiple possible paths since Flask might be running in different context
    possible_paths = [
        Path("crypto_bot/logs/trade_manager_state.json"),
        Path("/app/crypto_bot/logs/trade_manager_state.json"),
        Path("../crypto_bot/logs/trade_manager_state.json"),
        Path("../../crypto_bot/logs/trade_manager_state.json")
    ]

    reset_detected = False
    trade_manager_file = None

    for path in possible_paths:
        if path.exists():
            trade_manager_file = path
            logger.info(f"Found TradeManager state file at: {path}")
            break

    if trade_manager_file:
        try:
            with open(trade_manager_file, 'r') as f:
                tm_state = json.load(f)
                positions = tm_state.get("positions", {})
                logger.info(f"TradeManager state positions: {len(positions) if positions else 0}")
                if not positions or len(positions) == 0:
                    logger.info("TradeManager state file shows empty positions (reset detected)")
                    reset_detected = True
        except Exception as e:
            logger.warning(f"Error reading TradeManager state file {trade_manager_file}: {e}")
    else:
        logger.warning("TradeManager state file not found in any expected location")

    if reset_detected:
        # Return empty positions directly
        logger.info("Reset detected from state file, returning empty positions")
        payload = {"positions": [], "price_cache": [], "closed_positions": []}
    else:
        # Try portfolio service normally
        logger.info("No reset detected, trying portfolio service")
        try:
            # Use a shorter timeout for position fetching
            payload = get_gateway_json(PORTFOLIO_POSITIONS_PATH, timeout=5.0)
            if not isinstance(payload, dict):
                raise ApiGatewayError(
                    "Portfolio service returned an unexpected payload for /api/v1/portfolio/state"
                )
        except ApiGatewayError as exc:
            logger.warning(
                "Portfolio service unavailable (%s); falling back to TradeManager state",
                exc,
            )
            payload = _fallback_portfolio_state_from_trade_manager()  # Use local state
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error(
                "Unexpected error retrieving portfolio positions: %s", exc,
            )
            payload = _fallback_portfolio_state_from_trade_manager()  # Use local state

    # Extract positions from the state payload
    positions = payload.get("positions", [])
    if not isinstance(positions, list):
        raise ApiGatewayError("Portfolio state positions is not a list")

    # Build a quick lookup for the latest cached prices provided by the portfolio service
    price_cache_lookup: dict[str, float] = {}
    for cache_entry in payload.get("price_cache", []) or []:
        if not isinstance(cache_entry, dict):
            continue
        cache_symbol_raw = cache_entry.get("symbol")
        if not cache_symbol_raw:
            continue
        cache_symbol = str(cache_symbol_raw).strip().upper()
        cache_price = _coerce_optional_float(cache_entry.get("price"))
        if cache_symbol and cache_price:
            price_cache_lookup[cache_symbol] = cache_price

    def _apply_price_update(position: Dict[str, Any], latest_price: float, source: str) -> bool:
        try:
            entry_price = float(position.get("entry_price", 0) or 0.0)
            total_amount = float(position.get("size", 0) or 0.0)
            if entry_price <= 0 or total_amount <= 0 or latest_price <= 0:
                return False

            if position.get("side") == "long":
                pnl_value = (latest_price - entry_price) * total_amount
            else:
                pnl_value = (entry_price - latest_price) * total_amount

            pnl_pct = (
                (pnl_value / (entry_price * total_amount)) * 100
                if entry_price and total_amount
                else 0.0
            )

            position["pnl_value"] = pnl_value
            position["pnl"] = pnl_pct
            position["pnl_percentage"] = pnl_pct
            position["current_price"] = latest_price
            position["current_value"] = latest_price * total_amount
            price_delta = latest_price - entry_price
            price_moved_up = price_delta >= 0
            position["price_delta"] = price_delta
            position["trend_direction"] = "UPWARD" if price_moved_up else "DOWNWARD"
            position["trend_is_favorable"] = (
                price_moved_up if position.get("side") == "long" else not price_moved_up
            )
            position["price_source"] = source
            position["price_unavailable"] = False
            return True
        except Exception as exc:
            logger.debug(
                "Failed to apply price update for %s via %s: %s",
                position.get("symbol"),
                source,
                exc,
            )
            return False

    results: list[dict[str, Any]] = []
    def _resolve_strategy_label(
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
        trades: List[Any],
    ) -> Tuple[str, Optional[str]]:
        """Derive a human-friendly strategy label with fallbacks."""

        candidates: List[str] = []

        def _add_candidate(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, dict)):
                return
            text = str(value).strip()
            if text:
                candidates.append(text)

        def _add_nested_candidates(container: Dict[str, Any], keys: Tuple[str, ...]) -> None:
            for key in keys:
                nested_value = container.get(key)
                if isinstance(nested_value, dict):
                    _add_nested_candidates(
                        nested_value,
                        (
                            "name",
                            "label",
                            "display_name",
                            "strategy",
                            "strategy_name",
                            "selected_strategy",
                        ),
                    )
                else:
                    _add_candidate(nested_value)

        def _format_label(raw: str) -> str:
            cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)
            cleaned = re.sub(r"[_\-]+", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                return raw
            return cleaned.title()

        placeholder_terms = {
            "dry_run",
            "dryrun",
            "dry-run",
            "paper",
            "paper_trading",
            "paper-trading",
            "papertrading",
            "manual",
            "manual_market_sell",
            "manual-sell",
            "simulation",
            "simulated",
            "sim",
            "demo",
            "test",
            "unknown",
        }

        def _is_placeholder(text: str) -> bool:
            lowered = text.lower().strip()
            normalized = lowered.replace("-", "_")
            if normalized in placeholder_terms:
                return True
            if "dry" in lowered and "run" in lowered:
                return True
            if "paper" in lowered and "trade" in lowered:
                return True
            if lowered.startswith("manual"):
                return True
            return False

        # Collect potential identifiers in priority order
        _add_candidate(payload.get("strategy"))

        metadata_keys = (
            "strategy",
            "strategy_name",
            "strategyLabel",
            "strategy_label",
            "active_strategy",
            "primary_strategy",
            "selected_strategy",
            "applied_strategy",
            "strategy_code",
            "strategy_id",
            "strategy_key",
            "strategy_display",
        )
        for key in metadata_keys:
            value = metadata.get(key)
            if isinstance(value, dict):
                _add_nested_candidates(value, metadata_keys)
            else:
                _add_candidate(value)

        for nested_key in (
            "strategy_details",
            "strategy_context",
            "strategy_metadata",
            "routing",
            "trade_metadata",
            "decision",
        ):
            nested = metadata.get(nested_key)
            if isinstance(nested, dict):
                _add_nested_candidates(nested, metadata_keys)

        nested_strategy = metadata.get("strategy")
        if isinstance(nested_strategy, dict):
            _add_nested_candidates(nested_strategy, metadata_keys)

        for trade in reversed(trades):
            if not isinstance(trade, dict):
                continue
            _add_candidate(trade.get("strategy"))
            trade_metadata = trade.get("metadata")
            if isinstance(trade_metadata, dict):
                _add_nested_candidates(trade_metadata, metadata_keys)

        non_placeholder: List[str] = []
        placeholder: List[str] = []

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate:
                continue
            key = candidate.strip()
            if key in seen:
                continue
            seen.add(key)
            if _is_placeholder(key):
                placeholder.append(key)
            else:
                non_placeholder.append(key)

        if non_placeholder:
            best_raw = non_placeholder[0]
            return _format_label(best_raw), best_raw

        if placeholder:
            raw_placeholder = placeholder[0]
            lowered = raw_placeholder.lower()
            if "manual" in lowered:
                return "Manual Action", raw_placeholder
            if "paper" in lowered or "dry" in lowered:
                return "Paper Trading", raw_placeholder
            if "sim" in lowered:
                return "Simulation", raw_placeholder
            return _format_label(raw_placeholder), raw_placeholder

        return "Unknown", None

    for item in positions:
        try:
            symbol_raw = (
                item.get("symbol")
                or item.get("pair")
                or item.get("asset")
                or ""
            )
            symbol = str(symbol_raw).strip().upper()
            if not symbol or is_test_position(symbol):
                continue

            trades = item.get("trades")

            # Skip entries explicitly marked as closed
            is_open_flag = item.get("is_open")
            if isinstance(is_open_flag, str):
                is_open_flag = is_open_flag.strip().lower() not in {"false", "0", "no"}
            if is_open_flag is not None and not bool(is_open_flag):
                logger.debug("Skipping position %s flagged as closed", symbol)
                continue

            raw_amount = (
                item.get("total_amount")
                if item.get("total_amount") is not None
                else item.get("amount")
            )
            total_amount = _coerce_optional_float(raw_amount) or 0.0

            # Skip positions with zero or negative amounts
            if abs(total_amount) <= POSITION_DUST_THRESHOLD:
                logger.debug(
                    "Skipping position %s with negligible amount: %s",
                    symbol,
                    total_amount,
                )
                continue

            entry_price = (
                _coerce_optional_float(item.get("average_price"))
                or _coerce_optional_float(item.get("entry_price"))
                or 0.0
            )

            # Get current market price from portfolio data or cached prices
            current_price_raw = (
                _coerce_optional_float(item.get("current_price"))
                or _coerce_optional_float(item.get("mark_price"))
                or price_cache_lookup.get(symbol)
            )

            if current_price_raw and current_price_raw > 0:
                current_price = current_price_raw
                price_source = "portfolio"
            else:
                current_price = entry_price
                price_source = "entry_fallback"

            # Store symbol for batch price fetching if no current price available
            side_raw = (
                str(item.get("side") or item.get("position_side") or "")
                .strip()
                .lower()
            )
            side = "short" if side_raw in {"short", "sell"} else "long"

            if not trades:
                trades = [
                    {
                        "side": side,
                        "amount": total_amount,
                        "price": entry_price,
                    }
                ]

            # Skip if we don't have valid prices
            if entry_price <= 0 or current_price <= 0:
                logger.debug(f"Skipping position {symbol} with invalid prices: entry={entry_price}, current={current_price}")
                continue

            amount_abs = abs(total_amount)

            if side == "long":
                pnl_value = (current_price - entry_price) * amount_abs
            else:
                pnl_value = (entry_price - current_price) * amount_abs

            position_value = current_price * amount_abs
            pnl_pct = (
                (pnl_value / (entry_price * amount_abs)) * 100
                if entry_price and amount_abs
                else 0.0
            )

            stop_loss_price = _coerce_optional_float(item.get("stop_loss_price"))
            price_delta = current_price - entry_price
            price_moved_up = price_delta >= 0
            trend_direction = "UPWARD" if price_moved_up else "DOWNWARD"
            trend_is_favorable = (
                price_moved_up if side == "long" else not price_moved_up
            )
            chart_coordinates = compute_chart_coordinates(
                entry_price,
                current_price,
                stop_loss_price=stop_loss_price,
                include_current_price=True,
            )


            metadata_raw = item.get("metadata")
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            strategy_label, strategy_code = _resolve_strategy_label(item, metadata, trades or [])

            logger.debug(f"Adding position {symbol}: amount={total_amount}, pnl={pnl_pct:.2f}%")

            results.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "size": amount_abs,
                    "amount": amount_abs,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "current_value": position_value,
                    "pnl": pnl_pct,
                    "pnl_value": pnl_value,
                    "pnl_percentage": pnl_pct,
                    "chart_min": chart_coordinates.min_price,
                    "chart_max": chart_coordinates.max_price,
                    "trend_strength": "strong" if abs(pnl_pct) > 2 else "moderate" if abs(pnl_pct) > 1 else "weak",
                    "r_squared": None,  # TODO: Calculate real R² from actual price data when available
                    "highest_price": item.get("highest_price"),
                    "lowest_price": item.get("lowest_price"),
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": item.get("take_profit_price"),
                    "entry_time": item.get("entry_time"),
                    "price_delta": price_delta,
                    "trend_direction": trend_direction,
                    "trend_is_favorable": trend_is_favorable,
                    "strategy": strategy_label,
                    "strategy_label": strategy_label,
                    "strategy_code": strategy_code,
                    "price_source": price_source,
                    "price_unavailable": price_source == "entry_fallback",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to normalise position %s: %s", item, exc)
            continue

    if skip_enrichment:
        return results

    # Batch fetch current prices for positions that don't have them
    positions_needing_prices = [pos for pos in results if pos.get("price_unavailable")]
    symbols_to_fetch = [pos["symbol"] for pos in positions_needing_prices]

    updated_symbols: set[str] = set()

    if symbols_to_fetch:
        try:
            logger.info(
                "Batch fetching current prices for %d symbols: %s",
                len(symbols_to_fetch),
                symbols_to_fetch,
            )
            batch_payload = {
                "symbols": symbols_to_fetch,
                "limit": 1,  # Only need latest price
                "timeframe": "5m",
                "exchange": "kraken",
            }
            batch_response = safe_post_gateway_json("/api/v1/market-data/batch-candles", json=batch_payload)

            if batch_response and batch_response.get("results"):
                updated_count = 0
                for symbol, data in batch_response["results"].items():
                    if data and "error" not in data and data.get("candles") and len(data["candles"]) > 0:
                        latest_price = float(data["candles"][-1][4])
                        logger.debug("Fetched price for %s: $%.4f", symbol, latest_price)

                        for pos in positions_needing_prices:
                            if pos["symbol"] == symbol and _apply_price_update(pos, latest_price, "market"):
                                updated_symbols.add(symbol)
                                updated_count += 1
                                logger.info(
                                    "Updated %s via market-data service -> $%.4f (P&L %.2f%%)",
                                    symbol,
                                    latest_price,
                                    pos["pnl"],
                                )
                                break

                logger.info(
                    "Successfully updated prices for %d/%d symbols",
                    len(updated_symbols),
                    len(symbols_to_fetch),
                )
            else:
                logger.warning("Batch price fetch returned no results")
        except Exception as exc:
            logger.warning("Failed to batch fetch current prices: %s", exc)

    # Fallback: direct price fetch for any symbols still missing prices
    remaining_for_direct = [
        pos for pos in positions_needing_prices if pos["symbol"] not in updated_symbols
    ]
    if remaining_for_direct:
        logger.info(
            "Applying direct price fallback for %d symbols", len(remaining_for_direct)
        )
        for pos in remaining_for_direct:
            symbol = pos["symbol"]
            try:
                direct_price = float(get_current_price_for_symbol(symbol) or 0)
                if _apply_price_update(pos, direct_price, "direct"):
                    logger.info(
                        "Direct price fetch succeeded for %s -> $%.4f (P&L %.2f%%)",
                        symbol,
                        direct_price,
                        pos["pnl"],
                    )
                else:
                    logger.warning(
                        "Direct price fetch did not return a usable price for %s", symbol
                    )
            except Exception as direct_exc:
                logger.warning("Direct price fetch failed for %s: %s", symbol, direct_exc)

    return results


def _find_open_position(symbol: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Locate an open position for ``symbol`` using live services before falling back."""

    normalized = (symbol or "").strip().upper()
    if not normalized:
        return None, ""

    try:
        positions = fetch_positions_from_service(skip_enrichment=True)
        for payload in positions:
            pos_symbol = (payload.get("symbol") or "").strip().upper()
            if pos_symbol == normalized:
                return payload, "portfolio_service"
    except ApiGatewayError as exc:
        logger.warning("Portfolio service lookup failed for %s: %s", normalized, exc)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(
            "Unexpected portfolio lookup failure for %s: %s",
            normalized,
            exc,
        )

    try:
        from crypto_bot.utils.trade_manager import get_trade_manager

        trade_manager = get_trade_manager()
        price_cache = getattr(trade_manager, "price_cache", {}) or {}
        normalized_cache: Dict[str, float] = {}
        for cache_symbol, cache_price in price_cache.items():
            try:
                value = float(cache_price)
            except (TypeError, ValueError):
                continue
            normalized_cache[str(cache_symbol).strip().upper()] = value

        for tm_position in trade_manager.get_all_positions():
            if not getattr(tm_position, "is_open", True):
                continue
            pos_symbol = str(getattr(tm_position, "symbol", "")).strip().upper()
            if pos_symbol != normalized:
                continue

            pos_dict = tm_position.to_dict()
            pos_dict.setdefault("symbol", pos_symbol)
            pos_dict.setdefault("entry_price", pos_dict.get("average_price"))
            pos_dict.setdefault("amount", pos_dict.get("total_amount"))
            pos_dict.setdefault("size", pos_dict.get("total_amount"))

            cached_price = normalized_cache.get(pos_symbol)
            if cached_price and cached_price > 0:
                pos_dict.setdefault("current_price", cached_price)
                pos_dict.setdefault("mark_price", cached_price)
                pos_dict.setdefault("price_source", "cache")
            return pos_dict, "trade_manager"
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.error("TradeManager lookup failed for %s: %s", normalized, exc)

    return None, ""


def _resolve_close_price(symbol: str, position: Dict[str, Any]) -> float:
    """Determine a usable close price for ``symbol`` using cached data and live fallbacks."""

    candidates: List[float] = []
    for candidate in (
        position.get("current_price"),
        position.get("mark_price"),
        position.get("last_price"),
        position.get("average_price"),
        position.get("entry_price"),
    ):
        value = _coerce_optional_float(candidate)
        if value and value > 0:
            candidates.append(value)

    if candidates:
        return candidates[0]

    live_price = _coerce_optional_float(get_current_price_for_symbol(symbol))
    if live_price and live_price > 0:
        return live_price

    try:
        params = {
            "symbol": symbol,
            "limit": 1,
            "timeframe": "1m",
            "exchange": "kraken",
        }
        candle_response = get_gateway_json(
            "/market-data/get-candles", params=params, timeout=6.0
        )
        if isinstance(candle_response, dict):
            candles = candle_response.get("candles") or []
            if candles:
                latest = candles[-1]
                try:
                    close_price = float(latest[4])
                    if close_price > 0:
                        return close_price
                except (TypeError, ValueError, IndexError):
                    logger.debug(
                        "Malformed candle data when resolving price for %s: %s",
                        symbol,
                        latest,
                    )
    except ApiGatewayError as exc:
        logger.warning(
            "Market data service unavailable while resolving price for %s: %s",
            symbol,
            exc,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(
            "Unexpected error fetching market price for %s: %s",
            symbol,
            exc,
        )

    fallback_entry = _coerce_optional_float(position.get("entry_price"))
    if fallback_entry and fallback_entry > 0:
        return fallback_entry

    return 0.0


def fetch_portfolio_state() -> Dict[str, Any]:
    """Retrieve the full portfolio state from the portfolio service."""

    payload = get_gateway_json(PORTFOLIO_STATE_PATH)
    if not isinstance(payload, dict):
        raise ApiGatewayError("Portfolio service returned an unexpected payload for /state")
    return payload


def fetch_portfolio_pnl() -> Dict[str, Decimal]:
    """Return the current realised/unrealised PnL figures from the portfolio service."""

    try:
        payload = get_gateway_json(PORTFOLIO_PNL_PATH)
        if not isinstance(payload, dict):
            raise ApiGatewayError(
                "Portfolio service returned an unexpected payload for /pnl"
            )
    except ApiGatewayError as exc:
        logger.warning(
            "Portfolio PnL unavailable via gateway: %s", exc
        )
        return {
            "total": Decimal("0"),
            "realized": Decimal("0"),
            "unrealized": Decimal("0"),
        }

    def _to_decimal(value: Any) -> Decimal:
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")

    return {
        "total": _to_decimal(payload.get("total")),
        "realized": _to_decimal(payload.get("realized")),
        "unrealized": _to_decimal(payload.get("unrealized")),
    }

# Secure headers middleware
@app.after_request
def add_secure_headers(response):
    """Add secure headers to all responses."""
    from flask import request

    response = _standardize_error_response(response)

    # Get the origin from the request
    origin = request.headers.get("Origin")

    # Generate secure CSP header
    # For development, use a more permissive CSP
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "connect-src * http: https: wss:; "
        "font-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )

    # Generate secure CORS headers
    if origin and settings and settings.security:
        cors_headers = settings.security.get_cors_headers(origin)
        response.headers.update(cors_headers)

    # Additional security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # HSTS header for production
    if settings and settings.environment == "production":
        hsts_value = "max-age=31536000; includeSubDomains"
        response.headers["Strict-Transport-Security"] = hsts_value
    else:
        response.headers["Strict-Transport-Security"] = ""

    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Cache control for security
    if settings.environment == "development":
        response.headers["Cache-Control"] = (
            "no-cache, no-store, must-revalidate"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    else:
        # 5 minutes for production
        response.headers["Cache-Control"] = "public, max-age=300"

    return response


# Authentication Routes
@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    from flask import request, session, flash, redirect, url_for

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = auth.authenticate(username, password)
        if user:
            session["user"] = {
                "username": user["username"],
                "roles": user.get("roles", []),
            }
            session["login_time"] = user["login_time"]
            session["access_token"] = user.get("access_token")
            session["token_expires_at"] = user.get("token_expires_at")
            session["password_expires_at"] = user.get("password_expires_at")
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Handle user logout."""
    from flask import session, flash, redirect, url_for

    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    """API endpoint for login."""
    from flask import request, session, jsonify

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    user = auth.authenticate(username, password)
    if user:
        session["user"] = {
            "username": user["username"],
            "roles": user.get("roles", []),
        }
        session["login_time"] = user["login_time"]
        session["access_token"] = user.get("access_token")
        session["token_expires_at"] = user.get("token_expires_at")
        session["password_expires_at"] = user.get("password_expires_at")
        return jsonify(
            {
                "message": "Login successful",
                "user": {
                    "username": user["username"],
                    "roles": user.get("roles", []),
                },
            }
        )
    else:
        return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    """API endpoint for logout."""
    from flask import session, jsonify

    session.clear()
    return jsonify({"message": "Logged out successfully"})


@app.route("/api/auth/status")
def auth_status():
    """Get current authentication status."""
    from flask import session, jsonify

    if "user" in session:
        user = session["user"]
        return jsonify(
            {
                "authenticated": True,
                "user": {
                    "username": user.get("username"),
                    "roles": user.get("roles", []),
                },
            }
        )
    else:
        return jsonify({"authenticated": False})


@app.route("/api/bot-status")
def api_bot_status():
    """Return aggregated bot status information for dashboard polling."""

    timestamp = datetime.now(timezone.utc).isoformat()
    bot_running = False
    process_details = []
    process_error = None

    try:  # psutil is optional in some environments
        import psutil  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        psutil = None  # type: ignore
        process_error = "psutil unavailable"

    if psutil is not None:
        try:
            patterns = (
                "start_bot.py",
                "crypto_bot.main",
                "crypto_bot/main.py",
                "services.trading_engine.app",
                "uvicorn services.trading_engine.app",
                "uvicorn",
            )
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):  # type: ignore[attr-defined]
                cmdline = proc.info.get("cmdline") or []
                text = " ".join(map(str, cmdline)).lower()
                if any(pattern in text for pattern in patterns):
                    bot_running = True
                    process_details.append(
                        {
                            "pid": proc.info.get("pid"),
                            "name": proc.info.get("name"),
                            "cmdline": cmdline,
                        }
                    )
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.debug("Bot process probe failed: %s", exc)
            process_error = str(exc)

    source = "process"
    message = "Local process probe did not detect an active trading engine"
    state: Dict[str, Any] = {}
    mode = load_execution_mode() if load_execution_mode else "dry_run"
    next_run_at = None
    last_completed_at = None

    try:
        state = get_trading_engine_state(force_refresh=True) or {}
        if state:
            bot_running = bool(state.get("running"))
            metadata = state.get("metadata") or {}
            mode = metadata.get("mode") or mode
            next_run_at = state.get("next_run_at")
            last_completed_at = state.get("last_run_completed_at")
            source = "trading-engine"
            message = "Trading engine status retrieved via gateway"
    except ApiGatewayError as exc:
        logger.warning("Trading engine status via gateway unavailable: %s", exc)
        message = f"Gateway error: {exc}"
        source = "process-fallback" if bot_running else "unavailable"
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.exception("Unexpected error fetching bot status")
        message = f"Unexpected error: {exc}"
        source = "error"

    last_log_timestamp = None
    trading_log = LOG_DIR / "trading_engine.log"
    if trading_log.exists():
        try:
            last_log_timestamp = datetime.fromtimestamp(
                trading_log.stat().st_mtime, timezone.utc
            ).isoformat()
        except Exception:  # pragma: no cover - filesystem is best effort
            last_log_timestamp = None

    payload = {
        "bot_running": bot_running,
        "source": source,
        "message": message,
        "mode": mode,
        "state": state,
        "next_run_at": next_run_at,
        "last_run_completed_at": last_completed_at,
        "uptime": get_uptime(),
        "process_probe": {
            "processes": process_details,
            "error": process_error,
        },
        "last_log_timestamp": last_log_timestamp,
    }

    return jsonify({
        "success": True,
        "data": payload,
        "timestamp": timestamp,
    })


# Add monitoring API routes first to avoid conflicts with general CORS handler
@app.route("/api/monitoring/health", methods=["GET"])
def api_monitoring_health():
    """Return comprehensive system health status."""
    try:
        import json
        from pathlib import Path

        # Try to read the actual monitoring data first
        frontend_status_file = LOG_DIR / "frontend_monitoring_status.json"
        if frontend_status_file.exists():
            try:
                with open(frontend_status_file, "r") as f:
                    monitoring_data = json.load(f)

                # Add current timestamp and return
                monitoring_data["last_update"] = datetime.now().isoformat()

                return jsonify(
                    {
                        "success": True,
                        "data": monitoring_data,
                        "timestamp": int(time.time() * 1000),
                    }
                )
            except Exception as e:
                print(f"Error reading frontend monitoring status: {e}")

        # Fallback: Get system metrics if monitoring data not available
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Check if bot is running
        bot_running = False
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and len(cmdline) > 0:
                    cmd_str = " ".join(cmdline).lower()
                    if any(
                        pattern in cmd_str
                        for pattern in [
                            "start_bot.py",
                            "crypto_bot.main",
                            "crypto_bot/main.py",
                        ]
                    ):
                        bot_running = True
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Determine overall system status
        overall_status = "healthy"
        if cpu_percent > 80 or memory.percent > 80:
            overall_status = "warning"
        if cpu_percent > 95 or memory.percent > 95:
            overall_status = "critical"

        # Get component status
        components = {
            "evaluation_pipeline": {
                "status": "healthy" if bot_running else "critical",
                "message": "Trading bot active" if bot_running else "Trading bot not running",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "process_running": bot_running,
                    "recent_evaluations": 0
                },
            },
            "execution_pipeline": {
                "status": "healthy",
                "message": "Order execution pipeline active",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "recent_executions": 0,
                    "recent_errors": 0,
                    "pending_orders": 0
                },
            },
            "system_resources": {
                "status": overall_status,
                "message": f"System resources: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "memory_usage_mb": memory.used / 1024 / 1024,
                    "cpu_usage_percent": cpu_percent,
                    "system_memory_percent": memory.percent
                },
            },
            "monitoring_system": {
                "status": "unknown",
                "message": "Monitoring system status unknown",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "monitoring_running": False,
                    "health_check_running": False,
                },
            },
            "websocket_connections": {
                "status": "unknown",
                "message": "WebSocket connection status unknown",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "connectivity_ok": False,
                    "ws_active": False,
                    "connections": 0
                },
            },
            "strategy_router": {
                "status": "unknown",
                "message": "Strategy router status unknown",
                "last_check": datetime.now().isoformat(),
                "metrics": {"recent_routing": 0},
            },
            "enhanced_scanner": {
                "status": "unknown",
                "message": "Enhanced scanner status unknown",
                "last_check": datetime.now().isoformat(),
                "metrics": {"tokens_scanned": 0, "scanner_active": False},
            },
            "position_monitoring": {
                "status": "unknown",
                "message": "Position monitoring status unknown",
                "last_check": datetime.now().isoformat(),
                "metrics": {"age_seconds": 0, "recent_updates": 0},
            },
        }

        # Try to get recent metrics from monitoring files and log analysis
        try:
            # Count recent log entries to estimate activity
            import glob
            import re

            # Count recent log entries across various log files
            log_files = [
                LOG_DIR / "bot_*.log",
                LOG_DIR / "wallet.log",
                LOG_DIR / "telemetry.log",
            ]

            total_entries = 0
            recent_entries = 0
            current_time = time.time()

            for pattern in log_files:
                for log_file in glob.glob(str(pattern)):
                    try:
                        with open(log_file, "r") as f:
                            lines = f.readlines()
                            total_entries += len(lines)

                            # Count recent entries (last hour)
                            for line in lines:
                                # Extract timestamp from log line
                                timestamp_match = re.search(
                                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
                                    line,
                                )
                                if timestamp_match:
                                    try:
                                        log_time = datetime.strptime(
                                            timestamp_match.group(1),
                                            "%Y-%m-%d %H:%M:%S",
                                        )
                                        log_timestamp = log_time.timestamp()
                                        if (
                                            current_time - log_timestamp < 3600
                                        ):  # Last hour
                                            recent_entries += 1
                                    except (ValueError, AttributeError):
                                        # Skip malformed timestamps
                                        pass
                    except (OSError, IOError):
                        # Skip files that can't be read
                        pass

            # Estimate activity based on log entries
            if recent_entries > 0:
                components["evaluation_pipeline"]["metrics"].update(
                    {
                        "evaluation_count": recent_entries
                        // 10,  # Estimate evaluations
                        "successful_evaluations": recent_entries
                        // 15,  # Estimate successful
                        "failed_evaluations": recent_entries
                        // 50,  # Estimate failed
                    }
                )
                components["execution_pipeline"]["metrics"].update(
                    {
                        "execution_count": recent_entries
                        // 20,  # Estimate executions
                        "pending_orders": 0,  # Default to 0
                        "success_rate": 95.0,  # Default success rate
                    }
                )
        except Exception as e:
            print(f"Error analyzing log files: {e}")

        # Check WebSocket monitoring status
        try:
            ws_monitoring_file = LOG_DIR / "websocket_monitoring.json"
            if ws_monitoring_file.exists():
                with open(ws_monitoring_file, "r") as f:
                    ws_data = json.load(f)

                ws_active = ws_data.get("websocket_active", False)
                ws_status = ws_data.get("current_status", "unknown")
                error_msg = ws_data.get("error_message", "")

                if ws_active:
                    components["websocket_connections"]["status"] = "healthy"
                    components["websocket_connections"][
                        "message"
                    ] = "WebSocket connection active"
                    components["websocket_connections"]["metrics"][
                        "ws_active"
                    ] = True
                    components["websocket_connections"]["metrics"][
                        "connectivity_ok"
                    ] = True
                    components["websocket_connections"]["metrics"][
                        "connections"
                    ] = 1
                elif ws_status == "connection_failed":
                    components["websocket_connections"]["status"] = "warning"
                    components["websocket_connections"][
                        "message"
                    ] = f"WebSocket connection failed: {error_msg}"
                    components["websocket_connections"]["metrics"][
                        "connectivity_ok"
                    ] = False
                else:
                    components["websocket_connections"]["status"] = "warning"
                    components["websocket_connections"][
                        "message"
                    ] = f"WebSocket status: {ws_status}"
        except Exception as e:
            print(f"Error reading WebSocket monitoring file: {e}")

        # Check strategy routing status
        try:
            routing_stats_file = LOG_DIR / "strategy_routing_stats.json"
            if routing_stats_file.exists():
                with open(routing_stats_file, "r") as f:
                    routing_data = json.load(f)

                recent_routing = len(
                    routing_data.get("recent_routing_activity", [])
                )
                last_routing_time = routing_data.get("last_routing_time")

                if (
                    last_routing_time and time.time() - last_routing_time < 300
                ):  # Within last 5 minutes
                    components["strategy_router"]["status"] = "healthy"
                    components["strategy_router"][
                        "message"
                    ] = f"Strategy routing active ({recent_routing} recent activities)"
                    components["strategy_router"]["metrics"][
                        "recent_routing"
                    ] = recent_routing
                elif recent_routing > 0:
                    components["strategy_router"]["status"] = "warning"
                    components["strategy_router"][
                        "message"
                    ] = f"Strategy routing has {recent_routing} activities but not recent"
                    components["strategy_router"]["metrics"][
                        "recent_routing"
                    ] = recent_routing
                else:
                    components["strategy_router"]["status"] = "warning"
                    components["strategy_router"][
                        "message"
                    ] = "No recent strategy routing activity"
                    components["strategy_router"]["metrics"][
                        "recent_routing"
                    ] = 0
        except Exception as e:
            print(f"Error reading strategy routing stats: {e}")

        # Check enhanced scanner status
        try:
            scanner_status_file = LOG_DIR / "enhanced_scanner_status.json"
            if scanner_status_file.exists():
                with open(scanner_status_file, "r") as f:
                    scanner_data = json.load(f)

                tokens_scanned = scanner_data.get("tokens_scanned", 0)
                last_scan_time = scanner_data.get("last_scan_time")

                if (
                    last_scan_time and time.time() - last_scan_time < 300
                ):  # Within last 5 minutes
                    components["enhanced_scanner"]["status"] = "healthy"
                    components["enhanced_scanner"][
                        "message"
                    ] = f"Enhanced scanner active ({tokens_scanned} tokens scanned)"
                    components["enhanced_scanner"]["metrics"][
                        "tokens_scanned"
                    ] = tokens_scanned
                    components["enhanced_scanner"]["metrics"][
                        "scanner_active"
                    ] = True
                elif tokens_scanned > 0:
                    components["enhanced_scanner"]["status"] = "warning"
                    components["enhanced_scanner"][
                        "message"
                    ] = f"Enhanced scanner has scanned {tokens_scanned} tokens but not recently"
                    components["enhanced_scanner"]["metrics"][
                        "tokens_scanned"
                    ] = tokens_scanned
                    components["enhanced_scanner"]["metrics"][
                        "scanner_active"
                    ] = False
                else:
                    components["enhanced_scanner"]["status"] = "warning"
                    components["enhanced_scanner"][
                        "message"
                    ] = "Enhanced scanner not active"
                    components["enhanced_scanner"]["metrics"][
                        "tokens_scanned"
                    ] = 0
                    components["enhanced_scanner"]["metrics"][
                        "scanner_active"
                    ] = False
        except Exception as e:
            print(f"Error reading enhanced scanner status: {e}")

        return jsonify(
            {
                "success": True,
                "overall_status": overall_status,
                "components": components,
                "system_metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory.used / (1024 * 1024),
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                },
                "recent_metrics": [
                    {
                        "timestamp": int(time.time() * 1000),
                        "evaluation_count": components["evaluation_pipeline"][
                            "metrics"
                        ]["evaluation_count"],
                        "execution_count": components["execution_pipeline"][
                            "metrics"
                        ]["execution_count"],
                        "memory_mb": memory.used / (1024 * 1024),
                        "cpu_percent": cpu_percent,
                        "errors": components["evaluation_pipeline"]["metrics"][
                            "failed_evaluations"
                        ],
                        "websocket_connections": (
                            1
                            if components["websocket_connections"]["metrics"][
                                "ws_active"
                            ]
                            else 0
                        ),
                        "api_calls": (
                            recent_entries // 5
                            if "recent_entries" in locals()
                            else 0
                        ),
                    }
                ],
                "timestamp": int(time.time() * 1000),
            }
        )

    except Exception as e:
        print(f"Error in api_monitoring_health: {e}")
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "overall_status": "unknown",
                "components": {},
                "timestamp": int(time.time() * 1000),
            }
        )


@app.route("/api/monitoring/metrics", methods=["GET"])
def api_monitoring_metrics():
    """Return performance metrics and statistics."""
    try:
        import json
        from pathlib import Path

        # Try to read from frontend monitoring status first (contains recent_metrics)
        frontend_status_file = LOG_DIR / "frontend_monitoring_status.json"
        metrics_data = {
            "recent_metrics": [],
            "scan_metrics": {
                "tokens_scanned": 0,
                "execution_opportunities": 0,
                "scan_cache_hits": 0,
            },
            "evaluation_metrics": {
                "strategy_evaluations": 0,
                "successful_evaluations": 0,
                "failed_evaluations": 0,
            },
            "alerts_active": [],
        }

        if frontend_status_file.exists():
            try:
                with open(frontend_status_file, "r") as f:
                    status_data = json.load(f)
                    # Extract recent metrics from the monitoring data
                    metrics_data["recent_metrics"] = status_data.get("recent_metrics", [])

                    # Extract component metrics for evaluation counts
                    components = status_data.get("components", {})
                    if "evaluation_pipeline" in components:
                        eval_metrics = components["evaluation_pipeline"].get("metrics", {})
                        metrics_data["evaluation_metrics"]["strategy_evaluations"] = eval_metrics.get("recent_evaluations", 0)

                    if "enhanced_scanner" in components:
                        scanner_metrics = components["enhanced_scanner"].get("metrics", {})
                        metrics_data["scan_metrics"]["tokens_scanned"] = scanner_metrics.get("tokens_scanned", 0)
            except Exception as e:
                print(f"Error reading frontend monitoring status for metrics: {e}")

        # Try to load additional metrics from other files as fallback/supplement
        try:
            metrics_file = LOG_DIR / "monitoring_metrics.json"
            if metrics_file.exists():
                file_data = json.loads(metrics_file.read_text())
                metrics_data.update(file_data)
        except Exception as e:
            print(f"Error reading monitoring metrics file: {e}")

        return jsonify(
            {
                "success": True,
                "data": metrics_data,
                "timestamp": int(time.time() * 1000),
            }
        )

    except Exception as e:
        print(f"Error in api_monitoring_metrics: {e}")
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "data": {},
                "timestamp": int(time.time() * 1000),
            }
        )


@app.route("/api/monitoring/logs", methods=["GET"])
def api_monitoring_logs():
    """Return monitoring logs from various system components."""
    try:
        logs_data = {}

        # Define log files to read - updated for microservices architecture
        log_files = {
            "trading_engine": LOG_DIR / "trading_engine.log",
            "portfolio": LOG_DIR / "portfolio.log",
            "market_data": LOG_DIR / "market_data.log",
            "strategy_engine": LOG_DIR / "strategy_engine.log",
            "token_discovery": LOG_DIR / "token_discovery.log",
            "cex_scanner": LOG_DIR / "cex_scanner.log",
            "api_gateway": LOG_DIR / "api_gateway.log",
            "pipeline_monitor": LOG_DIR / "pipeline_monitor.log",
            "health_check": LOG_DIR / "health_check.log",
            "recovery_actions": LOG_DIR / "health_check.log",
            "monitoring_status": LOG_DIR / "pipeline_monitor.log",
        }

        for log_type, log_file in log_files.items():
            if log_file.exists():
                try:
                    # Read last 100 lines of each log file for better coverage
                    text = log_file.read_text(encoding='utf-8', errors='ignore')
                    lines = text.splitlines()[-100:]
                    # Filter out empty lines
                    lines = [line.strip() for line in lines if line.strip()]
                    logs_data[log_type] = lines
                except Exception as e:
                    logs_data[log_type] = [f"Error reading log file: {e}"]
            else:
                # Check for alternative log file locations
                alt_locations = [
                    LOG_DIR / f"{log_type}.out",
                    LOG_DIR / f"{log_type}_service.log",
                    Path("logs") / f"{log_type}.log",
                    Path(".") / f"{log_type}.log"
                ]
                
                found = False
                for alt_file in alt_locations:
                    if alt_file.exists():
                        try:
                            text = alt_file.read_text(encoding='utf-8', errors='ignore')
                            lines = text.splitlines()[-100:]
                            lines = [line.strip() for line in lines if line.strip()]
                            logs_data[log_type] = lines
                            found = True
                            break
                        except Exception:
                            continue
                
                if not found:
                    # No log file yet; return an empty list and let the UI show its placeholder
                    logs_data[log_type] = []

        return jsonify(
            {
                "success": True,
                "data": logs_data,
                "timestamp": int(time.time() * 1000),
                "total_entries": sum(
                    len(logs) for logs in logs_data.values()
                )
            }
        )

    except Exception as e:
        print(f"Error in api_monitoring_logs: {e}")
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "data": {},
                "timestamp": int(time.time() * 1000),
            }
        )


@app.route("/api/monitoring/status", methods=["GET"])
def api_monitoring_status():
    """Return monitoring system status and component health."""
    try:
        status_data = {
            "monitoring_running": False,
            "health_check_running": False,
            "frontend_running": True,  # Frontend is always running if this endpoint is called
            "timestamp": int(time.time() * 1000),
        }

        # Check if monitoring processes are running
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and len(cmdline) > 0:
                    cmd_str = " ".join(cmdline).lower()
                    # Check for various bot startup patterns
                    if any(
                        pattern in cmd_str
                        for pattern in [
                            "pipeline_monitor",
                            "enhanced_monitoring",
                            "start_bot.py",
                            "crypto_bot.main",
                            "crypto_bot/main.py",
                        ]
                    ):
                        status_data["monitoring_running"] = True
                    if any(
                        pattern in cmd_str
                        for pattern in [
                            "health_check",
                            "auto_health_check",
                            "pipeline_monitor",
                        ]
                    ):
                        status_data["health_check_running"] = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Check for recent monitoring activity
        try:
            health_status_file = LOG_DIR / "health_status.json"
            if health_status_file.exists():
                health_data = json.loads(health_status_file.read_text())
                status_data["last_health_check"] = health_data.get("timestamp")
                status_data["health_status"] = health_data
        except Exception as e:
            print(f"Error reading health status: {e}")

        return jsonify(
            {
                "success": True,
                "data": status_data,
                "timestamp": int(time.time() * 1000),
            }
        )

    except Exception as e:
        print(f"Error in api_monitoring_status: {e}")
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "data": {},
                "timestamp": int(time.time() * 1000),
            }
        )


@app.route("/api/monitoring/system-status", methods=["GET"])
def api_system_status():
    """Return system status for various services."""
    try:
        status_data = {
            "trading_engine": {
                "status": "online", "last_seen": int(time.time() * 1000)
            },
            "portfolio": {
                "status": "online", "last_seen": int(time.time() * 1000)
            },
            "market_data": {
                "status": "online", "last_seen": int(time.time() * 1000)
            },
            "strategy_engine": {
                "status": "online", "last_seen": int(time.time() * 1000)
            },
            "token_discovery": {
                "status": "online", "last_seen": int(time.time() * 1000)
            },
            "cex_scanner": {
                "status": "online", "last_seen": int(time.time() * 1000)
            },
            "api_gateway": {
                "status": "online", "last_seen": int(time.time() * 1000)
            },
        }

        # Check if log files have been updated recently to determine service status
        for service_name in status_data.keys():
            log_file = LOG_DIR / f"{service_name}.log"
            if log_file.exists():
                try:
                    mtime = log_file.stat().st_mtime
                    # If log file hasn't been updated in the last 5 minutes,
                    # mark as potentially offline
                    if time.time() - mtime > 300:  # 5 minutes
                        status_data[service_name]["status"] = "warning"
                    status_data[service_name]["last_seen"] = int(mtime * 1000)
                except Exception:
                    status_data[service_name]["status"] = "unknown"
            else:
                status_data[service_name]["status"] = "offline"

        return jsonify(
            {
                "success": True,
                "data": status_data,
                "timestamp": int(time.time() * 1000),
            }
        )

    except Exception as e:
        print(f"Error in api_system_status: {e}")
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "data": {},
                "timestamp": int(time.time() * 1000),
            }
        )


@app.route('/api/sync-health')
def get_sync_health():
    """Get synchronization health status."""
    try:
        # Get sync health from portfolio service
        sync_health = get_gateway_json("/api/v1/portfolio/sync-health")

        # Get position sync status
        position_health = get_gateway_json("/api/v1/portfolio/positions/sync-status")

        # Combine health data
        health_data = {
            'portfolio_sync': sync_health or {'status': 'unknown'},
            'position_sync': position_health or {'status': 'unknown'},
            'overall_status': 'healthy' if sync_health and position_health else 'degraded'
        }

        return jsonify({
            'success': True,
            'data': health_data
        })
    except Exception as e:
        logger.error(f"Error getting sync health: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/sync-positions', methods=['POST'])
def trigger_sync():
    """Manually trigger position synchronization."""
    try:
        # Trigger position sync via portfolio service
        sync_result = safe_post_gateway_json("/api/v1/portfolio/positions/sync", {})

        if sync_result:
            return jsonify({
                'success': True,
                'message': 'Position synchronization triggered successfully.',
                'data': sync_result
            })
        else:
            # Fallback message for when service is not available
            return jsonify({
                'success': True,
                'message': 'Position synchronization is handled automatically by the portfolio service.',
                'data': {
                    'next_steps': [
                        'Position sync is automatic in microservice architecture',
                        'Check /api/sync-health for synchronization status',
                        'Monitor portfolio service logs for sync operations'
                    ]
                }
            })

    except Exception as e:
        logger.error(f"Error triggering sync: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/sync-report/<operation_id>')
def get_sync_report(operation_id):
    """Get detailed report for a specific synchronization operation."""
    try:
        # Get sync report from portfolio service
        report = get_gateway_json(f"/api/v1/portfolio/sync-report/{operation_id}")

        if not report:
            return jsonify({
                'success': False,
                'error': 'Sync report not found or service unavailable'
            })

        if 'error' in report:
            return jsonify({
                'success': False,
                'error': report['error']
            })

        return jsonify({
            'success': True,
            'data': report
        })

    except Exception as e:
        logger.error(f"Error getting sync report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


# CORS preflight is now handled by the secure headers middleware above

# Cached trading engine state to reduce gateway calls during template rendering
TRADING_ENGINE_CACHE_TTL = 2.0
_trading_engine_state_cache: Optional[Dict[str, Any]] = None
_trading_engine_cache_ts: float = 0.0

# Global controller instance
CONTROLLER = None


def get_controller():
    """Get or create the TradingBotController instance."""
    global CONTROLLER
    if CONTROLLER is None:
        from crypto_bot.bot_controller import TradingBotController

        CONTROLLER = TradingBotController()
    return CONTROLLER


# Context processor to make high-level status information available to all templates
@app.context_processor
def inject_bot_status():
    """Return default values required by the base template on every page."""

    context: Dict[str, Any] = {
        "running": False,
        "mode": "dry_run",
        "uptime": "0:00:00",
        "paper_wallet_balance": 0.0,
        "available_balance": 0.0,
        "balance": 0.0,
        "header_total_unrealized_pnl": 0.0,
    }

    try:
        context["running"] = is_running()
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        logger.debug("Unable to determine bot runtime status: %s", exc)

    try:
        context["mode"] = load_execution_mode()
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        logger.debug("Falling back to default execution mode: %s", exc)

    try:
        context["uptime"] = get_uptime()
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        logger.debug("Failed to compute uptime: %s", exc)

    open_positions: List[Dict[str, Any]] = []
    try:
        open_positions = get_open_positions()
        context["header_total_unrealized_pnl"] = sum(
            float(position.get("pnl_value") or 0.0) for position in open_positions
        )
        context["available_balance"] = get_available_balance(open_positions)
    except Exception as exc:  # pragma: no cover - portfolio service may be unavailable
        logger.debug("Using fallback header metrics because positions are unavailable: %s", exc)

    try:
        wallet_summary = calculate_wallet_pnl()
        balance_value = float(wallet_summary.get("balance", 0.0))
        context["paper_wallet_balance"] = balance_value
        context["balance"] = balance_value
        if not context["available_balance"] and open_positions:
            context["available_balance"] = get_available_balance(open_positions)
        elif not context["available_balance"]:
            context["available_balance"] = balance_value
    except Exception as exc:  # pragma: no cover - portfolio service may be unavailable
        logger.debug("Using fallback wallet summary values: %s", exc)

    return context


LOG_FILE = LOG_DIR / "bot.log"
STATS_FILE = LOG_DIR / "strategy_stats.json"
SCAN_FILE = LOG_DIR / "asset_scores.json"
MODEL_REPORT = Path("crypto_bot/ml_signal_model/models/model_report.json")
TRADE_FILE = LOG_DIR / "trades.csv"
ERROR_FILE = LOG_DIR / "errors.log"
CONFIG_FILE = Path(resolve_config_path() if resolve_config_path else "crypto_bot/config.yaml")
REGIME_FILE = LOG_DIR / "regime_history.txt"
POSITIONS_FILE = LOG_DIR / "positions.log"

# Define project root for use in various functions
project_root = Path(__file__).parent.parent

# Environment variables will be loaded in the main block


def get_trading_engine_state(force_refresh: bool = False) -> Dict[str, Any]:
    """Return cached trading engine scheduler state from the API gateway."""

    global _trading_engine_state_cache, _trading_engine_cache_ts

    now = time.time()
    if (
        not force_refresh
        and _trading_engine_state_cache is not None
        and now - _trading_engine_cache_ts < TRADING_ENGINE_CACHE_TTL
    ):
        return _trading_engine_state_cache

    try:
        if get_gateway_json:
            state = get_gateway_json(TRADING_ENGINE_STATE_PATH)
            if isinstance(state, dict):
                _trading_engine_state_cache = state
                _trading_engine_cache_ts = now
                return state
    except (ApiGatewayError, Exception) as exc:
        logger.warning("Failed to fetch trading engine state: %s", exc)

    return _trading_engine_state_cache or {}


def is_running() -> bool:
    """Return True if the trading engine reports that it is running."""

    state = get_trading_engine_state()
    running = bool(state.get("running"))
    return running


def start_trading_engine(
    mode: str,
    *,
    interval_seconds: Optional[int] = None,
    immediate: bool = True,
) -> Dict[str, Any]:
    """Start the trading engine scheduler via the API gateway."""

    metadata: Dict[str, Any] = {"mode": mode}
    payload: Dict[str, Any] = {"immediate": immediate, "metadata": metadata}
    if interval_seconds is not None:
        payload["interval_seconds"] = interval_seconds

    try:
        response = safe_post_gateway_json(TRADING_ENGINE_START_PATH, json=payload)
        get_trading_engine_state(force_refresh=True)
        if isinstance(response, dict):
            return response
    except ApiGatewayError as exc:
        logger.error("Failed to start trading engine: %s", exc)
        return {"error": str(exc)}

    return {}


def stop_trading_engine() -> Dict[str, Any]:
    """Stop the trading engine scheduler via the API gateway."""

    try:
        response = safe_post_gateway_json(TRADING_ENGINE_STOP_PATH, json={})
        get_trading_engine_state(force_refresh=True)
        if isinstance(response, dict):
            return response
    except ApiGatewayError as exc:
        logger.error("Failed to stop trading engine: %s", exc)
        return {"error": str(exc)}

    return {}


def set_execution_mode(mode: str) -> None:
    """Set execution mode in config file."""
    canonical = _canonicalize_execution_mode(mode)
    if canonical is None:
        raise ValueError(f"Unsupported execution mode: {mode}")
    utils.set_execution_mode(canonical, CONFIG_FILE)


def load_execution_mode() -> str:
    """Load execution mode from config file."""
    if utils and hasattr(utils, 'load_execution_mode'):
        return _normalize_execution_mode(utils.load_execution_mode(CONFIG_FILE))
    return "dry_run"


def calculate_wallet_balance_from_trade_manager() -> float:
    """Retrieve wallet balance from the portfolio service."""

    pnl = calculate_wallet_pnl()
    return float(pnl.get("balance", 0.0))


def calculate_wallet_balance_from_csv() -> float:
    """DEPRECATED: Legacy CSV-based balance calculation - kept for backward compatibility."""
    logger.warning(
        "Using deprecated CSV-based balance calculation. TradeManager should be used instead."
    )
    try:
        if log_reader is not None:
            df = log_reader._read_trades(TRADE_FILE)
            if df.empty:
                return 10000.0
        else:
            return 10000.0

        # Calculate realized P&L from closed trades
        closed_trades = df[df["status"] == "closed"]
        realized_pnl = (
            closed_trades["pnl"].sum()
            if "pnl" in closed_trades.columns
            else 0.0
        )

        # Calculate unrealized P&L from open positions
        open_trades = df[df["status"] == "open"]
        unrealized_pnl = 0.0

        for _, trade in open_trades.iterrows():
            try:
                # Get current price for the symbol
                symbol = trade["symbol"]
                current_price = get_current_price(symbol)
                entry_price = trade["price"]
                amount = trade["amount"]
                side = trade["side"]

                if current_price and entry_price:
                    if side == "long":
                        pnl = (current_price - entry_price) * amount
                    else:  # short
                        pnl = (entry_price - current_price) * amount
                    unrealized_pnl += pnl
            except Exception as e:
                logger.warning(
                    f"Error calculating unrealized P&L for {symbol}: {e}"
                )
                continue

        total_pnl = realized_pnl + unrealized_pnl
        wallet_balance = 10000.0 + total_pnl
        logger.info(
            f"CSV-based calculation: realized=${realized_pnl:.2f}, unrealized=${unrealized_pnl:.2f}, total=${total_pnl:.2f}, balance=${wallet_balance:.2f}"
        )
        return wallet_balance

    except Exception as e:
        logger.error(f"Error calculating wallet balance from CSV: {e}")
        return 10000.0


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Best-effort parser for timestamps stored on positions."""

    if not value:
        return None
    if isinstance(value, datetime):
        return value
    # Handle numeric epochs (seconds)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Accept common ISO-8601 variants
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text)
        except ValueError:
            # Try parsing numeric strings
            try:
                return datetime.fromtimestamp(float(text))
            except (OSError, OverflowError, ValueError):
                return None
    return None


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Compatibility shim for legacy helpers expecting explicit ISO parsing."""

    return _parse_datetime(value)


def deduplicate_positions(positions: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Remove duplicate position entries while keeping the freshest view.

    Duplicate cards can appear when multiple data sources contribute position
    snapshots. We de-duplicate by (symbol, side) and prefer the entry with the
    most recent timestamp; if timestamps are missing we fall back to the
    highest notional value so larger/open positions are favoured.
    """

    if not positions:
        return []

    def _scoring_tuple(position: Dict[str, Any]) -> tuple[float, float]:
        ts = _parse_datetime(
            position.get("entry_time")
            or position.get("opened_at")
            or position.get("timestamp")
        )
        if ts:
            try:
                ts_score = ts.timestamp()
            except (OSError, OverflowError, ValueError):
                ts_score = float("-inf")
        else:
            ts_score = float("-inf")

        value = position.get("position_value")
        if value is None:
            value = position.get("current_value")
        if value is None and position.get("current_price") and position.get("amount"):
            try:
                value = float(position["current_price"]) * float(position["amount"])
            except (TypeError, ValueError):
                value = 0.0
        try:
            value_score = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            value_score = 0.0
        return ts_score, value_score

    deduped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for position in positions:
        raw_symbol = (
            position.get("symbol")
            or position.get("pair")
            or position.get("asset")
            or ""
        )
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            # fall back to any identifier we can find to avoid losing entries
            symbol = str(position.get("id") or position.get("asset") or "").strip().upper()
        if symbol:
            position["symbol"] = symbol

        raw_side = (
            str(position.get("side") or position.get("position_side") or "")
            .strip()
            .lower()
        )
        side = "short" if raw_side in {"short", "sell"} else "long"
        position["side"] = side
        key = (symbol, side)

        existing = deduped.get(key)
        if existing is None:
            deduped[key] = position
            continue

        if _scoring_tuple(position) >= _scoring_tuple(existing):
            deduped[key] = position

    if len(deduped) != len(positions):
        logger.debug(
            "Deduplicated open positions from %s to %s entries", len(positions), len(deduped)
        )

    return list(deduped.values())


def get_current_price(symbol: str) -> float:
    """Get current price for a symbol from various sources."""
    try:
        # Use the existing get_current_price_for_symbol function instead of missing price_manager
        return get_current_price_for_symbol(symbol)
    except Exception:
        # Fallback to basic price fetching
        try:
            import requests

            # Simple price fetching for common symbols
            if symbol == "BTC/USD":
                response = requests.get(
                    "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                    timeout=5,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("bitcoin", {}).get("usd", 0)
        except Exception:
            pass
        return 0.0


def get_paper_wallet_balance() -> float:
    """Get paper wallet balance from the single source of truth."""
    try:
        from crypto_bot.utils.balance_manager import get_single_balance

        balance = get_single_balance()
        print(
            f"Frontend got balance from single source of truth: ${balance:.2f}"
        )
        return balance
    except Exception as e:
        print(f"Error getting balance from single source: {e}")
        return 10000.0


def get_available_balance(open_positions: list) -> float:
    """Calculate available balance (wallet balance minus value of open positions)."""
    try:
        # Get current wallet balance
        total_balance = get_paper_wallet_balance()

        # Calculate value of open positions
        positions_value = 0.0
        for position in open_positions:
            if position.get("current_price") and position.get("amount"):
                # Position value = current_price * amount
                positions_value += (
                    position["current_price"] * position["amount"]
                )

        # Available balance = total balance - value of open positions
        available_balance = total_balance - positions_value

        print(
            f"Calculated available balance: ${available_balance:.2f} (total: ${total_balance:.2f}, positions: ${positions_value:.2f})"
        )
        return max(0.0, available_balance)  # Ensure non-negative

    except Exception as e:
        print(f"Error calculating available balance: {e}")
        return get_paper_wallet_balance()  # Fallback to total balance


def set_paper_wallet_balance(balance: float) -> None:
    """Set paper wallet balance in multiple locations for consistency."""
    try:
        # Update paper wallet state file (highest priority)
        paper_wallet_state_file = Path(
            "crypto_bot/logs/paper_wallet_state.yaml"
        )
        if paper_wallet_state_file.exists():
            try:
                with open(paper_wallet_state_file, "r") as f:
                    state = yaml.safe_load(f) or {}
                state["balance"] = balance
                state["initial_balance"] = (
                    balance  # Also update initial balance
                )
                with open(paper_wallet_state_file, "w") as f:
                    yaml.dump(state, f, default_flow_style=False)
                print(
                    f"Frontend updated paper wallet state file: ${balance:.2f}"
                )
            except Exception as e:
                print(
                    f"Frontend failed to update paper wallet state file: {e}"
                )
        else:
            # Create new state file
            state = {
                "balance": balance,
                "initial_balance": balance,
                "realized_pnl": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "positions": {},
            }
            paper_wallet_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(paper_wallet_state_file, "w") as f:
                yaml.dump(state, f, default_flow_style=False)
            print(
                f"Frontend created new paper wallet state file: ${balance:.2f}"
            )

        # Update paper_wallet.yaml
        paper_wallet_file = LOG_DIR / "paper_wallet.yaml"
        paper_config = {"initial_balance": balance}
        with open(paper_wallet_file, "w") as f:
            yaml.dump(paper_config, f, default_flow_style=False)
        print(f"Frontend updated paper_wallet.yaml: ${balance:.2f}")

        # Update user_config.yaml
        user_config_file = Path("crypto_bot/user_config.yaml")
        if user_config_file.exists():
            with open(user_config_file) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        config["paper_wallet_balance"] = balance
        with open(user_config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Frontend updated user_config.yaml: ${balance:.2f}")

        # Update legacy config if it exists
        legacy_config_path = Path("crypto_bot/paper_wallet_config.yaml")
        if legacy_config_path.exists():
            try:
                with open(legacy_config_path) as f:
                    legacy_config = yaml.safe_load(f) or {}
                legacy_config["initial_balance"] = balance
                with open(legacy_config_path, "w") as f:
                    yaml.dump(legacy_config, f, default_flow_style=False)
                print(
                    f"Frontend updated legacy config {legacy_config_path}: ${balance:.2f}"
                )
            except Exception as e:
                print(
                    f"Frontend failed to update legacy config {legacy_config_path}: {e}"
                )

        print(
            f"Frontend successfully updated paper wallet balance to: ${balance:.2f}"
        )

    except Exception as e:
        print(f"Error setting paper wallet balance: {e}")
        raise


def _resolve_default_paper_balance() -> float:
    """Return the configured starting paper balance."""

    candidates: list[float] = []

    user_config_file = Path("crypto_bot/user_config.yaml")
    if user_config_file.exists():
        try:
            with open(user_config_file) as handle:
                user_config = yaml.safe_load(handle) or {}
            value = user_config.get("paper_wallet_balance")
            if value is not None:
                candidates.append(float(value))
        except Exception as exc:
            logger.debug("Failed to read paper_wallet_balance from user_config.yaml: %s", exc)

    try:
        config_data = load_bot_config(CONFIG_FILE) if load_bot_config else {}
        risk_cfg = config_data.get("risk", {}) if isinstance(config_data, dict) else {}
        value = risk_cfg.get("starting_balance")
        if value is not None:
            candidates.append(float(value))
    except Exception as exc:
        logger.debug("Failed to read starting balance from config: %s", exc)

    for candidate in candidates:
        try:
            if candidate > 0:
                return float(candidate)
        except Exception:
            continue

    return 10000.0


def _reset_paper_wallet_state_file(balance: float) -> None:
    """Overwrite the paper wallet state file with a clean payload."""

    state_file = Path("crypto_bot/logs/paper_wallet_state.yaml")
    clean_state = {
        "balance": float(balance),
        "initial_balance": float(balance),
        "realized_pnl": 0.0,
        "total_trades": 0,
        "winning_trades": 0,
        "positions": {},
    }
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with state_file.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(clean_state, handle, default_flow_style=False)
        logger.info("Paper wallet state reset to clean balance %.2f", balance)
    except Exception as exc:
        logger.error("Failed to reset paper wallet state file: %s", exc)
        raise


def _coerce_optional_float(value: Any) -> Optional[float]:
    """Safely convert values from API payloads into floats."""

    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        value = candidate
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _parse_entry_timestamp(value: Any) -> float:
    """Normalise entry time information for deduplication comparisons."""

    if value is None:
        return float("-inf")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return float("-inf")
        try:
            # Handle ISO timestamps with or without ``Z`` suffix.
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).timestamp()
        except ValueError:
            try:
                return float(candidate)
            except ValueError:
                return float("-inf")
    return float("-inf")


def _deduplicate_positions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure only the freshest position per symbol is returned."""

    unique_positions: dict[str, dict[str, Any]] = {}
    for payload in positions:
        symbol_raw = (
            payload.get("symbol")
            or payload.get("pair")
            or payload.get("asset")
            or ""
        )
        symbol = str(symbol_raw).strip().upper()
        if not symbol:
            continue

        payload["symbol"] = symbol

        current = unique_positions.get(symbol)
        if current is None:
            unique_positions[symbol] = payload
            continue

        if _parse_entry_timestamp(payload.get("entry_time")) >= _parse_entry_timestamp(current.get("entry_time")):
            unique_positions[symbol] = payload

    return list(unique_positions.values())


def get_open_positions() -> list:
    """Get open positions via the portfolio service."""

    positions = fetch_positions_from_service()
    if not positions:
        logger.info("No open positions returned by portfolio service")
        return []

    unique_positions = _deduplicate_positions(positions)
    if len(unique_positions) != len(positions):
        logger.info(
            "Deduplicated open positions: %s raw entries -> %s unique symbols",
            len(positions),
            len(unique_positions),
        )
    else:
        logger.info("Fetched %s open positions from portfolio service", len(unique_positions))

    return unique_positions











def get_uptime() -> str:
    """Return human readable uptime derived from trading engine state."""

    state = get_trading_engine_state()
    metadata = state.get("metadata") or {}
    start_timestamp = _parse_iso_timestamp(metadata.get("started_at"))

    if start_timestamp is None and state.get("running"):
        start_timestamp = _parse_iso_timestamp(state.get("last_run_started_at"))

    if start_timestamp is None:
        return utils.get_uptime(None)

    return utils.get_uptime(start_timestamp)


@lru_cache(maxsize=1)
def _resolve_initial_balance() -> float:
    """Return the configured starting balance for the paper wallet."""

    try:
        config_path = resolve_config_path() if resolve_config_path else "crypto_bot/config.yaml"
        cfg = load_bot_config(config_path) if load_bot_config else {}
        paper_cfg = cfg.get("paper_wallet") or {}
        balance = paper_cfg.get("initial_balance")
        if balance is None:
            balance = cfg.get("risk", {}).get("starting_balance")
        return float(balance) if balance is not None else 10000.0
    except Exception:
        return 10000.0


def calculate_wallet_pnl() -> Dict[str, float]:
    """Calculate wallet PnL using the portfolio service."""

    pnl = fetch_portfolio_pnl()
    initial_balance = _resolve_initial_balance()

    total = float(pnl["total"])
    realized = float(pnl["realized"])
    unrealized = float(pnl["unrealized"])

    balance = initial_balance + total
    pnl_percentage = (total / initial_balance * 100) if initial_balance else 0.0

    return {
        "total_pnl": total,
        "pnl_percentage": pnl_percentage,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "balance": balance,
        "initial_balance": initial_balance,
    }





@app.route("/api/test")
def api_test():
    """Simple test endpoint to verify API is working."""
    return jsonify(
        {
            "status": "success",
            "message": "API is working",
            "timestamp": str(datetime.now()),
        }
    )


@app.route("/api/debug-positions")
def api_debug_positions():
    """Debug endpoint to check what data is available."""
    try:
        state = fetch_portfolio_state()
        positions = state.get("positions") or []
        closed = state.get("closed_positions") or []
        price_cache = state.get("price_cache") or []

        return jsonify(
            {
                "positions_count": len(positions),
                "closed_positions_count": len(closed),
                "price_cache_count": len(price_cache),
                "sample_position": positions[0] if positions else None,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/clear-old-positions", methods=["POST"])
def api_clear_old_positions():
    """Force close stale positions through the portfolio service."""

    from flask import request

    payload = request.get_json(silent=True) or {}
    max_age_hours = payload.get("max_age_hours", 24)
    symbols = payload.get("symbols")

    try:
        max_age_int = int(max_age_hours)
    except (TypeError, ValueError):
        max_age_int = 24
    max_age_int = max(0, min(max_age_int, 24 * 365))

    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]
    elif isinstance(symbols, list):
        symbols = [str(s).strip() for s in symbols if s]
    else:
        symbols = None

    request_body: Dict[str, Any] = {"max_age_hours": max_age_int}
    if symbols:
        request_body["symbols"] = symbols

    try:
        result = safe_post_gateway_json("/api/v1/portfolio/positions/close-stale", json=request_body)
    except ApiGatewayError as exc:
        logger.error("Failed to close stale positions via portfolio service: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 502

    logger.info(
        "Closed %s stale positions via portfolio service (mode=%s)",
        result.get("closed", 0),
        result.get("mode"),
    )

    response_payload = {
        "success": True,
        "closed": result.get("closed", 0),
        "symbols": result.get("symbols", []),
        "cutoff": result.get("cutoff"),
        "mode": result.get("mode"),
    }

    return jsonify(response_payload)


@app.route("/api/dashboard-metrics")
def api_dashboard_metrics():
    """Return comprehensive dashboard metrics."""
    try:
        # Get performance data
        performance = {
            "total_pnl": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "uptime": get_uptime(),
            "trades_today": 0,
            "trades_last_24h": 0,
            "winning_positions": 0,
            "losing_positions": 0,
        }
        pnl_data = calculate_wallet_pnl()
        performance.update(
            {
                "total_pnl": pnl_data.get("total_pnl", 0.0),
                "realized_pnl": pnl_data.get("realized_pnl", 0.0),
                "unrealized_pnl": pnl_data.get("unrealized_pnl", 0.0),
            }
        )
        performance["pnl_percentage"] = pnl_data.get("pnl_percentage", 0.0)
        performance["balance"] = pnl_data.get("balance", 0.0)
        performance["initial_balance"] = pnl_data.get("initial_balance", 0.0)
        logger.info("Retrieved portfolio P&L data via API gateway")

        # Try to get statistics from portfolio service
        try:
            stats_data = get_gateway_json("/api/v1/portfolio/statistics")
            if stats_data:
                performance.update({
                    "total_trades": stats_data.get("total_trades", performance["total_trades"]),
                    "win_rate": stats_data.get("win_rate", performance["win_rate"]),
                    "trades_today": stats_data.get("trades_today", performance["trades_today"]),
                    "trades_last_24h": stats_data.get(
                        "trades_last_24h", performance["trades_last_24h"]
                    ),
                    "winning_positions": stats_data.get(
                        "winning_positions", performance["winning_positions"]
                    ),
                    "losing_positions": stats_data.get(
                        "losing_positions", performance["losing_positions"]
                    ),
                    "unrealized_pnl": stats_data.get("unrealized_pnl", pnl_data.get("unrealized_pnl", 0.0)),
                    "realized_pnl": stats_data.get("realized_pnl", pnl_data.get("realized_pnl", 0.0)),
                })
                logger.info("Retrieved statistics from portfolio service")
        except ApiGatewayError as e:
            logger.warning(f"Could not get statistics from portfolio service: {e}")

        # Get allocation data
        allocation = {}
        try:
            config_path = resolve_config_path() if resolve_config_path else "crypto_bot/config.yaml"
            cfg = load_bot_config(config_path) if load_bot_config else {}
            allocation = cfg.get("strategy_allocation", {})
        except Exception:
            allocation = {}

        # Get recent trades (placeholder - could be enhanced)
        recent_trades = []
        
        return jsonify({
            "success": True,
            "performance": performance,
            "allocation": allocation,
            "recent_trades": recent_trades,
            "wallet_summary": pnl_data,
            "timestamp": int(time.time() * 1000)
        })

    except ApiGatewayError as exc:
        logger.error("Dashboard metrics unavailable: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 502
    except Exception as e:
        logger.error(f"Error fetching dashboard metrics: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/live-updates")
def api_live_updates():
    """Return live dashboard updates."""
    try:
        # Get bot status
        bot_status = {
            "running": False,
            "uptime": get_uptime()
        }
        
        # Check if trading engine is running via gateway
        try:
            state = get_trading_engine_state()
            bot_status.update({
                "running": bool(state.get("running", False)),
                "mode": state.get("mode", "unknown"),
                "last_cycle": state.get("last_cycle", None)
            })
        except Exception as e:
            logger.debug(f"Could not get trading engine state: {e}")

        # Get wallet balance
        paper_wallet_balance = None
        try:
            pnl_summary = calculate_wallet_pnl()
            paper_wallet_balance = pnl_summary.get("balance")
        except ApiGatewayError as exc:
            logger.warning("Portfolio service unavailable for live updates: %s", exc)
        except Exception as e:
            logger.debug(f"Could not calculate wallet balance: {e}")

        return jsonify({
            "success": True,
            "bot_status": bot_status,
            "paper_wallet_balance": paper_wallet_balance,
            "timestamp": int(time.time() * 1000)
        })

    except Exception as e:
        logger.error(f"Error fetching live updates: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/open-positions")
def api_open_positions():
    """Return open positions data for the dashboard."""

    try:
        positions = get_open_positions()
        return jsonify(positions)
    except Exception as exc:
        logger.error("Failed to fetch open positions: %s", exc)
        return jsonify({"error": str(exc)}), 502


@app.route("/api/portfolio/pnl")
def api_portfolio_pnl():
    """Expose portfolio PnL details expected by dashboard refresh logic."""

    try:
        pnl = fetch_portfolio_pnl()
        payload = {
            "success": True,
            "total": float(pnl.get("total", 0)),
            "realized": float(pnl.get("realized", 0)),
            "unrealized": float(pnl.get("unrealized", 0)),
        }
        return jsonify(payload)
    except ApiGatewayError as exc:
        logger.error("Portfolio PnL unavailable: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 502
    except Exception as exc:  # pragma: no cover - defensive path
        logger.error("Failed to retrieve portfolio PnL: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500




def fetch_current_price_for_symbol(symbol):
    """Fetch current price for a symbol using available exchanges."""
    import ccxt

    # Try multiple exchanges in order
    exchanges_to_try = [
        ('kraken', ccxt.kraken()),
        ('binance', ccxt.binance()),
        ('coinbase', ccxt.coinbase()),
        ('bitstamp', ccxt.bitstamp()),
    ]

    for exchange_name, exchange in exchanges_to_try:
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker.get("last", 0)
            if price and price > 0:
                logger.debug(f"Successfully fetched price for {symbol} from {exchange_name}: ${price}")
                return price
        except Exception as e:
            logger.debug(f"{exchange_name} failed for {symbol}: {e}")
            continue

    # If all exchanges fail, try to get price from TradeManager cache as last resort
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager
        tm = get_trade_manager()
        cached_price = tm.price_cache.get(symbol)
        if cached_price:
            logger.debug(f"Using cached price for {symbol}: ${cached_price}")
            return float(cached_price)
    except Exception as e:
        logger.debug(f"Failed to get cached price for {symbol}: {e}")

    # Return 0 if all methods fail
    logger.warning(f"All price fetching methods failed for {symbol}")
    return 0


@app.route("/api/wallet-balance")
def api_wallet_balance():
    """Return current wallet balance."""
    try:
        pnl = calculate_wallet_pnl()
        return jsonify(
            {
                "success": True,
                "balance": pnl.get("balance", 0.0),
                "total_pnl": pnl.get("total_pnl", 0.0),
                "realized_pnl": pnl.get("realized_pnl", 0.0),
                "unrealized_pnl": pnl.get("unrealized_pnl", 0.0),
            }
        )

    except ApiGatewayError as exc:
        logger.error(f"Wallet balance unavailable: {exc}")
        return jsonify({"success": False, "error": str(exc)}), 502
    except Exception as e:
        logger.error(f"Error calculating wallet balance: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/wallet-pnl")
def api_wallet_pnl():
    """Return current wallet PnL calculation from TradeManager."""
    try:
        pnl_data = calculate_wallet_pnl()
        return jsonify(pnl_data)

    except ApiGatewayError as exc:
        logger.error("Wallet PnL unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 502
    except Exception as fallback_error:
        logger.error(
            f"Failed to calculate wallet PnL: {fallback_error}"
        )
        return jsonify({"error": str(fallback_error)}), 500


@app.route("/api/execution-mode", methods=["POST"])
def api_set_execution_mode():
    """Update the execution mode persisted in configuration."""

    payload = request.get_json(silent=True) or {}
    requested_mode = payload.get("mode")
    resolved_mode = _canonicalize_execution_mode(requested_mode)
    if not resolved_mode:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Unsupported execution mode",
                    "requested_mode": requested_mode,
                }
            ),
            400,
        )

    current_mode = load_execution_mode()
    if _normalize_execution_mode(current_mode) == resolved_mode:
        logger.info("Execution mode already %s", resolved_mode)
        return jsonify(
            {
                "success": True,
                "mode": resolved_mode,
                "previous_mode": current_mode,
                "message": "Execution mode unchanged",
            }
        )

    logger.info(
        "Updating execution mode from %s to %s", current_mode, resolved_mode
    )

    set_execution_mode(resolved_mode)

    engine_stop = None
    engine_error = None
    try:
        engine_stop = stop_trading_engine()
    except Exception as exc:  # pragma: no cover - best effort stop
        engine_error = str(exc)
        logger.warning("Failed to stop trading engine after mode change: %s", exc)

    response_payload: Dict[str, Any] = {
        "success": True,
        "mode": resolved_mode,
        "previous_mode": current_mode,
    }
    if engine_stop:
        response_payload["engine_stop"] = engine_stop
    if engine_error:
        response_payload["engine_stop_error"] = engine_error

    return jsonify(response_payload)


@app.route("/api/paper/reset", methods=["POST"])
def api_reset_paper_trading():
    """Reset paper trading state, caches, and wallet balance."""

    mode = _normalize_execution_mode(load_execution_mode())
    if mode != "dry_run":
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Paper trading reset is only available in dry_run mode",
                    "mode": mode,
                }
            ),
            409,
        )

    details: Dict[str, Any] = {}
    errors: list[str] = []
    details["mode_before_reset"] = mode

    # Create a reset flag file to force local state for next position fetches
    reset_flag_file = Path("crypto_bot/logs/reset_flag.tmp")
    try:
        reset_flag_file.parent.mkdir(parents=True, exist_ok=True)
        reset_flag_file.write_text(str(time.time()))
        logger.info("Reset flag file created, will force local state for position fetches")
    except Exception as e:
        logger.warning(f"Failed to create reset flag file: {e}")

    try:
        details["engine_stop"] = stop_trading_engine()
    except Exception as exc:  # pragma: no cover - best effort stop
        logger.warning("Unable to stop trading engine before reset: %s", exc)
        errors.append(f"Failed to stop trading engine: {exc}")

    positions_cleared = 0
    trade_state_file = None
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager, reset_trade_manager

        trade_manager = get_trade_manager()
        trade_state_file = str(getattr(trade_manager, "storage_path", "")) or None
        with trade_manager.lock:
            positions_cleared = len(trade_manager.positions)
            trade_manager.trades.clear()
            trade_manager.positions.clear()
            trade_manager.closed_positions.clear()
            trade_manager.price_cache.clear()
            trade_manager.total_trades = 0
            trade_manager.total_volume = Decimal("0")
            trade_manager.total_fees = Decimal("0")
            trade_manager.total_realized_pnl = Decimal("0")
            trade_manager.save_state()

        try:
            trade_manager.shutdown()
        except Exception as exc:
            logger.warning("TradeManager shutdown after reset failed: %s", exc)
            errors.append(f"TradeManager shutdown failed: {exc}")
        finally:
            try:
                reset_trade_manager()
            except Exception as exc:
                logger.debug("Failed to reset TradeManager singleton: %s", exc)

        details["trade_manager"] = {
            "positions_cleared": positions_cleared,
            "state_file": trade_state_file,
        }
    except ImportError:
        details["trade_manager"] = "unavailable"
    except Exception as exc:
        logger.error("Failed to reset TradeManager state: %s", exc)
        errors.append(f"TradeManager reset failed: {exc}")

    try:
        from crypto_bot.utils.scan_cache_manager import get_scan_cache_manager

        get_scan_cache_manager().clear()
        details["scan_cache"] = "cleared"
    except Exception as exc:
        logger.warning("Failed to clear scan cache: %s", exc)
        errors.append(f"Scan cache clear failed: {exc}")

    try:
        from crypto_bot.utils import indicator_cache

        indicator_cache.CACHE.clear()
        details["indicator_cache"] = "cleared"
    except Exception as exc:
        logger.debug("Indicator cache clear failed: %s", exc)

    try:
        from crypto_bot.utils.price_fetcher import clear_price_cache

        clear_price_cache()
        details["price_cache"] = "cleared"
    except Exception as exc:
        logger.debug("Price cache clear failed: %s", exc)

    cex_state_file = Path("crypto_bot/logs/cex_scanner_state.json")
    try:
        cex_state_file.parent.mkdir(parents=True, exist_ok=True)
        baseline_state = {
            "seen_pairs": [],
            "last_scan": None,
            "exchange": "kraken",
            "initialised": False,
        }
        with cex_state_file.open("w", encoding="utf-8") as handle:
            json.dump(baseline_state, handle, indent=2)
        details["cex_scanner_state"] = "reset"
    except Exception as exc:
        logger.warning("Failed to reset CEX scanner state: %s", exc)
        errors.append(f"CEX scanner state reset failed: {exc}")

    # Clear trade history
    try:
        trades_file = Path("crypto_bot/logs/trades.csv")
        if trades_file.exists():
            with trades_file.open("w", encoding="utf-8") as handle:
                handle.write("")  # Clear the file completely
            details["trade_history"] = "cleared"
        else:
            details["trade_history"] = "not_found"
    except Exception as exc:
        logger.warning("Failed to clear trade history: %s", exc)
        errors.append(f"Trade history clear failed: {exc}")

    # Try to reset portfolio service database
    try:
        import requests
        # Create empty portfolio state matching Pydantic schema
        empty_state = {
            "trades": [],
            "positions": [],
            "closed_positions": [],
            "price_cache": [],
            "statistics": {
                "total_trades": 0,
                "total_volume": 0.0,
                "total_fees": 0.0,
                "total_realized_pnl": 0.0,
                "last_updated": datetime.now().isoformat()
            }
        }

        portfolio_url = "http://localhost:8003/state"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.put(portfolio_url, json=empty_state, headers=headers, timeout=5)
            if response.status_code == 200:
                details["portfolio_service"] = "reset_via_api"
                logger.info("Successfully reset portfolio service database")
            else:
                details["portfolio_service"] = f"api_error_{response.status_code}"
                logger.warning(f"Portfolio service API returned status {response.status_code}")
        except requests.exceptions.RequestException as exc:
            logger.debug(f"Portfolio service API not available: {exc}")
            details["portfolio_service"] = "api_not_available"
    except ImportError:
        logger.debug("Requests library not available for portfolio service reset")
        details["portfolio_service"] = "requests_unavailable"
    except Exception as exc:
        logger.debug(f"Failed to reset portfolio service state: {exc}")
        details["portfolio_service"] = "reset_failed"

    pair_cache_file = Path("cache/liquid_pairs.json")
    try:
        if pair_cache_file.exists():
            pair_cache_file.unlink()
            details["pair_cache"] = "cleared"
    except Exception as exc:
        logger.debug("Pair cache clear failed: %s", exc)

    # Clear additional cache files
    additional_cache_files = [
        Path("crypto_bot/logs/last_regime.json"),
        Path("crypto_bot/logs/system_status.json")
    ]
    for cache_file in additional_cache_files:
        try:
            if cache_file.exists():
                cache_file.unlink()
                details[f"{cache_file.name}"] = "cleared"
        except Exception as exc:
            logger.debug(f"Cache file {cache_file.name} clear failed: {exc}")

    # Reset additional state files
    additional_state_files = [
        Path("crypto_bot/logs/paper_wallet.yaml"),
        Path("crypto_bot/logs/paper_wallet_state.yaml"),
        Path("frontend/crypto_bot/logs/paper_wallet_state.yaml"),
        Path("frontend/crypto_bot/logs/trade_manager_state.json")
    ]

    for state_file in additional_state_files:
        try:
            if state_file.exists():
                # Create clean state based on file type
                if state_file.suffix == '.yaml':
                    clean_state = {
                        "balance": 10000.0,
                        "initial_balance": 10000.0,
                        "realized_pnl": 0.0,
                        "total_trades": 0,
                        "winning_trades": 0,
                        "positions": {}
                    }
                    import yaml
                    with state_file.open("w", encoding="utf-8") as handle:
                        yaml.safe_dump(clean_state, handle, default_flow_style=False)
                elif state_file.suffix == '.json':
                    clean_state = {
                        "trades": [],
                        "positions": {},
                        "closed_positions": [],
                        "price_cache": {},
                        "statistics": {
                            "total_trades": 0,
                            "total_volume": 0.0,
                            "total_fees": 0.0,
                            "total_realized_pnl": 0.0
                        }
                    }
                    with state_file.open("w", encoding="utf-8") as handle:
                        json.dump(clean_state, handle, indent=2)

                details[f"{state_file.name}"] = "reset"
            else:
                details[f"{state_file.name}"] = "not_found"
        except Exception as exc:
            logger.debug(f"Additional state file {state_file.name} reset failed: {exc}")

    # Reset BalanceManager (single source of truth)
    try:
        from crypto_bot.utils.balance_manager import BalanceManager
        BalanceManager.set_balance(10000.0)
        details["balance_manager"] = "reset_to_10000"
    except Exception as exc:
        logger.debug(f"BalanceManager reset failed: {exc}")
        errors.append(f"BalanceManager reset failed: {exc}")

    balance = _resolve_default_paper_balance()
    try:
        _reset_paper_wallet_state_file(balance)
        set_paper_wallet_balance(balance)
        details["paper_wallet_balance"] = balance
    except Exception as exc:
        errors.append(f"Paper wallet reset failed: {exc}")

    status_code = 200 if not errors else 207
    return (
        jsonify(
            {
                "success": not errors,
                "balance": balance,
                "details": details,
                "errors": errors,
                "positions_cleared": positions_cleared,
                "force_empty_positions": True,  # Flag to force frontend to show empty positions
            }
        ),
        status_code,
    )


@app.route("/start", methods=["POST"])
def start():
    mode = request.form.get("mode", "dry_run")
    set_execution_mode(mode)
    start_trading_engine(mode)
    return redirect(url_for("index"))


@app.route("/start_bot", methods=["POST"])
def start_bot():
    """Start the trading bot and return JSON status."""
    mode = (
        request.json.get("mode", "dry_run")
        if request.is_json
        else request.form.get("mode", "dry_run")
    )
    interval_seconds = None
    immediate = True
    if request.is_json:
        interval_seconds = request.json.get("interval_seconds")
        immediate = request.json.get("immediate", True)

    set_execution_mode(mode)
    result = start_trading_engine(
        mode,
        interval_seconds=interval_seconds,
        immediate=bool(immediate),
    )

    state = get_trading_engine_state(force_refresh=True)
    running = bool(state.get("running"))
    response = {
        "status": "started" if running else "pending",
        "running": running,
        "uptime": get_uptime(),
        "mode": mode,
        "state": state,
        "message": result.get("status") or "Trading engine start requested",
    }

    if "error" in result:
        response["status"] = "error"
        response["message"] = result["error"]
        return jsonify(response), 502

    return jsonify(response)


@app.route("/stop")
def stop():
    stop_trading_engine()
    return redirect(url_for("index"))


@app.route("/stop_bot", methods=["POST"])
def stop_bot():
    """Stop the trading bot and return JSON status."""
    result = stop_trading_engine()
    state = get_trading_engine_state(force_refresh=True)
    response = {
        "status": "stopped" if not state.get("running") else "pending",
        "running": bool(state.get("running")),
        "uptime": get_uptime(),
        "mode": load_execution_mode(),
        "message": result.get("status") or "Trading engine stop requested",
    }

    if "error" in result:
        response["status"] = "error"
        response["message"] = result["error"]
        return jsonify(response), 502

    return jsonify(response)


@app.route("/pause_bot", methods=["POST"])
def pause_bot():
    """Pause the trading bot and return JSON status."""
    result = stop_trading_engine()
    response = {
        "status": "paused" if "error" not in result else "error",
        "running": False,
        "uptime": get_uptime(),
        "mode": load_execution_mode(),
        "message": result.get("status") or "Trading engine pause requested",
    }
    if "error" in result:
        response["message"] = result["error"]
        return jsonify(response), 502
    return jsonify(response)


@app.route("/resume_bot", methods=["POST"])
def resume_bot():
    """Resume the trading bot and return JSON status."""
    mode = load_execution_mode()
    result = start_trading_engine(mode, immediate=False)
    state = get_trading_engine_state(force_refresh=True)
    response = {
        "status": "resumed" if state.get("running") else "pending",
        "running": bool(state.get("running")),
        "uptime": get_uptime(),
        "mode": mode,
        "message": result.get("status") or "Trading engine resume requested",
    }
    if "error" in result:
        response["status"] = "error"
        response["message"] = result["error"]
        return jsonify(response), 502
    return jsonify(response)


@app.route("/bot_logs")
def bot_logs_page():
    """Bot logs page with navigation."""
    mode = load_execution_mode()
    return render_template(
        "bot_logs.html",
        running=is_running(),
        mode=mode,
        uptime=get_uptime(),
        title="Bot Logs",
    )


@app.route("/logs_tail")
def logs_tail():
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().splitlines()[-200:]
        return "\n".join(lines)
    return ""


@app.route("/stats")
def stats():
    data = {}
    if STATS_FILE.exists():
        with open(STATS_FILE) as f:
            data = json.load(f)
    return render_template("stats.html", stats=data)


@app.route("/scans")
def scans():
    data = {}
    if SCAN_FILE.exists():
        with open(SCAN_FILE) as f:
            data = json.load(f)
    return render_template("scans.html", scans=data)


@app.route("/cli", methods=["GET", "POST"])
def cli():
    """Run CLI commands and display output."""
    output = None
    if request.method == "POST":
        base = request.form.get("base", "bot")
        cmd_args = request.form.get("command", "")
        venv_python = Path(__file__).parent.parent / "venv" / "bin" / "python3"
        if base == "backtest":
            cmd = f"{venv_python} -m crypto_bot.backtest.backtest_runner {cmd_args}"
        elif base == "custom":
            cmd = cmd_args
        else:
            cmd = f"{venv_python} start_bot.py noninteractive {cmd_args}"
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=False
            )
            output = proc.stdout + proc.stderr
        except Exception as exc:  # pragma: no cover - subprocess
            output = str(exc)
    return render_template("cli.html", output=output)


# Global cache for dashboard data (disabled to ensure freshness)
_dashboard_cache = {}
_CACHE_TIMEOUT = 0  # disable caching


@app.route("/")
def index():
    """Root route: redirect to the main dashboard."""
    return redirect(url_for("dashboard"))


def get_cached_dashboard_data():
    """Dashboard caching disabled."""
    return None


def set_cached_dashboard_data(data):
    """Dashboard caching disabled (no-op)."""
    return


def batch_fetch_prices(symbols):
    """Fetch prices for multiple symbols in batch to reduce API calls."""
    import ccxt

    prices = {}

    # Try Kraken first for major pairs
    kraken_symbols = []
    binance_symbols = []

    for symbol in symbols:
        # Normalize for Kraken
        kraken_symbol = symbol.replace("BTC/", "XBT/").replace("/BTC", "/XBT")
        kraken_symbols.append(kraken_symbol)
        binance_symbols.append(symbol)

    try:
        exchange = ccxt.kraken()
        # Fetch tickers in batch if supported
        kraken_tickers = exchange.fetch_tickers(kraken_symbols)
        for symbol, kraken_symbol in zip(symbols, kraken_symbols):
            if kraken_symbol in kraken_tickers:
                prices[symbol] = kraken_tickers[kraken_symbol].get("last", 0)
    except Exception as e:
        logger.debug(f"Kraken batch fetch failed: {e}")

    # Fill missing prices with Binance
    missing_symbols = [s for s in symbols if s not in prices]
    if missing_symbols:
        try:
            exchange = ccxt.binance()
            binance_tickers = exchange.fetch_tickers(missing_symbols)
            for symbol in missing_symbols:
                if symbol in binance_tickers:
                    prices[symbol] = binance_tickers[symbol].get("last", 0)
        except Exception as e:
            logger.debug(f"Binance batch fetch failed: {e}")

    return prices


@app.route("/favicon.ico")
def favicon():
    """Serve favicon.ico to prevent 404 errors."""
    return "", 204  # No Content response


@app.route("/api/v1/market-data/batch-candles", methods=["POST"])
def api_batch_candles():
    """Proxy batch candles request to market-data service."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract parameters
        symbols = data.get("symbols", [])
        limit = data.get("limit", 100)
        timeframe = data.get("timeframe", "5m")
        exchange = data.get("exchange", "kraken")

        if not symbols:
            return jsonify({"error": "No symbols provided"}), 400

        # Prepare batch payload
        batch_payload = {
            "symbols": symbols,
            "limit": limit,
            "timeframe": timeframe,
            "exchange": exchange
        }

        # Call the market-data service
        logger.info(f"Requesting batch candles for symbols: {symbols}")
        try:
            batch_response = safe_post_gateway_json("/api/v1/market-data/batch-candles", json=batch_payload)

            if batch_response:
                logger.info("Successfully received batch candles data")
                return jsonify(batch_response)
            else:
                logger.error("Market-data service returned empty response")
                return jsonify({"error": "Market-data service returned empty response"}), 502

        except Exception as gateway_error:
            logger.error(f"Market-data service unavailable: {gateway_error}")
            return jsonify({
                "error": "Market-data service is currently unavailable",
                "details": str(gateway_error)
            }), 503

    except Exception as e:
        logger.error(f"Error in api_batch_candles: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard")
def dashboard():
    """Dashboard with cache-busting timestamp."""
    try:
        # Get actual positions and portfolio data
        open_positions = get_open_positions()
        pnl_data = calculate_wallet_pnl()
        statistics_snapshot: Dict[str, Any] = {}

        try:
            raw_stats = get_gateway_json("/api/v1/portfolio/statistics")
            if isinstance(raw_stats, dict) and raw_stats.get("success") != False:
                statistics_snapshot = raw_stats
        except (ApiGatewayError, Exception) as exc:
            logger.warning("Portfolio statistics unavailable for dashboard render: %s", exc)

        
        # Calculate available balance
        available_balance = get_available_balance(open_positions)

        win_rate_value = float(statistics_snapshot.get("win_rate", 0.0) or 0.0)
        total_trades_value = int(
            statistics_snapshot.get("total_trades", len(open_positions))
        )
        trades_today_value = int(statistics_snapshot.get("trades_today", 0) or 0)
        
        # Calculate total unrealized P&L from positions
        total_unrealized_pnl = sum(pos.get("pnl_value", 0.0) for pos in open_positions)

        # Return dashboard with real data
        dashboard_data = {
            "running": True,
            "mode": "paper",
            "uptime": "0:00:00",
            "last_trade": None,
            "regime": "unknown",
            "last_reason": "test",
            "pnl": pnl_data.get("total_pnl", 0.0),
            "performance": {
                "total_pnl": pnl_data.get("total_pnl", 0.0),
                "win_rate": win_rate_value,
                "trades": len(open_positions),
                "trades_today": trades_today_value,
            },
            "allocation": {},
            "paper_wallet_balance": pnl_data.get("balance", 10000.0),
            "initial_balance": pnl_data.get("initial_balance", 10000.0),
            "available_balance": available_balance,
            "open_position_balance": sum(pos.get("current_value", 0) for pos in open_positions),
            "open_positions": open_positions,
            "pnl_data": pnl_data,
            "header_total_unrealized_pnl": total_unrealized_pnl,
            "cache_bust": int(time.time() * 1000),
            "regimes": [],
            "total_trades": total_trades_value,
            "win_rate": win_rate_value,
            "last_update": datetime.now()
        }
        if statistics_snapshot:
            dashboard_data["statistics"] = statistics_snapshot
        return render_template("dashboard.html", **dashboard_data)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        import traceback
        traceback.print_exc()
        return f"<h1>Dashboard Error</h1><pre>{str(e)}</pre><pre>{traceback.format_exc()}</pre>", 500


@app.route("/dashboard-test")
def dashboard_test():
    """Test route to debug dashboard issues."""
    return jsonify({"status": "working", "message": "Dashboard test route works"})

@app.route("/dashboard-simple")
def dashboard_simple():
    """Simple dashboard test with minimal data."""
    try:
        minimal_data = {
            "running": True,
            "mode": "paper",
            "uptime": "0:00:00",
            "last_trade": None,
            "regime": "unknown",
            "last_reason": "test",
            "pnl": 0.0,
            "performance": {"total_pnl": 0, "win_rate": 0, "trades": 0},
            "allocation": {},
            "paper_wallet_balance": 10000.0,
            "initial_balance": 10000.0,
            "available_balance": 10000.0,
            "open_position_balance": 0.0,
            "open_positions": [],
            "pnl_data": {"total_pnl": 0.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0},
            "cache_bust": int(time.time() * 1000),
            "regimes": [],
        }
        return render_template("dashboard.html", **minimal_data)
    except Exception as e:
        logger.error(f"Simple dashboard error: {e}")
        import traceback
        traceback.print_exc()
        return f"Simple Dashboard Error: {str(e)}", 500

@app.route("/model")
def model_page():
    report = {}
    if MODEL_REPORT.exists():
        with open(MODEL_REPORT) as f:
            report = json.load(f)
    return render_template("model.html", report=report)


@app.route("/train_model", methods=["POST"])
def train_model_route():
    file = request.files.get("csv")
    if file:
        tmp_path = LOG_DIR / "upload.csv"
        file.save(tmp_path)
        ml.train_from_csv(tmp_path)
        tmp_path.unlink()
    return redirect(url_for("model_page"))


@app.route("/validate_model", methods=["POST"])
def validate_model_route():
    file = request.files.get("csv")
    tmp_path = None
    if file:
        tmp_path = LOG_DIR / "validate.csv"
        file.save(tmp_path)
        metrics = ml.validate_from_csv(tmp_path)
        tmp_path.unlink()
    else:
        default_csv = LOG_DIR / "trades.csv"
        if default_csv.exists():
            metrics = ml.validate_from_csv(default_csv)
        else:
            metrics = ml.validate_from_csv(default_csv)
    if metrics:
        MODEL_REPORT.write_text(json.dumps(metrics))
    return redirect(url_for("model_page"))


@app.route("/api_config")
def api_config_page():
    """API configuration page."""
    # Load current API configuration
    api_config = {}
    user_config_file = Path("crypto_bot/user_config.yaml")
    if user_config_file.exists():
        with open(user_config_file) as f:
            api_config = yaml.safe_load(f) or {}

    return render_template("api_config.html", api_config=api_config)


@app.route("/monitoring")
def monitoring_page():
    """Monitoring dashboard page."""
    return render_template("monitoring.html")


@app.route("/logs")
def logs_page():
    """System logs dashboard page."""
    return render_template("logs.html")


@app.route("/config_settings")
def config_settings_page():
    """General configuration settings page."""
    # Load current configuration
    config_data = {}
    if CONFIG_FILE.exists():
        try:
            config_data = load_bot_config(CONFIG_FILE) if load_bot_config else {}
        except Exception:
            config_data = {}

    return render_template("config_settings.html", config_data=config_data)


@app.route("/api/save_api_config", methods=["POST"])
def save_api_config():
    """Save API configuration."""
    try:
        data = request.get_json()
        user_config_file = Path("crypto_bot/user_config.yaml")

        # Load existing config
        current_config = {}
        if user_config_file.exists():
            with open(user_config_file) as f:
                current_config = yaml.safe_load(f) or {}

        # Update with new values
        current_config.update(data)

        # Save back to file
        with open(user_config_file, "w") as f:
            yaml.dump(current_config, f, default_flow_style=False)

        return jsonify(
            {
                "status": "success",
                "message": "API configuration saved successfully",
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error saving configuration: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/save_config_settings", methods=["POST"])
def save_config_settings():
    """Save general configuration settings."""
    try:
        data = request.get_json()

        # Load existing config
        current_config = {}
        if CONFIG_FILE.exists():
            try:
                current_config = load_bot_config(CONFIG_FILE)
            except Exception:
                current_config = {}

        # Update with new values (merge nested structures)
        def deep_merge(d1, d2):
            for key, value in d2.items():
                if (
                    key in d1
                    and isinstance(d1[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(d1[key], value)
                else:
                    d1[key] = value
            return d1

        updated_config = deep_merge(current_config, data)

        # Save back to file
        save_config(updated_config, CONFIG_FILE)

        return jsonify(
            {
                "status": "success",
                "message": "Configuration saved successfully",
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error saving configuration: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/refresh_config", methods=["POST"])
def refresh_config():
    """Refresh configuration by reloading from files."""
    try:
        response = safe_post_gateway_json(TRADING_ENGINE_RELOAD_CONFIG_PATH, json={})
        message = (
            response.get("status") if isinstance(response, dict) else "Configuration reload triggered"
        )
        return jsonify(
            {"status": "success", "message": message}
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error refreshing configuration: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/refresh-dashboard", methods=["POST"])
def refresh_dashboard():
    """API endpoint to refresh dashboard data."""
    try:
        logger.info("Dashboard refresh requested")

        return jsonify(
            {
                "success": True,
                "message": "Dashboard refresh completed",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error refreshing dashboard: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route("/trades")
def trades_page():
    return render_template("trades.html")


@app.route("/trades_data")
def trades_data():
    """Return structured trade data for the trades page."""
    try:
        # Get trades from portfolio service
        portfolio_state = get_gateway_json("/api/v1/portfolio/state")

        if not portfolio_state:
            return jsonify({"error": "Portfolio service unavailable"}), 503

        trades = portfolio_state.get("trades", [])

        # Format trades for frontend compatibility
        formatted_trades = []
        for trade in trades:
            formatted_trade = {
                "id": trade.get("id", f"{trade.get('symbol', 'unknown')}_{trade.get('timestamp', 'unknown')}"),
                "symbol": trade.get("symbol", ""),
                "side": trade.get("side", ""),
                "type": trade.get("side", ""),  # For compatibility
                "amount": float(trade.get("amount", 0)),
                "quantity": float(trade.get("amount", 0)),  # For compatibility
                "price": float(trade.get("price", 0)),
                "execution_price": float(trade.get("price", 0)),  # For compatibility
                "timestamp": trade.get("timestamp", ""),
                "date": trade.get("timestamp", ""),  # For compatibility
                "status": trade.get("status", "completed"),
                "pnl": 0.0,  # Will be calculated by frontend if needed
                "pnl_percentage": 0.0,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_percentage": 0.0,
                "current_price": 0.0
            }
            formatted_trades.append(formatted_trade)

        # Sort trades by timestamp (most recent first)
        formatted_trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Return the most recent 200 trades to avoid overwhelming the frontend
        return jsonify(formatted_trades[-200:] if len(formatted_trades) > 200 else formatted_trades)
        
    except Exception as e:
        logger.error(f"Error in trades_data endpoint: {e}")
        return jsonify([])


@app.route("/trades_tail")
def trades_tail():
    trades = ""
    if TRADE_FILE.exists():
        trades = "\n".join(TRADE_FILE.read_text().splitlines()[-100:])
    errors = ""
    if ERROR_FILE.exists():
        errors = "\n".join(ERROR_FILE.read_text().splitlines()[-100:])
    return jsonify({"trades": trades, "errors": errors})


@app.route("/api/current-prices")
def api_current_prices():
    """Return current market prices for symbols with fresh data."""
    try:
        # Check if historical data is requested
        include_history = request.args.get("history", "").lower() == "true"
        symbol_filter = request.args.get("symbol")

        # Return real price history if requested
        if include_history and symbol_filter:
            # Get price history from market data service
            history_data = get_gateway_json(f"/market-data/history/{symbol_filter}")
            if history_data:
                return jsonify(
                    {
                        "symbol": symbol_filter,
                        "include_history": include_history,
                        "history": history_data,
                    }
                )
            else:
                return (
                    jsonify(
                        {
                            "error": "No price history data available",
                            "symbol": symbol_filter,
                            "include_history": include_history,
                        }
                    ),
                    404,
                )

        # Get symbols from active positions in portfolio service
        portfolio_state = get_gateway_json("/api/v1/portfolio/state")

        if not portfolio_state:
            return jsonify({"error": "Portfolio service unavailable"}), 503

        # Get symbols from open positions
        positions = portfolio_state.get("positions", [])
        symbols = [pos.get("symbol") for pos in positions if pos.get("is_open", True)]

        # Get current prices from market data service
        current_prices = {}
        if symbols:
            # Get fresh prices for all symbols
            prices_data = get_gateway_json(f"/market-data/prices?symbols={','.join(symbols)}")
            if prices_data:
                current_prices.update(prices_data)

        logger.info(f"Returning current prices for {len(current_prices)} symbols")
        return jsonify(current_prices)
    except Exception as e:
        logger.error(f"Error in api_current_prices: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/test-route")
def api_test_route():
    """Simple test route to check if routing is working."""
    import time

    return jsonify(
        {"message": "Test route works!", "timestamp": int(time.time() * 1000)}
    )


# Price history API removed - no chart functionality


def get_latest_candle_timestamp(symbol):
    """Get the timestamp of the most recent 5-minute candle for a symbol."""
    try:
        # Try to get real market data first
        import asyncio
        import time

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Import the enhanced OHLCV fetcher
            from crypto_bot.utils.enhanced_ohlcv_fetcher import (
                EnhancedOHLCVFetcher,
            )
            from crypto_bot.execution.cex_executor import get_exchange
            import yaml
            from pathlib import Path

            # Load user configuration to get the correct exchange
            user_config_path = (
                Path(__file__).resolve().parent.parent
                / "crypto_bot"
                / "user_config.yaml"
            )
            if user_config_path.exists():
                with open(user_config_path, "r") as f:
                    user_config = yaml.safe_load(f)
                exchange_name = user_config.get("exchange", "kraken")
            else:
                # Fallback to kraken if no user config found
                exchange_name = "kraken"

            # Create exchange instance with correct configuration
            config = {
                "exchange": exchange_name,
                "max_concurrent_ohlcv": 3,
                "max_concurrent_dex_ohlcv": 10,
                "min_volume_usd": 0,
            }
            exchange, _ = get_exchange(config)

            # Create fetcher instance
            fetcher = EnhancedOHLCVFetcher(exchange, config)

            # Normalize symbol for specific exchanges (e.g., Kraken uses XBT instead of BTC)
            try:
                ex_id = getattr(exchange, "id", "").lower()
            except Exception:
                ex_id = ""
            normalized_symbol = symbol
            if ex_id == "kraken":
                if isinstance(symbol, str):
                    normalized_symbol = symbol.replace("BTC/", "XBT/").replace(
                        "/BTC", "/XBT"
                    )

            # Fetch just the most recent candle
            cex_data, dex_data = loop.run_until_complete(
                fetcher.fetch_ohlcv_batch([normalized_symbol], "5m", 1)
            )
            # Combine CEX and DEX data for frontend display
            data_map = {**cex_data, **dex_data}

            # Prefer normalized symbol key if present
            symbol_key = (
                normalized_symbol if normalized_symbol in data_map else symbol
            )

            if symbol_key in data_map and data_map[symbol_key]:
                raw_data = data_map[symbol_key]

                if isinstance(raw_data, list) and len(raw_data) > 0:
                    # Get the most recent candle timestamp
                    latest_candle = raw_data[-1]  # Most recent candle
                    if len(latest_candle) >= 1:
                        return int(latest_candle[0])  # Return timestamp

        except Exception as e:
            print(f"Failed to fetch real candle timestamp for {symbol}: {e}")
        finally:
            loop.close()

    except Exception as e:
        print(f"Error getting candle timestamp for {symbol}: {e}")

    # Fallback: return current time rounded to nearest 5-minute interval
    current_time = int(time.time())
    # Round down to nearest 5-minute boundary
    return (current_time // 300) * 300




def _submit_execution_order(
    symbol: str, side: str, amount: float, *, metadata: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Submit a market order via the execution service. Returns (success, result, error)."""

    payload_metadata = {
        "source": "manual_market_sell",
        "order_type": "market",
        "requested_amount": amount,
    }
    if metadata:
        payload_metadata.update(metadata)

    request_payload = {
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "metadata": payload_metadata,
    }

    try:
        # Skip execution service call for paper trading - just return success
        logger.info("Skipping execution service call for paper trading mode")
        return True, {"status": "paper_trading", "exchange": "paper"}, None
    except ApiGatewayError as exc:
        logger.warning(
            "Execution service unavailable for %s %s %s: %s",
            side,
            amount,
            symbol,
            exc,
        )
        return False, None, str(exc)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(
            "Execution service call failed for %s %s %s: %s",
            side,
            amount,
            symbol,
            exc,
        )
        return False, None, str(exc)

    if not isinstance(result, dict):
        logger.error(
            "Execution service returned unexpected payload for %s: %r",
            symbol,
            result,
        )
        return False, None, "Unexpected response from execution service"

    client_order_id = result.get("client_order_id") or result.get("order_id")
    if not client_order_id:
        logger.error(
            "Execution response missing order id for %s: %r",
            symbol,
            result,
        )
        return False, result, "Execution service did not return an order id"

    return True, result, None


def _record_closing_trade(
    symbol: str,
    side: str,
    amount: float,
    price: float,
    *,
    order_result: Optional[Dict[str, Any]],
    position_source: str,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Record the manual closing trade via the portfolio service or TradeManager."""

    timestamp = datetime.now(timezone.utc)
    trade_id = uuid4().hex
    metadata: Dict[str, Any] = {
        "source": "manual_market_sell",
        "position_source": position_source,
    }
    if order_result:
        metadata["execution"] = {
            key: order_result.get(key)
            for key in ("client_order_id", "order_id", "status")
            if order_result.get(key) is not None
        }

    trade_payload = {
        "id": trade_id,
        "symbol": symbol,
        "side": side,
        "amount": str(Decimal(str(amount))),
        "price": str(Decimal(str(price))),
        "timestamp": timestamp.isoformat(),
        "strategy": "manual_market_sell",
        "exchange": order_result.get("exchange") if order_result else None,
        "fees": "0",
        "status": order_result.get("status") if order_result else "filled",
        "order_id": order_result.get("order_id") if order_result else None,
        "client_order_id": order_result.get("client_order_id") if order_result else None,
        "metadata": metadata,
    }

    # Skip portfolio service call and go straight to TradeManager for paper trading
    logger.info("Using TradeManager directly for trade recording in paper trading mode")

    # Fallback to TradeManager for local accounting
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager, create_trade

        trade = create_trade(
            symbol=symbol,
            side=side,
            amount=Decimal(str(amount)),
            price=Decimal(str(price)),
            strategy="manual_market_sell",
            exchange=(order_result.get("exchange") if order_result else "manual"),
            order_id=order_result.get("order_id") if order_result else None,
            client_order_id=order_result.get("client_order_id") if order_result else None,
            metadata=metadata,
        )

        trade_manager = get_trade_manager()
        trade_manager.record_trade(trade)
        # Skip save_state() for manual trades to avoid blocking the response
        # The trade will still be recorded in memory and saved on next automatic save
        logger.info("Trade recorded in TradeManager (save_state skipped for performance)")

        return True, trade.id, {"method": "trade_manager"}
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(
            "TradeManager fallback failed to record trade for %s: %s",
            symbol,
            exc,
        )
        return False, None, {"error": str(exc)}


@app.route("/api/sell-position", methods=["POST"])
def api_sell_position():
    """Sell a position via market order."""
    try:
        data = request.get_json()
        if not data:
            return (
                jsonify({"success": False, "error": "No data provided"}),
                400,
            )

        symbol_raw = data.get("symbol")
        amount = data.get("amount")

        if not symbol_raw or amount is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Symbol and amount are required",
                    }
                ),
                400,
            )

        symbol = str(symbol_raw).strip().upper()

        try:
            amount_value = float(amount)
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Amount must be a number",
                    }
                ),
                400,
            )

        if amount_value <= 0:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Amount must be greater than zero",
                    }
                ),
                400,
            )

        logger.info("API: Received sell request for %.10f %s", amount_value, symbol)

        position, position_source = _find_open_position(symbol)
        if not position:
            logger.warning("Sell request for %s but no open position found", symbol)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"No open position found for {symbol}",
                    }
                ),
                404,
            )

        position_amount = _coerce_optional_float(
            position.get("size")
            or position.get("amount")
            or position.get("total_amount")
        )
        if not position_amount or position_amount <= 0:
            logger.warning(
                "Sell request for %s but position has non-positive amount: %s",
                symbol,
                position_amount,
            )
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Position for {symbol} has no remaining size",
                    }
                ),
                400,
            )

        close_amount = min(position_amount, amount_value)
        if close_amount <= 0:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Requested amount must be greater than zero",
                    }
                ),
                400,
            )

        position_side_raw = str(position.get("side") or "long").lower()
        position_side = "short" if position_side_raw in {"short", "sell"} else "long"
        close_side = "sell" if position_side == "long" else "buy"

        close_price = _resolve_close_price(symbol, position)
        if close_price <= 0:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Unable to determine market price for {symbol}",
                    }
                ),
                502,
            )

        # Attempt execution, but don't fail if it doesn't work (paper trading mode)
        execution_success, execution_result, execution_error = _submit_execution_order(
            symbol,
            close_side,
            close_amount,
            metadata={"position_source": position_source},
        )

        order_metadata = execution_result if execution_success else None

        # Always try to record the trade, even if execution failed
        trade_recorded, trade_id, trade_details = _record_closing_trade(
            symbol,
            close_side,
            close_amount,
            close_price,
            order_result=order_metadata,
            position_source=position_source,
        )

        if not trade_recorded:
            logger.error(
                "Failed to record closing trade for %s after execution result %s",
                symbol,
                execution_result,
            )
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Failed to record closing trade",
                        "execution": {
                            "submitted": execution_success,
                            "error": execution_error,
                            "result": execution_result,
                        },
                        "trade": trade_details,
                    }
                ),
                500,
            )

        response_payload = {
            "success": True,
            "message": f"Closed {close_amount:.10f} {symbol}",
            "symbol": symbol,
            "close_amount": close_amount,
            "close_side": close_side,
            "price": close_price,
            "position_source": position_source,
            "trade_id": trade_id,
            "trade": trade_details,
            "execution": {
                "submitted": execution_success,
                "error": execution_error,
                "result": execution_result,
            },
            "timestamp": time.time(),
        }

        if execution_success and execution_result:
            response_payload["order_id"] = (
                execution_result.get("client_order_id")
                or execution_result.get("order_id")
            )
            response_payload["status"] = execution_result.get("status") or "submitted"

        return jsonify(response_payload)

    except Exception as e:
        logger.error(f"Error in api_sell_position: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Compatibility alias in case clients post to '/sell-position' without the /api prefix
@app.route("/sell-position", methods=["POST"])
def api_sell_position_alias():
    return api_sell_position()


@app.route("/api/candle-timestamp")
def api_candle_timestamp():
    """Return the timestamp of the most recent 5-minute candle for a symbol."""
    try:
        symbol = request.args.get("symbol")
        if not symbol:
            return jsonify({"error": "Symbol parameter is required"})

        # Get the most recent 5-minute candle timestamp from market data service
        candle_info = get_gateway_json(f"/market-data/candle-timestamp/{symbol}")

        if candle_info:
            return jsonify({
                "symbol": symbol,
                "timestamp": candle_info.get("timestamp"),
                "timeframe": "5m"
            })
        else:
            return jsonify({"error": "Unable to fetch candle timestamp"}), 503

    except Exception as e:
        logger.error(f"Error in api_candle_timestamp: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/candle-data")
def api_candle_data():
    """Return candle data for a symbol."""
    try:
        raw_symbol = request.args.get("symbol", "BTC/USD") or "BTC/USD"
        symbol = raw_symbol.strip().upper()

        limit = int(request.args.get("limit", 50))
        interval = (request.args.get("interval") or "5m").lower()
        exchange = (request.args.get("exchange") or "kraken").lower()

        # Clamp limit to a sensible range to avoid overloading downstream services
        limit = max(1, min(limit, 500))

        params = {
            "limit": limit,
            "timeframe": interval,
            "exchange_id": exchange,
        }

        use_cache_arg = request.args.get("use_cache")
        if use_cache_arg is not None:
            params["use_cache"] = use_cache_arg

        encoded_symbol = quote(symbol, safe="")

        # Get candle data from market data service via the API gateway
        candle_data = get_gateway_json(
            f"/candles/{encoded_symbol}", params=params
        )

        if candle_data:
            return jsonify(candle_data)
        return jsonify({"error": "Unable to fetch candle data"}), 503

    except Exception as e:
        logger.error(f"Error in api_candle_data: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/trend-data")
def api_trend_data():
    """Return trend analysis data for a symbol."""
    try:
        symbol = request.args.get("symbol")
        if not symbol:
            return jsonify({"error": "Symbol parameter is required"})

        # Get trend data from strategy engine service
        trend_data = get_gateway_json(f"/strategy-engine/trend/{symbol}")

        if trend_data:
            return jsonify(trend_data)
        else:
            return jsonify({"error": "Unable to fetch trend data"}), 503

    except Exception as e:
        logger.error(f"Error in api_trend_data: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/live-signals")
def api_live_signals():
    """Return live trading signals."""
    try:
        # Import the signals file functionality from api.py
        from pathlib import Path
        import json

        SIGNALS_FILE = Path(__file__).resolve().parent / ".." / "crypto_bot" / "signals.json"

        signals_data = {}
        if SIGNALS_FILE.exists():
            try:
                signals_data = json.loads(SIGNALS_FILE.read_text())
            except Exception:
                logger.warning("Failed to parse signals file, returning empty data")
                signals_data = {}

        return jsonify(signals_data)

    except Exception as e:
        logger.error(f"Error in api_live_signals: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/price-history")
def api_price_history():
    """Return 5-minute price history for the last 2 hours for trend chart."""
    try:
        symbol = request.args.get("symbol")
        if not symbol:
            return jsonify({"error": "Symbol parameter is required"})

        # Get price history from market data service
        history_data = get_gateway_json(f"/market-data/history/{symbol}?hours=2&timeframe=5m")

        if history_data:
            return jsonify(history_data)
        else:
            return jsonify({"error": "Unable to fetch price history"}), 503

    except Exception as e:
        logger.error(f"Error in api_price_history: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def api_v1_proxy(path):
    """Proxy /api/v1/* requests to the API gateway."""
    try:
        # Build the full path for the API gateway
        gateway_path = f"/api/v1/{path}"

        # Get query parameters
        query_string = request.query_string.decode('utf-8')
        if query_string:
            gateway_path += f"?{query_string}"

        # Handle different HTTP methods
        if request.method == "GET":
            return get_gateway_json(gateway_path)
        elif request.method in ["POST", "PUT"]:
            # For POST/PUT requests, pass the JSON data
            data = request.get_json() if request.is_json else None
            return safe_post_gateway_json(gateway_path, json=data)
        else:
            # For other methods, return method not allowed
            return {"error": f"Method {request.method} not supported"}, 405
    except Exception as exc:
        logger.error(f"Error proxying API request to {path}: {exc}")
        return {"error": f"Failed to proxy request: {str(exc)}"}, 500


# Main entry point for running the Flask application
if __name__ == "__main__":
    import os as _os
    import sys

    def find_free_port():
        """Find a free port to use for the server."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    # Set up frontend event subscriber for real-time trade updates
    try:
        trade_manager = get_single_source_trade_manager()

        def frontend_event_handler(event):
            """Handle trade events for frontend updates."""
            try:
                if hasattr(app, 'frontend_event_queue'):
                    app.frontend_event_queue.put(event)
                logger.debug(f"Frontend received event: {event.event_type}")
            except Exception as e:
                logger.error(f"Frontend event handler failed: {e}")

        # Create and register frontend subscriber
        frontend_subscriber = create_frontend_subscriber(frontend_event_handler)
        trade_manager.add_frontend_subscriber(frontend_subscriber)

        # Create event queue for frontend
        from queue import Queue
        app.frontend_event_queue = Queue()

        logger.info("Frontend event subscriber registered with SingleSourceTradeManager")
    except Exception as e:
        logger.warning(f"Failed to set up frontend event subscriber: {e}")

    # Get port from environment or find a free one
    env_port = _os.environ.get("LCT_PORT") or _os.environ.get("FLASK_PORT_OVERRIDE") or _os.environ.get("FLASK_PORT")
    try:
        port = int(env_port) if env_port else 5050  # Default to 5050 instead of random
    except Exception:
        port = 5050  # Default to 5050 instead of random
    print(f"FLASK_PORT={port}")  # This is what startup scripts look for
    print(f"Starting ASGI app on port {port} using Uvicorn...")
    print("Press Ctrl+C to stop the server")

    try:
        try:
            import uvicorn
            print(f"Starting ASGI app on port {port} using Uvicorn...")
            config = uvicorn.Config(asgi_app, host="0.0.0.0", port=port, log_level="info")
            server = uvicorn.Server(config)
            server.run()
        except ImportError:
            print(f"Uvicorn not available, falling back to Flask development server on port {port}...")
            app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error running server: {e}")
        sys.exit(1)
