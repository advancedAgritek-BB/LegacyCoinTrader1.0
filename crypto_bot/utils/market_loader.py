"""Utilities for loading trading symbols and fetching OHLCV data."""

from typing import Iterable, List, Dict, Any, Deque, Optional, Union, Tuple
import asyncio
import inspect
import time
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import ccxt
import aiohttp
import base58
import warnings
import contextlib
from collections import deque
from typing import Dict, Any

from .telegram import TelegramNotifier
from .logger import LOG_DIR, setup_logger


class CircuitBreaker:
    """Circuit breaker pattern for failing API endpoints."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,
        expected_exception: Exception = Exception,
    ):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Type of exception to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = setup_logger("circuit_breaker", LOG_DIR / "circuit_breaker.log")

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    async def async_call(self, coro_func, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await coro_func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.logger.info("Circuit breaker reset to CLOSED after successful call")
        elif self.state == "CLOSED":
            self.failure_count = 0  # Reset on consecutive successes

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            if self.state != "OPEN":
                self.state = "OPEN"
                self.logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
        elif self.state == "HALF_OPEN":
            # If we're in HALF_OPEN and still failing, go back to OPEN
            self.state = "OPEN"
            self.logger.warning("Circuit breaker returned to OPEN after HALF_OPEN failure")


class AdaptiveRateLimiter:
    """Intelligent rate limiting for API calls with adaptive delays."""
    
    def __init__(
        self,
        max_requests_per_minute: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        error_backoff_multiplier: float = 2.0,
        success_recovery_factor: float = 0.8,
        window_size: int = 1000
    ):
        """
        Initialize the adaptive rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests allowed per minute
            base_delay: Base delay in seconds between requests
            max_delay: Maximum delay in seconds
            error_backoff_multiplier: Multiplier for delay on errors
            success_recovery_factor: Factor to reduce delay on success
            window_size: Size of the request history window
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.error_backoff_multiplier = error_backoff_multiplier
        self.success_recovery_factor = success_recovery_factor
        
        # Request tracking
        self.request_times = deque(maxlen=window_size)
        self.error_count = 0
        self.consecutive_errors = 0
        self.current_delay = base_delay
        self.last_request_time = 0
        
        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.logger = setup_logger("adaptive_rate_limiter", LOG_DIR / "adaptive_rate_limiter.log")
    
    async def wait_if_needed(self) -> None:
        """
        Wait if necessary to respect rate limits and apply adaptive delays.
        """
        now = time.time()
        
        # Calculate current request rate over the last 60 seconds
        cutoff_time = now - 60
        recent_requests = sum(1 for req_time in self.request_times if req_time > cutoff_time)
        
        # Apply rate limiting if we're exceeding the limit
        if recent_requests >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.request_times[0]) if self.request_times else 60
            if wait_time > 0:
                self.logger.debug(f"Rate limit exceeded, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Apply adaptive delay based on current delay setting
        if self.current_delay > 0:
            time_since_last = now - self.last_request_time
            if time_since_last < self.current_delay:
                wait_time = self.current_delay - time_since_last
                self.logger.debug(f"Applying adaptive delay: {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_times.append(now)
        self.last_request_time = now
        self.total_requests += 1
    
    def record_error(self) -> None:
        """
        Record an error and increase the backoff delay.
        """
        self.total_errors += 1
        self.error_count += 1
        self.consecutive_errors += 1
        
        # Apply exponential backoff for consecutive errors
        if self.consecutive_errors > 0:
            backoff_delay = self.base_delay * (self.error_backoff_multiplier ** self.consecutive_errors)
            self.current_delay = min(self.max_delay, backoff_delay)
            
            self.logger.warning(
                f"API error recorded. Consecutive errors: {self.consecutive_errors}, "
                f"delay increased to {self.current_delay:.2f}s"
            )
    
    def record_success(self) -> None:
        """
        Record a successful request and gradually reduce the delay.
        """
        self.consecutive_errors = 0
        
        # Gradually reduce delay on success
        if self.current_delay > self.base_delay:
            self.current_delay = max(
                self.base_delay,
                self.current_delay * self.success_recovery_factor
            )
            self.logger.debug(f"Success recorded, delay reduced to {self.current_delay:.2f}s")
    
    def get_current_delay(self) -> float:
        """
        Get the current delay value.
        
        Returns:
            Current delay in seconds
        """
        return self.current_delay
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Returns:
            Dictionary with current statistics
        """
        now = time.time()
        recent_requests = sum(1 for req_time in self.request_times if req_time > now - 60)
        
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(self.total_requests, 1),
            "consecutive_errors": self.consecutive_errors,
            "current_delay": self.current_delay,
            "requests_last_minute": recent_requests,
            "max_requests_per_minute": self.max_requests_per_minute
        }


# Global rate limiter instance
_global_rate_limiter: Optional[AdaptiveRateLimiter] = None

# Global circuit breaker instance for API endpoints
_global_circuit_breaker: Optional[CircuitBreaker] = None


def get_rate_limiter() -> AdaptiveRateLimiter:
    """
    Get or create the global rate limiter instance.

    Returns:
        AdaptiveRateLimiter instance
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = AdaptiveRateLimiter()
    return _global_rate_limiter


def get_circuit_breaker() -> CircuitBreaker:
    """
    Get or create the global circuit breaker instance.

    Returns:
        CircuitBreaker instance
    """
    global _global_circuit_breaker
    if _global_circuit_breaker is None:
        _global_circuit_breaker = CircuitBreaker(
            failure_threshold=10,  # Allow more failures before opening
            recovery_timeout=600.0,  # 10 minutes before attempting recovery
            expected_exception=(ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadRequest)
        )
    return _global_circuit_breaker


def configure_rate_limiter(
    max_requests_per_minute: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    error_backoff_multiplier: Optional[float] = None,
    success_recovery_factor: Optional[float] = None,
    window_size: Optional[int] = None
) -> None:
    """
    Configure the global rate limiter with new settings.
    
    Args:
        max_requests_per_minute: Maximum requests allowed per minute
        base_delay: Base delay in seconds between requests
        max_delay: Maximum delay in seconds
        error_backoff_multiplier: Multiplier for delay on errors
        success_recovery_factor: Factor to reduce delay on success
        window_size: Size of the request history window
    """
    global _global_rate_limiter
    
    if _global_rate_limiter is None:
        _global_rate_limiter = AdaptiveRateLimiter()
    
    if max_requests_per_minute is not None:
        _global_rate_limiter.max_requests_per_minute = max_requests_per_minute
    if base_delay is not None:
        _global_rate_limiter.base_delay = base_delay
    if max_delay is not None:
        _global_rate_limiter.max_delay = max_delay
    if error_backoff_multiplier is not None:
        _global_rate_limiter.error_backoff_multiplier = error_backoff_multiplier
    if success_recovery_factor is not None:
        _global_rate_limiter.success_recovery_factor = success_recovery_factor
    if window_size is not None:
        # Create new deque with new window size
        old_times = list(_global_rate_limiter.request_times)
        _global_rate_limiter.request_times = deque(old_times, maxlen=window_size)


_last_snapshot_time = 0

logger = setup_logger(__name__, LOG_DIR / "bot.log")

failed_symbols: Dict[str, Dict[str, Any]] = {}
RETRY_DELAY = 300
MAX_RETRY_DELAY = 3600


def reset_failed_symbols(symbols: Optional[List[str]] = None) -> int:
    """Reset failed symbols cache, optionally for specific symbols only.

    Args:
        symbols: List of specific symbols to reset. If None, resets all.

    Returns:
        Number of symbols reset.
    """
    global failed_symbols

    if symbols is None:
        # Reset all symbols
        count = len(failed_symbols)
        failed_symbols.clear()
        logger.info(f"Reset all {count} failed symbols")
        return count
    else:
        # Reset specific symbols
        count = 0
        for symbol in symbols:
            if symbol in failed_symbols:
                del failed_symbols[symbol]
                count += 1
                logger.info(f"Reset failed symbol: {symbol}")
        logger.info(f"Reset {count} specific failed symbols")
        return count


def get_failed_symbols_info() -> Dict[str, Dict[str, Any]]:
    """Get information about currently failed/disabled symbols."""
    return failed_symbols.copy()
# Default timeout when fetching OHLCV data
OHLCV_TIMEOUT = 60
# Default timeout when fetching OHLCV data over WebSocket
WS_OHLCV_TIMEOUT = 60
# REST requests occasionally face Cloudflare delays up to a minute
REST_OHLCV_TIMEOUT = 90
# Number of consecutive failures allowed before disabling a symbol
MAX_OHLCV_FAILURES = 10
# Additional retryable error patterns
RETRYABLE_ERROR_PATTERNS = [
    "too many requests",
    "missing client certificate",
    "rate limit",
    "temporary",
    "unavailable",
    "timeout",
    "connection",
    "network",
    "dns",
    "resolve",
    "nodename"
]

# API error handling configuration
API_ERROR_CONFIG: Optional[Dict[str, Any]] = None

def load_api_error_config() -> Dict[str, Any]:
    """Load API error handling configuration."""
    global API_ERROR_CONFIG
    if API_ERROR_CONFIG is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "api_error_handling.yaml"
        try:
            with open(config_path, 'r') as f:
                API_ERROR_CONFIG = yaml.safe_load(f) or {}
            logger.info("Loaded API error handling configuration")
        except Exception as exc:
            logger.warning(f"Failed to load API error config: {exc}. Using defaults.")
            API_ERROR_CONFIG = {}
    return API_ERROR_CONFIG

def get_api_config_value(key_path: str, default=None):
    """Get a value from the API error configuration using dot notation."""
    config = load_api_error_config()
    keys = key_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
MAX_WS_LIMIT = 500
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
UNSUPPORTED_SYMBOL = object()
STATUS_UPDATES = True
SEMA: Optional[asyncio.Semaphore] = None

# Mapping of common symbols to CoinGecko IDs for OHLC fallback
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

# Cache GeckoTerminal pool addresses and metadata per symbol
# Mapping: symbol -> (pool_addr, volume, reserve, price, limit)
GECKO_POOL_CACHE: dict[str, tuple[str, float, float, float, int]] = {}
GECKO_SEMAPHORE = asyncio.Semaphore(25)

# Valid characters for Solana addresses
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# Quote currencies eligible for Coinbase fallback
SUPPORTED_USD_QUOTES = {"USD", "USDC", "USDT"}


def _is_valid_base_token(token: str) -> bool:
    """Return True if ``token`` looks like a Solana mint address."""
    if not isinstance(token, str):
        return False
    if not (32 <= len(token) <= 44):
        return False
    try:
        return len(base58.b58decode(token)) == 32 and all(
            c in BASE58_ALPHABET for c in token
        )
    except Exception:
        return False


def configure(
    ohlcv_timeout: Optional[Union[int, float]] = None,
    max_failures: Optional[int] = None,
    max_ws_limit: Optional[int] = None,
    status_updates: Optional[bool] = None,
    ws_ohlcv_timeout: Optional[Union[int, float]] = None,
    rest_ohlcv_timeout: Optional[Union[int, float]] = None,
    max_concurrent: Optional[int] = None,
    gecko_limit: Optional[int] = None,
) -> None:
    """Configure module-wide settings."""
    global OHLCV_TIMEOUT, MAX_OHLCV_FAILURES, MAX_WS_LIMIT, STATUS_UPDATES, SEMA, GECKO_SEMAPHORE
    if ohlcv_timeout is not None:
        try:
            val = max(1, int(ohlcv_timeout))
            OHLCV_TIMEOUT = val
            WS_OHLCV_TIMEOUT = val
            REST_OHLCV_TIMEOUT = val
        except (TypeError, ValueError):
            logger.warning(
                "Invalid ohlcv_timeout %s; using default %s",
                ohlcv_timeout,
                OHLCV_TIMEOUT,
            )
    if ws_ohlcv_timeout is not None:
        try:
            WS_OHLCV_TIMEOUT = max(1, int(ws_ohlcv_timeout))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid WS_OHLCV_TIMEOUT %s; using default %s",
                ws_ohlcv_timeout,
                WS_OHLCV_TIMEOUT,
            )
    if rest_ohlcv_timeout is not None:
        try:
            REST_OHLCV_TIMEOUT = max(1, int(rest_ohlcv_timeout))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid REST_OHLCV_TIMEOUT %s; using default %s",
                rest_ohlcv_timeout,
                REST_OHLCV_TIMEOUT,
            )
    if max_failures is not None:
        try:
            MAX_OHLCV_FAILURES = max(1, int(max_failures))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid MAX_OHLCV_FAILURES %s; using default %s",
                max_failures,
                MAX_OHLCV_FAILURES,
            )
    if max_ws_limit is None:
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
            cfg_val = cfg.get("max_ws_limit")
            if cfg_val is not None:
                max_ws_limit = cfg_val
        except Exception:
            pass
    if max_ws_limit is not None:
        try:
            MAX_WS_LIMIT = max(1, int(max_ws_limit))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid MAX_WS_LIMIT %s; using default %s",
                max_ws_limit,
                MAX_WS_LIMIT,
            )
    if status_updates is not None:
        STATUS_UPDATES = bool(status_updates)
    if max_concurrent is None:
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
            cfg_val = cfg.get("max_concurrent_ohlcv")
            if cfg_val is not None:
                max_concurrent = cfg_val
        except Exception:
            pass
    if max_concurrent is not None:
        try:
            val = int(max_concurrent)
            if val < 1:
                raise ValueError
            SEMA = asyncio.Semaphore(val)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid max_concurrent %s; disabling semaphore", max_concurrent
            )
            SEMA = None

    if gecko_limit is not None:
        try:
            val = int(gecko_limit)
            if val < 1:
                raise ValueError
            GECKO_SEMAPHORE = asyncio.Semaphore(val)
        except (TypeError, ValueError):
            logger.warning("Invalid gecko_limit %s; using default", gecko_limit)


def is_symbol_type(pair_info: dict, allowed: List[str]) -> bool:
    """Return ``True`` if ``pair_info`` matches one of the ``allowed`` types.

    The heuristic checks common CCXT fields like ``type`` and boolean flags
    (``spot``, ``future``, ``swap``) along with nested ``info`` metadata.  If no
    explicit type can be determined, a pair is treated as ``spot`` by default.
    """

    allowed_set = {t.lower() for t in allowed}

    market_type = str(pair_info.get("type", "")).lower()
    if market_type:
        return market_type in allowed_set

    for key in ("spot", "future", "swap", "option"):
        if pair_info.get(key) and key in allowed_set:
            return True

    info = pair_info.get("info", {}) or {}
    asset_class = str(info.get("assetClass", "")).lower()
    if asset_class:
        if asset_class in allowed_set:
            return True
        if asset_class in ("perpetual", "swap") and "swap" in allowed_set:
            return True
        if asset_class in ("future", "futures") and "future" in allowed_set:
            return True

    contract_type = str(info.get("contractType", "")).lower()
    if contract_type:
        if contract_type in allowed_set:
            return True
        if "perp" in contract_type and "swap" in allowed_set:
            return True

    # default to spot if no derivative hints are present
    if "spot" in allowed_set:
        derivative_keys = (
            "future",
            "swap",
            "option",
            "expiry",
            "contract",
            "settlement",
        )
        if not any(k in pair_info for k in derivative_keys) and not any(
            k in info for k in derivative_keys
        ):
            return True

    return False


def timeframe_seconds(exchange, timeframe: str) -> int:
    """Return timeframe length in seconds."""
    if hasattr(exchange, "parse_timeframe"):
        try:
            return int(exchange.parse_timeframe(timeframe))
        except Exception:
            pass
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    if unit == "w":
        return value * 604800
    if unit == "M":
        return value * 2592000
    raise ValueError(f"Unknown timeframe {timeframe}")


async def _call_with_retry(func, *args, timeout=None, **kwargs):
    """Call ``func`` with exponential back-off, adaptive rate limiting, and circuit breaker."""

    # Get the adaptive rate limiter and circuit breaker
    rate_limiter = get_rate_limiter()
    circuit_breaker = get_circuit_breaker()

    # Use circuit breaker to wrap the entire retry logic
    async def _execute_with_retry():
        # Use configuration values if available
        attempts = get_api_config_value('kraken.max_retries', 5)
        base_delay = get_api_config_value('kraken.base_retry_delay', 1.0)
        max_delay = get_api_config_value('kraken.max_retry_delay', 30.0)

        for attempt in range(attempts):
            try:
                # Wait for rate limiter before making the call
                await rate_limiter.wait_if_needed()

                if timeout is not None:
                    result = await asyncio.wait_for(
                        asyncio.shield(func(*args, **kwargs)), timeout
                    )
                else:
                    result = await func(*args, **kwargs)

                # Record successful call to adjust rate limiter
                rate_limiter.record_success()
                return result

            except asyncio.CancelledError:
                raise
            except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadRequest) as exc:
                # Record error to adjust rate limiter
                rate_limiter.record_error()

                # Handle various Kraken-specific errors
                error_msg = str(exc).lower()

                # Get retryable status codes and patterns from config
                retryable_status_codes = get_api_config_value('kraken.retryable_status_codes', [520, 522, 429, 400])
                retryable_patterns = get_api_config_value('kraken.retryable_error_patterns', RETRYABLE_ERROR_PATTERNS)

                # Special handling for mTLS certificate errors - treat as non-retryable
                is_mtls_error = "missing client certificate" in error_msg

                is_retryable_error = (
                    getattr(exc, "http_status", None) in retryable_status_codes or
                    (any(pattern in error_msg for pattern in retryable_patterns) and not is_mtls_error)
                )

                if is_retryable_error and attempt < attempts - 1:
                    # Check if circuit breaker is open
                    if circuit_breaker.state == "OPEN":
                        logger.warning("Circuit breaker is OPEN, skipping retry attempt")
                        raise exc

                    # Use adaptive delay instead of fixed exponential backoff
                    adaptive_delay = min(max_delay, rate_limiter.get_current_delay())
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{attempts}: {exc}. "
                        f"Retrying in {adaptive_delay:.1f}s (adaptive delay)"
                    )
                    await asyncio.sleep(adaptive_delay)
                    continue
                elif not is_retryable_error:
                    logger.error(f"Non-retryable error: {exc}")
                    raise
                else:
                    logger.error(f"Max retries ({attempts}) exceeded for: {exc}")
                    raise

    # Execute with circuit breaker protection
    return await circuit_breaker.async_call(_execute_with_retry)


async def load_kraken_symbols(
    exchange,
    exclude: Optional[Iterable[str]] = None,
    config: Optional[Dict] = None,
) -> Optional[List[str]]:
    """Return a list of active trading pairs on Kraken.

    Parameters
    ----------
    exchange : ccxt Exchange
        Exchange instance connected to Kraken.
    exclude : Iterable[str] | None
        Symbols to exclude from the result.
    """

    exclude_set = set(exclude or [])
    if config and "exchange_market_types" in config:
        allowed_types = set(config["exchange_market_types"])
    else:
        allowed_types = set(getattr(exchange, "exchange_market_types", []))
        if not allowed_types:
            allowed_types = {"spot"}

    markets = None
    if getattr(exchange, "has", {}).get("fetchMarketsByType"):
        fetcher = getattr(exchange, "fetch_markets_by_type", None) or getattr(
            exchange, "fetchMarketsByType", None
        )
        if fetcher:
            markets = {}
            for m_type in allowed_types:
                try:
                    if asyncio.iscoroutinefunction(fetcher):
                        fetched = await fetcher(m_type)
                    else:
                        fetched = await asyncio.to_thread(fetcher, m_type)
                except TypeError:
                    params = {"type": m_type}
                    if asyncio.iscoroutinefunction(fetcher):
                        fetched = await fetcher(params)
                    else:
                        fetched = await asyncio.to_thread(fetcher, params)
                except Exception as exc:  # pragma: no cover - safety
                    logger.warning("fetch_markets_by_type failed: %s", exc)
                    continue
                if isinstance(fetched, dict):
                    for sym, info in fetched.items():
                        info.setdefault("type", m_type)
                        markets[sym] = info
                elif isinstance(fetched, list):
                    for info in fetched:
                        sym = info.get("symbol")
                        if sym:
                            info.setdefault("type", m_type)
                            markets[sym] = info
    if markets is None:
        if asyncio.iscoroutinefunction(getattr(exchange, "load_markets", None)):
            markets = await exchange.load_markets()
        else:
            markets = await asyncio.to_thread(exchange.load_markets)

    df = pd.DataFrame.from_dict(markets, orient="index")
    df.index.name = "symbol"
    if "symbol" in df.columns:
        df.drop(columns=["symbol"], inplace=True)
    df.reset_index(inplace=True)

    df["active"] = df.get("active", True).fillna(True)
    df["reason"] = None
    df.loc[~df["active"], "reason"] = "inactive"

    mask_type = df.apply(lambda r: is_symbol_type(r.to_dict(), allowed_types), axis=1)
    df.loc[df["reason"].isna() & ~mask_type, "reason"] = (
        "type mismatch ("
        + df.get("type", "unknown").fillna("unknown").astype(str)
        + ")"
    )

    df.loc[df["reason"].isna() & df["symbol"].isin(exclude_set), "reason"] = "excluded"

    symbols: List[str] = []
    for row in df.itertuples():
        if row.reason:
            logger.debug("Skipping symbol %s: %s", row.symbol, row.reason)
        else:
            # Additional validation for Kraken symbols
            if getattr(exchange, "id", "").lower() == "kraken" and not is_valid_kraken_symbol(row.symbol):
                logger.debug("Skipping invalid Kraken symbol %s", row.symbol)
                continue
            # Normalize symbol before adding
            normalized_symbol = normalize_symbol(row.symbol)
            logger.debug("Including symbol %s (normalized: %s)", row.symbol, normalized_symbol)
            symbols.append(normalized_symbol)

    if not symbols:
        logger.warning("No active trading pairs were discovered")
        return None

    return symbols


async def fetch_ohlcv_async(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    since: Optional[int] = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
) -> Union[list, Exception]:
    """Return OHLCV data for ``symbol`` using async I/O."""

    if hasattr(exchange, "has") and not exchange.has.get("fetchOHLCV"):
        ex_id = getattr(exchange, "id", "unknown")
        logger.warning("Exchange %s lacks fetchOHLCV capability", ex_id)
        return []
    if getattr(exchange, "timeframes", None) and timeframe not in getattr(
        exchange, "timeframes", {}
    ):
        ex_id = getattr(exchange, "id", "unknown")
        logger.warning("Timeframe %s not supported on %s", timeframe, ex_id)
        return []

    if timeframe in ("4h", "1d"):
        use_websocket = False

    try:
        # Normalize the symbol before checking
        original_symbol = symbol
        normalized_symbol = normalize_symbol(symbol)

        # Check if we should use Kraken for this symbol
        use_kraken = should_use_kraken_for_symbol(symbol)
        exchange_id = getattr(exchange, "id", "unknown").lower()

        # If it's a Solana contract and we're on Kraken, use the normalized symbol
        if is_solana_contract_address(original_symbol) and exchange_id == "kraken":
            if use_kraken:
                symbol = normalized_symbol
                logger.debug("Mapped Solana contract %s to Kraken symbol %s", original_symbol, symbol)
            else:
                logger.warning(
                    "Skipping unsupported symbol %s on %s (no Kraken mapping available)",
                    original_symbol,
                    exchange_id,
                )
                failed_symbols[original_symbol] = {
                    "time": time.time(),
                    "delay": MAX_RETRY_DELAY,
                    "count": MAX_OHLCV_FAILURES,
                    "disabled": True,
                }
                return UNSUPPORTED_SYMBOL

        if hasattr(exchange, "symbols"):
            if not exchange.symbols and hasattr(exchange, "load_markets"):
                try:
                    if asyncio.iscoroutinefunction(
                        getattr(exchange, "load_markets", None)
                    ):
                        await exchange.load_markets()
                    else:
                        await asyncio.to_thread(exchange.load_markets)
                except Exception as exc:
                    logger.warning("load_markets failed: %s", exc)
            if exchange.symbols and symbol not in exchange.symbols:
                logger.warning(
                    "Skipping unsupported symbol %s on %s",
                    symbol,
                    exchange_id,
                )
                # Try Helius fallback for Solana tokens not supported by Kraken
                if is_solana_contract_address(original_symbol) and exchange_id == "kraken":
                    logger.debug("Trying Helius fallback for Solana token %s", original_symbol)
                    helius_data = await fetch_helius_ohlcv(original_symbol, timeframe=timeframe, limit=limit)
                    if helius_data:
                        logger.info("Successfully fetched %d candles for %s via Helius fallback", len(helius_data), original_symbol)
                        return helius_data

                failed_symbols[symbol] = {
                    "time": time.time(),
                    "delay": MAX_RETRY_DELAY,
                    "count": MAX_OHLCV_FAILURES,
                    "disabled": True,
                }
                return UNSUPPORTED_SYMBOL
        if (
            use_websocket
            and since is None
            and timeframe == "1m"
            and limit > MAX_WS_LIMIT
            and not force_websocket_history
        ):
            logger.info(
                "Skipping WebSocket OHLCV for %s limit %d exceeds %d",
                symbol,
                limit,
                MAX_WS_LIMIT,
            )
            use_websocket = False
            limit = min(limit, MAX_WS_LIMIT)
        if use_websocket and since is not None:
            try:
                seconds = timeframe_seconds(exchange, timeframe)
                candles_needed = int((time.time() - since) / seconds) + 1
                if candles_needed < limit:
                    limit = candles_needed
            except Exception:
                pass
        if use_websocket and hasattr(exchange, "watch_ohlcv"):
            params = inspect.signature(exchange.watch_ohlcv).parameters
            ws_limit = max(1, limit)  # Ensure minimum limit of 1
            kwargs = {"symbol": symbol, "timeframe": timeframe, "limit": ws_limit}
            if since is not None and "since" in params:
                kwargs["since"] = since
                tf_sec = timeframe_seconds(exchange, timeframe)
                try:
                    if since > 1e10:
                        now_ms = int(time.time() * 1000)
                        expected = max(0, (now_ms - since) // (tf_sec * 1000))
                        ws_limit = max(1, min(ws_limit, int(expected) + 2))
                    else:
                        expected = max(0, (time.time() - since) // tf_sec)
                        ws_limit = max(1, min(ws_limit, int(expected) + 1))
                    kwargs["limit"] = ws_limit
                except Exception:
                    # Fallback to safe limit if calculation fails
                    ws_limit = max(1, min(limit, 100))
                    kwargs["limit"] = ws_limit
            
            # Final safety check to ensure limit is positive
            if ws_limit <= 0:
                logger.warning(
                    "Invalid WebSocket limit %d for %s, falling back to REST",
                    ws_limit, symbol
                )
                use_websocket = False
                limit = min(limit, MAX_WS_LIMIT)
            else:
                kwargs["limit"] = ws_limit
            try:
                data = await _call_with_retry(
                    exchange.watch_ohlcv, timeout=WS_OHLCV_TIMEOUT, **kwargs
                )
            except asyncio.CancelledError:
                if hasattr(exchange, "close"):
                    if asyncio.iscoroutinefunction(getattr(exchange, "close")):
                        with contextlib.suppress(Exception):
                            await exchange.close()
                    else:
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(exchange.close)
                raise
            if ws_limit and len(data) < ws_limit and force_websocket_history:
                logger.warning(
                    "WebSocket OHLCV for %s %s returned %d of %d candles; disable force_websocket_history to allow REST fallback",
                    symbol,
                    timeframe,
                    len(data),
                    ws_limit,
                )
            if (
                ws_limit
                and len(data) < ws_limit
                and not force_websocket_history
                and hasattr(exchange, "fetch_ohlcv")
            ):
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                    params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                    kwargs_f = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": limit,
                    }
                    if since is not None and "since" in params_f:
                        kwargs_f["since"] = since
                    try:
                        data = await _call_with_retry(
                            exchange.fetch_ohlcv,
                            timeout=REST_OHLCV_TIMEOUT,
                            **kwargs_f,
                        )
                    except asyncio.CancelledError:
                        raise
                    expected = limit
                    if since is not None:
                        try:
                            tf_sec = timeframe_seconds(exchange, timeframe)
                            now_ms = int(time.time() * 1000)
                            expected = min(
                                limit, int((now_ms - since) // (tf_sec * 1000)) + 1
                            )
                        except Exception:
                            pass
                    if len(data) < expected:
                        # Only warn if we got significantly less data (less than 50% of expected)
                        if len(data) < expected * 0.5:
                            logger.warning(
                                "Incomplete OHLCV for %s: got %d of %d (significant data gap)",
                                symbol,
                                len(data),
                                expected,
                            )
                        else:
                            logger.debug(
                                "Incomplete OHLCV for %s: got %d of %d (minor data gap)",
                                symbol,
                                len(data),
                                expected,
                            )
                    return data
                params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
                if since is not None and "since" in params_f:
                    kwargs_f["since"] = since
                try:
                    data = await _call_with_retry(
                        asyncio.to_thread,
                        exchange.fetch_ohlcv,
                        **kwargs_f,
                        timeout=REST_OHLCV_TIMEOUT,
                    )
                except asyncio.CancelledError:
                    raise
                expected = limit
                if since is not None:
                    try:
                        tf_sec = timeframe_seconds(exchange, timeframe)
                        now_ms = int(time.time() * 1000)
                        expected = min(
                            limit, int((now_ms - since) // (tf_sec * 1000)) + 1
                        )
                    except Exception:
                        pass
                if len(data) < expected:
                    logger.warning(
                        "Incomplete OHLCV for %s: got %d of %d",
                        symbol,
                        len(data),
                        expected,
                    )
                return data
            expected = limit
            if since is not None:
                try:
                    tf_sec = timeframe_seconds(exchange, timeframe)
                    now_ms = int(time.time() * 1000)
                    expected = min(limit, int((now_ms - since) // (tf_sec * 1000)) + 1)
                except Exception:
                    pass
            if len(data) < expected:
                logger.warning(
                    "Incomplete OHLCV for %s: got %d of %d",
                    symbol,
                    len(data),
                    expected,
                )
                if since is not None and hasattr(exchange, "fetch_ohlcv"):
                    try:
                        kwargs_r = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "limit": limit,
                        }
                        if asyncio.iscoroutinefunction(
                            getattr(exchange, "fetch_ohlcv", None)
                        ):
                            try:
                                data_r = await _call_with_retry(
                                    exchange.fetch_ohlcv,
                                    timeout=REST_OHLCV_TIMEOUT,
                                    **kwargs_r,
                                )
                            except asyncio.CancelledError:
                                raise
                        else:
                            try:
                                data_r = await _call_with_retry(
                                    asyncio.to_thread,
                                    exchange.fetch_ohlcv,
                                    **kwargs_r,
                                    timeout=REST_OHLCV_TIMEOUT,
                                )
                            except asyncio.CancelledError:
                                raise
                        if len(data_r) > len(data):
                            data = data_r
                    except Exception:
                        pass
            return data
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
            params_f = inspect.signature(exchange.fetch_ohlcv).parameters
            kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
            if since is not None and "since" in params_f:
                kwargs_f["since"] = since
            try:
                data = await _call_with_retry(
                    exchange.fetch_ohlcv,
                    timeout=REST_OHLCV_TIMEOUT,
                    **kwargs_f,
                )
            except asyncio.CancelledError:
                raise
            expected = limit
            if since is not None:
                try:
                    tf_sec = timeframe_seconds(exchange, timeframe)
                    now_ms = int(time.time() * 1000)
                    expected = min(limit, int((now_ms - since) // (tf_sec * 1000)) + 1)
                except Exception:
                    pass
            if len(data) < expected:
                logger.warning(
                    "Incomplete OHLCV for %s: got %d of %d",
                    symbol,
                    len(data),
                    expected,
                )
            if since is not None:
                try:
                    kwargs_r = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": limit,
                    }
                    try:
                        data_r = await _call_with_retry(
                            exchange.fetch_ohlcv,
                            timeout=REST_OHLCV_TIMEOUT,
                            **kwargs_r,
                        )
                    except asyncio.CancelledError:
                        raise
                    if len(data_r) > len(data):
                        data = data_r
                except Exception:
                    pass
            return data
        params_f = inspect.signature(exchange.fetch_ohlcv).parameters
        kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
        if since is not None and "since" in params_f:
            kwargs_f["since"] = since
        try:
            data = await _call_with_retry(
                asyncio.to_thread,
                exchange.fetch_ohlcv,
                **kwargs_f,
                timeout=REST_OHLCV_TIMEOUT,
            )
        except asyncio.CancelledError:
            raise
        expected = limit
        if since is not None:
            try:
                tf_sec = timeframe_seconds(exchange, timeframe)
                now_ms = int(time.time() * 1000)
                expected = min(limit, int((now_ms - since) // (tf_sec * 1000)) + 1)
            except Exception:
                pass
        if len(data) < expected:
            logger.warning(
                "Incomplete OHLCV for %s: got %d of %d",
                symbol,
                len(data),
                expected,
            )
            if since is not None:
                try:
                    kwargs_r = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": limit,
                    }
                    try:
                        data_r = await _call_with_retry(
                            asyncio.to_thread,
                            exchange.fetch_ohlcv,
                            **kwargs_r,
                            timeout=REST_OHLCV_TIMEOUT,
                        )
                    except asyncio.CancelledError:
                        raise
                    if len(data_r) > len(data):
                        data = data_r
                except Exception:
                    pass
        return data
    except asyncio.TimeoutError as exc:
        ex_id = getattr(exchange, "id", "unknown")
        if use_websocket and hasattr(exchange, "watch_ohlcv"):
            logger.error(
                "WS OHLCV timeout for %s on %s (tf=%s limit=%s ws=%s): %s",
                symbol,
                ex_id,
                timeframe,
                limit,
                use_websocket,
                exc,
                exc_info=False,
            )
        else:
            logger.error(
                "REST OHLCV timeout for %s on %s (tf=%s limit=%s ws=%s): %s",
                symbol,
                ex_id,
                timeframe,
                limit,
                use_websocket,
                exc,
                exc_info=False,
            )
        if use_websocket and hasattr(exchange, "fetch_ohlcv"):
            logger.info(
                "Falling back to REST fetch_ohlcv for %s on %s limit %d",
                symbol,
                timeframe,
                limit,
            )
            try:
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                    params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                    kwargs_f = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": limit,
                    }
                    if since is not None and "since" in params_f:
                        kwargs_f["since"] = since
                    try:
                        return await _call_with_retry(
                            exchange.fetch_ohlcv,
                            timeout=REST_OHLCV_TIMEOUT,
                            **kwargs_f,
                        )
                    except asyncio.CancelledError:
                        raise
                params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
                if since is not None and "since" in params_f:
                    kwargs_f["since"] = since
                try:
                    return await _call_with_retry(
                        asyncio.to_thread,
                        exchange.fetch_ohlcv,
                        **kwargs_f,
                        timeout=REST_OHLCV_TIMEOUT,
                    )
                except asyncio.CancelledError:
                    raise
            except Exception as exc2:  # pragma: no cover - fallback
                ex_id = getattr(exchange, "id", "unknown")
                logger.error(
                    "REST fallback fetch_ohlcv failed for %s on %s (tf=%s limit=%s ws=%s): %s",
                    symbol,
                    ex_id,
                    timeframe,
                    limit,
                    use_websocket,
                    exc2,
                    exc_info=True,
                )
        return exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - network
        if (
            use_websocket
            and hasattr(exchange, "fetch_ohlcv")
            and not force_websocket_history
        ):
            try:
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                    params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                    kwargs_f = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "limit": limit,
                    }
                    if since is not None and "since" in params_f:
                        kwargs_f["since"] = since
                    try:
                        return await _call_with_retry(
                            exchange.fetch_ohlcv,
                            timeout=REST_OHLCV_TIMEOUT,
                            **kwargs_f,
                        )
                    except asyncio.CancelledError:
                        raise
                params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
                if since is not None and "since" in params_f:
                    kwargs_f["since"] = since
                try:
                    return await _call_with_retry(
                        asyncio.to_thread,
                        exchange.fetch_ohlcv,
                        **kwargs_f,
                        timeout=REST_OHLCV_TIMEOUT,
                    )
                except asyncio.CancelledError:
                    raise
            except Exception:
                pass
        return exc


async def fetch_dexscreener_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
) -> Optional[list]:
    """Deprecated: use :func:`fetch_geckoterminal_ohlcv` instead."""

    warnings.warn(
        "fetch_dexscreener_ohlcv is deprecated; use fetch_geckoterminal_ohlcv",
        DeprecationWarning,
        stacklevel=2,
    )
    return await fetch_geckoterminal_ohlcv(symbol, timeframe=timeframe, limit=limit)


async def fetch_geckoterminal_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    *,
    min_24h_volume: float = 0.0,
    return_price: bool = False,
) -> Optional[Union[Tuple[List, float], Tuple[List, float, float]]]:
    """Return OHLCV data and 24h volume for ``symbol`` from GeckoTerminal.

    When ``return_price`` is ``True`` the pool price is returned instead of the
    reserve liquidity value.
    """

    from urllib.parse import quote_plus

    # Validate symbol before making any requests
    try:
        token_mint, quote = symbol.split("/", 1)
    except ValueError:
        token_mint, quote = symbol, ""
    if quote != "USDC":
        return None
    if not _is_valid_base_token(token_mint):
        return None

    volume = 0.0
    reserve = 0.0
    price = 0.0
    data = {}
    is_cached = False

    # Use semaphore to limit concurrent API calls
    async with GECKO_SEMAPHORE:
        backoff = 1
        for attempt in range(3):
            cached = GECKO_POOL_CACHE.get(symbol)
            is_cached = cached is not None and cached[4] == limit
            try:
                async with aiohttp.ClientSession() as session:
                    if cached is None:
                        query = quote_plus(symbol)
                        search_url = (
                            "https://api.geckoterminal.com/api/v2/search/pools"
                            f"?query={query}&network=solana"
                        )

                        async with session.get(search_url, timeout=10) as resp:
                            if resp.status == 404:
                                logger.info(
                                    "pair not available on GeckoTerminal: %s", symbol
                                )
                                return None
                            resp.raise_for_status()
                            search_data = await resp.json()

                        items = search_data.get("data") or []
                        if not items:
                            logger.info("pair not available on GeckoTerminal: %s", symbol)
                            return None

                        first = items[0]
                        attrs = (
                            first.get("attributes", {}) if isinstance(first, dict) else {}
                        )

                        pool_id = str(first.get("id", ""))
                        pool_addr = pool_id.split("_", 1)[-1]
                        try:
                            volume = float(attrs.get("volume_usd", {}).get("h24", 0.0))
                        except Exception:
                            volume = 0.0
                        if volume < float(min_24h_volume):
                            return None
                        try:
                            price = float(attrs.get("base_token_price_quote_token", 0.0))
                        except Exception:
                            price = 0.0
                        try:
                            reserve = float(attrs.get("reserve_in_usd", 0.0))
                        except Exception:
                            reserve = 0.0

                        GECKO_POOL_CACHE[symbol] = (
                            pool_addr,
                            volume,
                            reserve,
                            price,
                            limit,
                        )
                    else:
                        pool_addr, volume, reserve, price, _ = cached

                    ohlcv_url = (
                        "https://api.geckoterminal.com/api/v2/networks/solana/pools/"
                        f"{pool_addr}/ohlcv/{timeframe}?aggregate=1&limit={limit}"
                    )

                    async with session.get(ohlcv_url, timeout=10) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                break
            except Exception as exc:  # pragma: no cover - network
                if attempt == 2:
                    logger.error("GeckoTerminal OHLCV error for %s: %s", symbol, exc)
                    return None
                await asyncio.sleep(backoff)
                backoff *= 2

    candles = (data.get("data") or {}).get("attributes", {}).get("ohlcv_list") or []

    result: list = []
    multiplier = 1000 if is_cached else 1
    for c in candles[-limit:]:
        try:
            result.append(
                [
                    int(c[0]) * multiplier,
                    float(c[1]),
                    float(c[2]),
                    float(c[3]),
                    float(c[4]),
                    float(c[5]),
                ]
            )
        except Exception:
            continue

    if return_price:
        return result, volume, price
    return result, volume, reserve


async def fetch_coingecko_ohlc(
    coin_id: str,
    timeframe: str = "1h",
    limit: int = 100,
) -> Optional[list]:
    """Return OHLC data from CoinGecko as [timestamp, open, high, low, close, 0]."""

    days = 1
    if timeframe.endswith("d"):
        days = 90
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception:  # pragma: no cover - network
        return None

    result: list = []
    for c in data[-limit:]:
        if not isinstance(c, list) or len(c) < 5:
            continue
        try:
            ts, o, h, l, cl = c[:5]
            result.append([int(ts), float(o), float(h), float(l), float(cl), 0.0])
        except Exception:
            continue
    return result


async def fetch_helius_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
) -> Optional[list]:
    """Fetch OHLCV data for Solana tokens using Helius API.

    This is used as a fallback for tokens not supported by Kraken.
    """
    import os

    # Check if we have a Helius API key
    helius_key = os.getenv("HELIUS_KEY")
    if not helius_key:
        logger.debug("No HELIUS_KEY found, skipping Helius OHLCV fetch")
        return None

    # Extract token mint address from symbol
    if "/" in symbol:
        base_token = symbol.split("/")[0]
    else:
        base_token = symbol

    # Check if it's a valid Solana token address
    if not _is_valid_base_token(base_token):
        logger.debug("Invalid Solana token address for Helius: %s", base_token)
        return None

    try:
        # Use GeckoTerminal as primary source for Solana tokens via Helius
        res = await fetch_geckoterminal_ohlcv(
            f"{base_token}/USDC",
            timeframe=timeframe,
            limit=limit,
            min_24h_volume=0.0  # Accept any volume for now
        )

        if res:
            if isinstance(res, tuple):
                data, vol = res
                if data and len(data) > 0:
                    logger.debug("Fetched %d candles for %s via Helius/GeckoTerminal", len(data), symbol)
                    return data
            elif isinstance(res, list) and len(res) > 0:
                logger.debug("Fetched %d candles for %s via Helius/GeckoTerminal", len(res), symbol)
                return res

    except Exception as exc:
        logger.debug("Helius OHLCV fetch failed for %s: %s", symbol, exc)

    return None


async def fetch_dex_ohlcv(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    *,
    min_volume_usd: float = 0.0,
    gecko_res: Optional[Union[list, tuple]] = None,
    use_gecko: bool = True,
) -> Optional[list]:
    """Fetch DEX OHLCV with fallback to CoinGecko, Coinbase then Kraken."""

    # Guard against None symbol values
    if symbol is None or not isinstance(symbol, str):
        logger.warning("Invalid symbol passed to fetch_dex_ohlcv: %s", symbol)
        return None

    res = gecko_res
    if res is None and use_gecko:
        try:
            res = await fetch_geckoterminal_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:  # pragma: no cover - network
            logger.error("GeckoTerminal OHLCV error for %s: %s", symbol, exc)
            res = None

    data = None
    if res:
        if isinstance(res, tuple):
            data, vol = res
        else:
            data = res
            vol = min_volume_usd
        if data and vol >= min_volume_usd:
            return data

    # Try Helius for Solana tokens
    if is_solana_contract_address(symbol.split("/")[0] if "/" in symbol else symbol):
        helius_data = await fetch_helius_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if helius_data:
            return helius_data

    base, _, quote = symbol.partition("/")
    coin_id = COINGECKO_IDS.get(base)
    if coin_id:
        data = await fetch_coingecko_ohlc(coin_id, timeframe=timeframe, limit=limit)
        if data:
            return data

    if quote.upper() in SUPPORTED_USD_QUOTES:
        try:
            cb = ccxt.coinbase({"enableRateLimit": True})
            data = await fetch_ohlcv_async(cb, symbol, timeframe=timeframe, limit=limit)
        finally:
            close = getattr(cb, "close", None)
            if close:
                try:
                    if asyncio.iscoroutinefunction(close):
                        await close()
                    else:
                        close()
                except Exception:
                    pass
        if data and not isinstance(data, Exception):
            return data

    data = await fetch_ohlcv_async(exchange, symbol, timeframe=timeframe, limit=limit)
    if isinstance(data, Exception):
        return None
    return data


async def fetch_order_book_async(
    exchange,
    symbol: str,
    depth: int = 2,
) -> Union[dict, Exception]:
    """Return order book snapshot for ``symbol`` with top ``depth`` levels."""

    if hasattr(exchange, "has") and not exchange.has.get("fetchOrderBook"):
        return {}

    try:
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_order_book", None)):
            return await asyncio.wait_for(
                exchange.fetch_order_book(symbol, limit=depth), OHLCV_TIMEOUT
            )
        return await asyncio.wait_for(
            asyncio.to_thread(exchange.fetch_order_book, symbol, depth),
            OHLCV_TIMEOUT,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - network
        return exc


async def load_ohlcv_parallel(
    exchange,
    symbols: Iterable[str],
    timeframe: str = "1h",
    limit: int = 100,
    since_map: Optional[Dict[str, int]] = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    max_concurrent: Optional[int] = None,
    notifier: Optional[TelegramNotifier] = None,
) -> Dict[str, list]:
    """Fetch OHLCV data for multiple symbols concurrently.

    Parameters
    ----------
    notifier : Optional[TelegramNotifier], optional
        If provided, failures will be sent using this notifier.
    """

    since_map = since_map or {}

    # Filter out None values to prevent downstream errors
    valid_symbols = [s for s in symbols if s is not None and isinstance(s, str)]

    now = time.time()
    filtered_symbols: List[str] = []
    for s in valid_symbols:
        info = failed_symbols.get(s)
        if not info:
            filtered_symbols.append(s)
            continue
        if info.get("disabled"):
            continue
        if now - info["time"] >= info["delay"]:
            filtered_symbols.append(s)
    symbols = filtered_symbols

    if not symbols:
        return {}

    if max_concurrent is not None:
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer or None")
        sem = asyncio.Semaphore(max_concurrent)
    elif SEMA is not None:
        sem = SEMA
    else:
        sem = None

    async def sem_fetch(sym: str):
        async def _fetch_and_sleep():
            data = await fetch_ohlcv_async(
                exchange,
                sym,
                timeframe=timeframe,
                limit=limit,
                since=since_map.get(sym),
                use_websocket=use_websocket,
                force_websocket_history=force_websocket_history,
            )
            rl = getattr(exchange, "rateLimit", None)
            if rl:
                await asyncio.sleep(rl / 1000)
            return data

        if sem:
            async with sem:
                return await _fetch_and_sleep()

        return await _fetch_and_sleep()

    # Ensure symbols is filtered again in case of None values
    symbols = [s for s in symbols if s is not None and isinstance(s, str)]
    tasks = [asyncio.create_task(sem_fetch(s)) for s in symbols]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    if any(isinstance(r, asyncio.CancelledError) for r in results):
        for t in tasks:
            if not t.done():
                t.cancel()
        raise asyncio.CancelledError()

    data: Dict[str, list] = {}
    ex_id = getattr(exchange, "id", "unknown")
    mode = "websocket" if use_websocket else "REST"
    for sym, res in zip(symbols, results):
        if res is UNSUPPORTED_SYMBOL:
            continue
        if isinstance(res, asyncio.CancelledError):
            raise res
        if isinstance(res, asyncio.TimeoutError):
            logger.error(
                "Timeout loading OHLCV for %s on %s limit %d: %s",
                sym,
                timeframe,
                limit,
                res,
                exc_info=True,
            )
            msg = (
                f"Timeout loading OHLCV for {sym} on {ex_id} "
                f"(tf={timeframe} limit={limit} mode={mode})"
            )
            logger.error(msg)
            # Only send notification for critical timeouts, not individual symbol failures
            # This prevents flooding the Telegram with error messages
            pass
            info = failed_symbols.get(sym)
            delay = RETRY_DELAY
            count = 1
            disabled = False
            if info is not None:
                delay = min(info["delay"] * 2, MAX_RETRY_DELAY)
                count = info.get("count", 0) + 1
                disabled = info.get("disabled", False)
            if count >= MAX_OHLCV_FAILURES:
                disabled = True
                if not info or not info.get("disabled"):
                    logger.info("Disabling %s after %d OHLCV failures", sym, count)
            failed_symbols[sym] = {
                "time": time.time(),
                "delay": delay,
                "count": count,
                "disabled": disabled,
            }
            continue
        if (
            isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError)
        ) or not res:
            logger.error(
                "Failed to load OHLCV for %s on %s limit %d: %s",
                sym,
                timeframe,
                limit,
                res,
                exc_info=isinstance(res, Exception),
            )
            msg = (
                f"Failed to load OHLCV for {sym} on {ex_id} "
                f"(tf={timeframe} limit={limit} mode={mode}): {res}"
            )
            logger.error(msg)
            # Only send notification for critical failures, not individual symbol failures
            # This prevents flooding the Telegram with error messages
            pass
            info = failed_symbols.get(sym)
            delay = RETRY_DELAY
            count = 1
            disabled = False
            if info is not None:
                delay = min(info["delay"] * 2, MAX_RETRY_DELAY)
                count = info.get("count", 0) + 1
                disabled = info.get("disabled", False)
            if count >= MAX_OHLCV_FAILURES:
                disabled = True
                if not info or not info.get("disabled"):
                    logger.info("Disabling %s after %d OHLCV failures", sym, count)
            failed_symbols[sym] = {
                "time": time.time(),
                "delay": delay,
                "count": count,
                "disabled": disabled,
            }
            continue
        if res and len(res[0]) > 6:
            res = [[c[0], c[1], c[2], c[3], c[4], c[6]] for c in res]
        data[sym] = res
        failed_symbols.pop(sym, None)
    return data


async def update_ohlcv_cache(
    exchange,
    cache: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
    timeframe: str = "1h",
    limit: int = 100,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    config: Optional[Dict] = None,
    max_concurrent: Optional[int] = None,
    notifier: Optional[TelegramNotifier] = None,
) -> Dict[str, pd.DataFrame]:
    """Update cached OHLCV DataFrames with new candles.

    Parameters
    ----------
    max_concurrent : Optional[int], optional
        Maximum number of concurrent OHLCV requests. ``None`` means no limit.
    """

    from crypto_bot.regime.regime_classifier import clear_regime_cache

    # Ensure we always request a reasonable number of candles
    limit = max(limit, 200)

    if max_concurrent is not None:
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer or None")

    global _last_snapshot_time
    config = config or {}
    snapshot_interval = config.get("ohlcv_snapshot_frequency_minutes", 1440) * 60
    now = time.time()
    snapshot_due = now - _last_snapshot_time >= snapshot_interval

    logger.info("Starting OHLCV update for timeframe %s", timeframe)

    # Filter out None values to prevent downstream errors
    symbols = [s for s in symbols if s is not None and isinstance(s, str)]

    since_map: Dict[str, Optional[int]] = {}
    if snapshot_due:
        _last_snapshot_time = now
        limit = max(config.get("ohlcv_snapshot_limit", limit), 200)
        since_map = {sym: None for sym in symbols}
    else:
        for sym in symbols:
            df = cache.get(sym)
            if df is not None and not df.empty:
                since_map[sym] = int(df["timestamp"].iloc[-1]) + 1
    now = time.time()
    filtered_symbols: List[str] = []
    for s in symbols:
        info = failed_symbols.get(s)
        if not info:
            filtered_symbols.append(s)
            continue
        if info.get("disabled"):
            continue
        if now - info["time"] >= info["delay"]:
            filtered_symbols.append(s)
    symbols = filtered_symbols
    if not symbols:
        return cache

    logger.info(
        "Fetching %d candles for %d symbols on %s",
        limit,
        len(symbols),
        timeframe,
    )

    data_map = await load_ohlcv_parallel(
        exchange,
        symbols,
        timeframe,
        limit,
        since_map,
        use_websocket,
        force_websocket_history,
        max_concurrent,
        notifier,
    )

    logger.info(
        "Fetched OHLCV for %d/%d symbols on %s",
        len([s for s in symbols if s in data_map]),
        len(symbols),
        timeframe,
    )

    for sym in symbols:
        data = data_map.get(sym)
        if not data:
            info = failed_symbols.get(sym)
            skip_retry = (
                info is not None
                and time.time() - info["time"] < info["delay"]
                and since_map.get(sym) is None
            )
            if skip_retry:
                continue
            failed_symbols.pop(sym, None)
            full = await load_ohlcv_parallel(
                exchange,
                [sym],
                timeframe,
                limit,
                None,
                use_websocket,
                force_websocket_history,
                max_concurrent,
                notifier,
            )
            data = full.get(sym)
            if data:
                failed_symbols.pop(sym, None)
        if data is None:
            # Skip this symbol but don't break the cache structure
            continue

        # Ensure data is a list of lists before creating DataFrame
        if not isinstance(data, list) or not data:
            logger.warning(f"Invalid data format for {sym}: {type(data)}")
            continue

        try:
            df_new = pd.DataFrame(
                data, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        except Exception as e:
            logger.error(f"Failed to create DataFrame for {sym}: {e}")
            continue
        min_candles_required = int(limit * 0.5)
        if len(df_new) < min_candles_required:
            since_val = since_map.get(sym)
            retry = await load_ohlcv_parallel(
                exchange,
                [sym],
                timeframe,
                limit,
                {sym: since_val},
                False,
                force_websocket_history,
                max_concurrent,
                notifier,
            )
            retry_data = retry.get(sym)
            if retry_data and len(retry_data) > len(data):
                data = retry_data
                df_new = pd.DataFrame(
                    data,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
            if len(df_new) < min_candles_required:
                logger.info(
                    "Incomplete data for %s: %d/%d candles (accumulating in background)",
                    sym,
                    len(df_new),
                    limit,
                )
                # Don't skip - allow incomplete data to be cached for gradual accumulation
        changed = False
        if sym in cache and not cache[sym].empty:
            last_ts = cache[sym]["timestamp"].iloc[-1]
            df_new = df_new[df_new["timestamp"] > last_ts]
            if df_new.empty:
                continue
            cache[sym] = pd.concat([cache[sym], df_new], ignore_index=True)
            changed = True
        else:
            cache[sym] = df_new
            changed = True
        if changed:
            cache[sym]["return"] = cache[sym]["close"].pct_change()
            clear_regime_cache(sym, timeframe)
    logger.info("Completed OHLCV update for timeframe %s", timeframe)
    return cache


async def update_multi_tf_ohlcv_cache(
    exchange,
    cache: Dict[str, Dict[str, pd.DataFrame]],
    symbols: Iterable[str],
    config: Dict,
    limit: int = 100,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    max_concurrent: Optional[int] = None,
    notifier: Optional[TelegramNotifier] = None,
    priority_queue: Optional[Deque[str]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Update OHLCV caches for multiple timeframes.

    Parameters
    ----------
    config : Dict
        Configuration containing a ``timeframes`` list.
    """
    from crypto_bot.regime.regime_classifier import clear_regime_cache

    limit = max(limit, 200)

    def add_priority(data: list, symbol: str) -> None:
        """Push ``symbol`` to ``priority_queue`` if volume spike detected."""
        if priority_queue is None or vol_thresh is None or not data:
            return
        try:
            vols = np.array([row[5] for row in data], dtype=float)
            mean = float(np.mean(vols)) if len(vols) else 0.0
            std = float(np.std(vols))
            if std <= 0:
                return
            z_max = float(np.max((vols - mean) / std))
            if z_max > vol_thresh:
                priority_queue.appendleft(symbol)
        except Exception:
            return

    tfs = config.get("timeframes", ["1h"])
    logger.info("Updating OHLCV cache for timeframes: %s", tfs)

    min_volume_usd = float(config.get("min_volume_usd", 0) or 0)
    vol_thresh = config.get("bounce_scalper", {}).get("vol_zscore_threshold")

    for tf in tfs:
        logger.info("Starting update for timeframe %s", tf)
        tf_cache = cache.get(tf, {})

        # Filter out None values to prevent partition errors
        valid_symbols = [s for s in symbols if s is not None and isinstance(s, str)]

        cex_symbols: List[str] = []
        dex_symbols: List[str] = []
        for s in valid_symbols:
            base, _, quote = s.partition("/")
            if quote.upper() == "USDC" and _is_valid_base_token(base):
                dex_symbols.append(s)
            else:
                cex_symbols.append(s)

        if cex_symbols:
            tf_cache = await update_ohlcv_cache(
                exchange,
                tf_cache,
                cex_symbols,
                timeframe=tf,
                limit=limit,
                use_websocket=use_websocket,
                force_websocket_history=force_websocket_history,
                max_concurrent=max_concurrent,
                notifier=notifier,
            )

        for sym in dex_symbols:
            data = None
            vol = 0.0
            try:
                res = await fetch_geckoterminal_ohlcv(
                    sym,
                    timeframe=tf,
                    limit=limit,
                    min_24h_volume=min_volume_usd,
                )
            except Exception as exc:  # pragma: no cover - network
                logger.error("GeckoTerminal OHLCV error for %s: %s", sym, exc)
                res = None

            if res:
                if isinstance(res, tuple):
                    data, vol, *_ = res
                else:
                    data = res
                    vol = min_volume_usd
                add_priority(data, sym)

            if not data or vol < min_volume_usd:
                data = await fetch_dex_ohlcv(
                    exchange,
                    sym,
                    timeframe=tf,
                    limit=limit,
                    min_volume_usd=min_volume_usd,
                    gecko_res=res,
                    use_gecko=False,
                )
                if isinstance(data, Exception) or not data:
                    continue

            if not data:
                continue

            df_new = pd.DataFrame(
                data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            changed = False
            if sym in tf_cache and not tf_cache[sym].empty:
                last_ts = tf_cache[sym]["timestamp"].iloc[-1]
                df_new = df_new[df_new["timestamp"] > last_ts]
                if df_new.empty:
                    continue
                tf_cache[sym] = pd.concat([tf_cache[sym], df_new], ignore_index=True)
                changed = True
            else:
                tf_cache[sym] = df_new
                changed = True
            if changed:
                tf_cache[sym]["return"] = tf_cache[sym]["close"].pct_change()
                clear_regime_cache(sym, tf)

        cache[tf] = tf_cache
        logger.info("Finished update for timeframe %s", tf)

    return cache


async def update_regime_tf_cache(
    exchange,
    cache: Dict[str, Dict[str, pd.DataFrame]],
    symbols: Iterable[str],
    config: Dict,
    limit: int = 100,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    max_concurrent: Optional[int] = None,
    notifier: Optional[TelegramNotifier] = None,
    df_map: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Update OHLCV caches for regime detection timeframes."""
    limit = max(limit, 200)
    regime_cfg = {**config, "timeframes": config.get("regime_timeframes", [])}
    tfs = regime_cfg["timeframes"]
    logger.info("Updating regime cache for timeframes: %s", tfs)

    missing_tfs: List[str] = []
    if df_map is not None:
        for tf in tfs:
            tf_data = df_map.get(tf)
            if tf_data is None:
                missing_tfs.append(tf)
                continue
            tf_cache = cache.setdefault(tf, {})
            for sym in symbols:
                df = tf_data.get(sym)
                if df is not None:
                    tf_cache[sym] = df
            cache[tf] = tf_cache
    else:
        missing_tfs = tfs

    if missing_tfs:
        fetch_cfg = {**regime_cfg, "timeframes": missing_tfs}
        cache = await update_multi_tf_ohlcv_cache(
            exchange,
            cache,
            symbols,
            fetch_cfg,
            limit=limit,
            use_websocket=use_websocket,
            force_websocket_history=force_websocket_history,
            max_concurrent=max_concurrent,
            notifier=notifier,
            priority_queue=None,
        )

    return cache

# Add this after the imports section, around line 20

# Invalid Kraken symbols that cause API errors
INVALID_KRAKEN_SYMBOLS = {
    'FXS/USD', 'WIF/USD', 'GAIA/USD', 'GAIA/EUR', 'CRV/USD',
    'FWOG/USD', 'FWOG/EUR', 'FORTH/USD', 'IP/USD', 'AI16Z/USD',
    'SAPIEN/USD', 'CFG/USD', 'PYTH/USD', 'BONK/USD', 'CRO/USDT',
    'CRO/USD', 'CRO/EUR', 'BCH/USD', 'BCH/EUR', 'ARB/USD', 'ARB/EUR',
    'ADA/USD', 'AAVE/USD', 'SOL/USD'
}

# Mapping of Solana contract addresses to standard symbols
SOLANA_CONTRACT_TO_SYMBOL = {
    'So11111111111111111111111111111111111111112': 'SOL',
    'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v': 'USDC',
    'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB': 'USDT',
    '2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo': 'PYTH',
    'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm': 'WIF',
    '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr': 'BONK',
    'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn': 'JTO',
    'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN': 'JUP',
    'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263': 'DBR',
    '27G8MtK7VtTcCHkpASjSDdkWWYfoqT6ggEuKidVJidD4': 'RAY',
    '5LafQUrVco6o7KMz42eqVEJ9LW31StPyGjeeu5sKoMtA': 'HBAR',
    'jupSoLaHXQiZZTSfEWMTRRgpnyFm8f6sZdosWBjx93v': 'JUPSOL',
    'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So': 'MSOL',
    '5mbK36SZ7J19An8jFochhQS4of8g6BwUjbeCSxBSoWdp': 'ORCA',
    'hntyVP6YFm1Hg25TN9WGLqM12b8TQmcknKrdu1oxWux': 'HNT',
    'A7GJgPaRgLR9M7DjXnX78Ab2PWQ5rZhtLdj2qGAnZnZa': 'ATLAS',
    'ukHH6c7mMyiWCf1b9pnWe25TSpkDDt3H5pQZgZ74J82': 'UKT',
    'MEW1gQWJ3nEXg2qgERiKu7FAFj79PHvQVREQUzScPP5': 'MEW',
    '24gG4br5xFBRmxdqpgirtxgcr7BaWoErQfc2uyDp2Qhh': 'HONEY',
    'DtR4D9FtVoTX2569gaL837ZgrB6wNjj6tkmnX9Rdk9B2': 'BONK',
    'CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump': 'PUMP',
    '25hAyBQfoDhfWx9ay6rarbgvWGwDdNqcHsXS3jQ3mTDJ': 'BONK',
    'KENJSUYLASHUMfHyy5o4Hp2FdNqZg1AsUPhfH2kYvEP': 'KEN',
    'J1Wpmugrooj1yMyQKrdZ2vwRXG5rhfx3vTnYE39gpump': 'PUMP',
    '6NspJqVFceCiU5D1YgVq7waYoC394Vhqxwg7cSJdFtVE': 'BONK',
    'J3NKxxXZcnNiMjKw9hYb2K4LUxgwB6t1FtPtQVsv3KFr': 'JUP',
    '9PR7nCP9DpcUotnDPVLUBUZKu5WAYkwrCUx9wDnSpump': 'PUMP',
    'CJMihkPYswa3k6az9SUbepjKnkJQ6KWpGG5p9qW9n7NV': 'JUP',
    'HgBRWfYxEfvPhtqkaeymCQtHCrKE46qQ43pKe8HCpump': 'PUMP',
    '7BgBvyjrZX1YKz4oh9mjb8ZScatkkwb8DzFx7LoiVkM3': 'MEW',
    'WskzsKqEW3ZsmrhPAevfVZb6PuuLzWov9mJWZsfDePC': 'WSK',
}

# Kraken-supported symbols that we can map Solana contracts to
KRAKEN_SUPPORTED_SOLANA_SYMBOLS = {
    'SOL/USD', 'SOL/EUR', 'SOL/USDT',
    'USDC/USD', 'USDC/EUR', 'USDC/USDT',
    'USDT/USD', 'USDT/EUR',
    'BONK/USD', 'BONK/EUR',
    'WIF/USD', 'WIF/EUR',
    'JUP/USD', 'JUP/EUR',
    'PYTH/USD', 'PYTH/EUR',
    'RAY/USD', 'RAY/EUR',
    'HBAR/USD', 'HBAR/EUR',
    'HNT/USD', 'HNT/EUR',
}

def map_solana_contract_to_symbol(contract_address: str) -> Optional[str]:
    """Map a Solana contract address to its standard symbol."""
    return SOLANA_CONTRACT_TO_SYMBOL.get(contract_address)


def is_solana_contract_address(symbol: str) -> bool:
    """Check if a symbol is a Solana contract address."""
    if not symbol or not isinstance(symbol, str):
        return False
    return _is_valid_base_token(symbol)


def normalize_symbol(symbol: str) -> str:
    """Normalize a symbol by mapping Solana contracts to standard symbols."""
    if not symbol or not isinstance(symbol, str):
        return symbol

    # If it's a Solana contract address, map it to standard symbol
    if is_solana_contract_address(symbol):
        mapped_symbol = map_solana_contract_to_symbol(symbol)
        if mapped_symbol:
            # Return the mapped symbol with USD as default quote
            return f"{mapped_symbol}/USD"

    # If it's already in symbol format (base/quote), return as-is
    if '/' in symbol:
        return symbol

    # If it's just a base symbol, add USD quote
    return f"{symbol}/USD"


def is_valid_kraken_symbol(symbol: str) -> bool:
    """Check if a symbol is valid for Kraken API."""
    if not symbol or not isinstance(symbol, str):
        return False

    # Check against known invalid symbols
    if symbol in INVALID_KRAKEN_SYMBOLS:
        return False

    # Basic format validation
    if '/' not in symbol:
        return False

    base, quote = symbol.split('/', 1)
    if not base or not quote:
        return False

    # Check for common invalid patterns
    invalid_patterns = [
        'USDUSD', 'USDTUSD', 'EURUSD', 'USDEUR',
        'BTCBTC', 'ETHETH', 'SOLSOL'
    ]

    for pattern in invalid_patterns:
        if pattern in symbol:
            return False

    return True


def should_use_kraken_for_symbol(symbol: str) -> bool:
    """Determine if we should use Kraken for a given symbol."""
    if not symbol or not isinstance(symbol, str):
        return False

    # Normalize the symbol first
    normalized = normalize_symbol(symbol)

    # Check if the normalized symbol is supported by Kraken
    return normalized in KRAKEN_SUPPORTED_SOLANA_SYMBOLS
