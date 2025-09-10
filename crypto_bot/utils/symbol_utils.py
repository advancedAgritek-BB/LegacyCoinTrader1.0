from typing import Optional
import asyncio
import time

from .logger import LOG_DIR, setup_logger
from .symbol_pre_filter import filter_symbols
from .telemetry import telemetry
from .token_validator import _is_valid_base_token
from .circuit_breaker import (
    get_circuit_breaker_manager,
    EXCHANGE_API_CONFIG,
    CircuitBreakerConfig
)
from .retry_handler import (
    get_retry_manager,
    EXCHANGE_API_RETRY_CONFIG,
    RetryConfig
)

# Global circuit breaker manager
circuit_breaker_manager = get_circuit_breaker_manager()

# Global retry manager
retry_manager = get_retry_manager()

# Circuit breaker configurations
SYMBOL_SCAN_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=300.0,  # 5 minutes
    expected_exception=Exception,
    success_threshold=2
)

# Retry configuration for symbol scanning
SYMBOL_SCAN_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=2.0,
    max_delay=30.0,
    strategy="exponential_backoff",
    retry_on_exceptions=(Exception,),
    timeout=60.0
)


def fix_symbol(sym: str) -> str:
    """Normalize different notations of Bitcoin."""
    if not isinstance(sym, str):
        return sym
    return sym.replace("XBT/", "BTC/").replace("XBT", "BTC")

logger = setup_logger("bot", LOG_DIR / "bot.log")


_cached_symbols: Optional[list] = None
_last_refresh: float = 0.0
def _normalize_to_slash_format(symbol: str) -> str:
    """Normalize symbol to CCXT slash format BASE/QUOTE when possible.

    Examples:
        BTCUSD -> BTC/USD
        ETHUSDT -> ETH/USDT
        SOLUSDC -> SOL/USDC
    Leaves symbols with existing slash unchanged.
    """
    if not isinstance(symbol, str):
        return symbol
    if "/" in symbol:
        return symbol
    # Common quote currency suffixes
    quotes = ("USD", "USDT", "USDC", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "BTC", "ETH")
    up = symbol.upper().replace("XBT", "BTC")
    for q in quotes:
        if up.endswith(q):
            base = up[: -len(q)]
            if base:
                return f"{base}/{q}"
    return symbol



async def get_filtered_symbols(exchange, config) -> list:
    """Return user symbols filtered by liquidity/volatility or fallback.

    Results are cached for ``symbol_refresh_minutes`` minutes to avoid
    unnecessary API calls.
    """
    global _cached_symbols, _last_refresh

    refresh_m = config.get("symbol_refresh_minutes", 30)
    now = time.time()

    if (
        _cached_symbols is not None
        and now - _last_refresh < refresh_m * 60
    ):
        return _cached_symbols

    if config.get("skip_symbol_filters"):
        syms = config.get("symbols", [config.get("symbol")])
        # Filter out None values to prevent issues
        syms = [s for s in syms if s is not None]
        # Normalize to slash format
        syms = [fix_symbol(_normalize_to_slash_format(s)) for s in syms]
        # If Kraken, drop obvious Solana contract addresses and unsupported
        try:
            ex_id = getattr(exchange, "id", "").lower()
            if ex_id == "kraken":
                # Ensure markets are loaded so symbols list is populated
                try:
                    if not getattr(exchange, "symbols", []) and hasattr(exchange, "load_markets"):
                        # Use circuit breaker and retry handler for exchange API calls
                        await circuit_breaker_manager.call_with_circuit_breaker(
                            "exchange_load_markets",
                            exchange.load_markets,
                            config=SYMBOL_SCAN_CONFIG
                        )
                except Exception as e:
                    logger.warning(f"Failed to load markets with circuit breaker: {e}")
                    # Fallback to retry handler
                    try:
                        retry_handler = await retry_manager.get_retry_handler(
                            "exchange_load_markets_fallback",
                            SYMBOL_SCAN_RETRY_CONFIG
                        )
                        if asyncio.iscoroutinefunction(exchange.load_markets):
                            await retry_handler.execute_with_retry(exchange.load_markets)
                        else:
                            await retry_handler.execute_with_retry(
                                asyncio.to_thread,
                                exchange.load_markets
                            )
                    except Exception as retry_e:
                        logger.warning(f"Failed to load markets with retry handler: {retry_e}")
                        # Final fallback to direct call
                        try:
                            if asyncio.iscoroutinefunction(exchange.load_markets):
                                await exchange.load_markets()
                            else:
                                await asyncio.to_thread(exchange.load_markets)
                        except Exception:
                            pass
                symbols_set = set(getattr(exchange, "symbols", []) or [])
                pruned = []
                for s in syms:
                    base, _, quote = s.partition("/")
                    # Exclude Solana contract addresses on Kraken (they're not supported)
                    import re
                    if re.match(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", base):
                        logger.debug(f"Skipping Solana address {base} on Kraken - not supported")
                        continue
                    # Also exclude symbols not in Kraken's supported list
                    if symbols_set and s not in symbols_set:
                        continue
                    pruned.append(s)
                syms = pruned
        except Exception as e:
            logger.warning(f"Symbol filtering failed: {e}")
            pass
        result = [(s, 0.0) for s in syms]
        _cached_symbols = result
        _last_refresh = now
        return result

    symbols = config.get("symbols", [config.get("symbol")])
    # Filter out None values to prevent TypeError in string operations
    symbols = [s for s in symbols if s is not None]
    # Normalize to slash format early
    symbols = [fix_symbol(_normalize_to_slash_format(s)) for s in symbols]
    # If Kraken, load markets once and pre-filter unsupported & Solana contracts
    try:
        ex_id = getattr(exchange, "id", "").lower()
        if ex_id == "kraken":
            try:
                if not getattr(exchange, "symbols", []) and hasattr(exchange, "load_markets"):
                    # Use retry handler for exchange API calls
                    retry_handler = await retry_manager.get_retry_handler(
                        "exchange_load_markets_main",
                        SYMBOL_SCAN_RETRY_CONFIG
                    )
                    if asyncio.iscoroutinefunction(exchange.load_markets):
                        await retry_handler.execute_with_retry(exchange.load_markets)
                    else:
                        await retry_handler.execute_with_retry(
                            asyncio.to_thread,
                            exchange.load_markets
                        )
            except Exception as e:
                logger.warning(f"Failed to load markets with retry handler: {e}")
                # Fallback to direct call
                try:
                    if asyncio.iscoroutinefunction(exchange.load_markets):
                        await exchange.load_markets()
                    else:
                        await asyncio.to_thread(exchange.load_markets)
                except Exception:
                    pass
            symbols_set = set(getattr(exchange, "symbols", []) or [])
        else:
            symbols_set = set()
    except Exception:
        symbols_set = set()
    cleaned_symbols = []
    for sym in symbols:
        if not isinstance(sym, str):
            logger.warning("Skipping non-string symbol: %s (type: %s)", sym, type(sym))
            continue
        base, _, quote = sym.partition("/")
        # If Kraken: drop Solana contracts (valid base tokens with USDC) and unsupported pairs
        if symbols_set:
            # Exclude Solana contract addresses on Kraken (they're not supported)
            import re
            if re.match(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", base):
                logger.debug(f"Skipping Solana address {base} on Kraken - not supported")
                continue
            if sym not in symbols_set:
                logger.debug("Dropping unsupported Kraken symbol %s", sym)
                continue
        # Generic sanity for Solana tokens in CEX list
        if quote.upper() == "USDC" and not _is_valid_base_token(base):
            logger.info("Dropping invalid USDC pair %s", sym)
            continue
        cleaned_symbols.append(sym)

    symbols = cleaned_symbols
    skipped_before = telemetry.snapshot().get("scan.symbols_skipped", 0)
    if asyncio.iscoroutinefunction(filter_symbols):
        scored = await filter_symbols(exchange, symbols, config)
    else:
        scored = await asyncio.to_thread(filter_symbols, exchange, symbols, config)
    skipped_main = telemetry.snapshot().get("scan.symbols_skipped", 0) - skipped_before
    if not scored:
        fallback = config.get("symbol")
        if fallback is None:
            logger.warning("No fallback symbol configured and no symbols passed filters")
            return []
        excluded = [s.upper() for s in config.get("excluded_symbols", [])]
        if fallback and fallback.upper() in excluded:
            logger.warning("Fallback symbol %s is excluded", fallback)
            logger.warning(
                "No symbols met volume/spread requirements; consider adjusting symbol_filter in config. Rejected %d symbols",
                skipped_main,
            )
            return []

        skipped_before = telemetry.snapshot().get("scan.symbols_skipped", 0)
        if asyncio.iscoroutinefunction(filter_symbols):
            check = await filter_symbols(exchange, [fallback], config)
        else:
            check = await asyncio.to_thread(filter_symbols, exchange, [fallback], config)
        skipped_fb = telemetry.snapshot().get("scan.symbols_skipped", 0) - skipped_before

        if not check:
            logger.warning(
                "Fallback symbol %s does not meet volume requirements", fallback
            )
            logger.warning(
                "No symbols met volume/spread requirements; consider adjusting symbol_filter in config. Rejected %d symbols initially, %d on fallback",
                skipped_main,
                skipped_fb,
            )
            return []

        logger.warning(
            "No symbols passed filters, falling back to %s",
            fallback,
        )
        scored = [(fallback, 0.0)]

    logger.info("%d symbols passed filtering", len(scored))

    if not scored:
        logger.warning(
            "No symbols met volume/spread requirements; consider adjusting symbol_filter in config. Rejected %d symbols",
            skipped_main,
        )

    if scored:
        _cached_symbols = scored
        _last_refresh = now

    return scored
