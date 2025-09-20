"""
Enhanced Symbol Validation for Production Deployment

This module provides production-grade symbol validation to prevent API errors
and circuit breaker trips caused by invalid or unsupported symbols.
"""

import asyncio
import re
import time
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from .logger import LOG_DIR, setup_logger
from .token_validator import _is_valid_base_token

logger = setup_logger(__name__, LOG_DIR / "symbol_validator.log")


class ProductionSymbolValidator:
    """
    Production-grade symbol validator with comprehensive filtering.

    Features:
    - Format validation (BASE/QUOTE)
    - Exchange compatibility checking
    - Liquidity validation
    - Volume threshold checking
    - Blacklist filtering
    - Contract address validation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.blacklist_patterns = self._load_blacklist_patterns()
        self.valid_quotes = self.config.get("valid_quote_currencies", ["USD", "USDT", "USDC", "EUR", "GBP"])
        self.min_volume_usd = self.config.get("min_volume_usd", 10000)
        self.min_liquidity_score = self.config.get("min_liquidity_score", 0.6)
        self.max_price_deviation = self.config.get("max_price_deviation", 0.1)
        self.strict_mode = self.config.get("strict_mode", True)

        # Cache for validation results
        self.validation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour
        self.last_cache_cleanup = time.time()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for symbol validation."""
        return {
            "valid_quote_currencies": ["USD", "USDT", "USDC", "EUR", "GBP"],
            "min_volume_usd": 10000,
            "min_liquidity_score": 0.6,
            "max_price_deviation": 0.1,
            "strict_mode": True,
            "validate_exchange_support": True,
            "validate_liquidity": True,
            "filter_contract_addresses": True
        }

    def _load_blacklist_patterns(self) -> Set[str]:
        """Load patterns for blacklisted symbols."""
        blacklist = {
            "INVALID", "TEST", "DEPRECATED", "DELISTED",
            "BANNED", "SUSPENDED", "DISABLED", "UNKNOWN"
        }

        # Add regex patterns for problematic symbols
        patterns = {
            r"^.{50,}$",  # Extremely long symbols (likely contract addresses)
            r".*[^A-Z0-9/-].*",  # Symbols with invalid characters
            r"^/.*",  # Symbols starting with slash
            r".*/$",  # Symbols ending with slash
            r".*//.*",  # Multiple slashes
            r".*INVALID.*",  # Any symbol containing INVALID substring
            r"^UNKNOWN(/|$).*",  # Symbols with UNKNOWN base indicator
        }

        return blacklist.union(patterns)

    def _matches_blacklist(self, symbol: str) -> bool:
        """Check if symbol matches any blacklist pattern."""
        # Check exact matches
        if symbol.upper() in self.blacklist_patterns:
            return True

        # Check regex patterns
        for pattern in self.blacklist_patterns:
            if pattern.startswith("^") and pattern.endswith("$"):
                if re.match(pattern, symbol):
                    return True

        return False

    def _validate_format(self, symbol: str) -> bool:
        """Validate symbol format (BASE/QUOTE)."""
        if not isinstance(symbol, str):
            return False

        if "/" not in symbol:
            return False

        parts = symbol.split("/")
        if len(parts) != 2:
            return False

        base, quote = parts
        if not base or not quote:
            return False

        # Check for valid characters
        if not re.match(r"^[A-Z0-9]+$", base):
            return False

        if not re.match(r"^[A-Z0-9]+$", quote):
            return False

        # Check quote currency
        if self.strict_mode and quote.upper() not in self.valid_quotes:
            return False

        return True

    def _validate_contract_address(self, symbol: str) -> bool:
        """Validate that symbol doesn't contain contract addresses."""
        if not self.config.get("filter_contract_addresses", True):
            return True

        base, quote = symbol.split("/")
        if quote.upper() == "USDC" and not _is_valid_base_token(base):
            logger.warning(f"Filtering out invalid USDC pair: {symbol}")
            return False

        return True

    def _is_cached(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check if symbol validation is cached."""
        if symbol in self.validation_cache:
            cached = self.validation_cache[symbol]
            if time.time() - cached.get("timestamp", 0) < self.cache_ttl:
                return cached

        # Cleanup old cache entries periodically
        if time.time() - self.last_cache_cleanup > 3600:  # Every hour
            self._cleanup_cache()

        return None

    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired = [
            symbol for symbol, data in self.validation_cache.items()
            if current_time - data.get("timestamp", 0) > self.cache_ttl
        ]

        for symbol in expired:
            del self.validation_cache[symbol]

        self.last_cache_cleanup = current_time
        logger.debug(f"Cleaned up {len(expired)} expired cache entries")

    def _cache_result(self, symbol: str, result: Dict[str, Any]):
        """Cache validation result."""
        result["timestamp"] = time.time()
        self.validation_cache[symbol] = result

    async def validate_symbol(self, symbol: str, exchange=None) -> Dict[str, Any]:
        """
        Validate a single symbol comprehensively.

        Returns:
            Dict with validation results and metadata
        """
        # Check cache first
        cached = self._is_cached(symbol)
        if cached:
            return cached

        result = {
            "symbol": symbol,
            "valid": True,
            "reason": "OK",
            "checks": []
        }

        try:
            # 1. Format validation
            if not self._validate_format(symbol):
                result.update({
                    "valid": False,
                    "reason": "Invalid format (expected BASE/QUOTE)"
                })
                result["checks"].append("format")
                self._cache_result(symbol, result)
                return result

            result["checks"].append("format:pass")

            # 2. Blacklist check
            if self._matches_blacklist(symbol):
                result.update({
                    "valid": False,
                    "reason": "Symbol blacklisted"
                })
                result["checks"].append("blacklist")
                self._cache_result(symbol, result)
                return result

            result["checks"].append("blacklist:pass")

            # 3. Contract address validation
            if not self._validate_contract_address(symbol):
                result.update({
                    "valid": False,
                    "reason": "Invalid contract address"
                })
                result["checks"].append("contract")
                self._cache_result(symbol, result)
                return result

            result["checks"].append("contract:pass")

            # 4. Exchange support validation
            if self.config.get("validate_exchange_support", True) and exchange:
                if not await self._validate_exchange_support(symbol, exchange):
                    result.update({
                        "valid": False,
                        "reason": "Not supported by exchange"
                    })
                    result["checks"].append("exchange")
                    self._cache_result(symbol, result)
                    return result

            result["checks"].append("exchange:pass")

            # 5. Liquidity validation
            if self.config.get("validate_liquidity", True) and exchange:
                liquidity_result = await self._validate_liquidity(symbol, exchange)
                if not liquidity_result["valid"]:
                    result.update({
                        "valid": False,
                        "reason": f"Insufficient liquidity: {liquidity_result['reason']}"
                    })
                    result["checks"].append("liquidity")
                    self._cache_result(symbol, result)
                    return result

                result["liquidity_score"] = liquidity_result.get("score", 0)
                result["volume_usd"] = liquidity_result.get("volume_usd", 0)

            result["checks"].append("liquidity:pass")

        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            result.update({
                "valid": False,
                "reason": f"Validation error: {str(e)}"
            })

        self._cache_result(symbol, result)
        return result

    async def _validate_exchange_support(self, symbol: str, exchange) -> bool:
        """Validate that exchange supports the symbol."""
        try:
            if hasattr(exchange, "markets"):
                if not exchange.markets:
                    await exchange.load_markets()
                return symbol in exchange.markets
            return True  # Assume supported if we can't check
        except Exception as e:
            logger.warning(f"Could not validate exchange support for {symbol}: {e}")
            return True  # Don't fail validation due to exchange check issues

    async def _validate_liquidity(self, symbol: str, exchange) -> Dict[str, Any]:
        """Validate symbol liquidity and volume."""
        result = {"valid": True, "reason": "OK"}

        try:
            # Get ticker data
            ticker = await asyncio.to_thread(exchange.fetch_ticker, symbol)

            volume_usd = ticker.get("quoteVolume", 0)
            if volume_usd < self.min_volume_usd:
                return {
                    "valid": False,
                    "reason": f"Volume ${volume_usd:.2f} < ${self.min_volume_usd}"
                }

            # Check spread
            ask = ticker.get("ask", 0)
            bid = ticker.get("bid", 0)
            if ask > 0 and bid > 0:
                spread_pct = abs(ask - bid) / ((ask + bid) / 2)
                if spread_pct > self.max_price_deviation:
                    return {
                        "valid": False,
                        "reason": f"Spread {spread_pct:.2%} > {self.max_price_deviation:.1%}"
                    }

            result.update({
                "score": min(volume_usd / (self.min_volume_usd * 10), 1.0),
                "volume_usd": volume_usd,
                "spread_pct": spread_pct if 'spread_pct' in locals() else 0
            })

        except Exception as e:
            logger.warning(f"Could not validate liquidity for {symbol}: {e}")
            # Don't fail validation due to liquidity check errors in non-strict mode
            if self.strict_mode:
                return {"valid": False, "reason": f"Liquidity check failed: {str(e)}"}

        return result

    async def validate_symbols_batch(self, symbols: List[str], exchange=None) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple symbols in parallel.

        Returns:
            Dict mapping symbols to validation results
        """
        logger.info(f"Validating {len(symbols)} symbols...")

        # Create tasks for parallel validation
        tasks = [self.validate_symbol(symbol, exchange) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        validation_results = {}
        valid_count = 0
        invalid_count = 0

        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Validation failed for {symbol}: {result}")
                validation_results[symbol] = {
                    "symbol": symbol,
                    "valid": False,
                    "reason": f"Exception: {str(result)}"
                }
                invalid_count += 1
            else:
                validation_results[symbol] = result
                if result["valid"]:
                    valid_count += 1
                else:
                    invalid_count += 1

        logger.info(f"Symbol validation complete: {valid_count} valid, {invalid_count} invalid")

        return validation_results

    def get_valid_symbols(self, validation_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Extract list of valid symbols from validation results."""
        return [
            symbol for symbol, result in validation_results.items()
            if result.get("valid", False)
        ]

    def get_validation_stats(self, validation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about validation results."""
        total = len(validation_results)
        valid = sum(1 for r in validation_results.values() if r.get("valid", False))
        invalid = total - valid

        reasons = {}
        for result in validation_results.values():
            if not result.get("valid", True):
                reason = result.get("reason", "Unknown")
                reasons[reason] = reasons.get(reason, 0) + 1

        return {
            "total_symbols": total,
            "valid_symbols": valid,
            "invalid_symbols": invalid,
            "validation_rate": valid / total if total > 0 else 0,
            "failure_reasons": reasons
        }


# Global validator instance
_production_validator = None

def get_production_validator(config: Optional[Dict[str, Any]] = None) -> ProductionSymbolValidator:
    """Get or create production symbol validator instance."""
    global _production_validator
    if _production_validator is None:
        _production_validator = ProductionSymbolValidator(config)
    return _production_validator


async def validate_symbols_production(
    symbols: List[str],
    exchange=None,
    config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Production-ready symbol validation function.

    Args:
        symbols: List of symbols to validate
        exchange: Exchange instance for validation
        config: Validation configuration

    Returns:
        List of valid symbols
    """
    validator = get_production_validator(config)
    results = await validator.validate_symbols_batch(symbols, exchange)
    return validator.get_valid_symbols(results)
