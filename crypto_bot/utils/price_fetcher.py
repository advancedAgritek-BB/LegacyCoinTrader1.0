"""
Price fetching utilities shared between frontend and crypto_bot modules.
This module helps break circular import dependencies.
"""

import time
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Global price cache
_price_cache: Dict[str, Tuple[float, float]] = {}
_CACHE_TTL = 300  # 5 minutes


def get_current_price_for_symbol(symbol: str) -> float:
    """Get current price for a symbol using available price sources with caching."""
    if not symbol or symbol.strip() == '':
        return 0.0

    symbol = symbol.strip().upper()

    # Check cache first
    now = time.time()
    if symbol in _price_cache:
        cached_price, timestamp = _price_cache[symbol]
        if now - timestamp < _CACHE_TTL:
            logger.debug(f"Using cached price for {symbol}: ${cached_price}")
            return cached_price
        else:
            # Cache expired, remove it
            del _price_cache[symbol]

    try:
        # Try Pyth network first for Solana tokens
        try:
            from crypto_bot.utils.pyth import get_pyth_price
            price = get_pyth_price(symbol)
            if price and price > 0:
                logger.debug(f"Got Pyth price for {symbol}: ${price}")
                _price_cache[symbol] = (price, time.time())
                return price
        except Exception as e:
            logger.debug(f"Pyth price fetch failed for {symbol}: {e}")

        # Try fallback sources
        try:
            from crypto_bot.utils.pyth import _get_fallback_price
            price = _get_fallback_price(symbol)
            if price and price > 0:
                logger.debug(f"Got fallback price for {symbol}: ${price}")
                _price_cache[symbol] = (price, time.time())
                return price
        except Exception as e:
            logger.debug(f"Fallback price fetch failed for {symbol}: {e}")

        # Try trade manager cache as final fallback
        try:
            from crypto_bot.utils.trade_manager import get_trade_manager
            trade_manager = get_trade_manager()
            if hasattr(trade_manager, 'price_cache') and symbol in trade_manager.price_cache:
                price = trade_manager.price_cache[symbol]
                if price and price > 0:
                    logger.debug(f"Using trade manager price for {symbol}: ${price}")
                    _price_cache[symbol] = (price, time.time())
                    return price
        except Exception as e:
            logger.debug(f"Trade manager price fetch failed for {symbol}: {e}")

    except Exception as e:
        logger.warning(f"Price fetch failed for {symbol}: {e}")

    # Return 0 if all methods fail
    _price_cache[symbol] = (0.0, time.time())
    return 0.0


def clear_price_cache():
    """Clear the price cache."""
    global _price_cache
    _price_cache.clear()
    logger.info("Price cache cleared")


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about the price cache."""
    now = time.time()
    valid_entries = sum(1 for _, timestamp in _price_cache.values()
                       if now - timestamp < _CACHE_TTL)
    expired_entries = len(_price_cache) - valid_entries

    return {
        "total_entries": len(_price_cache),
        "valid_entries": valid_entries,
        "expired_entries": expired_entries,
        "cache_ttl_seconds": _CACHE_TTL
    }
