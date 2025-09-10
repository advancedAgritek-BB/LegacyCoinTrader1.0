"""
Configuration validation utilities for the crypto trading bot.
"""

from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)

# Default supported timeframes for major exchanges
EXCHANGE_TIMEFRAMES = {
    'kraken': {
        '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
    },
    'binance': {
        '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
    },
    'coinbase': {
        '1m', '5m', '15m', '1h', '6h', '1d'
    },
    'kucoin': {
        '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'
    },
    'okx': {
        '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M'
    },
    'bybit': {
        '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M'
    }
}

def get_exchange_supported_timeframes(exchange_name: str) -> Set[str]:
    """
    Get supported timeframes for a specific exchange.
    
    Args:
        exchange_name: Name of the exchange (case-insensitive)
        
    Returns:
        Set of supported timeframe strings
    """
    exchange_name = exchange_name.lower()
    
    if exchange_name in EXCHANGE_TIMEFRAMES:
        return EXCHANGE_TIMEFRAMES[exchange_name].copy()
    
    # Default timeframes for unknown exchanges
    logger.warning(f"Unknown exchange '{exchange_name}', using default timeframes")
    return {'1m', '5m', '15m', '1h', '4h', '1d'}

def validate_timeframe(timeframe: str, exchange_name: str) -> bool:
    """
    Validate if a timeframe is supported by the exchange.
    
    Args:
        timeframe: Timeframe string to validate
        exchange_name: Name of the exchange
        
    Returns:
        True if timeframe is supported, False otherwise
    """
    supported = get_exchange_supported_timeframes(exchange_name)
    return timeframe in supported

def get_timeframe_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to seconds.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        Timeframe in seconds
        
    Raises:
        ValueError: If timeframe format is invalid
    """
    if not timeframe:
        raise ValueError("Timeframe cannot be empty")
    
    unit = timeframe[-1].lower()
    try:
        value = int(timeframe[:-1])
    except ValueError:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    elif unit == 'd':
        return value * 86400
    elif unit == 'w':
        return value * 604800
    elif unit == 'M':
        return value * 2592000  # Approximate month
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")

def get_optimal_timeframe(base_timeframe: str, exchange_name: str) -> str:
    """
    Get the optimal timeframe for an exchange, falling back to supported ones.
    
    Args:
        base_timeframe: Desired timeframe
        exchange_name: Name of the exchange
        
    Returns:
        Optimal supported timeframe
    """
    supported = get_exchange_supported_timeframes(exchange_name)
    
    if base_timeframe in supported:
        return base_timeframe
    
    # Try to find a close match
    base_seconds = get_timeframe_seconds(base_timeframe)
    
    closest_timeframe = None
    min_diff = float('inf')
    
    for tf in supported:
        try:
            tf_seconds = get_timeframe_seconds(tf)
            diff = abs(tf_seconds - base_seconds)
            if diff < min_diff:
                min_diff = diff
                closest_timeframe = tf
        except ValueError:
            continue
    
    if closest_timeframe:
        logger.info(f"Using closest supported timeframe '{closest_timeframe}' for '{base_timeframe}' on {exchange_name}")
        return closest_timeframe
    
    # Fallback to 1h if no match found
    logger.warning(f"No suitable timeframe found for '{base_timeframe}' on {exchange_name}, using '1h'")
    return '1h'
