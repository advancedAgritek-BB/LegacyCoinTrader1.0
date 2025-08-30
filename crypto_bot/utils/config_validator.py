"""
Configuration validation utility for the crypto bot.
Checks for unsupported timeframes, invalid configurations, and other issues.
"""

import logging
from typing import Dict, List, Optional, Set
import ccxt

logger = logging.getLogger(__name__)

# Kraken supported timeframes
KRAKEN_SUPPORTED_TIMEFRAMES = {
    '1m', '3m', '5m', '15m', '30m', '45m', '1h', '4h', '1d', '1w', '2w', '3w', '1M'
}

# Coinbase supported timeframes  
COINBASE_SUPPORTED_TIMEFRAMES = {
    '1m', '5m', '15m', '1h', '6h', '1d'
}

def get_exchange_supported_timeframes(exchange_name: str) -> Set[str]:
    """Get supported timeframes for a specific exchange."""
    exchange_name = exchange_name.lower()
    if exchange_name == 'kraken':
        return KRAKEN_SUPPORTED_TIMEFRAMES
    elif exchange_name == 'coinbase':
        return COINBASE_SUPPORTED_TIMEFRAMES
    else:
        # For unknown exchanges, return a minimal set
        return {'1m', '5m', '15m', '1h', '4h', '1d'}

def validate_timeframes(config: Dict, exchange_name: str) -> List[str]:
    """
    Validate timeframes in configuration against exchange support.
    
    Args:
        config: Bot configuration dictionary
        exchange_name: Name of the exchange (e.g., 'kraken', 'coinbase')
        
    Returns:
        List of validation error messages
    """
    errors = []
    supported_timeframes = get_exchange_supported_timeframes(exchange_name)
    
    # Check main timeframes list
    if 'timeframes' in config:
        for tf in config['timeframes']:
            if tf not in supported_timeframes:
                errors.append(
                    f"Timeframe '{tf}' is not supported on {exchange_name}. "
                    f"Supported: {', '.join(sorted(supported_timeframes))}"
                )
    
    # Check individual timeframe settings
    timeframe_fields = [
        'timeframe', 'scalp_timeframe', 'sideways_timeframe', 
        'trending_timeframe', 'volatile_timeframe', 'breakout_timeframe',
        'bounce_timeframe', 'mean_reverting_timeframe'
    ]
    
    for field in timeframe_fields:
        if field in config and config[field] not in supported_timeframes:
            errors.append(
                f"Field '{field}' has unsupported timeframe '{config[field]}' "
                f"for {exchange_name}. Supported: {', '.join(sorted(supported_timeframes))}"
            )
    
    # Check strategy router timeframes
    if 'strategy_router' in config:
        router = config['strategy_router']
        for field in timeframe_fields:
            if field in router and router[field] not in supported_timeframes:
                errors.append(
                    f"Strategy router field '{field}' has unsupported timeframe "
                    f"'{router[field]}' for {exchange_name}. "
                    f"Supported: {', '.join(sorted(supported_timeframes))}"
                )
    
    # Check enhanced backtesting timeframes
    if 'enhanced_backtesting' in config:
        backtest = config['enhanced_backtesting']
        if 'timeframes' in backtest:
            for tf in backtest['timeframes']:
                if tf not in supported_timeframes:
                    errors.append(
                        f"Enhanced backtesting timeframe '{tf}' is not supported "
                        f"on {exchange_name}. Supported: {', '.join(sorted(supported_timeframes))}"
                    )
    
    return errors

def validate_config(config: Dict) -> List[str]:
    """
    Comprehensive configuration validation.
    
    Args:
        config: Bot configuration dictionary
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Get exchange name
    exchange_name = config.get('exchange', 'unknown').lower()
    
    # Validate timeframes
    timeframe_errors = validate_timeframes(config, exchange_name)
    errors.extend(timeframe_errors)
    
    # Validate other configuration aspects
    if 'execution_mode' in config:
        mode = config['execution_mode']
        if mode not in ['dry_run', 'live', 'paper']:
            errors.append(f"Invalid execution_mode: {mode}. Must be 'dry_run', 'live', or 'paper'")
    
    # Validate risk parameters
    risk_fields = ['stop_loss_pct', 'take_profit_pct', 'sl_pct', 'tp_pct']
    for field in risk_fields:
        if field in config:
            value = config[field]
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"Invalid {field}: {value}. Must be a positive number")
    
    # Validate symbol filter settings
    if 'symbol_filter' in config:
        sf = config['symbol_filter']
        if 'min_volume_usd' in sf and sf['min_volume_usd'] <= 0:
            errors.append("symbol_filter.min_volume_usd must be positive")
        
        if 'max_spread_pct' in sf and (sf['max_spread_pct'] <= 0 or sf['max_spread_pct'] > 100):
            errors.append("symbol_filter.max_spread_pct must be between 0 and 100")
    
    return errors

def fix_timeframe_config(config: Dict, exchange_name: str) -> Dict:
    """
    Automatically fix timeframe configuration issues by replacing unsupported
    timeframes with supported alternatives.
    
    Args:
        config: Bot configuration dictionary
        exchange_name: Name of the exchange
        
    Returns:
        Updated configuration with fixed timeframes
    """
    supported_timeframes = get_exchange_supported_timeframes(exchange_name)
    
    # Mapping of unsupported timeframes to supported alternatives
    timeframe_mapping = {
        '10m': '15m',  # 10m -> 15m
        '30m': '15m',  # 30m -> 15m (if not supported)
        '6h': '4h',    # 6h -> 4h (if not supported)
        '2w': '1w',    # 2w -> 1w (if not supported)
        '3w': '1w',    # 3w -> 1w (if not supported)
    }
    
    config_copy = config.copy()
    
    # Fix main timeframes list
    if 'timeframes' in config_copy:
        fixed_timeframes = []
        for tf in config_copy['timeframes']:
            if tf in supported_timeframes:
                fixed_timeframes.append(tf)
            elif tf in timeframe_mapping:
                replacement = timeframe_mapping[tf]
                if replacement in supported_timeframes:
                    fixed_timeframes.append(replacement)
                    logger.warning(f"Replaced unsupported timeframe '{tf}' with '{replacement}'")
                else:
                    logger.warning(f"Skipped unsupported timeframe '{tf}' (no suitable replacement)")
            else:
                logger.warning(f"Skipped unsupported timeframe '{tf}' (no suitable replacement)")
        config_copy['timeframes'] = fixed_timeframes
    
    # Fix individual timeframe fields
    timeframe_fields = [
        'timeframe', 'scalp_timeframe', 'sideways_timeframe', 
        'trending_timeframe', 'volatile_timeframe', 'breakout_timeframe',
        'bounce_timeframe', 'mean_reverting_timeframe'
    ]
    
    for field in timeframe_fields:
        if field in config_copy:
            tf = config_copy[field]
            if tf not in supported_timeframes and tf in timeframe_mapping:
                replacement = timeframe_mapping[tf]
                if replacement in supported_timeframes:
                    config_copy[field] = replacement
                    logger.warning(f"Fixed {field}: replaced '{tf}' with '{replacement}'")
    
    # Fix strategy router timeframes
    if 'strategy_router' in config_copy:
        router = config_copy['strategy_router']
        for field in timeframe_fields:
            if field in router:
                tf = router[field]
                if tf not in supported_timeframes and tf in timeframe_mapping:
                    replacement = timeframe_mapping[tf]
                    if replacement in supported_timeframes:
                        router[field] = replacement
                        logger.warning(f"Fixed strategy_router.{field}: replaced '{tf}' with '{replacement}'")
    
    return config_copy

def log_config_summary(config: Dict) -> None:
    """Log a summary of the configuration for debugging purposes."""
    logger.info("Configuration Summary:")
    logger.info(f"Exchange: {config.get('exchange', 'unknown')}")
    logger.info(f"Execution Mode: {config.get('execution_mode', 'unknown')}")
    logger.info(f"Timeframes: {config.get('timeframes', [])}")
    logger.info(f"Main Timeframe: {config.get('timeframe', 'unknown')}")
    
    if 'strategy_router' in config:
        router = config['strategy_router']
        logger.info(f"Strategy Router Timeframes:")
        for field in ['breakout_timeframe', 'bounce_timeframe', 'mean_reverting_timeframe']:
            if field in router:
                logger.info(f"  {field}: {router[field]}")
    
    if 'enhanced_backtesting' in config:
        backtest = config['enhanced_backtesting']
        if 'timeframes' in backtest:
            logger.info(f"Enhanced Backtesting Timeframes: {backtest['timeframes']}")
