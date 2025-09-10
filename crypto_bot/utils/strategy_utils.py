"""
Utility functions for strategy execution and error handling.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any, Callable, Dict, Union
import concurrent.futures
import time
from pathlib import Path

from .logger import LOG_DIR

logger = logging.getLogger(__name__)


def compute_strategy_weights(path: str = str(LOG_DIR / "strategy_pnl.csv")) -> Dict[str, float]:
    """Return normalized allocation weights per strategy.

    We compute either win rate or Sharpe ratio from the PnL data stored at
    ``path``. The file must contain ``strategy`` and ``pnl`` columns. If the
    file is missing or empty an empty dict is returned.
    """
    file = Path(path)
    if not file.exists():
        return {}

    df = pd.read_csv(file)
    if df.empty or "strategy" not in df.columns or "pnl" not in df.columns:
        return {}

    scores = {}
    for strat, group in df.groupby("strategy"):
        wins = (group["pnl"] > 0).sum()
        total = len(group)
        win_rate = wins / total if total else 0.0
        std = group["pnl"].std()
        sharpe = group["pnl"].mean() / std * (total ** 0.5) if std else 0.0
        scores[strat] = max(win_rate, sharpe)

    total_score = sum(scores.values())
    if not total_score:
        return {s: 1 / len(scores) for s in scores} if scores else {}

    return {s: sc / total_score for s, sc in scores.items()}


def compute_drawdown(df: pd.DataFrame, lookback: int = 20) -> float:
    """Return maximum drawdown of ``close`` prices over ``lookback`` bars."""
    if df.empty or "close" not in df.columns:
        return 0.0
    series = df["close"].tail(lookback)
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdowns = series - running_max
    return float(drawdowns.min())


# Global tracking for strategy performance
_strategy_timeout_count: Dict[str, int] = {}
_strategy_last_timeout: Dict[str, float] = {}

def safe_strategy_execution(
    strategy_func: Callable,
    df: pd.DataFrame,
    config: Optional[dict] = None,
    timeout_seconds: int = 10
) -> Tuple[float, str]:
    """
    Execute a strategy function with proper error handling and timeout.
    
    Args:
        strategy_func: The strategy function to execute
        df: Input DataFrame
        config: Strategy configuration
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        Tuple of (score, direction) or (0.0, "none") on error
    """
    try:
        # Validate input data
        if df is None or df.empty:
            logger.warning("Strategy received empty DataFrame")
            return 0.0, "none"
        
        # Check if df is actually a DataFrame (not numpy array)
        if not isinstance(df, pd.DataFrame):
            logger.warning(f"Strategy received {type(df)} instead of DataFrame, converting...")
            try:
                if isinstance(df, np.ndarray):
                    # Convert numpy array to DataFrame with proper column names
                    if df.ndim == 2:
                        # 2D array - assume OHLCV data
                        if df.shape[1] >= 5:
                            df = pd.DataFrame(df, columns=['open', 'high', 'low', 'close', 'volume'])
                        else:
                            df = pd.DataFrame(df, columns=['close'] + [f'col_{i}' for i in range(df.shape[1]-1)])
                    elif df.ndim == 1:
                        # 1D array - assume it's a series
                        df = pd.DataFrame({'close': df})
                    else:
                        logger.error(f"Cannot handle numpy array with {df.ndim} dimensions")
                        return 0.0, "none"
                else:
                    logger.error(f"Cannot convert {type(df)} to DataFrame")
                    return 0.0, "none"
            except Exception as e:
                logger.error(f"Failed to convert data to DataFrame: {e}")
                return 0.0, "none"
        
        # Execute strategy with timeout
        if timeout_seconds > 0:
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(strategy_func, df, config)
                try:
                    result = future.result(timeout=timeout_seconds)
                    execution_time = time.time() - start_time
                    if execution_time > timeout_seconds * 0.8:  # Warning if >80% of timeout
                        logger.warning(f"Strategy {strategy_func.__name__} took {execution_time:.2f}s (close to {timeout_seconds}s timeout)")
                    else:
                        logger.debug(f"Strategy {strategy_func.__name__} executed in {execution_time:.2f}s")
                except concurrent.futures.TimeoutError:
                    execution_time = time.time() - start_time
                    strategy_name = strategy_func.__name__
                    
                    # Track timeout count
                    _strategy_timeout_count[strategy_name] = _strategy_timeout_count.get(strategy_name, 0) + 1
                    _strategy_last_timeout[strategy_name] = time.time()
                    
                    # Check if strategy should be disabled
                    max_timeouts = config.get("strategy_performance", {}).get("max_consecutive_timeouts", 3) if config else 3
                    if _strategy_timeout_count[strategy_name] >= max_timeouts:
                        logger.error(f"Strategy {strategy_name} disabled after {_strategy_timeout_count[strategy_name]} consecutive timeouts")
                        return 0.0, "none"
                    
                    logger.warning(f"Strategy {strategy_name} TIMEOUT after {execution_time:.2f}s (limit: {timeout_seconds}s) - timeout #{_strategy_timeout_count[strategy_name]}")
                    return 0.0, "none"
        else:
            start_time = time.time()
            result = strategy_func(df, config)
            execution_time = time.time() - start_time
            logger.debug(f"Strategy {strategy_func.__name__} executed in {execution_time:.2f}s")
        
        # Validate result
        if not isinstance(result, tuple) or len(result) != 2:
            logger.warning(f"Strategy {strategy_func.__name__} returned invalid result: {result}")
            return 0.0, "none"
        
        score, direction = result
        
        # Validate score
        if not isinstance(score, (int, float)) or not np.isfinite(score):
            logger.warning(f"Strategy {strategy_func.__name__} returned invalid score: {score}")
            return 0.0, "none"
        
        # Validate direction
        if not isinstance(direction, str):
            logger.warning(f"Strategy {strategy_func.__name__} returned invalid direction: {direction}")
            return 0.0, "none"
        
        return float(score), str(direction)
        
    except Exception as e:
        logger.warning(f"Strategy {strategy_func.__name__} failed: {e}")
        return 0.0, "none"


def reset_strategy_timeout_count(strategy_name: str) -> None:
    """Reset the timeout count for a specific strategy."""
    if strategy_name in _strategy_timeout_count:
        _strategy_timeout_count[strategy_name] = 0
        logger.info(f"Reset timeout count for strategy: {strategy_name}")


def get_strategy_performance_stats() -> Dict[str, Dict[str, Any]]:
    """Get performance statistics for all strategies."""
    stats = {}
    for strategy_name in _strategy_timeout_count:
        stats[strategy_name] = {
            "timeout_count": _strategy_timeout_count.get(strategy_name, 0),
            "last_timeout": _strategy_last_timeout.get(strategy_name, 0),
            "disabled": _strategy_timeout_count.get(strategy_name, 0) >= 3
        }
    return stats


def validate_strategy_config(config: dict) -> Dict[str, Any]:
    """Validate strategy configuration and provide recommendations."""
    validation = {
        "valid": True,
        "warnings": [],
        "recommendations": [],
        "issues": []
    }
    
    # Check strategy timeout configuration
    strategy_timeout = config.get("strategy_timeout_seconds", 10)
    if strategy_timeout < 15:
        validation["warnings"].append(f"Strategy timeout ({strategy_timeout}s) is quite low")
        validation["recommendations"].append("Consider increasing strategy_timeout_seconds to 30-60 seconds")
    
    if strategy_timeout > 120:
        validation["warnings"].append(f"Strategy timeout ({strategy_timeout}s) is very high")
        validation["recommendations"].append("Consider reducing strategy_timeout_seconds to 30-60 seconds")
    
    # Check performance monitoring configuration
    perf_config = config.get("strategy_performance", {})
    if not perf_config.get("enable_performance_logging", False):
        validation["recommendations"].append("Enable strategy performance logging for better monitoring")
    
    # Check for disabled strategies
    disabled_strategies = [name for name, count in _strategy_timeout_count.items() if count >= 3]
    if disabled_strategies:
        validation["warnings"].append(f"Strategies disabled due to timeouts: {disabled_strategies}")
        validation["recommendations"].append("Review and optimize disabled strategies or increase timeout")
    
    return validation


def validate_dataframe(df: pd.DataFrame, min_rows: int = 20, required_columns: Optional[list] = None) -> bool:
    """
    Validate DataFrame for strategy execution.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        required_columns: List of required columns
        
    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        return False
    
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
    
    return True


def filter_low_quality_symbols(
    symbols: list,
    df_cache: dict,
    min_candles: int = 50,
    min_completeness: float = 0.8
) -> list:
    """
    Filter out symbols with insufficient or low-quality data.
    
    Args:
        symbols: List of symbols to filter
        df_cache: DataFrame cache
        min_candles: Minimum number of candles required
        min_completeness: Minimum data completeness ratio
        
    Returns:
        Filtered list of symbols
    """
    filtered_symbols = []
    
    for symbol in symbols:
        # Check if symbol has data in cache
        if symbol not in df_cache:
            continue
        
        df = df_cache[symbol]
        
        # Check minimum candles
        if len(df) < min_candles:
            logger.debug(f"Skipping {symbol}: only {len(df)} candles (min: {min_candles})")
            continue
        
        # Check data completeness (non-null values)
        if min_completeness > 0:
            completeness = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
            if completeness < min_completeness:
                logger.debug(f"Skipping {symbol}: completeness {completeness:.2f} (min: {min_completeness})")
                continue
        
        filtered_symbols.append(symbol)
    
    return filtered_symbols
