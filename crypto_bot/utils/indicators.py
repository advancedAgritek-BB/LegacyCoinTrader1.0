"""Utility functions for computing technical indicators."""

from __future__ import annotations

from typing import NamedTuple, Optional

import pandas as pd


class BollingerBands(NamedTuple):
    """Container for Bollinger Band calculations."""

    middle: pd.Series
    upper: pd.Series
    lower: pd.Series
    width: pd.Series


def _as_series(data: pd.Series | pd.Index | list | tuple) -> pd.Series:
    """Return ``data`` as a pandas Series."""
    if isinstance(data, pd.Series):
        return data
    return pd.Series(data)


def calculate_bollinger_bands(
    close: pd.Series | pd.Index | list | tuple,
    window: int = 20,
    num_std: float = 2.0,
    *,
    min_periods: Optional[int] = None,
) -> BollingerBands:
    """Calculate Bollinger Bands for ``close`` prices."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_periods is None:
        min_periods = window
    series = _as_series(close).astype(float)
    rolling = series.rolling(window=window, min_periods=min_periods)
    middle = rolling.mean()
    std = rolling.std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    width = upper - lower
    return BollingerBands(middle=middle, upper=upper, lower=lower, width=width)


def calculate_rsi(
    close: pd.Series | pd.Index | list | tuple,
    window: int = 14,
    *,
    min_periods: Optional[int] = None,
    method: str = "sma",
) -> pd.Series:
    """Calculate the Relative Strength Index."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_periods is None:
        min_periods = window
    series = _as_series(close).astype(float)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    if method.lower() == "ema":
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=min_periods).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=min_periods).mean()
    elif method.lower() == "sma":
        avg_gain = gain.rolling(window=window, min_periods=min_periods).mean()
        avg_loss = loss.rolling(window=window, min_periods=min_periods).mean()
    else:
        raise ValueError("method must be 'sma' or 'ema'")
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(
    data: pd.DataFrame,
    window: int = 14,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    min_periods: Optional[int] = None,
    use_ema: bool = False,
) -> pd.Series:
    """Calculate the Average True Range."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_periods is None:
        min_periods = window
    high = data[high_col].astype(float)
    low = data[low_col].astype(float)
    close = data[close_col].astype(float)
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    if use_ema:
        atr = true_range.ewm(span=window, adjust=False, min_periods=min_periods).mean()
    else:
        atr = true_range.rolling(window=window, min_periods=min_periods).mean()
    return atr


__all__ = [
    "BollingerBands",
    "calculate_atr",
    "calculate_bollinger_bands",
    "calculate_rsi",
]
