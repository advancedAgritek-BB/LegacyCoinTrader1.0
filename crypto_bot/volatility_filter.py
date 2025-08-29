"""Helpers for evaluating market volatility."""

from __future__ import annotations

import os

import pandas as pd
import numpy as np
import requests

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.indicator_cache import cache_series
from pathlib import Path


logger = setup_logger(__name__, LOG_DIR / "volatility.log")

DEFAULT_FUNDING_URL = ""


def fetch_funding_rate(symbol: str) -> float:
    """Return the current funding rate for ``symbol``."""
    # Check for mock funding rate first
    mock = os.getenv("MOCK_FUNDING_RATE")
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            logger.warning(f"Invalid MOCK_FUNDING_RATE value: {mock}, using 0.0")
            return 0.0

    # Check if funding rate fetching is disabled
    base_url = os.getenv("FUNDING_RATE_URL", DEFAULT_FUNDING_URL)
    if not base_url:
        logger.debug(f"Funding rate fetching disabled for {symbol}, returning 0.0")
        return 0.0

    try:
        url = f"{base_url}{symbol}" if "?" in base_url else f"{base_url}?pair={symbol}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            # Kraken-style response
            if "result" in data and isinstance(data["result"], dict):
                first = next(iter(data["result"].values()), {})
                rate = first.get("fundingRate") or first.get("fr") or first.get("rate")
                if rate is not None:
                    return float(rate)

            # Alternative API response formats
            if "rates" in data and isinstance(data["rates"], list) and data["rates"]:
                last = data["rates"][-1]
                if isinstance(last, dict):
                    rate = last.get("relativeFundingRate") or last.get("fundingRate") or last.get("rate")
                    if rate is not None:
                        return float(rate)

            # Direct rate field
            rate = data.get("rate") or data.get("fundingRate") or data.get("funding_rate")
            if rate is not None:
                return float(rate)

        logger.debug(f"No funding rate found in response for {symbol}, returning 0.0")
        return 0.0

    except requests.exceptions.RequestException as exc:
        logger.debug(f"Network error fetching funding rate for {symbol}: {exc}, returning 0.0")
        return 0.0
    except (ValueError, KeyError, TypeError) as exc:
        logger.debug(f"Data parsing error fetching funding rate for {symbol}: {exc}, returning 0.0")
        return 0.0
    except Exception as exc:
        logger.warning(f"Unexpected error fetching funding rate for {symbol}: {exc}, returning 0.0")
        return 0.0


def calc_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Calculate the Average True Range using cached values."""
    try:
        # Validate input parameters
        if window < 1:
            logger.warning(f"Invalid window size for ATR: {window}, using default of 14")
            window = 14

        if df is None or df.empty:
            logger.debug("DataFrame is None or empty for ATR calculation")
            return 0.0

        # Check if required columns exist
        required_cols = ["high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"DataFrame missing required columns for ATR: {required_cols}")
            return 0.0

        if len(df) < window + 1:
            logger.debug(f"DataFrame too short for ATR calculation: {len(df)} rows, need {window + 1}")
            return 0.0

        # Use only the most recent data needed
        lookback = min(len(df), window + 1)
        recent = df.iloc[-lookback:].copy()

        # Validate data quality
        if recent.empty:
            logger.debug("Recent data slice is empty")
            return 0.0

        # Check for NaN values and replace with forward fill
        for col in required_cols:
            if recent[col].isna().any():
                logger.debug(f"Found NaN values in {col} column, attempting to fill")
                recent[col] = recent[col].fillna(method='ffill')
                if recent[col].isna().any():
                    logger.debug(f"Still have NaN values in {col} after forward fill")
                    return 0.0

        # Calculate True Range components
        high_low = recent["high"] - recent["low"]
        high_close = (recent["high"] - recent["close"].shift(1)).abs()
        low_close = (recent["low"] - recent["close"].shift(1)).abs()

        # Calculate True Range as the maximum of the three components
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Remove any remaining NaN values (first value will be NaN due to shift)
        tr = tr.dropna()

        if len(tr) < window:
            logger.debug(f"Not enough valid TR values: {len(tr)} available, need {window}")
            return 0.0

        # Calculate ATR using exponential moving average for better responsiveness
        # Fall back to simple moving average if EMA fails
        try:
            atr_value = float(tr.ewm(span=window, adjust=False).mean().iloc[-1])
        except Exception:
            logger.debug("EMA calculation failed, using SMA")
            atr_value = float(tr.rolling(window, min_periods=1).mean().iloc[-1])

        # Validate final ATR value
        if pd.isna(atr_value) or atr_value <= 0 or not np.isfinite(atr_value):
            logger.debug(f"Invalid ATR value calculated: {atr_value}")
            return 0.0

        # Cache the result for future use
        try:
            atr_series = pd.Series([atr_value], index=[df.index[-1]])
            cache_series(f"atr_{window}", df.iloc[-1:], atr_series, 1)
        except Exception as cache_error:
            logger.debug(f"Failed to cache ATR value: {cache_error}")

        return atr_value

    except Exception as e:
        logger.warning(f"Unexpected error calculating ATR: {e}")
        return 0.0


def too_flat(df: pd.DataFrame, min_atr_pct: float) -> bool:
    """Return True if ATR is below ``min_atr_pct`` of price."""
    atr = calc_atr(df)
    price = df["close"].iloc[-1]
    if price == 0:
        return True
    return bool(atr / price < min_atr_pct)


def too_hot(symbol: str, max_funding_rate: float) -> bool:
    """Return True when funding rate exceeds ``max_funding_rate``."""
    rate = fetch_funding_rate(symbol)
    return bool(rate > max_funding_rate)

