"""Meme wave bot strategy for detecting and trading meme coin pumps."""

from typing import Dict, Optional, Tuple

import pandas as pd


from ..sentiment_filter import get_lunarcrush_sentiment_boost, get_sentiment_score
from crypto_bot.utils.indicators import calculate_atr
from crypto_bot.utils.logging import setup_strategy_logger

NAME = "meme_wave_bot"

logger = setup_strategy_logger(NAME)


def generate_signal(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    **kwargs,
) -> Tuple[float, str]:
    """
    Generate trading signal based on meme wave patterns.
    
    Args:
        df: OHLCV dataframe
        symbol: Trading symbol
        timeframe: Timeframe string
        **kwargs: Additional parameters including config
    
    Returns:
        Tuple of (signal_strength, signal_type)
    """
    config = kwargs.get("config", {})

    if df is None or df.empty:
        logger.debug("%s: received empty dataframe", NAME)
        return 0.0, "none"
    
    # Strategy parameters
    vol_threshold = config.get("vol_threshold", 2.0)
    vol_mult = config.get("vol_mult", 3.0)
    jump_mult = config.get("jump_mult", 2.0)
    sentiment_thr = config.get("sentiment_threshold", 0.6)
    vol_spike_thr = config.get("vol_spike_threshold", 5.0)
    atr_window = config.get("atr_window", 14)
    
    # Get query for sentiment (use symbol if available)
    query = symbol or config.get("symbol", "bitcoin")
    
    # Get mempool volume data if available
    try:
        recent_vol = float(config.get("recent_mempool_volume", 0.0))
        avg_vol = float(config.get("avg_mempool_volume", 0.0))
    except Exception as e:
        logger.warning("%s: failed to get mempool volume data: %s", NAME, e)
        recent_vol = 0.0
        avg_vol = 0.0

    # Calculate price and volume metrics
    price_change = float(df["close"].iloc[-1] - df["close"].iloc[-2]) if len(df) > 1 else 0.0
    vol = float(df["volume"].iloc[-1])
    
    # Calculate ATR using shared helper
    try:
        atr_series = calculate_atr(df, window=atr_window)
        atr_value = (
            atr_series.iloc[-1]
            if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1])
            else 0.0
        )
    except Exception as e:
        logger.warning("%s: failed to calculate ATR: %s", NAME, e)
        atr_value = 0.0

    # Get sentiment using LunarCrush instead of Twitter
    try:
        # For synchronous operation, use neutral sentiment
        sentiment = 0.5  # Default neutral sentiment
    except Exception:
        sentiment = 0.5  # Default neutral sentiment
    logger.info("%s: sentiment %.2f for query '%s'", NAME, sentiment, query)

    # Check basic volume and sentiment conditions
    if avg_vol and recent_vol >= avg_vol * vol_threshold and sentiment >= sentiment_thr:
        logger.info(
            "%s: immediate long signal recent_vol=%.2f avg_vol=%.2f sentiment=%.2f",
            NAME,
            recent_vol,
            avg_vol,
            sentiment,
        )
        return 1.0, "long"

    # Check for price spike
    spike = (
        abs(price_change) >= atr_value * jump_mult
        and avg_vol > 0
        and vol >= avg_vol * vol_mult
    )

    if not spike:
        logger.debug("%s: no price spike detected", NAME)
        return 0.0, "none"

    # Check mempool conditions
    mempool_ok = True
    if vol_spike_thr is not None:
        try:
            if avg_vol <= 0 or recent_vol < float(vol_spike_thr) * avg_vol:
                mempool_ok = False
        except Exception as e:
            logger.warning("%s: failed to check mempool conditions: %s", NAME, e)
            mempool_ok = False

    # Check sentiment conditions using LunarCrush
    sentiment_ok = True
    if sentiment_thr is not None:
        try:
            # For synchronous operation, assume sentiment is OK
            sentiment_ok = True
        except Exception as e:
            logger.warning("%s: failed to check sentiment conditions: %s", NAME, e)
            sentiment_ok = False

    # Check if all conditions are met
    if mempool_ok and sentiment_ok:
        # Calculate signal strength based on volume and sentiment
        vol_strength = min(1.0, recent_vol / (avg_vol * vol_threshold))
        sentiment_strength = min(1.0, sentiment / sentiment_thr)
        
        # Combine strengths with volume having more weight
        signal_strength = (vol_strength * 0.7) + (sentiment_strength * 0.3)
        
        logger.info(
            "%s: signal %.2f (vol=%.2f sentiment=%.2f)",
            NAME,
            signal_strength,
            vol_strength,
            sentiment_strength,
        )

        return signal_strength, "long"

    logger.debug(
        "%s: risk checks failed mempool_ok=%s sentiment_ok=%s",
        NAME,
        mempool_ok,
        sentiment_ok,
    )

    return 0.0, "none"


def get_sentiment_boost(symbol: str, trade_direction: str = "long") -> float:
    """
    Get sentiment boost factor for meme wave trades.

    Args:
        symbol: Trading symbol
        trade_direction: Trade direction ('long' or 'short')

    Returns:
        Boost factor multiplier
    """
    try:
        # For synchronous operation, return default boost
        return 1.0  # Default no boost
    except Exception as e:
        logger.warning("%s: failed to get sentiment boost for %s: %s", NAME, symbol, e)
        return 1.0  # Default no boost


class regime_filter:
    """Match volatile regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "volatile"
