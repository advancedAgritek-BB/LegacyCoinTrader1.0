from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.indicators import calculate_atr, calculate_bollinger_bands, calculate_rsi
from crypto_bot.utils import stats
from crypto_bot.utils.logging_utils import setup_strategy_logger
from crypto_bot.utils.ml_utils import init_ml_or_warn, load_model
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.cooldown_manager import in_cooldown, mark_cooldown

NAME = "dip_hunter"

logger = setup_strategy_logger(NAME)
score_logger = setup_strategy_logger(f"{NAME}.score")

ML_AVAILABLE = init_ml_or_warn()
if ML_AVAILABLE:
    MODEL = load_model("dip_hunter")
else:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    **kwargs,
) -> Union[Tuple[float, str], Tuple[float, str, dict]]:
    """Detect deep dips for mean reversion long entries.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV data.
    symbol : str, optional
        Asset symbol. Kept for compatibility with other strategies.
    timeframe : str, optional
        Data timeframe. Unused but accepted for interface compatibility.
    **kwargs : dict
        May contain ``higher_df`` and ``config`` for advanced behaviour.
    """

    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    higher_df = kwargs.get("higher_df")
    config = kwargs.get("config")

    symbol = symbol or (config.get("symbol", "") if config else "")
    params = config.get("dip_hunter", {}) if config else {}
    cooldown_enabled = bool(params.get("cooldown_enabled", False))

    if cooldown_enabled and symbol and in_cooldown(symbol, "buy"):
        logger.info(
            "%s: cooldown active for %s on %s", NAME, symbol, timeframe or "N/A"
        )
        score_logger.info(
            "Signal for %s:%s -> %.3f, %s",
            symbol or "unknown",
            timeframe or "N/A",
            0.0,
            "cooldown",
        )
        return 0.0, "none"

    rsi_window = int(params.get("rsi_window", 14))
    rsi_oversold = float(params.get("rsi_oversold", 30.0))
    dip_pct = float(params.get("dip_pct", 0.03))
    dip_bars = int(params.get("dip_bars", 3))
    vol_window = int(params.get("vol_window", 20))
    vol_mult = float(params.get("vol_mult", 1.5))
    adx_window = int(params.get("adx_window", 14))
    adx_threshold = float(params.get("adx_threshold", 25.0))
    bb_window = int(params.get("bb_window", 20))
    ema_trend = int(params.get("ema_trend", 200))
    ml_weight = float(params.get("ml_weight", 0.5))
    atr_normalization = bool(params.get("atr_normalization", True))
    ema_slow = int(params.get("ema_slow", 20))

    min_bars = max(100, adx_window, rsi_window, ema_slow) + 5
    required_bars = max(min_bars, 2 * adx_window + 1)

    lookback = max(rsi_window, vol_window, adx_window, bb_window, dip_bars)
    recent = df.tail(required_bars)
    if len(recent) < required_bars:
        logger.debug(
            "%s: insufficient candles (have %d need %d)",
            NAME,
            len(recent),
            required_bars,
        )
        return 0.0, "none"

    rsi = calculate_rsi(recent["close"], window=rsi_window)

    # ADX requires at least twice the window length for a stable reading
    if len(recent) < 2 * adx_window:
        logger.debug("%s: insufficient data for ADX window", NAME)
        return 0.0, "none"

    # Calculate ADX manually (simplified)
    high_diff = recent["high"].diff()
    low_diff = recent["low"].diff()

    dm_plus = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
    dm_minus = ((low_diff > high_diff) & (low_diff > 0)) * (-low_diff)

    atr = calculate_atr(recent, window=adx_window)
    di_plus = 100 * (dm_plus.rolling(window=adx_window).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(window=adx_window).mean() / atr)
    dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus))
    adx = dx.rolling(window=adx_window).mean()

    bb = calculate_bollinger_bands(recent["close"], window=bb_window, num_std=2)
    bb_pct = (recent["close"] - bb.lower) / (bb.upper - bb.lower)
    vol_ma = recent["volume"].rolling(vol_window).mean()

    rsi = cache_series("rsi_dip", df, rsi, lookback)
    adx = cache_series("adx_dip", df, adx, lookback)
    bb_pct = cache_series("bb_pct_dip", df, bb_pct, lookback)
    vol_ma = cache_series("vol_ma_dip", df, vol_ma, lookback)

    recent = recent.copy()
    recent["rsi"] = rsi
    recent["adx"] = adx
    recent["bb_pct"] = bb_pct
    recent["vol_ma"] = vol_ma

    latest = recent.iloc[-1]

    if len(recent) < dip_bars + 1:
        logger.debug("%s: not enough bars to assess dip", NAME)
        return 0.0, "none"
    recent_returns = recent["close"].pct_change().iloc[-dip_bars:]
    dip_size = recent_returns.sum()
    is_dip = dip_size <= -dip_pct

    if cooldown_enabled and symbol and in_cooldown(symbol, "buy"):
        logger.info(
            "%s: cooldown triggered after calculations for %s", NAME, symbol
        )
        score_logger.info(
            "Signal for %s:%s -> %.3f, %s",
            symbol or "unknown",
            timeframe or "N/A",
            0.0,
            "none",
        )
        return 0.0, "none"

    bb_condition = latest["bb_pct"] < 0
    if not bb_condition and is_dip and latest["bb_pct"] < 0.3:
        logger.debug(
            "%s: bb_pct %.2f still elevated, treating as oversold due to dip %.4f",
            NAME,
            latest["bb_pct"],
            dip_size,
        )
        bb_condition = True
    oversold = latest["rsi"] < rsi_oversold and bb_condition
    vol_spike = (
        latest["volume"] > latest["vol_ma"] * vol_mult if latest["vol_ma"] > 0 else False
    )

    range_bound = latest["adx"] < adx_threshold
    if higher_df is not None and not higher_df.empty:
        h_lookback = max(ema_trend, 1)
        h_recent = higher_df.iloc[-(h_lookback + 1) :]
        # Calculate EMA manually
        ema_h = h_recent["close"].ewm(span=ema_trend, adjust=False).mean()
        ema_h = cache_series("ema_trend_h", higher_df, ema_h, h_lookback)
        in_trend = higher_df["close"].iloc[-1] > ema_h.iloc[-1]
    else:
        # Default to a neutral stance when higher timeframe data is unavailable
        in_trend = True

    favorable_regime = range_bound or in_trend

    if is_dip and oversold and vol_spike and favorable_regime:
        dip_score = min(abs(dip_size) / dip_pct, 1.0)
        oversold_score = min((rsi_oversold - latest["rsi"]) / rsi_oversold, 1.0)
        vol_z = stats.zscore(recent["volume"], vol_window).iloc[-1]
        vol_score = min(max(vol_z / 2, 0), 1.0)
        score = dip_score * 0.4 + oversold_score * 0.3 + vol_score * 0.3

        if MODEL:
            try:  # pragma: no cover - best effort
                # Ensure DataFrame is still valid before ML call
                if not isinstance(df, pd.DataFrame) or not hasattr(df, 'empty'):
                    logger.warning("DataFrame corrupted before ML prediction, skipping ML score")
                    ml_score = 0.5  # Default neutral score
                else:
                    ml_score = MODEL.predict(df)
                    # Validate that DataFrame is still intact after ML call
                    if not isinstance(df, pd.DataFrame) or not hasattr(df, 'empty'):
                        logger.warning("DataFrame corrupted after ML prediction, using default ML score")
                        ml_score = 0.5  # Default neutral score
                    else:
                        # Ensure ml_score is a valid number
                        if isinstance(ml_score, (list, np.ndarray)):
                            ml_score = float(ml_score[0]) if len(ml_score) > 0 else 0.5
                        else:
                            ml_score = float(ml_score)
                score = score * (1 - ml_weight) + ml_score * ml_weight
            except Exception as e:
                logger.warning(f"ML prediction failed, using base score: {e}")
                # Continue with base score if ML fails
                pass

        if atr_normalization:
            score = normalize_score_by_volatility(df, score)

        score = max(0.0, min(score, 1.0))
        logger.info(
            "%s: long signal %.3f dip=%.4f oversold=%s vol_spike=%s",
            NAME,
            score,
            dip_size,
            oversold,
            vol_spike,
        )
        score_logger.info(
            "Signal for %s:%s -> %.3f, %s",
            symbol or "unknown",
            timeframe or "N/A",
            score,
            "long",
        )
        if cooldown_enabled and symbol:
            mark_cooldown(symbol, "buy")
        return score, "long"

    logger.debug(
        "%s: conditions not met dip=%s oversold=%s vol_spike=%s regime=%s",
        NAME,
        is_dip,
        oversold,
        vol_spike,
        favorable_regime,
    )

    return 0.0, "none"


class regime_filter:
    """Match mean-reverting regime for Dip Hunter."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "mean-reverting"
