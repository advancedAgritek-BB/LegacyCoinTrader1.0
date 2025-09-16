from typing import Optional, Tuple

import logging
import pandas as pd

from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.ml_utils import init_ml_or_warn, load_model
import numpy as np

from crypto_bot.strategy.base import FunctionStrategy

logger = logging.getLogger(__name__)

NAME = "momentum_bot"

ML_AVAILABLE = init_ml_or_warn()
MODEL: Optional[object]
if ML_AVAILABLE:
    MODEL = load_model("momentum_bot")
else:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    **kwargs,
) -> Tuple[float, str]:
    """Donchian breakout with volume confirmation."""
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config")

    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("momentum_bot", {}) if config else {}
    window = int(params.get("donchian_window", 20))
    vol_window = int(params.get("volume_window", 20))
    vol_z_min = float(params.get("volume_z_min", 0.5))
    rsi_threshold = float(params.get("rsi_threshold", 55))
    macd_min = float(params.get("macd_min", 0.0))
    macd_fast = int(params.get("fast_length", 12))
    macd_slow = int(params.get("slow_length", 26))
    rsi_window = 14

    min_len = max(window, vol_window, macd_slow, rsi_window)
    if len(df) < min_len:
        return 0.0, "none"

    lookback = min(len(df), min_len)
    recent = df.iloc[-(lookback + 1) :]

    dc_low = recent["low"].rolling(window).min().shift(1)
    vol_ma = recent["volume"].rolling(vol_window).mean()
    vol_std = recent["volume"].rolling(vol_window).std()
    # Calculate RSI manually
    delta = recent["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate MACD manually
    ema_fast = recent["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = recent["close"].ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()  # Standard signal line
    macd_hist = macd_line - macd_signal
    macd = macd_line  # Return the main MACD line

    dc_low = cache_series("momentum_dc_low", df, dc_low, lookback)
    vol_ma = cache_series("momentum_vol_ma", df, vol_ma, lookback)
    vol_std = cache_series("momentum_vol_std", df, vol_std, lookback)
    volume_z = cache_series(
        "momentum_volume_z", df, (recent["volume"] - vol_ma) / vol_std, lookback
    )
    rsi = cache_series("momentum_rsi", df, rsi, lookback)
    macd = cache_series("momentum_macd", df, macd, lookback)

    recent = recent.copy()
    recent["dc_low"] = dc_low
    recent["vol_ma"] = vol_ma
    recent["vol_std"] = vol_std
    recent["volume_z"] = volume_z
    recent["rsi"] = rsi
    recent["macd"] = macd

    latest = recent.iloc[-1]

    macd_val = latest["macd"]
    rsi_val = latest["rsi"]
    volume_z = latest["volume_z"]

    score = 0.0
    direction = "none"

    long_cond = macd_val > 0 or rsi_val > 50
    short_cond = (
        latest["close"] < dc_low.iloc[-1]
        and latest["rsi"] < 100 - rsi_threshold
        and latest["macd"] < -macd_min
    )

    if long_cond:
        score = 0.8
        direction = "long"
        logger.info(
            f"momentum_bot long signal: MACD={macd_val}, RSI={rsi_val}"
        )
    elif short_cond and volume_z > vol_z_min:
        score = min(1.0, volume_z / 2)
        direction = "short"

    if score > 0:
        if MODEL is not None:
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
                    score = (score + ml_score) / 2
            except Exception as e:
                logger.warning(f"ML prediction failed, using base score: {e}")
                # Continue with base score if ML fails
                pass
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(recent, score)

    logger.info(
        "RSI %.2f MACD %.5f score %.2f direction %s",
        float(latest.get("rsi", float("nan"))),
        float(latest.get("macd", float("nan"))),
        score,
        direction,
    )
    return score, direction


class regime_filter:
    """Match trending and volatile regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime in {"trending", "volatile"}


class Strategy(FunctionStrategy):
    """Adapter exposing :func:`generate_signal` via :class:`BaseStrategy`."""

    def __init__(self) -> None:
        super().__init__(
            generate_signal,
            name=NAME,
            extras={"regime_filter": regime_filter},
        )

    def signal(self, *args, **kwargs):
        """Compatibility wrapper for pipelines expecting ``signal``."""
        return self(*args, **kwargs)


__all__ = ["generate_signal", "regime_filter", "Strategy"]
