"""Statistical arbitrage pair trading strategy."""

from typing import Optional, Tuple

import logging
import pandas as pd
import numpy as np

from crypto_bot.utils import stats
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.ml_utils import init_ml_or_warn, load_model

logger = logging.getLogger(__name__)

ML_AVAILABLE = init_ml_or_warn()

NAME = "stat_arb_bot"
if ML_AVAILABLE:
    MODEL = load_model("stat_arb_bot")
else:  # pragma: no cover - fallback
    MODEL = None

_ZSCORE_THRESHOLD_DEFAULT = 2.0
_LOOKBACK_DEFAULT = 20
_CORRELATION_THRESHOLD = 0.8


def generate_signal(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    **kwargs,
) -> Tuple[float, str]:
    """Return (score, direction) based on the price spread z-score."""
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config")

    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return 0.0, "none"

    threshold = (
        float(config.get("zscore_threshold", _ZSCORE_THRESHOLD_DEFAULT))
        if config
        else _ZSCORE_THRESHOLD_DEFAULT
    )
    lookback = (
        int(config.get("lookback", _LOOKBACK_DEFAULT)) if config else _LOOKBACK_DEFAULT
    )

    if len(df_a) < lookback or len(df_b) < lookback:
        return 0.0, "none"

    corr = df_a["close"].corr(df_b["close"])
    if pd.isna(corr) or corr < _CORRELATION_THRESHOLD:
        return 0.0, "none"

    spread = df_a["close"] - df_b["close"]
    z = stats.zscore(spread, lookback)
    z = cache_series("stat_arb_z", df_a, z, lookback)
    if z.empty:
        return 0.0, "none"

    z_last = z.iloc[-1]
    if abs(z_last) <= threshold:
        return 0.0, "none"

    direction = "long" if z_last < 0 else "short"
    score = float(abs(z_last))

    if score > 0:
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                # Ensure DataFrame is still valid before ML call
                if not isinstance(df_a, pd.DataFrame) or not hasattr(df_a, 'empty'):
                    logger.warning("DataFrame corrupted before ML prediction, skipping ML score")
                    ml_score = 0.5  # Default neutral score
                else:
                    ml_score = MODEL.predict(df_a)
                    # Validate that DataFrame is still intact after ML call
                    if not isinstance(df_a, pd.DataFrame) or not hasattr(df_a, 'empty'):
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
            score = normalize_score_by_volatility(df_a, score)

    return score, direction


class regime_filter:
    """Match mean-reverting regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "mean-reverting"
