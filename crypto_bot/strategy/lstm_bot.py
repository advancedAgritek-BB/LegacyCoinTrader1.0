from typing import Optional, Tuple

import numpy as np
import pandas as pd

from crypto_bot.utils.logging import setup_strategy_logger
from crypto_bot.utils.ml_utils import init_ml_or_warn, load_model

NAME = "lstm_bot"

logger = setup_strategy_logger(NAME)

ML_AVAILABLE = init_ml_or_warn()
if ML_AVAILABLE:
    MODEL = load_model("lstm_bot")
else:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    **kwargs,
) -> Tuple[float, str]:
    """Return LSTM-based momentum signal."""
    # Handle type conversion from dict to DataFrame
    if isinstance(df, dict):
        try:
            df = pd.DataFrame.from_dict(df)
        except Exception:
            logger.warning("%s: failed to convert dictionary input to DataFrame", NAME)
            return 0.0, "none"

    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config")

    if df is None or not isinstance(df, pd.DataFrame):
        logger.debug("%s: invalid dataframe input", NAME)
        return 0.0, "none"

    if df.empty:
        logger.debug("%s: received empty dataframe", NAME)
        return 0.0, "none"

    params = config or {}
    _ = params.get("model_path")  # preserved for backward compatibility
    seq_len = int(params.get("sequence_length", 50))
    threshold = float(params.get("threshold_pct", 0.0))

    if len(df) < seq_len:
        logger.debug(
            "%s: insufficient history (have %d, need %d)", NAME, len(df), seq_len
        )
        return 0.0, "none"

    score = 0.0
    if MODEL is not None:
        try:  # pragma: no cover - best effort
            # Ensure DataFrame is still valid before ML call
            if not isinstance(df, pd.DataFrame) or not hasattr(df, 'empty'):
                logger.warning("DataFrame corrupted before ML prediction, skipping ML score")
                score = 0.0
            else:
                # Get the tail of the DataFrame for LSTM prediction
                df_tail = df.tail(seq_len)
                if not isinstance(df_tail, pd.DataFrame) or not hasattr(df_tail, 'empty'):
                    logger.warning("DataFrame tail corrupted before ML prediction, skipping ML score")
                    score = 0.0
                else:
                    prediction = MODEL.predict(df_tail)
                    # Validate that DataFrame is still intact after ML call
                    if not isinstance(df, pd.DataFrame) or not hasattr(df, 'empty'):
                        logger.warning("DataFrame corrupted after ML prediction, using default score")
                        score = 0.0
                    else:
                        # Ensure prediction is a valid number
                        if isinstance(prediction, (list, np.ndarray)):
                            score = float(prediction[0]) if len(prediction) > 0 else 0.0
                        else:
                            score = float(prediction)
        except Exception as e:
            logger.warning(f"ML prediction failed, using default score: {e}")
            score = 0.0

    direction = "none"
    if score > threshold:
        direction = "long"
        logger.info("%s: long signal %.3f (threshold %.3f)", NAME, score, threshold)
    elif score < -threshold:
        direction = "short"
        logger.info("%s: short signal %.3f (threshold %.3f)", NAME, score, threshold)
    else:
        logger.debug("%s: score %.3f within threshold %.3f", NAME, score, threshold)
        score = 0.0

    return score, direction


class regime_filter:
    """Match all regimes."""

    @staticmethod
    def matches(_regime: str) -> bool:
        return True
