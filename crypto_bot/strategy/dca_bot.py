from typing import Optional, Tuple

import pandas as pd

from crypto_bot.utils.logging import setup_strategy_logger

STRATEGY_NAME = __name__.split(".")[-1]
logger = setup_strategy_logger(STRATEGY_NAME)


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Simple dollar-cost averaging signal."""
    if df is None or df.empty:
        logger.debug("%s received empty dataframe; skipping DCA signal.", STRATEGY_NAME)
        return 0.0, "none"

    ma = df["close"].rolling(20).mean().iloc[-1]
    last_close = df["close"].iloc[-1]
    if last_close < ma * 0.9:
        score = 0.8
        logger.info(
            "%s generated accumulation signal (price %.2f, MA %.2f, score %.2f).",
            STRATEGY_NAME,
            last_close,
            ma,
            score,
        )
        return score, "long"

    logger.debug(
        "%s no accumulation signal (price %.2f vs MA %.2f).",
        STRATEGY_NAME,
        last_close,
        ma,
    )
    return 0.0, "none"


class regime_filter:
    """DCA bot works across regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return True
