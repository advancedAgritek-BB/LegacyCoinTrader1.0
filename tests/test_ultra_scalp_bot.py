import importlib
from unittest.mock import patch

import numpy as np
import pandas as pd


ULTRA_MODULE = importlib.import_module("crypto_bot.strategy.ultra_scalp_bot")

NORMALIZATION_CONFIG = {
    "min_score": 0.0,
    "volume_window": 6,
    "min_volume_zscore": -2.0,
    "min_atr_pct": 0.0001,
    "atr_window": 5,
}


def _build_scalp_frame(factor: float) -> pd.DataFrame:
    periods = 120
    index = pd.date_range("2024-01-01", periods=periods, freq="min")

    close = 100 + np.linspace(0, 3, periods)
    close[-15:] += np.linspace(0, 4, 15)
    open_ = np.concatenate(([close[0]], close[:-1]))

    range_base = np.full(periods, 0.4)
    range_base[-20:] = range_base[-20:] * factor
    high = close + range_base
    low = close - range_base

    volume = np.full(periods, 900.0)
    volume += np.linspace(0, 60, periods)
    volume[-8:] += np.linspace(150, 300, 8)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _score_without_normalization(df: pd.DataFrame) -> tuple[float, str]:
    with patch.object(
        ULTRA_MODULE,
        "normalize_score_by_volatility",
        side_effect=lambda data, score, *args, **kwargs: score,
    ):
        return ULTRA_MODULE.generate_signal(df, config=NORMALIZATION_CONFIG)


def test_ultra_scalp_bot_normalization_boosts_high_volatility_scores():
    low_df = _build_scalp_frame(0.4)
    high_df = _build_scalp_frame(2.0)

    raw_low, dir_low_raw = _score_without_normalization(low_df)
    raw_high, dir_high_raw = _score_without_normalization(high_df)

    score_low, dir_low = ULTRA_MODULE.generate_signal(low_df, config=NORMALIZATION_CONFIG)
    score_high, dir_high = ULTRA_MODULE.generate_signal(high_df, config=NORMALIZATION_CONFIG)

    assert dir_low_raw == dir_high_raw == "short"
    assert dir_low == dir_high == "short"

    assert score_low < raw_low
    assert score_high > raw_high
    assert score_high > score_low
