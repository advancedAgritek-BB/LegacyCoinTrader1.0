import importlib
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_bot.strategy import volatility_harvester


VOLATILITY_MODULE = importlib.import_module("crypto_bot.strategy.volatility_harvester")


TEST_CONFIG = {
    "atr_threshold": 0.0005,
    "atr_multiplier": 1.1,
    "volume_zscore_threshold": 0.5,
    "volume_spike": 0.5,
    "range_expansion_threshold": 0.5,
}


def _build_sample_ohlcv(direction: str) -> pd.DataFrame:
    periods = 60
    index = pd.date_range("2024-01-01", periods=periods, freq="T")

    if direction == "long":
        close = 100 + np.linspace(0, 5, periods)
        close[-5:] += np.linspace(0.5, 3.0, 5)
        close[-1] += 2.0
        high = close + 1.5
        low = close - 1.5
        high[-1] = close[-1] + 4.0
        low[-1] = close[-1] - 2.0
    else:
        close = 110 - np.linspace(0, 5, periods)
        close[-5:] -= np.linspace(0.5, 3.0, 5)
        close[-1] -= 2.0
        high = close + 1.5
        low = close - 1.5
        high[-1] = close[-1] + 2.0
        low[-1] = close[-1] - 4.5

    open_ = np.concatenate(([close[0]], close[:-1]))
    volume = 100 + (np.arange(periods) % 5)
    volume[-1] = 320

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


def test_volatility_harvester_breakout_long_signal():
    df = _build_sample_ohlcv("long")
    score, direction = volatility_harvester.generate_signal(df, config=TEST_CONFIG)

    assert direction == "long"
    assert score > 0


def test_volatility_harvester_breakdown_short_signal():
    df = _build_sample_ohlcv("short")
    score, direction = volatility_harvester.generate_signal(df, config=TEST_CONFIG)

    assert direction == "short"
    assert score > 0


NORMALIZATION_CONFIG = {
    "atr_threshold": 0.0003,
    "atr_multiplier": 5.0,
    "volume_zscore_threshold": 10.0,
    "volume_spike": 10.0,
    "range_expansion_threshold": 5.0,
    "min_price_change": 0.0001,
}


def _build_normalization_frame(factor: float) -> pd.DataFrame:
    periods = 120
    index = pd.date_range("2024-01-01", periods=periods, freq="min")

    close = 100 + np.linspace(0, 1.5, periods)
    close[-15:] += np.linspace(0, 3, 15)
    open_ = np.concatenate(([close[0]], close[:-1]))

    base_range = np.full(periods, 0.8)
    base_range[-20:] = base_range[-20:] * factor
    high = close + base_range
    low = close - base_range

    volume = np.full(periods, 600.0)
    volume += np.linspace(0, 20, periods)
    volume[-5:] += np.linspace(120, 200, 5)

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
        VOLATILITY_MODULE,
        "normalize_score_by_volatility",
        side_effect=lambda data, score, *args, **kwargs: score,
    ):
        return VOLATILITY_MODULE.generate_signal(df, config=NORMALIZATION_CONFIG)


def test_volatility_harvester_normalization_scales_with_atr():
    low_df = _build_normalization_frame(0.25)
    high_df = _build_normalization_frame(2.0)

    raw_low, dir_low_raw = _score_without_normalization(low_df)
    raw_high, dir_high_raw = _score_without_normalization(high_df)

    score_low, dir_low = VOLATILITY_MODULE.generate_signal(low_df, config=NORMALIZATION_CONFIG)
    score_high, dir_high = VOLATILITY_MODULE.generate_signal(high_df, config=NORMALIZATION_CONFIG)

    assert dir_low_raw == dir_high_raw == "long"
    assert dir_low == dir_high == "long"

    assert score_low < raw_low
    assert score_high > raw_high
    assert score_high > score_low
