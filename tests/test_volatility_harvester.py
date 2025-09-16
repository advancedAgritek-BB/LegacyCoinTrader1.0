import numpy as np
import pandas as pd

from crypto_bot.strategy import volatility_harvester


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
