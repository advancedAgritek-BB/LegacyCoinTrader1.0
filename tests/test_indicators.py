import pandas as pd
import numpy as np
import pytest

from crypto_bot.utils.indicators import (
    BollingerBands,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_rsi,
)


pytestmark = pytest.mark.regression


def test_calculate_rsi_matches_manual():
    close = pd.Series([1, 2, 3, 2, 4, 5, 6], dtype=float)
    window = 3

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    expected = 100 - (100 / (1 + rs))

    result = calculate_rsi(close, window=window)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_atr_matches_manual():
    data = pd.DataFrame(
        {
            "high": [10, 12, 13, 11, 14, 15],
            "low": [8, 9, 10, 9, 11, 12],
            "close": [9, 11, 12, 10, 13, 14],
        }
    )
    window = 3

    high_low = data["high"] - data["low"]
    high_close = (data["high"] - data["close"].shift(1)).abs()
    low_close = (data["low"] - data["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    expected = tr.rolling(window=window, min_periods=window).mean()

    result = calculate_atr(data, window=window)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_bollinger_bands_matches_manual():
    close = pd.Series(np.linspace(100, 110, num=10))
    window = 4
    num_std = 2.5

    rolling = close.rolling(window=window, min_periods=window)
    middle = rolling.mean()
    std = rolling.std()
    upper = middle + std * num_std
    lower = middle - std * num_std
    width = upper - lower

    bands = calculate_bollinger_bands(close, window=window, num_std=num_std)

    assert isinstance(bands, BollingerBands)
    pd.testing.assert_series_equal(bands.middle, middle, check_names=False)
    pd.testing.assert_series_equal(bands.upper, upper, check_names=False)
    pd.testing.assert_series_equal(bands.lower, lower, check_names=False)
    pd.testing.assert_series_equal(bands.width, width, check_names=False)


def test_invalid_window_raises_value_error():
    series = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        calculate_rsi(series, window=0)
    with pytest.raises(ValueError):
        calculate_atr(pd.DataFrame({"high": series, "low": series, "close": series}), window=0)
    with pytest.raises(ValueError):
        calculate_bollinger_bands(series, window=0)
