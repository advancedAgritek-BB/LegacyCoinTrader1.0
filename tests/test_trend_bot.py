import pandas as pd
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import ta

spec = importlib.util.spec_from_file_location(
    "trend_bot",
    Path(__file__).resolve().parents[1]
    / "crypto_bot"
    / "strategy"
    / "trend_bot.py",
)
trend_bot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(trend_bot)


def _df_trend(volume_last: float, high_equals_close: bool = False):
    # Create more realistic price data with ups and downs for RSI calculation
    np.random.seed(42)
    base_price = 100.0
    # Create price data that stays positive and has enough variation for RSI
    close = pd.Series([base_price + i * 0.5 + np.random.normal(0, 1) for i in range(60)], dtype=float)
    close = close.abs() + 10  # Ensure all prices are positive and above 10
    
    if high_equals_close:
        high = close
    else:
        high = close + np.random.uniform(0.1, 1.0, len(close))
    low = close - np.random.uniform(0.1, 1.0, len(close))
    low = low.clip(lower=1.0)  # Ensure low prices are positive
    
    # Create volume data that ensures volume_ok=True for the test
    volume = pd.Series([100.0] * 59 + [volume_last])
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})
    
    # Calculate RSI for the test data
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    
    return df


def _df_adx_range():
    np.random.seed(42)  # Use same seed as _df_trend for consistent behavior
    # Create strongly trending upward price data to get high RSI and ADX
    base_price = 100.0
    # Create stronger upward trend to get overbought RSI
    close = pd.Series([base_price + i * 1.0 + np.random.normal(0, 0.5) for i in range(60)], dtype=float)
    close = close.abs() + 10  # Ensure all prices are positive
    
    high = close + np.random.uniform(0.1, 1.0, len(close))
    low = close - np.random.uniform(0.1, 1.0, len(close))
    low = low.clip(lower=1.0)  # Ensure low prices are positive
    
    # Use high volume to ensure volume condition is met
    volume = pd.Series([100.0] * 59 + [300.0])  # High final volume like _df_trend(150.0)
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})
    
    return df


def test_no_signal_when_volume_below_ma():
    df = _df_trend(50.0)
    score, direction = trend_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_long_signal_with_filters():
    df = _df_trend(150.0)
    cfg = {"donchian_confirmation": False, "indicator_lookback": 20}
    score, direction = trend_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0.0


def test_donchian_confirmation_blocks_false_breakout():
    df = _df_trend(150.0)
    cfg = {"donchian_confirmation": True}
    score, direction = trend_bot.generate_signal(df, cfg)
    assert direction == "none"
    assert score == 0.0


def test_donchian_confirmation_allows_breakout():
    df = _df_trend(150.0, high_equals_close=True)
    cfg = {"donchian_confirmation": True}
    score, direction = trend_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0.0


def test_rsi_zscore(monkeypatch):
    df = _df_trend(150.0)
    monkeypatch.setattr(
        trend_bot.stats,
        "zscore",
        lambda s, lookback=20: pd.Series([2] * len(s), index=s.index),
    )
    cfg = {"indicator_lookback": 20, "rsi_overbought_pct": 90, "donchian_confirmation": False}
    score, direction = trend_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0


def test_adx_threshold(monkeypatch):
    df = _df_adx_range()
    adx_values = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=7).adx()
    adx_val = adx_values[-1] if hasattr(adx_values, '__getitem__') else adx_values.iloc[-1]
    assert 25 <= adx_val <= 30

    monkeypatch.setattr(
        trend_bot.stats,
        "zscore",
        lambda s, lookback=250: pd.Series([2] * len(s), index=s.index),
    )

    score, direction = trend_bot.generate_signal(
        df, {"donchian_confirmation": False, "adx_threshold": adx_val + 1}
    )
    assert direction == "none"
    assert score == 0.0

    score, direction = trend_bot.generate_signal(
        df, {"donchian_confirmation": False, "adx_threshold": adx_val - 1}
    )
    assert direction != "none"
    assert score > 0


def test_torch_signal_default_weight(monkeypatch):
    df = _df_trend(150.0, high_equals_close=True)
    base_score, _ = trend_bot.generate_signal(df, {"donchian_confirmation": False})

    class _FakeTorch:
        @staticmethod
        def predict_signal(_df):
            return 0.2

    monkeypatch.setitem(sys.modules, "crypto_bot.torch_signal_model", _FakeTorch)
    cfg = {"donchian_confirmation": False, "torch_signal_model": {"enabled": True}}
    score, _ = trend_bot.generate_signal(df, cfg)
    expected = base_score * 0.3 + 0.2 * 0.7
    assert score == pytest.approx(expected)
