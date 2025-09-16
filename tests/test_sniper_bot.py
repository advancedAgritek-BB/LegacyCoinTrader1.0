import pandas as pd

from crypto_bot.strategy import sniper_bot


def _last_metadata():
    return getattr(sniper_bot.generate_signal, "last_metadata", {})


def _df_with_volume_and_price(close_list, volume_list):
    return pd.DataFrame({
        "open": close_list,
        "high": close_list,
        "low": close_list,
        "close": close_list,
        "volume": volume_list,
    })


def test_sniper_triggers_on_breakout():
    df = _df_with_volume_and_price(
        [1.0, 1.05, 1.1, 1.2],
        [10, 12, 11, 200]
    )
    result = sniper_bot.generate_signal(df)
    assert isinstance(result, tuple)
    assert len(result) == 2
    score, direction = result
    metadata = _last_metadata()
    assert direction == "long"
    assert score > 0.8
    assert not metadata["event"]


def test_sniper_ignores_low_volume():
    df = _df_with_volume_and_price(
        [1.0, 1.05, 1.1, 1.2],
        [1, 1, 1, 2]
    )
    score, direction = sniper_bot.generate_signal(df)
    metadata = _last_metadata()
    assert direction == "none"
    assert score == 0.0
    assert not metadata["event"]


def test_sniper_respects_history_length():
    df = _df_with_volume_and_price([1.0] * 100, [10] * 100)
    score, direction = sniper_bot.generate_signal(df)
    metadata = _last_metadata()
    assert direction == "none"
    assert score == 0.0
    assert not metadata["event"]


def test_direction_override_short():
    df = _df_with_volume_and_price(
        [1.0, 1.05, 1.1, 1.2],
        [10, 12, 11, 200]
    )
    config = {"direction": "short"}
    score, direction = sniper_bot.generate_signal(df, config)
    metadata = _last_metadata()
    assert direction == "short"
    assert score > 0.8
    assert not metadata["event"]
    assert config["_sniper_metadata"]["atr"] >= 0.0


def test_auto_short_on_price_drop():
    df = _df_with_volume_and_price(
        [1.0, 0.95, 0.9, 0.8],
        [10, 12, 11, 200]
    )
    result = sniper_bot.generate_signal(df)
    assert isinstance(result, tuple)
    assert len(result) == 2
    score, direction = result
    metadata = _last_metadata()
    assert direction == "short"
    assert score > 0.8
    assert not metadata["event"]


def test_high_freq_short_window():
    df = _df_with_volume_and_price(
        [1.0, 1.1, 1.2],
        [10, 12, 40]
    )
    score, direction = sniper_bot.generate_signal(
        df, high_freq=True, config={"min_volume": 1}
    )
    metadata = _last_metadata()
    assert direction == "long"
    assert score > 0.8
    assert not metadata["event"]


def test_symbol_filter_blocks_disallowed():
    df = _df_with_volume_and_price(
        [1.0, 1.1, 1.2],
        [10, 12, 40]
    )
    score, direction = sniper_bot.generate_signal(
        df, {"symbol": "XRP/USD"}
    )
    metadata = _last_metadata()
    assert direction == "none"
    assert score == 0.0
    assert not metadata["event"]


def test_event_trigger():
    df = pd.DataFrame({
        "open": [1, 1, 1, 1, 1],
        "high": [1.1, 1.1, 1.1, 1.1, 5],
        "low": [0.9, 0.9, 0.9, 0.9, 1],
        "close": [1, 1, 1, 1, 5],
        "volume": [10, 10, 10, 10, 50],
    })
    score, direction = sniper_bot.generate_signal(
        df, config={"atr_window": 4, "volume_window": 4, "min_volume": 1}
    )
    metadata = _last_metadata()
    assert direction == "long"
    assert score > 0
    assert metadata["event"]


def test_defaults_trigger_on_small_breakout():
    df = _df_with_volume_and_price(
        [1.0, 1.0, 1.0, 1.06],
        [100, 100, 100, 160],
    )
    score, direction = sniper_bot.generate_signal(df)
    metadata = _last_metadata()
    assert direction == "long"
    assert score > 0
    assert not metadata["event"]


def test_fallback_short_no_breakout():
    df = _df_with_volume_and_price(
        [1.0, 0.99, 0.98, 0.97],
        [120, 110, 100, 130],
    )
    score, direction = sniper_bot.generate_signal(df)
    metadata = _last_metadata()
    assert direction == "short"
    assert score > 0
    assert not metadata["event"]


def test_price_fallback_long_signal():
    bars = 15
    df = pd.DataFrame({
        "open": [1.0] * bars,
        "high": [1.05] * (bars - 1) + [1.25],
        "low": [0.95] * (bars - 1) + [1.0],
        "close": [1.0] * (bars - 1) + [1.25],
        "volume": [100] * (bars - 1) + [250],
    })
    score, direction = sniper_bot.generate_signal(
        df, {"price_fallback": True}
    )
    metadata = _last_metadata()
    assert direction == "long"
    assert score > 0
    assert metadata["atr"] > 0
    assert metadata["event"]
