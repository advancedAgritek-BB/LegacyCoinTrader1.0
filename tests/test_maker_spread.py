import pandas as pd

from crypto_bot.strategy import maker_spread


def _build_flat_market_dataframe(length: int = 40) -> pd.DataFrame:
    closes = [100.0 for _ in range(length)]
    volumes = [1_000.0 for _ in range(length)]
    return pd.DataFrame({"close": closes, "volume": volumes})


def test_maker_spread_returns_long_direction_for_sideways_market():
    df = _build_flat_market_dataframe()

    score, direction = maker_spread.generate_signal(df)

    assert direction == "long"
    assert score > 0


def test_maker_spread_suppresses_trending_market():
    closes = [100.0 * (1.05 ** i) for i in range(40)]
    volumes = [1_000.0 for _ in range(40)]
    df = pd.DataFrame({"close": closes, "volume": volumes})

    score, direction = maker_spread.generate_signal(df)

    assert direction == "none"
    assert score == 0.0
