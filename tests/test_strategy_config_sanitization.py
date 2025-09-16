import importlib
import numbers

import numpy as np
import pandas as pd
import pytest


ROUTER_CONFIG = {
    "strategy_router": {"allocation": 0.35, "mode": "dynamic"},
    "momentum_exploiter": {
        "lookback": 12,
        "threshold": 0.02,
        "momentum_window": 6,
    },
    "volatility_harvester": {
        "atr_window": 12,
        "atr_threshold": 0.0012,
        "volume_zscore_threshold": 0.6,
    },
    "ultra_scalp_bot": {
        "min_score": 0.055,
        "ema_fast": 4,
        "macd_fast": 5,
    },
    "unrelated_key": "should_be_ignored",
}


def _build_market_frame(periods: int = 180) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="min")

    base = 100 + np.linspace(0, 3.5, periods)
    oscillation = np.sin(np.linspace(0, 6, periods)) * 0.25

    open_ = base + oscillation
    close = base + np.cos(np.linspace(0, 6, periods)) * 0.2
    high = np.maximum(open_, close) + 0.6
    low = np.minimum(open_, close) - 0.6
    volume = 1200 + np.linspace(0, 400, periods) + np.sin(np.linspace(0, 5, periods)) * 120

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


@pytest.mark.parametrize(
    ("module_name", "strategy_key"),
    [
        ("momentum_exploiter", "momentum_exploiter"),
        ("volatility_harvester", "volatility_harvester"),
        ("ultra_scalp_bot", "ultra_scalp_bot"),
    ],
)
def test_strategy_generators_accept_router_config(module_name: str, strategy_key: str) -> None:
    module = importlib.import_module(f"crypto_bot.strategy.{module_name}")
    df = _build_market_frame()

    assert isinstance(ROUTER_CONFIG[strategy_key], dict)
    result = module.generate_signal(df, config=ROUTER_CONFIG)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], numbers.Real)
    assert isinstance(result[1], str)
