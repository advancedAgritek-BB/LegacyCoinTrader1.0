from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import pandas as pd
import requests

from crypto_bot.strategy._config_utils import apply_defaults, extract_params
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.pair_cache import load_liquid_pairs
from crypto_bot.utils.gas_estimator import fetch_priority_fee_gwei
from .base import CallableStrategy

ALLOWED_PAIRS = load_liquid_pairs() or []


@dataclass
class DexScalperConfig:
    """Configuration options for the DEX scalper strategy."""

    ema_fast: int = 3
    ema_slow: int = 10
    min_signal_score: float = 0.1
    gas_threshold_gwei: float = 10.0
    atr_normalization: bool = True

    @classmethod
    def from_dict(cls, data: object) -> "DexScalperConfig":
        params = extract_params(
            data,
            {
                "ema_fast",
                "ema_slow",
                "min_signal_score",
                "gas_threshold_gwei",
                "atr_normalization",
            },
            ("dex_scalper",),
        )
        return apply_defaults(cls, params)


def fetch_priority_fee_gwei(endpoint: Optional[str] = None) -> float:
    """Return the median Ethereum priority fee in gwei.

    The ``MOCK_ETH_PRIORITY_FEE_GWEI`` environment variable overrides
    network requests for testing purposes. ``endpoint`` defaults to the
    ``ETH_RPC_URL`` environment variable when not provided. Errors are
    swallowed and ``0.0`` is returned.
    """

    mock = os.getenv("MOCK_ETH_PRIORITY_FEE_GWEI")
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0

    endpoint = endpoint or os.getenv("ETH_RPC_URL")
    if not endpoint:
        return 0.0

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_feeHistory",
        "params": [5, "latest", [50]],
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        reward = data.get("result", {}).get("reward")
        if isinstance(reward, list):
            fees = []
            for block in reward:
                if isinstance(block, list) and block:
                    val = block[0]
                    try:
                        fees.append(int(val, 16))
                    except Exception:
                        pass
            if fees:
                fees.sort()
                median = fees[len(fees) // 2]
                return median / 1_000_000_000
    except Exception:
        pass
    return 0.0


def generate_signal(
    df,
    config: Optional[DexScalperConfig | Mapping[str, Any]] = None,
) -> Tuple[float, str]:
    """Short-term momentum strategy using EMA divergence on DEX pairs."""
    # Handle type conversion from dict to DataFrame
    if isinstance(df, dict):
        try:
            df = pd.DataFrame.from_dict(df)
        except Exception:
            return 0.0, "none"

    if df is None or not isinstance(df, pd.DataFrame):
        return 0.0, "none"

    if df.empty:
        return 0.0, "none"

    cfg = DexScalperConfig.from_dict(config)
    fast_window = int(cfg.ema_fast)
    slow_window = int(cfg.ema_slow)
    min_score = float(cfg.min_signal_score)
    gas_threshold_gwei = float(cfg.gas_threshold_gwei)

    if gas_threshold_gwei > 0:
        fee = fetch_priority_fee_gwei()
        if fee > gas_threshold_gwei:
            return 0.0, "none"

    if len(df) < slow_window:
        return 0.0, "none"

    lookback = slow_window
    recent = df.iloc[-(lookback + 1) :]

    # Calculate EMA manually
    ema_fast = recent["close"].ewm(span=fast_window, adjust=False).mean()
    ema_slow = recent["close"].ewm(span=slow_window, adjust=False).mean()

    ema_fast = cache_series("ema_fast", df, ema_fast, lookback)
    ema_slow = cache_series("ema_slow", df, ema_slow, lookback)

    df = recent.copy()
    df["ema_fast"] = ema_fast
    df["ema_slow"] = ema_slow

    latest = df.iloc[-1]
    if (
        latest["close"] == 0
        or pd.isnull(latest["ema_fast"])
        or pd.isnull(latest["ema_slow"])
    ):
        return 0.0, "none"

    momentum = latest["ema_fast"] - latest["ema_slow"]
    score = min(abs(momentum) / latest["close"], 1.0)

    if score < min_score:
        return 0.0, "none"

    if momentum > 0:
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        return score, "long"
    elif momentum < 0:
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        return score, "short"
    return 0.0, "none"


class regime_filter:
    """DEX scalper works for any regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return True

strategy = CallableStrategy('dex_scalper', generate_signal)
