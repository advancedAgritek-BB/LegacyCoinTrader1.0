from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import logging
import pandas as pd

from crypto_bot.strategy._config_utils import apply_defaults, extract_params
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.indicators import calculate_rsi
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.ml_utils import init_ml_or_warn, load_model
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MomentumBotConfig:
    """Configuration for the momentum breakout strategy."""

    donchian_window: int = 20
    volume_window: int = 20
    volume_z_min: float = 0.5
    rsi_threshold: float = 55.0
    macd_min: float = 0.0
    fast_length: int = 12
    slow_length: int = 26
    atr_normalization: bool = True

    @classmethod
    def from_dict(cls, data: object) -> "MomentumBotConfig":
        params = extract_params(
            data,
            {
                "donchian_window",
                "volume_window",
                "volume_z_min",
                "rsi_threshold",
                "macd_min",
                "fast_length",
                "slow_length",
                "atr_normalization",
            },
            ("momentum_bot", "momentum"),
        )
        return apply_defaults(cls, params)

NAME = "momentum_bot"

ML_AVAILABLE = init_ml_or_warn()
MODEL: Optional[object]
if ML_AVAILABLE:
    MODEL = load_model("momentum_bot")
else:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    **kwargs,
) -> Tuple[float, str]:
    """Donchian breakout with volume confirmation."""
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config")
    cfg = MomentumBotConfig.from_dict(config)

    if df is None or df.empty:
        return 0.0, "none"

    window = int(cfg.donchian_window)
    vol_window = int(cfg.volume_window)
    vol_z_min = float(cfg.volume_z_min)
    rsi_threshold = float(cfg.rsi_threshold)
    macd_min = float(cfg.macd_min)
    macd_fast = int(cfg.fast_length)
    macd_slow = int(cfg.slow_length)
    rsi_window = 14

    min_len = max(window, vol_window, macd_slow, rsi_window)
    if len(df) < min_len:
        return 0.0, "none"

    lookback = min(len(df), min_len)
    recent = df.iloc[-(lookback + 1) :]

    dc_low = recent["low"].rolling(window).min().shift(1)
    vol_ma = recent["volume"].rolling(vol_window).mean()
    vol_std = recent["volume"].rolling(vol_window).std()
    rsi = calculate_rsi(recent["close"], window=rsi_window)

    # Calculate MACD manually
    ema_fast = recent["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = recent["close"].ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()  # Standard signal line
    macd_hist = macd_line - macd_signal
    macd = macd_line  # Return the main MACD line

    dc_low = cache_series("momentum_dc_low", df, dc_low, lookback)
    vol_ma = cache_series("momentum_vol_ma", df, vol_ma, lookback)
    vol_std = cache_series("momentum_vol_std", df, vol_std, lookback)
    volume_z = cache_series(
        "momentum_volume_z", df, (recent["volume"] - vol_ma) / vol_std, lookback
    )
    rsi = cache_series("momentum_rsi", df, rsi, lookback)
    macd = cache_series("momentum_macd", df, macd, lookback)

    recent = recent.copy()
    recent["dc_low"] = dc_low
    recent["vol_ma"] = vol_ma
    recent["vol_std"] = vol_std
    recent["volume_z"] = volume_z
    recent["rsi"] = rsi
    recent["macd"] = macd

    latest = recent.iloc[-1]

    macd_val = latest["macd"]
    rsi_val = latest["rsi"]
    volume_z = latest["volume_z"]

    score = 0.0
    direction = "none"

    long_cond = macd_val > 0 or rsi_val > 50
    short_cond = (
        latest["close"] < dc_low.iloc[-1]
        and latest["rsi"] < 100 - rsi_threshold
        and latest["macd"] < -macd_min
    )

    if long_cond:
        score = 0.8
        direction = "long"
        logger.info(
            f"momentum_bot long signal: MACD={macd_val}, RSI={rsi_val}"
        )
    elif short_cond and volume_z > vol_z_min:
        score = min(1.0, volume_z / 2)
        direction = "short"

    if score > 0:
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                # Ensure DataFrame is still valid before ML call
                if not isinstance(df, pd.DataFrame) or not hasattr(df, 'empty'):
                    logger.warning("DataFrame corrupted before ML prediction, skipping ML score")
                    ml_score = 0.5  # Default neutral score
                else:
                    ml_score = MODEL.predict(df)
                    # Validate that DataFrame is still intact after ML call
                    if not isinstance(df, pd.DataFrame) or not hasattr(df, 'empty'):
                        logger.warning("DataFrame corrupted after ML prediction, using default ML score")
                        ml_score = 0.5  # Default neutral score
                    else:
                        # Ensure ml_score is a valid number
                        if isinstance(ml_score, (list, np.ndarray)):
                            ml_score = float(ml_score[0]) if len(ml_score) > 0 else 0.5
                        else:
                            ml_score = float(ml_score)
                    score = (score + ml_score) / 2
            except Exception as e:
                logger.warning(f"ML prediction failed, using base score: {e}")
                # Continue with base score if ML fails
                pass
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(recent, score)

    logger.info(
        "RSI %.2f MACD %.5f score %.2f direction %s",
        float(latest.get("rsi", float("nan"))),
        float(latest.get("macd", float("nan"))),
        score,
        direction,
    )
    return score, direction


class regime_filter:
    """Match trending and volatile regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime in {"trending", "volatile"}


class Strategy:
    """Strategy wrapper so :func:`load_strategies` can auto-register it."""

    def __init__(self) -> None:
        self.name = "momentum_bot"
        self.generate_signal = generate_signal
        self.regime_filter = regime_filter

    def signal(self, *args, **kwargs):
        """Compatibility wrapper for pipelines expecting ``signal``."""
        return self.generate_signal(*args, **kwargs)


__all__ = ["generate_signal", "regime_filter", "Strategy"]
