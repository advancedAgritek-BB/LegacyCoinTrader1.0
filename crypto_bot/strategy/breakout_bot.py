from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency
    from scipy import stats as scipy_stats
    if not hasattr(scipy_stats, "norm"):
        raise ImportError
except Exception:  # pragma: no cover - fallback
    class _Norm:
        @staticmethod
        def ppf(_x):
            return 0.0

    class _FakeStats:
        norm = _Norm()

    scipy_stats = _FakeStats()

from crypto_bot.strategy._config_utils import apply_defaults, extract_params
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_rsi,
)
from crypto_bot.utils import stats
from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "breakout_bot.log")


@dataclass
class BreakoutBotConfig:
    """Configuration options for the breakout strategy."""

    bb_length: int = 12
    bb_std: float = 2.0
    kc_length: int = 12
    kc_mult: float = 1.5
    donchian_window: int = 30
    atr_buffer_mult: float = 0.05
    volume_window: int = 20
    vol_confirmation: bool = True
    vol_multiplier: float = 1.2
    squeeze_threshold: float = 0.03
    momentum_filter: bool = False
    adx_threshold: float = 20.0
    indicator_lookback: int = 250
    bb_squeeze_pct: float = 20.0
    atr_normalization: bool = True

    @classmethod
    def from_dict(cls, data: object) -> "BreakoutBotConfig":
        params = extract_params(
            data,
            {
                "bb_length",
                "bb_std",
                "kc_length",
                "kc_mult",
                "donchian_window",
                "dc_length",
                "atr_buffer_mult",
                "volume_window",
                "vol_confirmation",
                "vol_multiplier",
                "volume_mult",
                "squeeze_threshold",
                "momentum_filter",
                "adx_threshold",
                "indicator_lookback",
                "bb_squeeze_pct",
                "atr_normalization",
            },
            ("breakout_bot", "breakout"),
        )
        if "donchian_window" not in params and "dc_length" in params:
            params["donchian_window"] = params["dc_length"]
        if "vol_multiplier" not in params and "volume_mult" in params:
            params["vol_multiplier"] = params["volume_mult"]
        return apply_defaults(cls, params)


def _squeeze(
    df: pd.DataFrame,
    bb_len: int,
    bb_std: float,
    kc_len: int,
    kc_mult: float,
    threshold: float,
    lookback: int,
    squeeze_pct: float,
) -> Tuple[pd.Series, pd.Series]:
    """Return squeeze boolean series and ATR values."""
    hist = max(bb_len, kc_len)
    recent = df.iloc[-(hist + 1) :]

    close = recent["close"]

    bb = calculate_bollinger_bands(close, window=bb_len, num_std=bb_std)
    bb_mid = bb.middle
    bb_upper = bb.upper
    bb_lower = bb.lower
    bb_width = bb.width

    atr = calculate_atr(recent, window=kc_len)
    kc_width = 2 * atr * kc_mult

    if len(bb_width) >= lookback:
        width_z = stats.zscore(bb_width, lookback)
        thresh = scipy_stats.norm.ppf(squeeze_pct / 100)
        squeeze = (width_z < thresh) & (bb_width < kc_width)
    else:
        squeeze = (bb_width / bb_mid < threshold) & (bb_width < kc_width)

    bb_width = cache_series("bb_width", df, bb_width, hist)
    bb_mid = cache_series("bb_mid", df, bb_mid, hist)
    atr = cache_series("atr_kc", df, atr, hist)
    kc_width = cache_series("kc_width", df, kc_width, hist)
    squeeze = cache_series("squeeze", df, squeeze.astype(float), hist) > 0

    return squeeze, atr


def generate_signal(
    df: pd.DataFrame,
    config: Optional[BreakoutBotConfig | Mapping[str, Any]] = None,
    higher_df: Optional[pd.DataFrame] = None,
) -> Tuple[float, str]:
    """Breakout strategy using Bollinger/Keltner squeeze confirmation.

    Returns
    -------
    Tuple[float, str]
        Returns ``(score, direction)`` where score is the signal strength
        and direction is the trade direction.
    """
    if df is None or df.empty:
        return 0.0, "none"

    cfg = BreakoutBotConfig.from_dict(config)
    bb_len = int(cfg.bb_length)
    bb_std = float(cfg.bb_std)
    kc_len = int(cfg.kc_length)
    kc_mult = float(cfg.kc_mult)
    donchian_window = int(cfg.donchian_window)
    atr_buffer_mult = float(cfg.atr_buffer_mult)
    vol_window = int(cfg.volume_window)
    vol_confirmation = bool(cfg.vol_confirmation)
    vol_multiplier = float(cfg.vol_multiplier)
    threshold = float(cfg.squeeze_threshold)
    momentum_filter = bool(cfg.momentum_filter)
    _ = float(cfg.adx_threshold)  # placeholder for future use
    lookback_cfg = int(cfg.indicator_lookback)
    squeeze_pct = float(cfg.bb_squeeze_pct)

    lookback = max(bb_len, kc_len, donchian_window, vol_window, 14)
    if len(df) < lookback:
        return 0.0, "none"

    recent = df.iloc[-(lookback + 1) :]

    squeeze, atr = _squeeze(
        recent,
        bb_len,
        bb_std,
        kc_len,
        kc_mult,
        threshold,
        lookback_cfg,
        squeeze_pct,
    )
    if pd.isna(squeeze.iloc[-1]) or not squeeze.iloc[-1]:
        return 0.0, "none"

    if higher_df is not None and not higher_df.empty:
        h_sq, _ = _squeeze(
            higher_df.iloc[-(lookback + 1) :],
            bb_len,
            bb_std,
            kc_len,
            kc_mult,
            threshold,
            lookback_cfg,
            squeeze_pct,
        )
        # Higher timeframe squeeze is informative but no longer mandatory

    close = recent["close"]
    high = recent["high"]
    low = recent["low"]
    volume = recent["volume"]

    dc_high = high.rolling(donchian_window).max().shift(1)
    dc_low = low.rolling(donchian_window).min().shift(1)
    vol_ma = volume.rolling(vol_window).mean()

    rsi = calculate_rsi(close, window=14)

    # Calculate MACD manually
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    dc_high = cache_series("dc_high", df, dc_high, lookback)
    dc_low = cache_series("dc_low", df, dc_low, lookback)
    vol_ma = cache_series("vol_ma_breakout", df, vol_ma, lookback)
    rsi = cache_series("rsi_breakout", df, rsi, lookback)
    macd_hist = cache_series("macd_hist", df, macd_hist, lookback)

    recent = recent.copy()
    recent["dc_high"] = dc_high
    recent["dc_low"] = dc_low
    recent["vol_ma"] = vol_ma
    recent["rsi"] = rsi
    recent["macd_hist"] = macd_hist

    if vol_confirmation:
        vol_ok = (
            vol_ma.iloc[-1] > 0 and volume.iloc[-1] > vol_ma.iloc[-1] * vol_multiplier
        )
    else:
        vol_ok = True
    atr_last = atr.iloc[-1]
    upper_break = dc_high.iloc[-1] + atr_last * atr_buffer_mult
    lower_break = dc_low.iloc[-1] - atr_last * atr_buffer_mult

    long_cond = close.iloc[-1] > upper_break
    short_cond = close.iloc[-1] < lower_break

    if momentum_filter:
        long_cond = long_cond and (rsi.iloc[-1] > 50 or macd_hist.iloc[-1] > 0)
        short_cond = short_cond and (rsi.iloc[-1] < 50 or macd_hist.iloc[-1] < 0)

    direction = "none"
    score = 0.0
    if long_cond and vol_ok:
        direction = "long"
        score = 1.0
    elif short_cond and vol_ok:
        direction = "short"
        score = 1.0

    if score > 0 and cfg.atr_normalization:
        score = normalize_score_by_volatility(recent, score)

    return score, direction


class regime_filter:
    """Match breakout regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "breakout"
