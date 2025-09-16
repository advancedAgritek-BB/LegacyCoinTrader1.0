from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import pandas as pd

from crypto_bot.strategy._config_utils import apply_defaults, extract_params
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.pair_cache import load_liquid_pairs
from crypto_bot.utils.indicators import calculate_atr
from crypto_bot.volatility_filter import calc_atr

DEFAULT_PAIRS = ["BTC/USD", "ETH/USD"]
ALLOWED_PAIRS = load_liquid_pairs() or DEFAULT_PAIRS


@dataclass
class SniperBotConfig:
    """Configuration for the sniper bot strategy."""

    symbol: str = ""
    breakout_pct: float = 0.05
    volume_multiple: float = 1.5
    max_history: int = 30
    initial_window: int = 3
    min_volume: float = 100.0
    direction: str = "auto"
    atr_window: int = 14
    volume_window: int = 5
    price_fallback: bool = True
    fallback_atr_mult: float = 1.5
    fallback_volume_mult: float = 1.2
    atr_normalization: bool = True

    @classmethod
    def from_dict(cls, data: object) -> "SniperBotConfig":
        params = extract_params(
            data,
            {
                "symbol",
                "breakout_pct",
                "volume_multiple",
                "max_history",
                "initial_window",
                "min_volume",
                "direction",
                "atr_window",
                "volume_window",
                "price_fallback",
                "fallback_atr_mult",
                "fallback_volume_mult",
                "atr_normalization",
            },
            ("sniper_bot", "sniper"),
        )
        return apply_defaults(cls, params)


def generate_signal(
    df: pd.DataFrame,
    config: Optional[SniperBotConfig | Mapping[str, Union[float, int, str]]] = None,
    *,
    breakout_pct: float = 0.05,
    volume_multiple: float = 1.5,
    max_history: int = 30,
    initial_window: int = 3,
    min_volume: float = 100.0,
    direction: str = "auto",
    high_freq: bool = False,
    atr_window: int = 14,
    volume_window: int = 5,
    price_fallback: bool = True,
    fallback_atr_mult: float = 1.5,
    fallback_volume_mult: float = 1.2,
) -> Tuple[float, str]:
    """Detect pumps for newly listed tokens using early price and volume action.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data ordered oldest -> newest.
    config : dict, optional
        Configuration values overriding the keyword defaults.
    breakout_pct : float, optional
        Minimum percent change from the first close considered a breakout.
    volume_multiple : float, optional
        Minimum multiple of the average volume of the first ``initial_window``
        candles considered abnormal.
    max_history : int, optional
        Maximum history length still considered a new listing.
    initial_window : int, optional
        Number of early candles used to compute baseline volume.
    min_volume : float, optional
        Minimum trade volume for the latest candle to consider a signal.
    direction : {"auto", "long", "short"}, optional
        Force a trade direction or infer automatically.
    high_freq : bool, optional
        When ``True`` the function expects 1m candles and shortens
        ``max_history`` and ``initial_window`` so signals can trigger
        right after a listing.
    atr_window : int, optional
        Window length used to compute ATR for event detection.
    volume_window : int, optional
        Window length used to compute average volume for event detection.
    price_fallback : bool, optional
        Enable ATR based fallback when breakout conditions fail.
        Defaults to ``True``.
    fallback_atr_mult : float, optional
        Required candle body multiple of ATR for the fallback.
        Defaults to ``1.5``.
    fallback_volume_mult : float, optional
        Required volume multiple for the fallback.
        Defaults to ``1.2``.

    Returns
    -------
    Tuple[float, str, float, bool]
        Score between 0 and 1, trade direction, ATR value and event flag.
    """
    config_provided = bool(config)
    cfg = SniperBotConfig.from_dict(config)
    symbol = cfg.symbol
    if symbol and ALLOWED_PAIRS and symbol not in ALLOWED_PAIRS:
        return 0.0, "none"

    if config_provided:
        breakout_pct = float(cfg.breakout_pct)
        volume_multiple = float(cfg.volume_multiple)
        max_history = int(cfg.max_history)
        initial_window = int(cfg.initial_window)
        min_volume = float(cfg.min_volume)
        direction = cfg.direction
        atr_window = int(cfg.atr_window)
        volume_window = int(cfg.volume_window)
        price_fallback = bool(cfg.price_fallback)
        fallback_atr_mult = float(cfg.fallback_atr_mult)
        fallback_volume_mult = float(cfg.fallback_volume_mult)

    if high_freq:
        max_history = min(max_history, 20)
        initial_window = max(1, initial_window // 2)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < initial_window:
        return 0.0, "none"

    price_change = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    if direction == "auto" and price_change < 0:
        atr = calc_atr(df)
        score = 1.0
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        return score, "short"

    base_volume = df["volume"].iloc[:initial_window].mean()
    vol_ratio = df["volume"].iloc[-1] / base_volume if base_volume > 0 else 0

    atr_window = min(atr_window, len(df))

    atr_series = calculate_atr(df, window=atr_window)
    atr = float(atr_series.iloc[-1]) if len(atr_series) and not pd.isna(atr_series.iloc[-1]) else 0.0

    if len(df) > volume_window:
        prev_vol = df["volume"].iloc[-(volume_window + 1):-1]
    else:
        prev_vol = df["volume"].iloc[:-1]
    avg_vol = prev_vol.mean() if not prev_vol.empty else 0.0
    body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    event = False
    if atr > 0 and avg_vol > 0:
        if body >= 2 * atr and df["volume"].iloc[-1] >= 2 * avg_vol:
            event = True

    if df["volume"].iloc[-1] < min_volume:
        return 0.0, "none"

    if (
        len(df) <= max_history
        and abs(price_change) >= breakout_pct
        and vol_ratio >= volume_multiple
    ):
        price_score = min(abs(price_change) / breakout_pct, 1.0)
        vol_score = min(vol_ratio / volume_multiple, 1.0)
        score = (price_score + vol_score) / 2
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        if direction not in {"auto", "long", "short"}:
            direction = "auto"
        trade_direction = direction
        if direction == "auto":
            trade_direction = "short" if price_change < 0 else "long"
        return score, trade_direction

    trade_direction = direction
    score = 0.0

    if price_fallback:
        atr = calc_atr(df)
        body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
        avg_vol = df["volume"].iloc[:-1].mean()
        if (
            atr > 0
            and body > atr * fallback_atr_mult
            and avg_vol > 0
            and df["volume"].iloc[-1] > avg_vol * fallback_volume_mult
        ):
            score = 1.0
            if cfg.atr_normalization:
                score = normalize_score_by_volatility(df, score)
            if direction not in {"auto", "long", "short"}:
                direction = "auto"
            trade_direction = direction
            if direction == "auto":
                trade_direction = (
                    "short"
                    if df["close"].iloc[-1] < df["open"].iloc[-1]
                    else "long"
                )
            return score, trade_direction

    return 0.0, "none"


class regime_filter:
    """Match volatile regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "volatile"
