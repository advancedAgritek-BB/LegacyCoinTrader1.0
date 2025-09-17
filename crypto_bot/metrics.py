"""Metrics helpers for trading orchestration."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from crypto_bot.utils.metrics_logger import log_cycle as log_cycle_metrics
from crypto_bot.volatility_filter import calc_atr


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe strings like ``1h`` or ``15m`` to seconds."""

    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    if unit == "w":
        return value * 604800
    if unit == "M":
        return value * 2_592_000
    raise ValueError(f"Unknown timeframe {timeframe}")


def compute_average_atr(
    symbols: Sequence[str],
    df_cache: Mapping[str, Mapping[str, pd.DataFrame]],
    timeframe: str,
) -> float:
    """Return the average ATR for symbols present in ``df_cache``."""

    atr_values: list[float] = []
    tf_cache = df_cache.get(timeframe, {})
    for sym in symbols:
        df = tf_cache.get(sym)
        if df is None or df.empty:
            continue
        atr_values.append(calc_atr(df))
    return sum(atr_values) / len(atr_values) if atr_values else 0.0


def is_market_pumping(
    symbols: Sequence[str],
    df_cache: Mapping[str, Mapping[str, pd.DataFrame]],
    timeframe: str = "1h",
    lookback_hours: int = 24,
) -> bool:
    """Return ``True`` when the average % change over ``lookback_hours`` exceeds ~10%."""

    tf_cache = df_cache.get(timeframe, {})
    if not tf_cache:
        return False
    try:
        sec = timeframe_to_seconds(timeframe)
    except Exception:
        return False
    candles = int(lookback_hours * 3600 / sec) if sec else 0
    changes: list[float] = []
    for sym in symbols:
        df = tf_cache.get(sym)
        if df is None or df.empty or "close" not in df:
            continue
        closes = df["close"]
        if len(closes) == 0:
            continue
        start_idx = -candles - 1 if candles and len(closes) > candles else 0
        try:
            start = float(closes[start_idx])
            end = float(closes[-1])
        except Exception:
            continue
        if start == 0:
            continue
        changes.append((end - start) / start)
    avg_change = sum(changes) / len(changes) if changes else 0.0
    return avg_change >= 0.10


def emit_cycle_timing(
    logger,
    *,
    symbol_t: float,
    ohlcv_t: float,
    analyze_t: float,
    total_t: float,
    metrics_path: Path | None = None,
    ohlcv_fetch_latency: float = 0.0,
    execution_latency: float = 0.0,
) -> None:
    """Log timing information and optionally append to metrics CSV."""

    logger.info(
        "\u23f1\ufe0f Cycle timing - Symbols: %.2fs, OHLCV: %.2fs, Analyze: %.2fs, Total: %.2fs",
        symbol_t,
        ohlcv_t,
        analyze_t,
        total_t,
    )
    if metrics_path:
        log_cycle_metrics(
            symbol_t,
            ohlcv_t,
            analyze_t,
            total_t,
            ohlcv_fetch_latency,
            execution_latency,
            metrics_path,
        )


def update_df_cache(
    cache: MutableMapping[str, MutableMapping[str, pd.DataFrame]],
    timeframe: str,
    symbol: str,
    df: pd.DataFrame,
    *,
    max_size: int,
) -> None:
    """Update an OHLCV cache with LRU eviction."""

    tf_cache = cache.setdefault(timeframe, OrderedDict())
    if not isinstance(tf_cache, OrderedDict):
        tf_cache = OrderedDict(tf_cache)
        cache[timeframe] = tf_cache
    tf_cache[symbol] = df
    tf_cache.move_to_end(symbol)
    if len(tf_cache) > max_size:
        tf_cache.popitem(last=False)


__all__ = [
    "timeframe_to_seconds",
    "compute_average_atr",
    "is_market_pumping",
    "emit_cycle_timing",
    "update_df_cache",
]
