from __future__ import annotations

"""Simple cross-chain arbitrage strategy."""

import asyncio
from typing import Optional, Tuple, Mapping, List, Dict

import pandas as pd

from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot.solana import fetch_solana_prices
from crypto_bot.utils.logging import setup_strategy_logger
from crypto_bot.utils.volatility import normalize_score_by_volatility

NAME = "cross_chain_arb_bot"
STRATEGY_NAME = NAME
logger = setup_strategy_logger(STRATEGY_NAME)

def _fetch_prices(symbols: List[str]) -> Dict[str, float]:
    """Return Solana prices synchronously."""
    if not symbols:
        return {}
    try:  # pragma: no cover - best effort
        # Try to get the current event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we can't use asyncio.run
            # Return empty dict for now, the async version will handle this
            return {}
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(fetch_solana_prices(symbols))
    except Exception:
        return {}


def generate_signal(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    **kwargs,
) -> Tuple[float, str]:
    """Return arbitrage signal comparing CEX OHLCV data to Solana prices."""
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config") or {}
    mempool_monitor: Optional[SolanaMempoolMonitor] = kwargs.get("mempool_monitor")
    mempool_cfg: Optional[Mapping[str, object]] = kwargs.get("mempool_cfg")

    if df is None or df.empty:
        logger.debug(
            "%s missing OHLCV data for %s; skipping arbitrage check.",
            STRATEGY_NAME,
            symbol or "<unknown>",
        )
        return 0.0, "none"

    params = config.get("cross_chain_arb_bot", {})
    pair = str(params.get("pair", ""))
    try:
        threshold = float(params.get("spread_threshold", 0.0))
    except (TypeError, ValueError):
        threshold = 0.0

    symbol = config.get("symbol", "")
    if symbol and not pair:
        pair = str(symbol)
    if not pair:
        logger.debug("%s no trading pair configured; aborting signal.", STRATEGY_NAME)
        return 0.0, "none"

    cfg = mempool_cfg or config.get("mempool_monitor", {})
    if mempool_monitor and cfg.get("enabled"):
        try:
            fee_thr = float(cfg.get("suspicious_fee_threshold", 0.0))
        except (TypeError, ValueError):
            fee_thr = 0.0
        try:
            # For synchronous calls, skip mempool monitoring
            suspicious = False
        except Exception:
            suspicious = False
        if suspicious:
            logger.info(
                "%s mempool risk triggered for %s; suppressing signal.",
                STRATEGY_NAME,
                pair,
            )
            return 0.0, "none"

    try:
        # Use synchronous price fetching
        prices = _fetch_prices([pair])
    except Exception:
        prices = {}
    dex_price = prices.get(pair)
    if dex_price is None or dex_price <= 0:
        logger.debug(
            "%s no DEX price available for %s; skipping.", STRATEGY_NAME, pair
        )
        return 0.0, "none"

    cex_price = float(df["close"].iloc[-1])
    if cex_price <= 0:
        logger.debug(
            "%s invalid CEX price %.4f for %s; skipping.",
            STRATEGY_NAME,
            cex_price,
            pair,
        )
        return 0.0, "none"

    diff = (dex_price - cex_price) / cex_price
    if abs(diff) < threshold:
        logger.debug(
            "%s spread %.4f below threshold %.4f for %s.",
            STRATEGY_NAME,
            diff,
            threshold,
            pair,
        )
        return 0.0, "none"

    score = min(abs(diff), 1.0)
    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    direction = "long" if diff > 0 else "short"
    logger.info(
        "%s generated %s arbitrage signal for %s (spread %.4f, score %.3f).",
        STRATEGY_NAME,
        direction,
        pair,
        diff,
        score,
    )
    return score, direction


class regime_filter:
    """Match sideways and volatile regimes."""

    @staticmethod
    def matches(regime: str) -> bool:  # pragma: no cover - simple
        return regime in {"sideways", "volatile"}
