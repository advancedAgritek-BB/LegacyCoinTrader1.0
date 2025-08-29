"""Scoring utilities for Solana pool events."""

from __future__ import annotations

import asyncio
import logging
from typing import Mapping, Optional

from .watcher import NewPoolEvent

logger = logging.getLogger(__name__)


def score_event(event: NewPoolEvent, cfg: Mapping[str, float]) -> float:
    """Return a numeric score for ``event`` based on heuristic weights."""

    liq_weight = float(cfg.get("weight_liquidity", 1.0))
    tx_weight = float(cfg.get("weight_tx", 1.0))
    social_weight = float(cfg.get("weight_social", 1.0))
    rug_weight = float(cfg.get("weight_rug", 1.0))

    liquidity_score = event.liquidity * liq_weight
    tx_score = event.tx_count * tx_weight
    social_score = float(cfg.get("social_score", 0)) * social_weight
    rug_penalty = float(cfg.get("rug_risk", 0)) * rug_weight

    return liquidity_score + tx_score + social_score - rug_penalty


async def score_event_with_sentiment(
    event: NewPoolEvent, 
    cfg: Mapping[str, float],
    symbol: Optional[str] = None
) -> float:
    """
    Enhanced scoring that incorporates LunarCrush sentiment analysis.
    
    This provides a more comprehensive score by including social sentiment
    alongside traditional liquidity and transaction metrics.
    """
    # Get base score
    base_score = score_event(event, cfg)
    
    # If no symbol provided, return base score
    if not symbol:
        return base_score
    
    # Get sentiment boost
    sentiment_boost = 1.0
    try:
        from crypto_bot.sentiment_filter import get_lunarcrush_sentiment_boost
        
        # For new pools, assume bullish intent
        sentiment_boost = await get_lunarcrush_sentiment_boost(symbol, "long")
        
        logger.info(f"Applied sentiment boost {sentiment_boost:.2f} to {symbol} pool score")
        
    except Exception as exc:
        logger.debug(f"Failed to get sentiment boost for {symbol}: {exc}")
    
    # Apply sentiment weight from config
    sentiment_weight = float(cfg.get("weight_sentiment", 0.5))
    
    # Calculate final score with sentiment enhancement
    enhanced_score = base_score * (1.0 + (sentiment_boost - 1.0) * sentiment_weight)
    
    logger.info(
        f"Pool scoring for {symbol}: base={base_score:.2f}, "
        f"sentiment_boost={sentiment_boost:.2f}, final={enhanced_score:.2f}"
    )
    
    return enhanced_score


async def get_token_sentiment_score(symbol: str) -> Optional[float]:
    """
    Get standalone sentiment score for a token (0-100 scale).
    
    This can be used for filtering and ranking tokens independently
    of pool events.
    """
    try:
        from .scanner import score_token_by_sentiment
        
        sentiment_data = await score_token_by_sentiment(symbol)
        if sentiment_data:
            return sentiment_data.get("composite_score", 0.0)
        
    except Exception as exc:
        logger.warning(f"Failed to get sentiment score for {symbol}: {exc}")
    
    return None
