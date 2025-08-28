from __future__ import annotations

"""Utilities for scanning new Solana tokens."""

import asyncio
import logging
import os
from typing import Mapping, List, Tuple, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


async def get_solana_new_tokens(cfg: Mapping[str, object]) -> List[str]:
    """Return a list of new token mint addresses using ``cfg`` options."""

    url = str(cfg.get("url", ""))
    if not url:
        return []

    key = os.getenv("HELIUS_KEY", "")
    if "${HELIUS_KEY}" in url:
        url = url.replace("${HELIUS_KEY}", key)
    if "YOUR_KEY" in url:
        url = url.replace("YOUR_KEY", key)

    limit = int(cfg.get("limit", 0))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.error("Solana scanner error: %s", exc)
        return []

    tokens = data.get("tokens") or data.get("mints") or data
    results: List[str] = []
    if isinstance(tokens, list):
        for item in tokens:
            if isinstance(item, str):
                results.append(item)
            elif isinstance(item, Mapping):
                mint = item.get("mint") or item.get("tokenMint") or item.get("token_mint")
                if mint:
                    results.append(str(mint))
    elif isinstance(tokens, Mapping):
        for mint in tokens.values():
            if isinstance(mint, str):
                results.append(mint)
    if limit:
        results = results[:limit]
    return results


async def get_sentiment_enhanced_tokens(
    cfg: Mapping[str, object], 
    min_galaxy_score: float = 60.0,
    min_sentiment: float = 0.6,
    limit: int = 20
) -> List[Tuple[str, Dict]]:
    """
    Get Solana tokens with enhanced sentiment data from LunarCrush.
    
    Returns list of (mint_address, sentiment_data) tuples for tokens that
    meet the sentiment criteria for potential early entries.
    """
    try:
        from crypto_bot.sentiment_filter import get_lunarcrush_client, SentimentDirection
        
        client = get_lunarcrush_client()
        
        # Get trending tokens with sentiment data
        trending_tokens = await client.get_trending_solana_tokens(limit=limit * 2)
        
        enhanced_results = []
        for symbol, sentiment_data in trending_tokens:
            # Filter for strong bullish sentiment
            if (sentiment_data.sentiment_direction == SentimentDirection.BULLISH and
                sentiment_data.galaxy_score >= min_galaxy_score and
                sentiment_data.sentiment >= min_sentiment):
                
                # Try to get mint address if available (this would need token mapping)
                mint_address = f"SOL_{symbol}_PLACEHOLDER"  # Placeholder for now
                
                sentiment_dict = {
                    "symbol": symbol,
                    "galaxy_score": sentiment_data.galaxy_score,
                    "alt_rank": sentiment_data.alt_rank,
                    "sentiment": sentiment_data.sentiment,
                    "sentiment_direction": sentiment_data.sentiment_direction.value,
                    "social_mentions": sentiment_data.social_mentions,
                    "social_volume": sentiment_data.social_volume,
                    "bullish_strength": sentiment_data.bullish_strength,
                    "last_updated": sentiment_data.last_updated
                }
                
                enhanced_results.append((mint_address, sentiment_dict))
                
                if len(enhanced_results) >= limit:
                    break
        
        logger.info(f"Found {len(enhanced_results)} sentiment-enhanced Solana tokens")
        return enhanced_results
        
    except Exception as exc:
        logger.error(f"Failed to get sentiment-enhanced tokens: {exc}")
        return []


async def score_token_by_sentiment(symbol: str) -> Optional[Dict]:
    """
    Score a token based on LunarCrush sentiment metrics.
    
    Returns sentiment scoring dict or None if token not found/error.
    """
    try:
        from crypto_bot.sentiment_filter import get_lunarcrush_client, SentimentDirection
        
        client = get_lunarcrush_client()
        sentiment_data = await client.get_sentiment(symbol)
        
        # Calculate composite score (0-100)
        galaxy_weight = 0.4
        sentiment_weight = 0.3
        social_weight = 0.2
        rank_weight = 0.1
        
        # Normalize alt_rank (lower is better, so invert)
        rank_score = max(0, 100 - (sentiment_data.alt_rank / 10))
        
        # Normalize social metrics (log scale to handle large variations)
        import math
        social_score = min(100, math.log10(max(1, sentiment_data.social_mentions)) * 10)
        
        composite_score = (
            sentiment_data.galaxy_score * galaxy_weight +
            sentiment_data.sentiment * 100 * sentiment_weight +
            social_score * social_weight +
            rank_score * rank_weight
        )
        
        return {
            "composite_score": composite_score,
            "galaxy_score": sentiment_data.galaxy_score,
            "sentiment": sentiment_data.sentiment,
            "sentiment_direction": sentiment_data.sentiment_direction.value,
            "alt_rank": sentiment_data.alt_rank,
            "social_mentions": sentiment_data.social_mentions,
            "social_volume": sentiment_data.social_volume,
            "bullish_strength": sentiment_data.bullish_strength,
            "recommendation": _get_recommendation(sentiment_data, composite_score)
        }
        
    except Exception as exc:
        logger.warning(f"Failed to score token {symbol} by sentiment: {exc}")
        return None


def _get_recommendation(sentiment_data, composite_score: float) -> str:
    """Get trading recommendation based on sentiment analysis."""
    from crypto_bot.sentiment_filter import SentimentDirection
    
    if composite_score >= 80 and sentiment_data.sentiment_direction == SentimentDirection.BULLISH:
        return "STRONG_BUY"
    elif composite_score >= 60 and sentiment_data.sentiment_direction == SentimentDirection.BULLISH:
        return "BUY"
    elif composite_score >= 40 and sentiment_data.sentiment_direction != SentimentDirection.BEARISH:
        return "HOLD"
    elif sentiment_data.sentiment_direction == SentimentDirection.BEARISH:
        return "AVOID"
    else:
        return "NEUTRAL"
