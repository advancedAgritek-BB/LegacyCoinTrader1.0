"""Utilities for gauging market sentiment."""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import requests

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from pathlib import Path


logger = setup_logger(__name__, LOG_DIR / "sentiment.log")


FNG_URL = "https://api.alternative.me/fng/?limit=1"
# LunarCrush is the primary sentiment source - Twitter sentiment has been removed
LUNARCRUSH_BASE_URL = "https://lunarcrush.com/api4/public"
LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY", "hpn7960ebtf31fplz8j0eurxqmdn418mequk61bq")


class SentimentDirection(Enum):
    """Direction of sentiment."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SentimentData:
    """Structured sentiment data from LunarCrush."""
    galaxy_score: float = 0.0
    alt_rank: int = 1000
    sentiment: float = 0.5  # 0-1 range
    sentiment_direction: SentimentDirection = SentimentDirection.NEUTRAL
    social_mentions: int = 0
    social_volume: float = 0.0
    last_updated: float = 0.0
    
    @property
    def is_fresh(self) -> bool:
        """Check if data is less than 5 minutes old."""
        return time.time() - self.last_updated < 300
    
    @property
    def bullish_strength(self) -> float:
        """Get bullish strength as a multiplier (1.0 = neutral, >1.0 = bullish)."""
        if self.sentiment_direction == SentimentDirection.BULLISH:
            return 1.0 + (self.sentiment - 0.5) * 2  # Range: 1.0 to 2.0
        elif self.sentiment_direction == SentimentDirection.BEARISH:
            return 0.5 + self.sentiment  # Range: 0.5 to 1.0
        return 1.0


class LunarCrushClient:
    """Client for LunarCrush API sentiment analysis."""
    
    def __init__(self, api_key: str = LUNARCRUSH_API_KEY):
        self.api_key = api_key
        self.base_url = LUNARCRUSH_BASE_URL
        self._cache: Dict[str, SentimentData] = {}
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def _get_cache_key(self, symbol: str) -> str:
        """Get cache key for symbol."""
        return symbol.lower().replace("/", "").replace("-", "")
    
    async def get_sentiment(self, symbol: str, force_refresh: bool = False) -> SentimentData:
        """Get sentiment data for a symbol."""
        cache_key = self._get_cache_key(symbol)
        
        # Return cached data if fresh and not forcing refresh
        if not force_refresh and cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached.is_fresh:
                return cached
        
        try:
            sentiment_data = await self._fetch_sentiment(symbol)
            self._cache[cache_key] = sentiment_data
            return sentiment_data
        except Exception as exc:
            logger.warning(f"Failed to fetch LunarCrush sentiment for {symbol}: {exc}")
            # Return cached data if available, otherwise default
            return self._cache.get(cache_key, SentimentData())
    
    async def _fetch_sentiment(self, symbol: str) -> SentimentData:
        """Fetch sentiment data from LunarCrush API."""
        # Clean symbol for API call
        clean_symbol = symbol.replace("/USD", "").replace("/USDT", "").replace("/BTC", "").lower()
        
        url = f"{self.base_url}/coins/{clean_symbol}/v1"
        
        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or "data" not in data:
                raise ValueError("Invalid API response")
            
            coin_data = data["data"]
            
            # Parse sentiment data
            galaxy_score = float(coin_data.get("galaxy_score", 0))
            alt_rank = int(coin_data.get("alt_rank", 1000))
            sentiment_raw = coin_data.get("sentiment", 0.5)
            social_mentions = int(coin_data.get("social_mentions", 0))
            social_volume = float(coin_data.get("social_volume", 0))
            
            # Normalize sentiment to 0-1 range
            if isinstance(sentiment_raw, (int, float)):
                sentiment = max(0.0, min(1.0, float(sentiment_raw) / 100.0))
            else:
                sentiment = 0.5
            
            # Determine sentiment direction based on multiple factors
            if galaxy_score > 70 and sentiment > 0.6:
                direction = SentimentDirection.BULLISH
            elif galaxy_score < 30 or sentiment < 0.4:
                direction = SentimentDirection.BEARISH
            else:
                direction = SentimentDirection.NEUTRAL
            
            return SentimentData(
                galaxy_score=galaxy_score,
                alt_rank=alt_rank,
                sentiment=sentiment,
                sentiment_direction=direction,
                social_mentions=social_mentions,
                social_volume=social_volume,
                last_updated=time.time()
            )
            
        except Exception as exc:
            logger.error(f"Failed to fetch LunarCrush sentiment for {symbol}: {exc}")
            raise
    
    async def get_trending_tokens(self, limit: int = 20) -> List[Dict]:
        """Get trending tokens based on social sentiment and volume."""
        try:
            url = f"{self.base_url}/trending"
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or "data" not in data:
                return []
            
            trending = []
            for coin in data["data"][:limit]:
                trending.append({
                    "symbol": coin.get("symbol", ""),
                    "name": coin.get("name", ""),
                    "galaxy_score": float(coin.get("galaxy_score", 0)),
                    "alt_rank": int(coin.get("alt_rank", 1000)),
                    "social_mentions": int(coin.get("social_mentions", 0)),
                    "social_volume": float(coin.get("social_volume", 0))
                })
            
            return trending
            
        except Exception as exc:
            logger.error(f"Failed to fetch trending tokens: {exc}")
            return []


# Global client instance
_lunarcrush_client: Optional[LunarCrushClient] = None


def get_lunarcrush_client() -> LunarCrushClient:
    """Get or create the global LunarCrush client."""
    global _lunarcrush_client
    if _lunarcrush_client is None:
        _lunarcrush_client = LunarCrushClient()
    return _lunarcrush_client


def fetch_fng_index() -> int:
    """Return the current Fear & Greed index (0-100)."""
    mock = os.getenv("MOCK_FNG_VALUE")
    if mock is not None:
        try:
            return int(mock)
        except ValueError:
            return 50
    try:
        resp = requests.get(FNG_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return int(data.get("data", [{}])[0].get("value", 50))
    except Exception as exc:
        logger.error("Failed to fetch FNG index: %s", exc)
    return 50


async def get_sentiment_score(symbol: str = "bitcoin") -> float:
    """
    Get sentiment score for a symbol using LunarCrush (0.0-1.0 range).
    
    This replaces the old fetch_twitter_sentiment function.
    Returns a normalized sentiment score between 0.0 and 1.0.
    """
    try:
        client = get_lunarcrush_client()
        sentiment_data = await client.get_sentiment(symbol)
        return sentiment_data.sentiment
    except Exception as exc:
        logger.warning(f"Failed to get LunarCrush sentiment for {symbol}: {exc}")
        return 0.5  # Default neutral sentiment


def too_bearish(min_fng: int, min_sentiment: float) -> bool:
    """Return ``True`` when sentiment is below thresholds."""
    fng = fetch_fng_index()
    # Note: This function needs to be updated to be async or use cached sentiment
    # For now, using a default sentiment value
    sentiment_score = 0.5  # Default neutral sentiment
    logger.info("FNG %s, sentiment %s", fng, sentiment_score)
    return fng < min_fng or sentiment_score < min_sentiment


def boost_factor(bull_fng: int, bull_sentiment: float) -> float:
    """Return a trade size boost factor based on strong sentiment."""
    fng = fetch_fng_index()
    # Note: This function needs to be updated to be async or use cached sentiment
    # For now, using a default sentiment value
    sentiment_score = 0.5  # Default neutral sentiment
    if fng > bull_fng and sentiment_score > bull_sentiment:
        factor = 1 + ((fng - bull_fng) + (sentiment_score - bull_sentiment)) / 200
        logger.info("Applying boost factor %.2f", factor)
        return factor
    return 1.0


async def get_lunarcrush_sentiment_boost(
    symbol: str, 
    trade_direction: str,
    min_galaxy_score: float = 60.0,
    min_sentiment: float = 0.6
) -> float:
    """
    Get sentiment boost factor from LunarCrush that enhances trades in the correct direction.
    
    Returns a multiplier:
    - 1.0 = neutral (no boost or hindrance)
    - >1.0 = positive boost for bullish trades when sentiment is bullish
    - 1.0 = no boost for bearish trades when sentiment is bullish (doesn't hinder)
    - 1.0 = no boost for bullish trades when sentiment is bearish (doesn't hinder)
    """
    try:
        client = get_lunarcrush_client()
        sentiment_data = await client.get_sentiment(symbol)
        
        logger.info(
            f"LunarCrush sentiment for {symbol}: "
            f"Galaxy Score: {sentiment_data.galaxy_score}, "
            f"Sentiment: {sentiment_data.sentiment:.2f}, "
            f"Direction: {sentiment_data.sentiment_direction.value}"
        )
        
        # Only boost trades that align with positive sentiment
        # Never hinder trades with negative multipliers
        if (trade_direction.lower() in ["long", "buy"] and 
            sentiment_data.sentiment_direction == SentimentDirection.BULLISH and
            sentiment_data.galaxy_score >= min_galaxy_score and 
            sentiment_data.sentiment >= min_sentiment):
            
            # Calculate boost based on galaxy score and sentiment strength
            galaxy_boost = (sentiment_data.galaxy_score - min_galaxy_score) / 40.0  # 0 to 1 range
            sentiment_boost = (sentiment_data.sentiment - min_sentiment) / 0.4  # 0 to 1 range
            
            # Combine boosts with a maximum of 50% increase
            total_boost = min(0.5, (galaxy_boost + sentiment_boost) / 4)
            boost_factor = 1.0 + total_boost
            
            logger.info(f"Applying LunarCrush boost factor {boost_factor:.2f} for {symbol}")
            return boost_factor
        
        # For bearish trades, we don't boost but also don't hinder
        # For misaligned sentiment, return neutral
        return 1.0
        
    except Exception as exc:
        logger.warning(f"Failed to get LunarCrush sentiment boost for {symbol}: {exc}")
        return 1.0  # Fail safely with no boost


async def check_sentiment_alignment(
    symbol: str, 
    trade_direction: str,
    require_alignment: bool = False
) -> bool:
    """
    Check if sentiment aligns with trade direction.
    
    Args:
        symbol: Trading symbol
        trade_direction: 'long'/'buy' or 'short'/'sell'
        require_alignment: If True, require strong sentiment alignment
    
    Returns:
        True if sentiment supports the trade or if alignment is not required
    """
    try:
        client = get_lunarcrush_client()
        sentiment_data = await client.get_sentiment(symbol)
        
        is_bullish_trade = trade_direction.lower() in ["long", "buy"]
        sentiment_is_bullish = sentiment_data.sentiment_direction == SentimentDirection.BULLISH
        sentiment_is_bearish = sentiment_data.sentiment_direction == SentimentDirection.BEARISH
        
        if not require_alignment:
            # If alignment is not required, only block trades that strongly oppose sentiment
            if is_bullish_trade and sentiment_is_bearish and sentiment_data.sentiment < 0.3:
                logger.info(f"Blocking bullish trade on {symbol} due to very bearish sentiment")
                return False
            return True
        
        # If alignment is required, check for positive alignment
        if is_bullish_trade and sentiment_is_bullish:
            return True
        elif not is_bullish_trade and sentiment_is_bearish:
            return True
        
        logger.info(f"Sentiment not aligned for {symbol}: trade={trade_direction}, sentiment={sentiment_data.sentiment_direction.value}")
        return False
        
    except Exception as exc:
        logger.warning(f"Failed to check sentiment alignment for {symbol}: {exc}")
        return True  # Fail safely by allowing the trade

