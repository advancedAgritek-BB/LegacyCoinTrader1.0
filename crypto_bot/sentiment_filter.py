"""Utilities for gauging market sentiment."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import requests

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from services.configuration import ManagedSecretsClient, load_manifest
from crypto_bot.solana.token_registry import get_jupiter_registry, index_registry_by_symbol


logger = setup_logger(__name__, LOG_DIR / "sentiment.log")


FNG_URL = "https://api.alternative.me/fng/?limit=1"
# Cache Fear & Greed values for reuse when upstream is unavailable
_FNG_CACHE_TTL = 300  # five minutes
_FNG_CACHE = {"value": 50, "fetched_at": 0.0}
_FNG_FAILURE_COUNT = 0
# LunarCrush is the primary sentiment source - Twitter sentiment has been removed
LUNARCRUSH_BASE_URL = "https://lunarcrush.com/api4/public"
_MANAGED_MANIFEST = load_manifest()
_MANAGED_SECRETS = ManagedSecretsClient(_MANAGED_MANIFEST)
LUNARCRUSH_API_KEY = _MANAGED_SECRETS.get("LUNARCRUSH_API_KEY")


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
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = _MANAGED_SECRETS.get("LUNARCRUSH_API_KEY")

        self.api_key = api_key or ""
        self.base_url = LUNARCRUSH_BASE_URL
        self._cache: Dict[str, SentimentData] = {}
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            logger.warning(
                "LUNARCRUSH_API_KEY is not configured; sentiment requests will "
                "return cached data or defaults."
            )
        self._session.headers.update(headers)
    
    def _get_cache_key(self, symbol: str) -> str:
        """Get cache key for symbol."""
        return symbol.lower().replace("/", "").replace("-", "")
    
    async def get_sentiment(self, symbol: str, force_refresh: bool = False) -> SentimentData:
        """Get sentiment data for a symbol."""
        if not self.api_key:
            logger.debug(
                "Skipping LunarCrush sentiment fetch for %s because the API key is not configured.",
                symbol,
            )
            return SentimentData()
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
        if not self.api_key:
            raise RuntimeError(
                "LUNARCRUSH_API_KEY must be configured to fetch sentiment data"
            )
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
        if not self.api_key:
            logger.debug(
                "Skipping LunarCrush trending tokens fetch because the API key is not configured."
            )
            return []

        def _fetch() -> List[Dict]:
            try:
                url = f"{self.base_url}/trending"
                response = self._session.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data or "data" not in data:
                    return []

                trending: List[Dict] = []
                for coin in data["data"][:limit]:
                    trending.append({
                        "symbol": coin.get("symbol", ""),
                        "name": coin.get("name", ""),
                        "galaxy_score": float(coin.get("galaxy_score", 0)),
                        "alt_rank": int(coin.get("alt_rank", 1000)),
                        "social_mentions": int(coin.get("social_mentions", 0)),
                        "social_volume": float(coin.get("social_volume", 0)),
                        "sentiment": coin.get("sentiment"),
                        "sentiment_score": coin.get("sentiment_score"),
                    })

                return trending

            except Exception as exc:
                logger.error(f"Failed to fetch trending tokens: {exc}")
                return []

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as exc:
            logger.error(f"Failed to retrieve trending tokens asynchronously: {exc}")
            return []

    async def get_trending_solana_tokens(
        self, *, limit: int = 20
    ) -> List[Tuple[str, SentimentData, Dict[str, object]]]:
        """Return trending Solana tokens with mapped mint addresses."""

        if not self.api_key:
            logger.debug(
                "Skipping LunarCrush Solana trend fetch because the API key is not configured."
            )
            return []

        raw_trending = await self.get_trending_tokens(limit * 2)
        if not raw_trending:
            return []

        registry = await get_jupiter_registry()
        if not registry:
            return []

        symbol_index = index_registry_by_symbol(registry, chain_id=101)
        results: List[Tuple[str, SentimentData, Dict[str, object]]] = []

        for entry in raw_trending:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol") or "").upper()
            if not symbol:
                continue
            candidates = symbol_index.get(symbol)
            if not candidates:
                continue
            mint_address = str(candidates[0].get("address") or "").strip()
            if not mint_address:
                continue

            galaxy_score = float(entry.get("galaxy_score", 0.0) or 0.0)
            alt_rank = int(entry.get("alt_rank", 1000) or 1000)
            sentiment_raw = entry.get("sentiment") or entry.get("sentiment_score") or 50.0
            try:
                sentiment_value = float(sentiment_raw)
            except (TypeError, ValueError):
                sentiment_value = 50.0
            if sentiment_value > 1:
                sentiment_value /= 100.0
            sentiment_value = max(0.0, min(1.0, sentiment_value))

            if galaxy_score >= 70 and sentiment_value >= 0.6:
                direction = SentimentDirection.BULLISH
            elif galaxy_score <= 30 or sentiment_value <= 0.4:
                direction = SentimentDirection.BEARISH
            else:
                direction = SentimentDirection.NEUTRAL

            sentiment_data = SentimentData(
                galaxy_score=galaxy_score,
                alt_rank=alt_rank,
                sentiment=sentiment_value,
                sentiment_direction=direction,
                social_mentions=int(entry.get("social_mentions", 0) or 0),
                social_volume=float(entry.get("social_volume", 0.0) or 0.0),
                last_updated=time.time(),
            )

            metadata: Dict[str, object] = {
                "symbol": symbol,
                "name": entry.get("name"),
                "mint": mint_address,
            }
            results.append((mint_address, sentiment_data, metadata))

            if len(results) >= limit:
                break

        return results


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
    global _FNG_FAILURE_COUNT, _FNG_CACHE
    mock = os.getenv("MOCK_FNG_VALUE")
    if mock is not None:
        try:
            return int(mock)
        except ValueError:
            return 50

    now = time.time()
    cached_age = now - _FNG_CACHE.get("fetched_at", 0.0)
    if cached_age < _FNG_CACHE_TTL:
        return int(_FNG_CACHE.get("value", 50))

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = requests.get(FNG_URL, timeout=5 + attempt)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                value = int(data.get("data", [{}])[0].get("value", 50))
            else:
                value = 50

            _FNG_CACHE.update({"value": value, "fetched_at": time.time()})
            _FNG_FAILURE_COUNT = 0
            return value
        except Exception as exc:  # pragma: no cover - network dependency
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))

    # All attempts failed - degrade gracefully using cached/default value
    _FNG_FAILURE_COUNT += 1
    fallback_value = int(_FNG_CACHE.get("value", 50))
    log_message = (
        "Failed to fetch FNG index after retries; using cached value %s"
        % fallback_value
    )
    if last_error is not None:
        log_message = f"Failed to fetch FNG index after retries: {last_error}; using cached value {fallback_value}"
    if _FNG_FAILURE_COUNT == 1:
        logger.warning(log_message)
    else:
        logger.debug(log_message)
    return fallback_value


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
