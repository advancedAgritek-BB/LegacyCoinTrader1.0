"""Utilities for gauging market sentiment."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import requests

from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "sentiment.log")


FNG_URL = "https://api.alternative.me/fng/?limit=1"
# LunarCrush is the primary sentiment source - Twitter sentiment has been removed
LUNARCRUSH_BASE_URL = "https://lunarcrush.com/api4/public"
LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY", "hpn7960ebtf31fplz8j0eurxqmdn418mequk61bq")
TRENDING_CACHE_TTL = 120


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
        self._trending_cache: Dict[str, Tuple[float, List[Tuple[str, SentimentData]]]] = {}
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
    
    def _trending_cache_key(self, chain: Optional[str]) -> str:
        """Return the cache key for the trending cache."""
        return (chain or "all").lower()

    @staticmethod
    def _coerce_number(value: object) -> Optional[float]:
        """Attempt to coerce the provided value into a float."""
        if isinstance(value, Mapping):
            for key in ("score", "value", "avg", "average", "percent", "count", "total", "rank"):
                if key in value:
                    coerced = LunarCrushClient._coerce_number(value[key])
                    if coerced is not None:
                        return coerced
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_number(source: Mapping[str, object], keys: Iterable[str]) -> Optional[float]:
        """Extract a numeric value for any of the provided keys."""
        for key in keys:
            for candidate in (key, key.lower(), key.upper()):
                if candidate in source:
                    value = LunarCrushClient._coerce_number(source[candidate])
                    if value is not None:
                        return value
        return None

    def _extract_metric(self, entry: Mapping[str, object], keys: Iterable[str]) -> Optional[float]:
        """Extract a numeric metric from a trending entry."""
        containers: List[Mapping[str, object]] = []
        if isinstance(entry, Mapping):
            containers.append(entry)
        metrics = entry.get("metrics")
        if isinstance(metrics, Mapping):
            containers.append(metrics)
        for container in containers:
            value = self._extract_number(container, keys)
            if value is not None:
                return value
        return None

    def _extract_social_metric(self, entry: Mapping[str, object], keys: Iterable[str]) -> Optional[float]:
        """Extract a social metric (mentions/volume) from a trending entry."""
        metrics = entry.get("metrics")
        candidates: List[Mapping[str, object]] = []
        if isinstance(metrics, Mapping):
            candidates.append(metrics)
            social = metrics.get("social")
            if isinstance(social, Mapping):
                candidates.append(social)
        if isinstance(entry, Mapping):
            candidates.append(entry)
        for container in candidates:
            value = self._extract_number(container, keys)
            if value is not None:
                return value
        return None

    @staticmethod
    def _normalize_sentiment_value(value: Optional[float]) -> float:
        """Normalize sentiment from 0-100/0-1 ranges into 0-1."""
        if value is None:
            return 0.5
        sentiment = float(value)
        if sentiment > 1:
            sentiment /= 100.0
        return max(0.0, min(1.0, sentiment))

    @staticmethod
    def _determine_direction(galaxy_score: float, sentiment: float) -> SentimentDirection:
        """Determine sentiment direction based on score thresholds."""
        if galaxy_score > 70 and sentiment > 0.6:
            return SentimentDirection.BULLISH
        if galaxy_score < 30 or sentiment < 0.4:
            return SentimentDirection.BEARISH
        return SentimentDirection.NEUTRAL

    @staticmethod
    def _iter_trending_entries(payload: object) -> Iterable[Mapping[str, object]]:
        """Yield coin entries from the trending payload."""
        if not isinstance(payload, Mapping):
            return []
        data = payload.get("data")
        if isinstance(data, list):
            entries = data
        elif isinstance(data, Mapping):
            for key in ("coins", "items", "data"):
                candidate = data.get(key)
                if isinstance(candidate, list):
                    entries = candidate
                    break
            else:
                entries = []
        else:
            entries = []
        return [entry for entry in entries if isinstance(entry, Mapping)]

    @staticmethod
    def _entry_matches_chain(entry: Mapping[str, object], chain: Optional[str]) -> bool:
        """Return True when the trending entry matches the requested chain."""
        if not chain:
            return True

        chain_lower = chain.lower()
        found_metadata = False

        for key in ("chain", "blockchain", "network"):
            value = entry.get(key)
            if isinstance(value, str):
                found_metadata = True
                if value.lower() == chain_lower:
                    return True

        for key in ("chains", "networks", "tags"):
            value = entry.get(key)
            if isinstance(value, list):
                found_metadata = True
                for item in value:
                    if isinstance(item, str) and item.lower() == chain_lower:
                        return True

        metrics = entry.get("metrics")
        if isinstance(metrics, Mapping):
            for key in ("chain", "blockchain", "network"):
                value = metrics.get(key)
                if isinstance(value, str):
                    found_metadata = True
                    if value.lower() == chain_lower:
                        return True

            for key in ("chains", "networks"):
                value = metrics.get(key)
                if isinstance(value, list):
                    found_metadata = True
                    for item in value:
                        if isinstance(item, str) and item.lower() == chain_lower:
                            return True

        # If no chain metadata was available we optimistically include the entry.
        return not found_metadata

    def _normalize_trending_entry(self, entry: Mapping[str, object]) -> Optional[Tuple[str, SentimentData]]:
        """Convert a trending entry into ``(symbol, SentimentData)``."""
        raw_symbol = entry.get("symbol") or entry.get("s") or entry.get("ticker")
        if not raw_symbol:
            return None

        symbol = str(raw_symbol).upper()
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), Mapping) else {}

        galaxy_score = self._extract_metric(entry, ("galaxy_score", "galaxyScore")) or 0.0
        alt_rank = self._extract_metric(entry, ("alt_rank", "altRank"))
        sentiment_raw = self._extract_metric(entry, ("average_sentiment", "sentiment", "sentiment_score", "sentimentScore"))
        social_mentions = self._extract_social_metric(entry, ("social_mentions", "socialMentions", "mentions"))
        social_volume = self._extract_social_metric(entry, ("social_volume", "socialVolume", "volume"))

        sentiment = self._normalize_sentiment_value(sentiment_raw)
        direction = self._determine_direction(float(galaxy_score), sentiment)

        try:
            alt_rank_int = int(alt_rank) if alt_rank is not None else 1000
        except (TypeError, ValueError):
            alt_rank_int = 1000

        social_mentions_int = 0
        if social_mentions is not None:
            try:
                social_mentions_int = int(social_mentions)
            except (TypeError, ValueError):
                social_mentions_int = 0

        social_volume_float = 0.0
        if social_volume is not None:
            try:
                social_volume_float = float(social_volume)
            except (TypeError, ValueError):
                social_volume_float = 0.0

        return symbol, SentimentData(
            galaxy_score=float(galaxy_score),
            alt_rank=alt_rank_int,
            sentiment=sentiment,
            sentiment_direction=direction,
            social_mentions=social_mentions_int,
            social_volume=social_volume_float,
            last_updated=time.time(),
        )

    async def get_trending_tokens(
        self,
        limit: int = 20,
        chain: Optional[str] = None,
    ) -> List[Tuple[str, SentimentData]]:
        """Return trending tokens from the LunarCrush v4 API.

        Args:
            limit: Maximum number of entries to return.
            chain: Optional blockchain filter (e.g. ``"solana"``).
        """

        try:
            limit = max(0, int(limit))
        except (TypeError, ValueError):
            limit = 0

        if limit <= 0:
            return []

        cache_key = self._trending_cache_key(chain)
        now = time.time()
        cached = self._trending_cache.get(cache_key)
        if cached and now - cached[0] < TRENDING_CACHE_TTL:
            return [(symbol, data) for symbol, data in cached[1][:limit]]

        params = {"limit": max(limit, 1)}
        if chain:
            params["chains"] = chain

        url = f"{self.base_url}/trending"

        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()

            entries = list(self._iter_trending_entries(payload))
            results: List[Tuple[str, SentimentData]] = []

            for entry in entries:
                if not self._entry_matches_chain(entry, chain):
                    continue
                normalized = self._normalize_trending_entry(entry)
                if not normalized:
                    continue
                results.append(normalized)
                if len(results) >= limit:
                    break

            self._trending_cache[cache_key] = (now, results)
            return [(symbol, data) for symbol, data in results[:limit]]

        except Exception as exc:
            logger.error(f"Failed to fetch trending tokens: {exc}")
            if cached:
                return [(symbol, data) for symbol, data in cached[1][:limit]]
            return []

    async def get_trending_solana_tokens(self, limit: int = 20) -> List[Tuple[str, SentimentData]]:
        """Return trending Solana tokens as ``(symbol, SentimentData)`` tuples."""
        return await self.get_trending_tokens(limit=limit, chain="solana")


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

