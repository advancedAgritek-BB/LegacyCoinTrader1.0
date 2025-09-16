"""
Social Sentiment Analyzer for Memecoin Pump Prediction

This module provides comprehensive social media sentiment analysis specifically
designed for memecoin pump detection, integrating multiple social platforms
and sentiment sources.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import numpy as np
import aiohttp
import re

from crypto_bot.sentiment_filter import SentimentData, get_lunarcrush_client

logger = logging.getLogger(__name__)


@dataclass
class SocialSignal:
    """Individual social media signal."""
    
    platform: str  # twitter, telegram, discord, reddit
    content: str
    author: str
    timestamp: float
    sentiment_score: float  # -1 to 1
    influence_score: float  # 0 to 1
    engagement_score: float  # 0 to 1
    confidence: float  # 0 to 1
    keywords: List[str] = field(default_factory=list)
    token_mentions: List[str] = field(default_factory=list)


@dataclass
class SentimentAnalysis:
    """Comprehensive sentiment analysis for a token."""
    
    token_mint: str
    token_symbol: str
    
    # Aggregate Metrics
    overall_sentiment: float  # -1 to 1
    sentiment_momentum: float  # Rate of change
    social_volume: int  # Total mentions
    unique_authors: int
    
    # Platform Breakdown
    twitter_sentiment: float
    telegram_sentiment: float
    discord_sentiment: float
    reddit_sentiment: float
    
    # Influence Metrics
    influencer_mentions: int
    whale_mentions: int
    verified_mentions: int
    
    # Engagement Metrics
    total_likes: int
    total_shares: int
    total_comments: int
    viral_score: float  # 0 to 1
    
    # Temporal Analysis
    sentiment_1h: float
    sentiment_6h: float
    sentiment_24h: float
    volume_spike_factor: float  # Current vs baseline volume
    
    # Quality Indicators
    organic_score: float  # How organic vs artificial
    credibility_score: float  # Source credibility
    pump_indicators: List[str] = field(default_factory=list)
    
    # Composite Scores
    bullish_score: float = 0.0  # 0 to 1
    pump_probability: float = 0.0  # 0 to 1
    urgency_score: float = 0.0  # 0 to 1
    
    signals: List[SocialSignal] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class SocialSentimentAnalyzer:
    """
    Advanced social sentiment analyzer for memecoin pump prediction.
    
    Features:
    - Multi-platform monitoring (Twitter, Telegram, Discord, Reddit)
    - Influencer and whale tracking
    - Real-time sentiment scoring
    - Pump pattern recognition
    - Viral momentum detection
    - Authenticity filtering
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.sentiment_config = config.get("social_sentiment", {})
        
        # API Configuration
        self.twitter_bearer_token = self.sentiment_config.get("twitter_bearer_token", "")
        self.lunarcrush_api_key = self.sentiment_config.get("lunarcrush_api_key", "")
        self.social_platforms = self.sentiment_config.get("platforms", [
            "twitter", "telegram", "discord", "reddit"
        ])
        
        # Sentiment thresholds
        self.min_sentiment_confidence = self.sentiment_config.get("min_confidence", 0.6)
        self.min_social_volume = self.sentiment_config.get("min_volume", 10)
        self.influencer_threshold = self.sentiment_config.get("influencer_threshold", 1000)
        
        # Tracking
        self.token_sentiments: Dict[str, SentimentAnalysis] = {}
        self.influencer_cache: Dict[str, Dict] = {}  # Cached influencer data
        self.keyword_patterns = self._build_keyword_patterns()
        
        # Rate limiting
        self.api_calls: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.rate_limits = {
            "twitter": {"calls": 100, "window": 900},  # 100 calls per 15 min
            "lunarcrush": {"calls": 20, "window": 60},  # 20 calls per minute
        }
        
        # Background monitoring
        self.monitoring_tasks: List[asyncio.Task] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        
        # Statistics
        self.stats = {
            "tokens_monitored": 0,
            "signals_processed": 0,
            "pumps_predicted": 0,
            "api_calls_made": 0,
            "sentiment_accuracy": 0.0
        }
        
    async def start(self):
        """Start the social sentiment analyzer."""
        self.running = True
        self.session = aiohttp.ClientSession()
        
        # Start background monitoring tasks
        self._start_monitoring_tasks()
        
        logger.info("Social sentiment analyzer started")
        
    async def stop(self):
        """Stop the social sentiment analyzer."""
        self.running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        if self.session:
            await self.session.close()
            
        logger.info("Social sentiment analyzer stopped")
        
    async def analyze_token_sentiment(self, token_mint: str, token_symbol: str) -> Optional[SentimentAnalysis]:
        """
        Perform comprehensive sentiment analysis for a token.
        
        Args:
            token_mint: Token mint address
            token_symbol: Token symbol
            
        Returns:
            SentimentAnalysis if analysis successful, None otherwise
        """
        try:
            # Initialize analysis
            analysis = SentimentAnalysis(
                token_mint=token_mint,
                token_symbol=token_symbol
            )
            
            # Collect signals from all platforms
            all_signals = []
            
            # Twitter analysis
            twitter_signals = await self._analyze_twitter_sentiment(token_symbol)
            all_signals.extend(twitter_signals)
            
            # LunarCrush analysis
            lunarcrush_data = await self._get_lunarcrush_data(token_symbol)
            if lunarcrush_data:
                lunarcrush_signals = self._process_lunarcrush_data(lunarcrush_data, token_symbol)
                all_signals.extend(lunarcrush_signals)
                
            # Telegram analysis
            telegram_signals = await self._analyze_telegram_sentiment(token_symbol)
            all_signals.extend(telegram_signals)
            
            # Reddit analysis
            reddit_signals = await self._analyze_reddit_sentiment(token_symbol)
            all_signals.extend(reddit_signals)
            
            # Process all signals
            if all_signals:
                analysis.signals = all_signals
                self._calculate_sentiment_metrics(analysis)
                self._detect_pump_indicators(analysis)
                self._calculate_composite_scores(analysis)
                
                # Cache the analysis
                self.token_sentiments[token_mint] = analysis
                self.stats["tokens_monitored"] += 1
                
                logger.info(
                    f"Sentiment analysis completed for {token_symbol}: "
                    f"Sentiment: {analysis.overall_sentiment:.2f}, "
                    f"Volume: {analysis.social_volume}, "
                    f"Pump Probability: {analysis.pump_probability:.2f}"
                )
                
                return analysis
                
            return None
            
        except Exception as exc:
            logger.error(f"Sentiment analysis failed for {token_symbol}: {exc}")
            return None
            
    async def _analyze_twitter_sentiment(self, token_symbol: str) -> List[SocialSignal]:
        """Analyze Twitter sentiment for a token."""
        signals = []
        
        try:
            if not self.twitter_bearer_token:
                return signals
                
            # Check rate limits
            if not self._check_rate_limit("twitter"):
                return signals
                
            # Search for tweets
            query = f"${token_symbol} OR {token_symbol} crypto"
            tweets = await self._search_twitter(query)
            
            for tweet in tweets:
                sentiment_score = self._calculate_text_sentiment(tweet["text"])
                influence_score = self._calculate_influence_score(tweet["author"])
                engagement_score = self._calculate_engagement_score(tweet)
                
                signal = SocialSignal(
                    platform="twitter",
                    content=tweet["text"],
                    author=tweet["author"]["username"],
                    timestamp=self._parse_twitter_timestamp(tweet["created_at"]),
                    sentiment_score=sentiment_score,
                    influence_score=influence_score,
                    engagement_score=engagement_score,
                    confidence=min(abs(sentiment_score), 0.8),
                    keywords=self._extract_keywords(tweet["text"]),
                    token_mentions=[token_symbol]
                )
                
                signals.append(signal)
                
            self.stats["signals_processed"] += len(signals)
            
        except Exception as exc:
            logger.debug(f"Twitter sentiment analysis failed: {exc}")
            
        return signals
        
    async def _get_lunarcrush_data(self, token_symbol: str) -> Optional[SentimentData]:
        """Get data from LunarCrush client."""
        try:
            if not self.lunarcrush_api_key:
                return None

            if not self._check_rate_limit("lunarcrush"):
                return None

            client = get_lunarcrush_client()
            data = await client.get_sentiment(token_symbol)
            self._record_api_call("lunarcrush")
            return data

        except Exception as exc:
            logger.debug(f"LunarCrush API call failed: {exc}")

        return None

    def _process_lunarcrush_data(self, data: SentimentData, token_symbol: str) -> List[SocialSignal]:
        """Process LunarCrush data into social signals."""
        signals = []

        try:
            sentiment_score = max(-1.0, min(1.0, (data.sentiment * 2) - 1))
            influence_score = min(data.galaxy_score / 100.0, 1.0) if data.galaxy_score else 0.0
            social_activity = data.social_volume + data.social_mentions
            engagement_score = min(social_activity / 2000.0, 1.0) if social_activity else 0.0

            signal = SocialSignal(
                platform="lunarcrush",
                content=(
                    "LunarCrush metrics: "
                    f"galaxy_score={data.galaxy_score:.2f}, "
                    f"sentiment={data.sentiment:.2f}, "
                    f"social_mentions={data.social_mentions}, "
                    f"social_volume={data.social_volume:.2f}"
                ),
                author="lunarcrush_aggregate",
                timestamp=time.time(),
                sentiment_score=sentiment_score,
                influence_score=influence_score,
                engagement_score=engagement_score,
                confidence=0.8,
                keywords=[
                    "lunarcrush",
                    data.sentiment_direction.value,
                    f"mentions:{data.social_mentions}",
                    f"volume:{int(data.social_volume)}"
                ],
                token_mentions=[token_symbol]
            )

            signals.append(signal)

        except Exception as exc:
            logger.debug(f"LunarCrush data processing failed: {exc}")
            
        return signals
        
    async def _analyze_telegram_sentiment(self, token_symbol: str) -> List[SocialSignal]:
        """Analyze Telegram sentiment (placeholder - would integrate with Telegram APIs)."""
        # This would integrate with Telegram monitoring services
        # Placeholder implementation
        return []
        
    async def _analyze_reddit_sentiment(self, token_symbol: str) -> List[SocialSignal]:
        """Analyze Reddit sentiment (placeholder - would integrate with Reddit API)."""
        # This would integrate with Reddit API
        # Placeholder implementation
        return []
        
    async def _search_twitter(self, query: str) -> List[Dict]:
        """Search Twitter for tweets matching query."""
        try:
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}
            params = {
                "query": query,
                "max_results": 100,
                "tweet.fields": "created_at,public_metrics,author_id",
                "user.fields": "public_metrics,verified",
                "expansions": "author_id"
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self._record_api_call("twitter")
                    
                    # Process Twitter API response
                    tweets = data.get("data", [])
                    users = {user["id"]: user for user in data.get("includes", {}).get("users", [])}
                    
                    processed_tweets = []
                    for tweet in tweets:
                        author_data = users.get(tweet.get("author_id"), {})
                        processed_tweet = {
                            "text": tweet.get("text", ""),
                            "created_at": tweet.get("created_at", ""),
                            "author": {
                                "username": author_data.get("username", "unknown"),
                                "followers": author_data.get("public_metrics", {}).get("followers_count", 0),
                                "verified": author_data.get("verified", False)
                            },
                            "metrics": tweet.get("public_metrics", {})
                        }
                        processed_tweets.append(processed_tweet)
                        
                    return processed_tweets
                    
        except Exception as exc:
            logger.debug(f"Twitter search failed: {exc}")
            
        return []
        
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text using keyword analysis."""
        text_lower = text.lower()
        
        # Positive keywords
        positive_keywords = [
            "moon", "bullish", "pump", "rocket", "gem", "diamond", "hands",
            "hodl", "buy", "calls", "up", "green", "gains", "profit", "win"
        ]
        
        # Negative keywords
        negative_keywords = [
            "dump", "bearish", "crash", "sell", "down", "red", "loss", "scam",
            "rug", "dead", "rip", "paper", "hands", "exit", "avoid"
        ]
        
        # Neutral/pump warning keywords
        warning_keywords = [
            "caution", "risky", "volatile", "careful", "dyor", "nfa"
        ]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        warning_count = sum(1 for keyword in warning_keywords if keyword in text_lower)
        
        # Calculate sentiment score
        total_keywords = positive_count + negative_count + warning_count
        if total_keywords == 0:
            return 0.0
            
        sentiment = (positive_count - negative_count - warning_count * 0.5) / total_keywords
        return max(-1.0, min(1.0, sentiment))
        
    def _calculate_influence_score(self, author: Dict) -> float:
        """Calculate influence score based on author metrics."""
        followers = author.get("followers", 0)
        verified = author.get("verified", False)
        
        # Base score from follower count
        if followers >= 100000:
            score = 1.0
        elif followers >= 10000:
            score = 0.8
        elif followers >= 1000:
            score = 0.6
        elif followers >= 100:
            score = 0.4
        else:
            score = 0.2
            
        # Bonus for verified accounts
        if verified:
            score = min(1.0, score * 1.2)
            
        return score
        
    def _calculate_engagement_score(self, tweet: Dict) -> float:
        """Calculate engagement score for a tweet."""
        metrics = tweet.get("metrics", {})
        likes = metrics.get("like_count", 0)
        retweets = metrics.get("retweet_count", 0)
        replies = metrics.get("reply_count", 0)
        
        # Weighted engagement score
        engagement = likes + (retweets * 3) + (replies * 2)
        
        # Normalize to 0-1 scale
        if engagement >= 1000:
            return 1.0
        elif engagement >= 100:
            return 0.8
        elif engagement >= 10:
            return 0.6
        elif engagement >= 1:
            return 0.4
        else:
            return 0.2
            
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        text_lower = text.lower()
        keywords = []
        
        # Check for common crypto keywords
        crypto_keywords = [
            "moon", "pump", "dump", "hodl", "diamond", "hands", "rocket",
            "gem", "ath", "btfd", "dyor", "nfa", "bullish", "bearish"
        ]
        
        for keyword in crypto_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
                
        # Extract cashtags ($SYMBOL)
        cashtags = re.findall(r'\$[A-Z]{2,10}', text.upper())
        keywords.extend(cashtags)
        
        return keywords
        
    def _calculate_sentiment_metrics(self, analysis: SentimentAnalysis):
        """Calculate aggregate sentiment metrics."""
        signals = analysis.signals
        
        if not signals:
            return
            
        # Overall sentiment (weighted by confidence and influence)
        weighted_sentiments = []
        total_weight = 0
        
        for signal in signals:
            weight = signal.confidence * signal.influence_score
            weighted_sentiments.append(signal.sentiment_score * weight)
            total_weight += weight
            
        if total_weight > 0:
            analysis.overall_sentiment = sum(weighted_sentiments) / total_weight
        else:
            analysis.overall_sentiment = 0.0
            
        # Platform breakdown
        platform_sentiments = defaultdict(list)
        for signal in signals:
            platform_sentiments[signal.platform].append(signal.sentiment_score)
            
        analysis.twitter_sentiment = np.mean(platform_sentiments.get("twitter", [0]))
        analysis.telegram_sentiment = np.mean(platform_sentiments.get("telegram", [0]))
        analysis.discord_sentiment = np.mean(platform_sentiments.get("discord", [0]))
        analysis.reddit_sentiment = np.mean(platform_sentiments.get("reddit", [0]))
        
        # Volume metrics
        analysis.social_volume = len(signals)
        analysis.unique_authors = len(set(signal.author for signal in signals))
        
        # Influence metrics
        analysis.influencer_mentions = len([s for s in signals if s.influence_score > 0.8])
        analysis.verified_mentions = len([s for s in signals if "verified" in s.keywords])
        
        # Temporal analysis
        current_time = time.time()
        recent_1h = [s for s in signals if current_time - s.timestamp <= 3600]
        recent_6h = [s for s in signals if current_time - s.timestamp <= 21600]
        recent_24h = [s for s in signals if current_time - s.timestamp <= 86400]
        
        analysis.sentiment_1h = np.mean([s.sentiment_score for s in recent_1h]) if recent_1h else 0
        analysis.sentiment_6h = np.mean([s.sentiment_score for s in recent_6h]) if recent_6h else 0
        analysis.sentiment_24h = np.mean([s.sentiment_score for s in recent_24h]) if recent_24h else 0
        
        # Volume spike calculation
        baseline_volume = max(analysis.social_volume / 24, 1)  # Assume 24h baseline
        current_volume = len(recent_1h)
        analysis.volume_spike_factor = current_volume / baseline_volume
        
    def _detect_pump_indicators(self, analysis: SentimentAnalysis):
        """Detect pump-specific indicators in social signals."""
        pump_indicators = []
        
        # High volume spike
        if analysis.volume_spike_factor > 3.0:
            pump_indicators.append("volume_spike")
            
        # Influencer involvement
        if analysis.influencer_mentions > 2:
            pump_indicators.append("influencer_mentions")
            
        # Sudden sentiment shift
        if analysis.sentiment_1h > 0.5 and analysis.sentiment_6h < 0.2:
            pump_indicators.append("sentiment_shift")
            
        # Pump keywords
        pump_keywords = ["moon", "rocket", "pump", "gem", "100x"]
        keyword_mentions = 0
        for signal in analysis.signals:
            for keyword in pump_keywords:
                if keyword in signal.keywords:
                    keyword_mentions += 1
                    
        if keyword_mentions > len(analysis.signals) * 0.3:  # 30% of signals mention pump keywords
            pump_indicators.append("pump_keywords")
            
        # Coordination indicators
        similar_timing = 0
        for i, signal1 in enumerate(analysis.signals):
            for signal2 in analysis.signals[i+1:]:
                if abs(signal1.timestamp - signal2.timestamp) < 300:  # Within 5 minutes
                    similar_timing += 1
                    
        if similar_timing > len(analysis.signals) * 0.5:
            pump_indicators.append("coordinated_timing")
            
        analysis.pump_indicators = pump_indicators
        
    def _calculate_composite_scores(self, analysis: SentimentAnalysis):
        """Calculate final composite scores."""
        # Bullish score
        bullish_factors = [
            max(0, analysis.overall_sentiment),  # Positive sentiment only
            min(1.0, analysis.social_volume / 100),  # Volume factor
            min(1.0, analysis.influencer_mentions / 5),  # Influencer factor
            min(1.0, analysis.volume_spike_factor / 5),  # Spike factor
        ]
        analysis.bullish_score = np.mean(bullish_factors)
        
        # Pump probability
        pump_factors = [
            analysis.bullish_score,
            len(analysis.pump_indicators) / 5,  # Normalized indicator count
            min(1.0, analysis.sentiment_momentum),  # Momentum factor
            min(1.0, analysis.viral_score)  # Viral factor
        ]
        analysis.pump_probability = np.mean(pump_factors)
        
        # Urgency score (how immediate the opportunity is)
        urgency_factors = [
            analysis.sentiment_1h / max(analysis.sentiment_6h, 0.1),  # Recent momentum
            min(1.0, analysis.volume_spike_factor / 3),  # Volume urgency
            min(1.0, len([s for s in analysis.signals if time.time() - s.timestamp < 1800]) / 10)  # Recent activity
        ]
        analysis.urgency_score = np.mean(urgency_factors)
        
    def _build_keyword_patterns(self) -> Dict[str, List[str]]:
        """Build keyword patterns for different signal types."""
        return {
            "pump": ["moon", "rocket", "pump", "parabolic", "breakout", "squeeze"],
            "gem": ["gem", "diamond", "hidden", "undervalued", "sleeper"],
            "technical": ["support", "resistance", "breakout", "pattern", "chart"],
            "fundamental": ["utility", "partnership", "listing", "development"],
            "warning": ["rug", "scam", "dump", "exit", "caution", "risky"]
        }
        
    def _check_rate_limit(self, api: str) -> bool:
        """Check if API rate limit allows new call."""
        if api not in self.rate_limits:
            return True
            
        limit_config = self.rate_limits[api]
        current_time = time.time()
        call_times = self.api_calls[api]
        
        # Remove old calls outside the window
        while call_times and current_time - call_times[0] > limit_config["window"]:
            call_times.popleft()
            
        return len(call_times) < limit_config["calls"]
        
    def _record_api_call(self, api: str):
        """Record an API call for rate limiting."""
        self.api_calls[api].append(time.time())
        self.stats["api_calls_made"] += 1
        
    def _parse_twitter_timestamp(self, timestamp_str: str) -> float:
        """Parse Twitter timestamp to Unix timestamp."""
        # This would parse ISO 8601 format from Twitter API
        # Placeholder implementation
        return time.time()
        
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        # Periodic sentiment monitoring for tracked tokens
        task = asyncio.create_task(self._periodic_monitoring())
        self.monitoring_tasks.append(task)
        
    async def _periodic_monitoring(self):
        """Periodically update sentiment for tracked tokens."""
        while self.running:
            try:
                for token_mint, analysis in list(self.token_sentiments.items()):
                    if time.time() - analysis.last_updated > 300:  # Update every 5 minutes
                        await self.analyze_token_sentiment(token_mint, analysis.token_symbol)
                        
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as exc:
                logger.error(f"Periodic monitoring error: {exc}")
                await asyncio.sleep(60)
                
    def get_token_sentiment(self, token_mint: str) -> Optional[SentimentAnalysis]:
        """Get cached sentiment analysis for a token."""
        return self.token_sentiments.get(token_mint)
        
    def get_statistics(self) -> Dict:
        """Get analyzer statistics."""
        return self.stats.copy()
        
    def get_top_bullish_tokens(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get tokens with highest bullish scores."""
        scored_tokens = [
            (mint, analysis.bullish_score)
            for mint, analysis in self.token_sentiments.items()
        ]
        scored_tokens.sort(key=lambda x: x[1], reverse=True)
        return scored_tokens[:limit]
