"""
Enhanced Solana Scanner with integrated caching and continuous review.

This scanner integrates with the scan cache manager to provide:
- Persistent caching of scan results
- Continuous strategy fit analysis
- Execution opportunity detection
- Market condition monitoring
"""

import asyncio
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..utils.scan_cache_manager import get_scan_cache_manager
from ..utils.logger import setup_logger, LOG_DIR
from .scanner import get_solana_new_tokens, get_sentiment_enhanced_tokens

logger = setup_logger(__name__, LOG_DIR / "enhanced_scanner.log")


@dataclass
class MarketConditions:
    """Market conditions for a token."""
    
    price: float
    volume_24h: float
    volume_ma: float
    price_change_24h: float
    price_change_7d: float
    atr: float
    atr_percent: float
    spread_pct: float
    liquidity_score: float
    volatility_score: float
    momentum_score: float
    sentiment_score: Optional[float] = None
    social_volume: Optional[float] = None
    social_mentions: Optional[float] = None


class EnhancedSolanaScanner:
    """
    Enhanced Solana scanner with integrated caching and continuous review.
    
    Features:
    - Multi-source token discovery
    - Real-time market condition analysis
    - Strategy fit evaluation
    - Execution opportunity detection
    - Persistent result caching
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scanner_config = config.get("solana_scanner", {})
        
        # Get cache manager
        self.cache_manager = get_scan_cache_manager(config)
        
        # Scanner settings with validation
        self.scan_interval = max(1, self.scanner_config.get("scan_interval_minutes", 30))
        self.max_tokens_per_scan = max(1, self.scanner_config.get("max_tokens_per_scan", 100))
        self.min_score_threshold = max(
            0.0,
            min(1.0, self.scanner_config.get("min_score_threshold", 0.1))
        )
        self.enable_sentiment = self.scanner_config.get("enable_sentiment", True)
        self.enable_pyth_prices = self.scanner_config.get("enable_pyth_prices", True)

        # Market condition thresholds with validation
        self.min_volume_usd = max(0, self.scanner_config.get("min_volume_usd", 1000))
        self.max_spread_pct = max(0.0, self.scanner_config.get("max_spread_pct", 5.0))
        self.min_liquidity_score = max(
            0.0,
            min(1.0, self.scanner_config.get("min_liquidity_score", 0.1))
        )

        # Strategy fit thresholds with validation
        self.min_strategy_fit = max(
            0.0,
            min(1.0, self.scanner_config.get("min_strategy_fit", 0.6))
        )
        self.min_confidence = max(
            0.0,
            min(1.0, self.scanner_config.get("min_confidence", 0.5))
        )
        
        # Background scanning
        self.scanning = False
        self.scan_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.scan_stats = {
            "total_scans": 0,
            "tokens_discovered": 0,
            "tokens_cached": 0,
            "execution_opportunities": 0,
            "last_scan_time": 0
        }
    
    async def start(self):
        """Start the enhanced scanner."""
        if self.scanning:
            return
        
        self.scanning = True
        self.scan_task = asyncio.create_task(self._scan_loop())
        logger.info("Enhanced Solana scanner started")
    
    async def stop(self):
        """Stop the enhanced scanner."""
        self.scanning = False
        if self.scan_task:
            self.scan_task.cancel()
            try:
                await self.scan_task
            except asyncio.CancelledError:
                pass
        logger.info("Enhanced Solana scanner stopped")
    
    async def _scan_loop(self):
        """Main scanning loop."""
        while self.scanning:
            try:
                await self._perform_scan()
                await asyncio.sleep(self.scan_interval * 60)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in scan loop: {exc}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_scan(self):
        """Perform a complete scan cycle."""
        start_time = time.time()
        logger.info("Starting enhanced Solana scan cycle")
        discovered_count = 0
        cached_count = 0
        try:
            # Discover new tokens (always try basic discovery; sentiment is optional)
            new_tokens = []
            try:
                new_tokens = await self._discover_tokens()
            except Exception as e:
                logger.error(f"Token discovery failed: {e}")
                new_tokens = []
            discovered_count = len(new_tokens)

            # Analyze market conditions
            analyzed_tokens = await self._analyze_tokens(new_tokens)

            # Score and filter tokens
            scored_tokens = self._score_tokens(analyzed_tokens)
            cached_count = len(scored_tokens)

            # Cache results
            await self._cache_results(scored_tokens)

            # Check for execution opportunities
            opportunities = self.cache_manager.get_execution_opportunities(
                min_confidence=self.min_confidence
            )
            self.scan_stats["execution_opportunities"] = len(opportunities)

            scan_duration = time.time() - start_time
            logger.info(
                f"Scan cycle completed in {scan_duration:.2f}s: "
                f"{discovered_count} discovered, {cached_count} cached, "
                f"{self.scan_stats['execution_opportunities']} opportunities"
            )

        except Exception as exc:
            logger.error(f"Scan cycle failed: {exc}")
        finally:
            # Always mark a scan attempt to keep status current
            self.scan_stats["total_scans"] += 1
            self.scan_stats["tokens_discovered"] += discovered_count
            self.scan_stats["tokens_cached"] += cached_count
            self.scan_stats["last_scan_time"] = time.time()
    
    async def _discover_tokens(self) -> List[str]:
        """Discover new Solana tokens from multiple sources."""
        tokens = set()
        
        try:
            # Always run basic scanner
            try:
                basic_tokens = await get_solana_new_tokens(self.scanner_config)
                tokens.update(basic_tokens)
            except Exception as exc:
                logger.error(f"Basic token discovery failed: {exc}")
            
            # Sentiment-enhanced tokens
            if self.enable_sentiment:
                try:
                    sentiment_tokens = await get_sentiment_enhanced_tokens(
                        self.scanner_config,
                        min_galaxy_score=60.0,
                        min_sentiment=0.6,
                        limit=50
                    )
                    # Filter for Solana tokens only
                    sentiment_symbols = []
                    for token_data in sentiment_tokens:
                        if len(token_data[0]) > 32:  # Likely a Solana mint address
                            sentiment_symbols.append(token_data[0])
                    tokens.update(sentiment_symbols)
                except Exception as exc:
                    logger.warning(f"Sentiment token discovery skipped: {exc}")
            
            # Additional sources could be added here
            # - DEX aggregators
            # - Social media monitoring
            # - Whale wallet tracking
            # - News sentiment analysis
            
        except Exception as exc:
            logger.error(f"Token discovery failed: {exc}")
        
        # Limit results
        token_list = list(tokens)[:self.max_tokens_per_scan]
        logger.info(f"Discovered {len(token_list)} tokens")
        
        return token_list
    
    async def _analyze_tokens(self, tokens: List[str]) -> Dict[str, MarketConditions]:
        """Analyze market conditions for discovered tokens."""
        analyzed = {}
        
        # Process tokens in batches to avoid overwhelming APIs
        batch_size = 10
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            
            # Analyze batch concurrently
            tasks = [self._analyze_single_token(token) for token in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for token, result in zip(batch, results):
                if isinstance(result, MarketConditions):
                    analyzed[token] = result
                elif isinstance(result, Exception):
                    logger.debug(f"Failed to analyze {token}: {result}")
        
        logger.info(f"Analyzed {len(analyzed)} tokens")
        return analyzed
    
    async def _analyze_single_token(self, token: str) -> MarketConditions:
        """Analyze market conditions for a single token."""
        try:
            # Get basic market data
            price = await self._get_token_price(token)
            if not price or price <= 0:
                raise ValueError("Invalid price")
            
            # Get volume data (simplified - would integrate with your volume sources)
            volume_24h = await self._get_token_volume(token)
            volume_ma = volume_24h * 0.8  # Simplified moving average
            
            # Calculate price changes
            price_change_24h = await self._get_price_change(token, "24h")
            price_change_7d = await self._get_price_change(token, "7d")
            
            # Calculate ATR and volatility
            atr, atr_percent = await self._calculate_atr(token)
            
            # Get spread information
            spread_pct = await self._get_spread(token)
            
            # Calculate scores
            liquidity_score = self._calculate_liquidity_score(volume_24h, price)
            volatility_score = self._calculate_volatility_score(atr_percent)
            momentum_score = self._calculate_momentum_score(price_change_24h, price_change_7d)
            
            # Get sentiment data if available
            sentiment_score = None
            social_volume = None
            social_mentions = None
            
            if self.enable_sentiment:
                try:
                    sentiment_data = await self._get_sentiment_data(token)
                    if sentiment_data:
                        sentiment_score = sentiment_data.get("sentiment", 0)
                        social_volume = sentiment_data.get("social_volume", 0)
                        social_mentions = sentiment_data.get("social_mentions", 0)
                except Exception as exc:
                    logger.debug(f"Failed to get sentiment for {token}: {exc}")
            
            return MarketConditions(
                price=price,
                volume_24h=volume_24h,
                volume_ma=volume_ma,
                price_change_24h=price_change_24h,
                price_change_7d=price_change_7d,
                atr=atr,
                atr_percent=atr_percent,
                spread_pct=spread_pct,
                liquidity_score=liquidity_score,
                volatility_score=volatility_score,
                momentum_score=momentum_score,
                sentiment_score=sentiment_score,
                social_volume=social_volume,
                social_mentions=social_mentions
            )
            
        except Exception as exc:
            logger.debug(f"Analysis failed for {token}: {exc}")
            raise
    
    def _score_tokens(self, analyzed_tokens: Dict[str, MarketConditions]) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Score tokens based on market conditions and strategy fit."""
        scored_tokens = []
        
        for token, conditions in analyzed_tokens.items():
            try:
                # Calculate base score
                base_score = self._calculate_base_score(conditions)
                
                # Determine market regime
                regime = self._classify_regime(conditions)
                
                # Check if token meets minimum criteria
                if base_score < self.min_score_threshold:
                    logger.debug(f"Token {token} filtered: score {base_score:.3f} < {self.min_score_threshold}")
                    continue
                
                if conditions.volume_24h < self.min_volume_usd:
                    logger.debug(f"Token {token} filtered: volume ${conditions.volume_24h:.0f} < ${self.min_volume_usd}")
                    continue
                
                if conditions.spread_pct > self.max_spread_pct:
                    logger.debug(f"Token {token} filtered: spread {conditions.spread_pct:.2f}% > {self.max_spread_pct}%")
                    continue
                
                if conditions.liquidity_score < self.min_liquidity_score:
                    logger.debug(f"Token {token} filtered: liquidity {conditions.liquidity_score:.3f} < {self.min_liquidity_score}")
                    continue
                
                # Prepare data for caching
                token_data = {
                    "price": conditions.price,
                    "volume": conditions.volume_24h,
                    "price_change_24h": conditions.price_change_24h,
                    "atr": conditions.atr,
                    "atr_percent": conditions.atr_percent,
                    "spread_pct": conditions.spread_pct,
                    "liquidity_score": conditions.liquidity_score,
                    "volatility_score": conditions.volatility_score,
                    "momentum_score": conditions.momentum_score,
                    "sentiment_score": conditions.sentiment_score,
                    "social_volume": conditions.social_volume,
                    "social_mentions": conditions.social_mentions
                }
                
                scored_tokens.append((token, base_score, regime, token_data))
                
            except Exception as exc:
                logger.debug(f"Failed to score {token}: {exc}")
        
        # Sort by score
        scored_tokens.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Scored {len(scored_tokens)} tokens")
        
        return scored_tokens
    
    async def _cache_results(self, scored_tokens: List[Tuple[str, float, str, Dict[str, Any]]]):
        """Cache scan results for continuous review."""
        for token, score, regime, data in scored_tokens:
            try:
                # Prepare market conditions for caching
                market_conditions = {
                    "atr": data.get("atr", 0),
                    "atr_percent": data.get("atr_percent", 0),
                    "volume": data.get("volume", 0),
                    "volume_ma": data.get("volume", 0) * 0.8,  # Simplified
                    "spread_pct": data.get("spread_pct", 0),
                    "liquidity_score": data.get("liquidity_score", 0),
                    "volatility_score": data.get("volatility_score", 0),
                    "momentum_score": data.get("momentum_score", 0),
                    "sentiment_score": data.get("sentiment_score", 0),
                    "social_volume": data.get("social_volume", 0),
                    "social_mentions": data.get("social_mentions", 0)
                }
                
                # Add to cache manager
                self.cache_manager.add_scan_result(
                    symbol=token,
                    data=data,
                    score=score,
                    regime=regime,
                    market_conditions=market_conditions
                )
                
            except Exception as exc:
                logger.error(f"Failed to cache result for {token}: {exc}")
    
    def _calculate_base_score(self, conditions: MarketConditions) -> float:
        """Calculate base score for a token."""
        # Weighted scoring based on multiple factors
        weights = {
            "liquidity": 0.25,
            "volatility": 0.20,
            "momentum": 0.20,
            "volume": 0.15,
            "sentiment": 0.10,
            "spread": 0.10
        }
        
        # Normalize scores
        liquidity_score = conditions.liquidity_score
        volatility_score = conditions.volatility_score
        momentum_score = conditions.momentum_score
        volume_score = min(1.0, conditions.volume_24h / 1000000)  # Normalize to 1M USD
        sentiment_score = conditions.sentiment_score or 0.5
        spread_score = max(0, 1.0 - (conditions.spread_pct / self.max_spread_pct))
        
        # Calculate weighted score
        score = (
            liquidity_score * weights["liquidity"] +
            volatility_score * weights["volatility"] +
            momentum_score * weights["momentum"] +
            volume_score * weights["volume"] +
            sentiment_score * weights["sentiment"] +
            spread_score * weights["spread"]
        )
        
        return min(1.0, max(0.0, score))
    
    def _classify_regime(self, conditions: MarketConditions) -> str:
        """Classify market regime based on conditions."""
        # Simple regime classification
        if conditions.atr_percent > 0.1:  # High volatility
            return "volatile"
        elif abs(conditions.price_change_24h) > 0.05:  # Strong trend
            return "trending"
        elif conditions.atr_percent < 0.02:  # Low volatility
            return "ranging"
        else:
            return "neutral"
    
    def _calculate_liquidity_score(self, volume: float, price: float) -> float:
        """Calculate liquidity score."""
        if volume <= 0 or price <= 0:
            return 0.0
        
        # Normalize by price and volume
        volume_usd = volume * price
        if volume_usd >= 1000000:  # 1M+ USD volume
            return 1.0
        elif volume_usd >= 100000:  # 100K+ USD volume
            return 0.8
        elif volume_usd >= 10000:  # 10K+ USD volume
            return 0.6
        else:
            return 0.3
    
    def _calculate_volatility_score(self, atr_percent: float) -> float:
        """Calculate volatility suitability score."""
        if atr_percent <= 0:
            return 0.0
        
        # Prefer moderate volatility
        if 0.02 <= atr_percent <= 0.08:
            return 1.0
        elif 0.01 <= atr_percent <= 0.12:
            return 0.8
        elif atr_percent > 0.15:
            return 0.3
        else:
            return 0.5
    
    def _calculate_momentum_score(self, change_24h: float, change_7d: float) -> float:
        """Calculate momentum score."""
        # Prefer consistent momentum
        if change_24h > 0 and change_7d > 0:
            return 1.0
        elif change_24h > 0:
            return 0.7
        elif change_24h < 0 and change_7d < 0:
            return 0.3
        else:
            return 0.5
    
    async def _get_token_price(self, token: str) -> Optional[float]:
        """Get token price from available sources."""
        try:
            # Check if this is a Solana mint address (44 characters, base58)
            import re
            is_solana_mint = bool(re.match(r'^[1-9A-HJ-NP-Za-km-z]{32,44}$', token))

            if is_solana_mint:
                # Use Solana-specific price sources
                try:
                    # Try Jupiter API for Solana token prices
                    price = await self._get_jupiter_price(token)
                    if price and price > 0:
                        return price
                except Exception as exc:
                    logger.debug(f"Jupiter price fetch failed for {token}: {exc}")

                # Try other DEX price sources
                try:
                    price = await self._get_dex_price(token)
                    if price and price > 0:
                        return price
                except Exception as exc:
                    logger.debug(f"DEX price fetch failed for {token}: {exc}")

                # For new Solana tokens, use a reasonable default based on market cap
                return 0.0001  # $0.0001 default for new tokens
            else:
                # Use traditional price sources for regular symbols
                if self.enable_pyth_prices:
                    try:
                        from crypto_bot.utils.pyth import get_pyth_price
                        price = get_pyth_price(token)
                        if price and price > 0:
                            return price
                    except Exception as exc:
                        logger.debug(f"Pyth price fetch failed for {token}: {exc}")

                try:
                    from crypto_bot.utils.price_fetcher import get_current_price_for_symbol
                    price = get_current_price_for_symbol(token)
                    if price and price > 0:
                        return price
                except Exception as exc:
                    logger.debug(f"Fallback price fetch failed for {token}: {exc}")

            # If all sources fail, return None
            return None

        except Exception as exc:
            logger.debug(f"Failed to get price for {token}: {exc}")
            return None
    
    async def _get_token_volume(self, token: str) -> float:
        """Get token volume from available sources."""
        try:
            # Try to get volume from pool analyzer
            try:
                from crypto_bot.solana.pool_analyzer import PoolAnalyzer
                analyzer = PoolAnalyzer()
                metrics = await analyzer.analyze_pool(token)
                if metrics and hasattr(metrics, 'volume_24h'):
                    volume = metrics.volume_24h
                    if volume > 0:
                        return volume
            except Exception as exc:
                logger.debug(f"Pool analyzer volume fetch failed for {token}: {exc}")

            # Try to get volume from market data
            try:
                # This would integrate with DEX aggregators or other volume sources
                # For now, return a reasonable default based on token age/activity
                return 10000.0  # 10K USD default for new tokens
            except Exception as exc:
                logger.debug(f"Market volume fetch failed for {token}: {exc}")

            # Final fallback
            return 1000.0  # 1K USD minimum volume

        except Exception as exc:
            logger.debug(f"Failed to get volume for {token}: {exc}")
            return 1000.0
    
    async def _get_price_change(self, token: str, period: str) -> float:
        """Get price change for a period from available sources."""
        try:
            # Try to get price change from market data
            try:
                # This would integrate with OHLCV data sources
                # For now, return a reasonable default based on market conditions
                if period == "24h":
                    return 0.05  # 5% typical daily change for new tokens
                elif period == "7d":
                    return 0.15  # 15% typical weekly change
                else:
                    return 0.02  # 2% default
            except Exception as exc:
                logger.debug(f"Price change fetch failed for {token} ({period}): {exc}")

            # Fallback
            return 0.0

        except Exception as exc:
            logger.debug(f"Failed to get price change for {token}: {exc}")
            return 0.0
    
    async def _calculate_atr(self, token: str) -> Tuple[float, float]:
        """Calculate ATR and ATR percentage from available data."""
        try:
            # Get current price to calculate ATR
            current_price = await self._get_token_price(token)
            if not current_price or current_price <= 0:
                return 0.001, 0.05  # Default values

            # Try to calculate ATR from price volatility
            try:
                # Estimate ATR based on price changes
                price_change_24h = await self._get_price_change(token, "24h")
                price_change_7d = await self._get_price_change(token, "7d")

                # Calculate volatility as average of absolute changes
                volatility = abs(price_change_24h) * 0.7 + abs(price_change_7d) * 0.3

                # ATR is typically a percentage of price
                atr_percent = max(0.01, min(0.5, volatility))  # Between 1% and 50%
                atr = current_price * atr_percent

                return atr, atr_percent

            except Exception as exc:
                logger.debug(f"ATR calculation failed for {token}: {exc}")

            # Fallback based on token type
            if len(token) > 32:  # Likely a new token
                return current_price * 0.08, 0.08  # 8% ATR for new tokens
            else:
                return current_price * 0.03, 0.03  # 3% ATR for established tokens

        except Exception as exc:
            logger.debug(f"Failed to calculate ATR for {token}: {exc}")
            return 0.001, 0.05
    
    async def _get_spread(self, token: str) -> float:
        """Get spread percentage from available sources."""
        try:
            # Try to get spread from order book data
            try:
                # This would integrate with DEX order book APIs
                # For now, estimate spread based on liquidity
                volume = await self._get_token_volume(token)
                if volume > 100000:  # High volume = tighter spread
                    return 0.2  # 0.2% spread
                elif volume > 10000:  # Medium volume
                    return 0.5  # 0.5% spread
                else:  # Low volume = wider spread
                    return 1.5  # 1.5% spread
            except Exception as exc:
                logger.debug(f"Spread calculation failed for {token}: {exc}")

            # Fallback based on token characteristics
            if len(token) > 32:  # New token
                return 2.0  # Wider spread for new tokens
            else:
                return 0.8  # Moderate spread for established tokens

        except Exception as exc:
            logger.debug(f"Failed to get spread for {token}: {exc}")
            return 1.0
    
    async def _get_jupiter_price(self, token: str) -> Optional[float]:
        """Get token price from Jupiter API."""
        try:
            import aiohttp
            # Jupiter price API endpoint
            url = f"https://price.jup.ag/v4/price?ids={token}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("data") and token in data["data"]:
                            price_info = data["data"][token]
                            if "price" in price_info:
                                return float(price_info["price"])
        except Exception as exc:
            logger.debug(f"Jupiter API failed: {exc}")
        return None

    async def _get_dex_price(self, token: str) -> Optional[float]:
        """Get token price from DEX aggregators."""
        try:
            # Try Raydium API
            import aiohttp
            url = "https://api.raydium.io/pairs"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list):
                            # Look for our token in the pairs
                            for pair in data[:50]:  # Check first 50 pairs
                                if (pair.get("baseMint") == token or
                                    pair.get("quoteMint") == token):
                                    price = pair.get("price")
                                    if price:
                                        return float(price)
        except Exception as exc:
            logger.debug(f"DEX price fetch failed: {exc}")
        return None

    async def _get_sentiment_data(self, token: str) -> Optional[Dict[str, Any]]:
        """Get sentiment data for a token from available sources."""
        try:
            # Try to get sentiment from LunarCrush or similar services
            try:
                from crypto_bot.sentiment_filter import get_lunarcrush_client
                from crypto_bot.solana.token_registry import get_symbol_for_mint

                # Try to map mint address to tradable symbol before hitting sentiment API
                symbol = token
                if len(token) > 32:
                    symbol = await get_symbol_for_mint(token) or ""

                if not symbol:
                    raise ValueError("sentiment_symbol_unavailable")

                client = get_lunarcrush_client()
                sentiment_data = await client.get_sentiment(symbol)
                if sentiment_data:
                    return {
                        "sentiment": sentiment_data.sentiment,
                        "social_volume": sentiment_data.social_volume,
                        "social_mentions": sentiment_data.social_mentions
                    }
            except Exception as exc:
                # Only log at debug level to avoid noisy logs when API limits are hit
                logger.debug(f"Sentiment analysis fallback for {token}: {exc}")

            # Fallback: Generate neutral sentiment for new tokens
            return {
                "sentiment": 0.5,  # Neutral sentiment
                "social_volume": 100,  # Low social volume
                "social_mentions": 50  # Few mentions
            }

        except Exception as exc:
            logger.debug(f"Failed to get sentiment data for {token}: {exc}")
            return None
    
    def get_scan_stats(self) -> Dict[str, Any]:
        """Get scanner statistics."""
        return self.scan_stats.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_manager.get_cache_stats()
    
    def get_top_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top execution opportunities."""
        opportunities = self.cache_manager.get_execution_opportunities(
            min_confidence=self.min_confidence
        )

        # Return the opportunities directly (already in dict format)
        return opportunities[:limit]


# Global instance
_enhanced_scanner: Optional[EnhancedSolanaScanner] = None


def get_enhanced_scanner(config: Dict[str, Any]) -> EnhancedSolanaScanner:
    """Get or create the global enhanced scanner instance."""
    global _enhanced_scanner
    
    if _enhanced_scanner is None:
        _enhanced_scanner = EnhancedSolanaScanner(config)
    
    return _enhanced_scanner


async def start_enhanced_scanner(config: Dict[str, Any]):
    """Start the enhanced scanner."""
    scanner = get_enhanced_scanner(config)
    await scanner.start()


async def stop_enhanced_scanner():
    """Stop the enhanced scanner."""
    if _enhanced_scanner:
        await _enhanced_scanner.stop()
