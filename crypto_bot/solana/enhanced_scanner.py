"""
Enhanced Solana Scanner with integrated caching and continuous review.

This scanner integrates with the scan cache manager to provide:
- Persistent caching of scan results
- Continuous strategy fit analysis
- Execution opportunity detection
- Market condition monitoring
"""

import asyncio
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np

from ..utils.scan_cache_manager import get_scan_cache_manager, ScanResult
from ..utils.logger import setup_logger, LOG_DIR
from .scanner import get_solana_new_tokens, get_sentiment_enhanced_tokens
from .pyth_utils import get_pyth_price
from .score import calculate_token_score

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
        
        # Scanner settings
        self.scan_interval = self.scanner_config.get("scan_interval_minutes", 30)
        self.max_tokens_per_scan = self.scanner_config.get("max_tokens_per_scan", 100)
        self.min_score_threshold = self.scanner_config.get("min_score_threshold", 0.3)
        self.enable_sentiment = self.scanner_config.get("enable_sentiment", True)
        self.enable_pyth_prices = self.scanner_config.get("enable_pyth_prices", True)
        
        # Market condition thresholds
        self.min_volume_usd = self.scanner_config.get("min_volume_usd", 10000)
        self.max_spread_pct = self.scanner_config.get("max_spread_pct", 2.0)
        self.min_liquidity_score = self.scanner_config.get("min_liquidity_score", 0.5)
        
        # Strategy fit thresholds
        self.min_strategy_fit = self.scanner_config.get("min_strategy_fit", 0.6)
        self.min_confidence = self.scanner_config.get("min_confidence", 0.5)
        
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
        
        try:
            # Discover new tokens
            new_tokens = await self._discover_tokens()
            
            # Analyze market conditions
            analyzed_tokens = await self._analyze_tokens(new_tokens)
            
            # Score and filter tokens
            scored_tokens = await self._score_tokens(analyzed_tokens)
            
            # Cache results
            await self._cache_results(scored_tokens)
            
            # Update statistics
            self.scan_stats["total_scans"] += 1
            self.scan_stats["tokens_discovered"] += len(new_tokens)
            self.scan_stats["tokens_cached"] += len(scored_tokens)
            self.scan_stats["last_scan_time"] = time.time()
            
            # Check for execution opportunities
            opportunities = self.cache_manager.get_execution_opportunities(
                min_confidence=self.min_confidence
            )
            self.scan_stats["execution_opportunities"] = len(opportunities)
            
            scan_duration = time.time() - start_time
            logger.info(
                f"Scan cycle completed in {scan_duration:.2f}s: "
                f"{len(new_tokens)} discovered, {len(scored_tokens)} cached, "
                f"{len(opportunities)} opportunities"
            )
            
        except Exception as exc:
            logger.error(f"Scan cycle failed: {exc}")
    
    async def _discover_tokens(self) -> List[str]:
        """Discover new Solana tokens from multiple sources."""
        tokens = set()
        
        try:
            # Basic scanner
            basic_tokens = await get_solana_new_tokens(self.scanner_config)
            tokens.update(basic_tokens)
            
            # Sentiment-enhanced tokens
            if self.enable_sentiment:
                try:
                    sentiment_tokens = await get_sentiment_enhanced_tokens(
                        self.scanner_config,
                        min_galaxy_score=60.0,
                        min_sentiment=0.6,
                        limit=50
                    )
                    sentiment_symbols = [t[0] for t in sentiment_tokens]
                    tokens.update(sentiment_symbols)
                except Exception as exc:
                    logger.debug(f"Sentiment token discovery failed: {exc}")
            
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
    
    async def _score_tokens(self, analyzed_tokens: Dict[str, MarketConditions]) -> List[Tuple[str, float, str, Dict[str, Any]]]:
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
                    continue
                
                if conditions.volume_24h < self.min_volume_usd:
                    continue
                
                if conditions.spread_pct > self.max_spread_pct:
                    continue
                
                if conditions.liquidity_score < self.min_liquidity_score:
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
            # Try Pyth first
            if self.enable_pyth_prices:
                pyth_price = await get_pyth_price(token)
                if pyth_price and pyth_price > 0:
                    return pyth_price
            
            # Fallback to other sources
            # This would integrate with your existing price fetching logic
            
            return None
            
        except Exception as exc:
            logger.debug(f"Failed to get price for {token}: {exc}")
            return None
    
    async def _get_token_volume(self, token: str) -> float:
        """Get token volume (simplified implementation)."""
        # This would integrate with your volume data sources
        # For now, return a placeholder value
        return 50000.0  # 50K USD placeholder
    
    async def _get_price_change(self, token: str, period: str) -> float:
        """Get price change for a period (simplified implementation)."""
        # This would integrate with your historical data
        # For now, return a placeholder value
        return 0.02  # 2% placeholder
    
    async def _calculate_atr(self, token: str) -> Tuple[float, float]:
        """Calculate ATR and ATR percentage (simplified implementation)."""
        # This would integrate with your OHLCV data
        # For now, return placeholder values
        atr = 0.001  # 0.1% placeholder
        atr_percent = 0.05  # 5% placeholder
        return atr, atr_percent
    
    async def _get_spread(self, token: str) -> float:
        """Get spread percentage (simplified implementation)."""
        # This would integrate with your order book data
        # For now, return a placeholder value
        return 0.5  # 0.5% placeholder
    
    async def _get_sentiment_data(self, token: str) -> Optional[Dict[str, Any]]:
        """Get sentiment data for a token (simplified implementation)."""
        # This would integrate with your sentiment analysis
        # For now, return placeholder data
        return {
            "sentiment": 0.7,
            "social_volume": 1000,
            "social_mentions": 500
        }
    
    def get_scan_stats(self) -> Dict[str, Any]:
        """Get scanner statistics."""
        return self.scan_stats.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_manager.get_cache_stats()
    
    async def get_top_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top execution opportunities."""
        opportunities = self.cache_manager.get_execution_opportunities(
            min_confidence=self.min_confidence
        )
        
        # Convert to dict format for easier handling
        result = []
        for opp in opportunities[:limit]:
            result.append({
                "symbol": opp.symbol,
                "strategy": opp.strategy,
                "direction": opp.direction,
                "confidence": opp.confidence,
                "entry_price": opp.entry_price,
                "stop_loss": opp.stop_loss,
                "take_profit": opp.take_profit,
                "risk_reward_ratio": opp.risk_reward_ratio,
                "timestamp": opp.timestamp
            })
        
        return result


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
