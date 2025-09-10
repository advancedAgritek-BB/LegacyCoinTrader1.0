"""
Advanced Memecoin Pump Detection System

This module provides sophisticated detection of high-probability pump opportunities
by analyzing multiple signals including liquidity patterns, transaction velocity,
social sentiment, and market microstructure.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import numpy as np
import aiohttp
import json

from .watcher import NewPoolEvent
from .safety import is_safe
from .score import score_event_with_sentiment
from ..utils.telemetry import telemetry

logger = logging.getLogger(__name__)


@dataclass
class PumpSignal:
    """Represents a detected pump signal."""
    
    signal_type: str
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    timestamp: float
    details: Dict = field(default_factory=dict)


@dataclass
class PoolAnalysis:
    """Comprehensive analysis of a liquidity pool."""
    
    pool_address: str
    token_mint: str
    
    # Liquidity Metrics
    initial_liquidity: float
    current_liquidity: float
    liquidity_change_rate: float
    liquidity_stability_score: float
    
    # Transaction Metrics
    transaction_velocity: float  # txs per minute
    unique_wallets: int
    avg_transaction_size: float
    whale_activity_score: float
    
    # Price & Volume Metrics
    price_momentum: float
    volume_spike_factor: float
    volume_consistency: float
    price_stability: float
    
    # Social & Sentiment
    social_buzz_score: float
    sentiment_score: float
    influencer_mentions: int
    
    # Technical Indicators
    rsi: float
    bollinger_position: float
    volume_profile_score: float
    
    # Risk Factors
    rug_risk_score: float
    dev_activity_score: float
    tokenomics_score: float
    
    # Composite Scores
    pump_probability: float = 0.0
    timing_score: float = 0.0
    risk_adjusted_score: float = 0.0
    
    signals: List[PumpSignal] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class PumpDetector:
    """
    Advanced pump detection engine for memecoin liquidity pools.
    
    Features:
    - Multi-signal analysis combining on-chain and social data
    - Real-time transaction pattern analysis
    - Liquidity quality assessment
    - Social sentiment integration
    - Risk-adjusted scoring
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.detection_config = config.get("pump_detection", {})
        
        # Detection thresholds
        self.min_pump_probability = self.detection_config.get("min_pump_probability", 0.7)
        self.min_timing_score = self.detection_config.get("min_timing_score", 0.6)
        self.max_risk_score = self.detection_config.get("max_risk_score", 0.4)
        
        # Signal weights
        self.signal_weights = self.detection_config.get("signal_weights", {
            "liquidity_analysis": 0.25,
            "transaction_velocity": 0.20,
            "social_sentiment": 0.15,
            "price_momentum": 0.15,
            "volume_analysis": 0.15,
            "risk_factors": -0.10
        })
        
        # Pool tracking
        self.tracked_pools: Dict[str, PoolAnalysis] = {}
        self.transaction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.social_cache: Dict[str, Dict] = {}
        
        # API sessions
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            "pools_analyzed": 0,
            "pumps_detected": 0,
            "successful_predictions": 0,
            "false_positives": 0
        }
        
    async def start(self):
        """Initialize the pump detector."""
        self.session = aiohttp.ClientSession()
        logger.info("Pump detector started")
        
    async def stop(self):
        """Clean shutdown of the pump detector."""
        if self.session:
            await self.session.close()
        logger.info("Pump detector stopped")
        
    async def analyze_pool(self, event: NewPoolEvent) -> Optional[PoolAnalysis]:
        """
        Perform comprehensive analysis of a liquidity pool.
        
        Args:
            event: New pool event to analyze
            
        Returns:
            PoolAnalysis if pool shows pump potential, None otherwise
        """
        try:
            # Initialize analysis
            analysis = PoolAnalysis(
                pool_address=event.pool_address,
                token_mint=event.token_mint,
                initial_liquidity=event.liquidity,
                current_liquidity=event.liquidity
            )
            
            # Perform multi-signal analysis
            await self._analyze_liquidity_metrics(analysis, event)
            await self._analyze_transaction_patterns(analysis)
            await self._analyze_price_momentum(analysis)
            await self._analyze_social_sentiment(analysis)
            await self._analyze_risk_factors(analysis)
            await self._calculate_technical_indicators(analysis)
            
            # Calculate composite scores
            self._calculate_composite_scores(analysis)
            
            # Check if pool meets minimum criteria
            if (analysis.pump_probability >= self.min_pump_probability and
                analysis.timing_score >= self.min_timing_score and
                analysis.rug_risk_score <= self.max_risk_score):
                
                self.tracked_pools[event.pool_address] = analysis
                self.stats["pumps_detected"] += 1
                
                telemetry.inc("pump_detector.pumps_detected")
                telemetry.gauge("pump_detector.pump_probability", analysis.pump_probability)
                
                logger.info(
                    f"High-probability pump detected: {event.token_mint[:8]}... "
                    f"Probability: {analysis.pump_probability:.2f}, "
                    f"Timing: {analysis.timing_score:.2f}, "
                    f"Risk: {analysis.rug_risk_score:.2f}"
                )
                
                return analysis
                
            self.stats["pools_analyzed"] += 1
            return None
            
        except Exception as exc:
            logger.error(f"Failed to analyze pool {event.pool_address}: {exc}")
            return None
            
    async def _analyze_liquidity_metrics(self, analysis: PoolAnalysis, event: NewPoolEvent):
        """Analyze liquidity quality and stability."""
        try:
            # Get current pool state
            pool_data = await self._fetch_pool_data(event.pool_address)
            if not pool_data:
                return
                
            # Calculate liquidity metrics
            analysis.current_liquidity = pool_data.get("liquidity", event.liquidity)
            analysis.liquidity_change_rate = self._calculate_liquidity_change_rate(analysis)
            
            # Liquidity stability analysis
            analysis.liquidity_stability_score = self._analyze_liquidity_stability(pool_data)
            
            # Add liquidity signal
            if analysis.liquidity_stability_score > 0.7:
                signal = PumpSignal(
                    signal_type="stable_liquidity",
                    strength=analysis.liquidity_stability_score,
                    confidence=0.8,
                    timestamp=time.time(),
                    details={"liquidity": analysis.current_liquidity}
                )
                analysis.signals.append(signal)
                
        except Exception as exc:
            logger.debug(f"Liquidity analysis failed: {exc}")
            
    async def _analyze_transaction_patterns(self, analysis: PoolAnalysis):
        """Analyze transaction velocity and patterns."""
        try:
            # Get recent transactions
            txs = await self._fetch_recent_transactions(analysis.pool_address)
            if not txs:
                return
                
            # Calculate transaction metrics
            analysis.transaction_velocity = len(txs) / 5.0  # per minute over 5 min
            analysis.unique_wallets = len(set(tx.get("signer", "") for tx in txs))
            analysis.avg_transaction_size = np.mean([tx.get("amount", 0) for tx in txs])
            
            # Whale activity detection
            large_txs = [tx for tx in txs if tx.get("amount", 0) > analysis.avg_transaction_size * 5]
            analysis.whale_activity_score = len(large_txs) / max(len(txs), 1)
            
            # Velocity signal
            if analysis.transaction_velocity > 2.0:  # > 2 tx/min
                signal = PumpSignal(
                    signal_type="high_velocity",
                    strength=min(analysis.transaction_velocity / 10.0, 1.0),
                    confidence=0.75,
                    timestamp=time.time(),
                    details={"velocity": analysis.transaction_velocity}
                )
                analysis.signals.append(signal)
                
        except Exception as exc:
            logger.debug(f"Transaction analysis failed: {exc}")
            
    async def _analyze_price_momentum(self, analysis: PoolAnalysis):
        """Analyze price momentum and volume patterns."""
        try:
            # Get price data
            price_data = await self._fetch_price_data(analysis.token_mint)
            if not price_data:
                return
                
            prices = [p["price"] for p in price_data[-20:]]  # Last 20 data points
            volumes = [p["volume"] for p in price_data[-20:]]
            
            # Calculate momentum
            if len(prices) >= 2:
                analysis.price_momentum = (prices[-1] - prices[0]) / prices[0]
                
            # Volume spike analysis
            if len(volumes) >= 5:
                recent_avg = np.mean(volumes[-5:])
                historical_avg = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_avg
                analysis.volume_spike_factor = recent_avg / max(historical_avg, 1)
                
            # Momentum signal
            if analysis.price_momentum > 0.05:  # 5% gain
                signal = PumpSignal(
                    signal_type="price_momentum",
                    strength=min(analysis.price_momentum * 10, 1.0),
                    confidence=0.8,
                    timestamp=time.time(),
                    details={"momentum": analysis.price_momentum}
                )
                analysis.signals.append(signal)
                
        except Exception as exc:
            logger.debug(f"Price momentum analysis failed: {exc}")
            
    async def _analyze_social_sentiment(self, analysis: PoolAnalysis):
        """Analyze social media buzz and sentiment."""
        try:
            # Check cache first
            cache_key = f"{analysis.token_mint}_{int(time.time() // 300)}"  # 5-min cache
            if cache_key in self.social_cache:
                social_data = self.social_cache[cache_key]
            else:
                social_data = await self._fetch_social_data(analysis.token_mint)
                self.social_cache[cache_key] = social_data
                
            if not social_data:
                return
                
            # Extract social metrics
            analysis.social_buzz_score = social_data.get("buzz_score", 0)
            analysis.sentiment_score = social_data.get("sentiment", 0.5)
            analysis.influencer_mentions = social_data.get("influencer_mentions", 0)
            
            # Social signal
            if analysis.social_buzz_score > 0.6:
                signal = PumpSignal(
                    signal_type="social_buzz",
                    strength=analysis.social_buzz_score,
                    confidence=0.6,
                    timestamp=time.time(),
                    details={"buzz": analysis.social_buzz_score, "sentiment": analysis.sentiment_score}
                )
                analysis.signals.append(signal)
                
        except Exception as exc:
            logger.debug(f"Social sentiment analysis failed: {exc}")
            
    async def _analyze_risk_factors(self, analysis: PoolAnalysis):
        """Analyze rug pull and other risk factors."""
        try:
            # Get token metadata
            token_data = await self._fetch_token_metadata(analysis.token_mint)
            if not token_data:
                return
                
            # Risk scoring
            freeze_authority = token_data.get("freeze_authority")
            mint_authority = token_data.get("mint_authority")
            
            # Calculate risk scores
            risk_factors = []
            
            # Authority risks
            if freeze_authority and freeze_authority != "11111111111111111111111111111111":
                risk_factors.append(0.3)
            if mint_authority and mint_authority != "11111111111111111111111111111111":
                risk_factors.append(0.2)
                
            # Developer activity score
            dev_score = await self._analyze_developer_activity(analysis.token_mint)
            analysis.dev_activity_score = dev_score
            if dev_score < 0.3:
                risk_factors.append(0.2)
                
            # Calculate composite rug risk
            analysis.rug_risk_score = min(sum(risk_factors), 1.0)
            
            # Tokenomics analysis
            analysis.tokenomics_score = await self._analyze_tokenomics(token_data)
            
        except Exception as exc:
            logger.debug(f"Risk analysis failed: {exc}")
            
    async def _calculate_technical_indicators(self, analysis: PoolAnalysis):
        """Calculate technical indicators."""
        try:
            # Get OHLCV data
            ohlcv_data = await self._fetch_ohlcv_data(analysis.token_mint)
            if not ohlcv_data or len(ohlcv_data) < 14:
                return
                
            prices = [d["close"] for d in ohlcv_data]
            volumes = [d["volume"] for d in ohlcv_data]
            
            # RSI calculation
            analysis.rsi = self._calculate_rsi(prices)
            
            # Bollinger Bands position
            analysis.bollinger_position = self._calculate_bb_position(prices)
            
            # Volume profile score
            analysis.volume_profile_score = self._calculate_volume_profile_score(volumes)
            
        except Exception as exc:
            logger.debug(f"Technical indicators calculation failed: {exc}")
            
    def _calculate_composite_scores(self, analysis: PoolAnalysis):
        """Calculate final composite scores."""
        try:
            # Pump probability calculation
            scores = {
                "liquidity_analysis": analysis.liquidity_stability_score,
                "transaction_velocity": min(analysis.transaction_velocity / 5.0, 1.0),
                "social_sentiment": analysis.social_buzz_score * analysis.sentiment_score,
                "price_momentum": min(abs(analysis.price_momentum) * 10, 1.0),
                "volume_analysis": min(analysis.volume_spike_factor / 3.0, 1.0),
                "risk_factors": analysis.rug_risk_score
            }
            
            # Weighted pump probability
            weighted_score = 0.0
            for factor, weight in self.signal_weights.items():
                weighted_score += scores.get(factor, 0) * weight
                
            analysis.pump_probability = max(0.0, min(1.0, weighted_score))
            
            # Timing score based on multiple factors
            timing_factors = [
                analysis.transaction_velocity / 10.0,
                min(analysis.volume_spike_factor / 2.0, 1.0),
                analysis.social_buzz_score,
                1.0 - analysis.rug_risk_score
            ]
            analysis.timing_score = np.mean(timing_factors)
            
            # Risk-adjusted score
            risk_penalty = 1.0 - analysis.rug_risk_score
            analysis.risk_adjusted_score = analysis.pump_probability * risk_penalty
            
        except Exception as exc:
            logger.error(f"Composite score calculation failed: {exc}")
            
    # Helper methods for data fetching and calculations
    
    async def _fetch_pool_data(self, pool_address: str) -> Optional[Dict]:
        """Fetch current pool data."""
        try:
            if not self.session:
                return None
                
            # This would integrate with your Helius/Raydium API
            # Placeholder implementation
            return {"liquidity": 10000, "volume_24h": 50000}
            
        except Exception as exc:
            logger.debug(f"Failed to fetch pool data: {exc}")
            return None
            
    async def _fetch_recent_transactions(self, pool_address: str) -> List[Dict]:
        """Fetch recent transactions for the pool."""
        try:
            # This would integrate with Helius transaction APIs
            # Placeholder implementation
            return [
                {"signer": "wallet1", "amount": 1000, "timestamp": time.time()},
                {"signer": "wallet2", "amount": 500, "timestamp": time.time() - 60}
            ]
            
        except Exception as exc:
            logger.debug(f"Failed to fetch transactions: {exc}")
            return []
            
    async def _fetch_price_data(self, token_mint: str) -> List[Dict]:
        """Fetch recent price data."""
        try:
            # This would integrate with your price feeds (Pyth, Jupiter, etc.)
            # Placeholder implementation
            return [
                {"price": 1.0, "volume": 1000, "timestamp": time.time() - i * 60}
                for i in range(20)
            ]
            
        except Exception as exc:
            logger.debug(f"Failed to fetch price data: {exc}")
            return []
            
    async def _fetch_social_data(self, token_mint: str) -> Optional[Dict]:
        """Fetch social media data and sentiment."""
        try:
            # This would integrate with LunarCrush, social APIs, etc.
            # Placeholder implementation
            return {
                "buzz_score": 0.7,
                "sentiment": 0.8,
                "influencer_mentions": 5,
                "social_volume": 1000
            }
            
        except Exception as exc:
            logger.debug(f"Failed to fetch social data: {exc}")
            return None
            
    async def _fetch_token_metadata(self, token_mint: str) -> Optional[Dict]:
        """Fetch token metadata and authority info."""
        try:
            # This would integrate with Solana RPC
            # Placeholder implementation
            return {
                "freeze_authority": "11111111111111111111111111111111",
                "mint_authority": "11111111111111111111111111111111",
                "supply": 1000000000,
                "decimals": 9
            }
            
        except Exception as exc:
            logger.debug(f"Failed to fetch token metadata: {exc}")
            return None
            
    async def _fetch_ohlcv_data(self, token_mint: str) -> List[Dict]:
        """Fetch OHLCV data for technical analysis."""
        try:
            # This would integrate with your OHLCV data sources
            # Placeholder implementation
            return [
                {
                    "open": 1.0 + i * 0.01,
                    "high": 1.05 + i * 0.01,
                    "low": 0.95 + i * 0.01,
                    "close": 1.02 + i * 0.01,
                    "volume": 1000 * (1 + i * 0.1)
                }
                for i in range(20)
            ]
            
        except Exception as exc:
            logger.debug(f"Failed to fetch OHLCV data: {exc}")
            return []
            
    def _calculate_liquidity_change_rate(self, analysis: PoolAnalysis) -> float:
        """Calculate rate of liquidity change."""
        if analysis.initial_liquidity <= 0:
            return 0.0
        return (analysis.current_liquidity - analysis.initial_liquidity) / analysis.initial_liquidity
        
    def _analyze_liquidity_stability(self, pool_data: Dict) -> float:
        """Analyze liquidity stability patterns."""
        # Simplified implementation
        liquidity = pool_data.get("liquidity", 0)
        if liquidity > 50000:  # High liquidity
            return 0.9
        elif liquidity > 10000:  # Medium liquidity
            return 0.6
        else:
            return 0.3
            
    async def _analyze_developer_activity(self, token_mint: str) -> float:
        """Analyze developer activity and legitimacy."""
        # This would check GitHub, social presence, etc.
        # Placeholder implementation
        return 0.7
        
    async def _analyze_tokenomics(self, token_data: Dict) -> float:
        """Analyze tokenomics health."""
        # This would analyze supply distribution, vesting, etc.
        # Placeholder implementation
        return 0.8
        
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_bb_position(self, prices: List[float], period: int = 20) -> float:
        """Calculate position within Bollinger Bands."""
        if len(prices) < period:
            return 0.5
            
        recent_prices = prices[-period:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        current_price = prices[-1]
        upper_band = mean_price + (2 * std_price)
        lower_band = mean_price - (2 * std_price)
        
        if upper_band == lower_band:
            return 0.5
            
        position = (current_price - lower_band) / (upper_band - lower_band)
        return max(0.0, min(1.0, position))
        
    def _calculate_volume_profile_score(self, volumes: List[float]) -> float:
        """Calculate volume profile score."""
        if len(volumes) < 5:
            return 0.5
            
        recent_vol = np.mean(volumes[-5:])
        historical_vol = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_vol
        
        if historical_vol == 0:
            return 1.0 if recent_vol > 0 else 0.0
            
        return min(recent_vol / historical_vol / 2.0, 1.0)
        
    def get_statistics(self) -> Dict:
        """Get detector statistics."""
        stats = self.stats.copy()
        stats["success_rate"] = (
            stats["successful_predictions"] / max(stats["pumps_detected"], 1)
        )
        stats["precision"] = (
            stats["successful_predictions"] / 
            max(stats["successful_predictions"] + stats["false_positives"], 1)
        )
        return stats
        
    def update_prediction_outcome(self, pool_address: str, successful: bool):
        """Update prediction outcome for learning."""
        if pool_address in self.tracked_pools:
            if successful:
                self.stats["successful_predictions"] += 1
            else:
                self.stats["false_positives"] += 1
