"""
Real-Time Liquidity Pool Quality Analyzer

This module provides comprehensive analysis of Solana liquidity pools,
evaluating their quality, stability, and trading viability for memecoin sniping.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import deque
import numpy as np
import aiohttp
import json

from .watcher import NewPoolEvent
from ..utils.telemetry import telemetry

logger = logging.getLogger(__name__)


@dataclass
class PoolMetrics:
    """Comprehensive pool quality metrics."""
    
    # Basic Information
    pool_address: str
    token_mint: str
    creation_time: float
    
    # Liquidity Analysis
    total_liquidity_usd: float
    token_liquidity: float
    sol_liquidity: float
    liquidity_ratio: float  # token/SOL ratio
    liquidity_concentration: float  # How concentrated liquidity is
    
    # Trading Metrics
    volume_24h: float
    volume_1h: float
    trade_count_24h: int
    trade_count_1h: int
    avg_trade_size: float
    max_trade_size: float
    
    # Price Metrics
    current_price: float
    price_change_1h: float
    price_change_24h: float
    price_volatility: float
    price_impact_1pct: float  # Price impact of 1% trade
    
    # Spread and Slippage
    bid_ask_spread: float
    slippage_0_1pct: float  # Slippage for 0.1% trade
    slippage_1pct: float    # Slippage for 1% trade
    slippage_5pct: float    # Slippage for 5% trade
    
    # Market Depth
    depth_2pct_bid: float   # Liquidity within 2% of mid price
    depth_2pct_ask: float
    depth_5pct_bid: float   # Liquidity within 5% of mid price
    depth_5pct_ask: float
    depth_imbalance: float  # Bid/ask depth imbalance
    
    # Trader Analysis
    unique_traders_24h: int
    whale_dominance: float  # Percentage of volume from large traders
    new_trader_ratio: float # Ratio of new vs returning traders
    trader_retention: float # How many traders come back
    
    # Technical Health
    pool_health_score: float  # Overall pool health (0-1)
    liquidity_quality: float # Quality of liquidity provision
    market_efficiency: float # How efficient price discovery is
    stability_score: float   # Price and liquidity stability
    
    # Risk Indicators
    manipulation_risk: float # Risk of price manipulation
    rug_pull_indicators: List[str] = field(default_factory=list)
    suspicious_activity: float  # Suspicious trading patterns
    
    # Composite Scores
    sniping_viability: float = 0.0  # How good for sniping (0-1)
    entry_difficulty: float = 0.0   # Difficulty of entry (0-1)
    exit_liquidity: float = 0.0     # Ease of exit (0-1)
    
    last_updated: float = field(default_factory=time.time)


class LiquidityPoolAnalyzer:
    """
    Real-time analyzer for Solana liquidity pool quality and trading viability.
    
    Features:
    - Comprehensive liquidity analysis
    - Real-time trading pattern detection
    - Market microstructure analysis
    - Risk assessment and red flag detection
    - Sniping viability scoring
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.analyzer_config = config.get("pool_analyzer", {})
        
        # Analysis parameters
        self.min_liquidity_usd = self.analyzer_config.get("min_liquidity_usd", 10000)
        self.max_slippage_threshold = self.analyzer_config.get("max_slippage_1pct", 0.05)
        self.min_health_score = self.analyzer_config.get("min_health_score", 0.6)
        
        # Quality scoring weights
        self.quality_weights = self.analyzer_config.get("quality_weights", {
            "liquidity_depth": 0.25,
            "trading_activity": 0.20,
            "price_stability": 0.15,
            "market_efficiency": 0.15,
            "trader_diversity": 0.15,
            "risk_factors": -0.10
        })
        
        # Pool tracking
        self.analyzed_pools: Dict[str, PoolMetrics] = {}
        self.pool_history: Dict[str, deque] = {}
        
        # API sessions
        self.session: Optional[aiohttp.ClientSession] = None
        self.helius_key = self.config.get("helius_key", "")
        
        # Caching
        self.price_cache: Dict[str, Tuple[float, float]] = {}  # token -> (price, timestamp)
        self.orderbook_cache: Dict[str, Tuple[Dict, float]] = {}
        
        # Statistics
        self.stats = {
            "pools_analyzed": 0,
            "high_quality_pools": 0,
            "analysis_failures": 0,
            "avg_analysis_time": 0.0
        }
        
    async def start(self):
        """Initialize the pool analyzer."""
        self.session = aiohttp.ClientSession()
        logger.info("Liquidity pool analyzer started")
        
    async def stop(self):
        """Clean shutdown of the analyzer."""
        if self.session:
            await self.session.close()
        logger.info("Liquidity pool analyzer stopped")
        
    async def analyze_pool_quality(self, event: NewPoolEvent) -> Optional[PoolMetrics]:
        """
        Perform comprehensive analysis of pool quality.
        
        Args:
            event: New pool event to analyze
            
        Returns:
            PoolMetrics if pool is viable, None otherwise
        """
        start_time = time.time()
        
        try:
            # Initialize metrics
            metrics = PoolMetrics(
                pool_address=event.pool_address,
                token_mint=event.token_mint,
                creation_time=event.timestamp or time.time(),
                total_liquidity_usd=event.liquidity
            )
            
            # Perform comprehensive analysis
            await self._analyze_liquidity_structure(metrics)
            await self._analyze_trading_activity(metrics)
            await self._analyze_price_metrics(metrics)
            await self._analyze_market_depth(metrics)
            await self._analyze_trader_patterns(metrics)
            await self._calculate_technical_health(metrics)
            await self._assess_risk_indicators(metrics)
            
            # Calculate composite scores
            self._calculate_quality_scores(metrics)
            
            # Check if pool meets quality thresholds
            if self._meets_quality_criteria(metrics):
                self.analyzed_pools[event.pool_address] = metrics
                self.stats["high_quality_pools"] += 1
                
                # Track in history
                if event.pool_address not in self.pool_history:
                    self.pool_history[event.pool_address] = deque(maxlen=100)
                self.pool_history[event.pool_address].append(metrics)
                
                telemetry.inc("pool_analyzer.high_quality_detected")
                telemetry.gauge("pool_analyzer.sniping_viability", metrics.sniping_viability)
                
                logger.info(
                    f"High-quality pool detected: {event.token_mint[:8]}... "
                    f"Liquidity: ${metrics.total_liquidity_usd:,.0f}, "
                    f"Health: {metrics.pool_health_score:.2f}, "
                    f"Viability: {metrics.sniping_viability:.2f}"
                )
                
                return metrics
                
            self.stats["pools_analyzed"] += 1
            return None
            
        except Exception as exc:
            self.stats["analysis_failures"] += 1
            logger.error(f"Pool analysis failed for {event.pool_address}: {exc}")
            return None
            
        finally:
            analysis_time = time.time() - start_time
            self.stats["avg_analysis_time"] = (
                (self.stats["avg_analysis_time"] * self.stats["pools_analyzed"] + analysis_time) /
                (self.stats["pools_analyzed"] + 1)
            )
            
    async def _analyze_liquidity_structure(self, metrics: PoolMetrics):
        """Analyze the structure and quality of liquidity."""
        try:
            # Get detailed pool data
            pool_data = await self._fetch_detailed_pool_data(metrics.pool_address)
            if not pool_data:
                return
                
            # Extract liquidity components
            reserves = pool_data.get("reserves", {})
            metrics.token_liquidity = float(reserves.get("token", 0))
            metrics.sol_liquidity = float(reserves.get("sol", 0))
            
            # Calculate ratios and concentration
            if metrics.sol_liquidity > 0:
                metrics.liquidity_ratio = metrics.token_liquidity / metrics.sol_liquidity
                
            # Liquidity concentration analysis
            lp_positions = pool_data.get("lp_positions", [])
            if lp_positions:
                total_lp = sum(pos.get("amount", 0) for pos in lp_positions)
                if total_lp > 0:
                    # Calculate Gini coefficient for liquidity concentration
                    amounts = sorted([pos.get("amount", 0) for pos in lp_positions])
                    metrics.liquidity_concentration = self._calculate_gini_coefficient(amounts)
                    
        except Exception as exc:
            logger.debug(f"Liquidity structure analysis failed: {exc}")
            
    async def _analyze_trading_activity(self, metrics: PoolMetrics):
        """Analyze trading activity and patterns."""
        try:
            # Get trading data
            trading_data = await self._fetch_trading_data(metrics.pool_address)
            if not trading_data:
                return
                
            trades_24h = trading_data.get("trades_24h", [])
            trades_1h = trading_data.get("trades_1h", [])
            
            # Calculate trading metrics
            metrics.trade_count_24h = len(trades_24h)
            metrics.trade_count_1h = len(trades_1h)
            
            if trades_24h:
                volumes_24h = [trade.get("volume_usd", 0) for trade in trades_24h]
                metrics.volume_24h = sum(volumes_24h)
                metrics.avg_trade_size = np.mean(volumes_24h)
                metrics.max_trade_size = max(volumes_24h)
                
            if trades_1h:
                volumes_1h = [trade.get("volume_usd", 0) for trade in trades_1h]
                metrics.volume_1h = sum(volumes_1h)
                
        except Exception as exc:
            logger.debug(f"Trading activity analysis failed: {exc}")
            
    async def _analyze_price_metrics(self, metrics: PoolMetrics):
        """Analyze price behavior and volatility."""
        try:
            # Get price data
            price_data = await self._fetch_price_history(metrics.token_mint)
            if not price_data:
                return
                
            prices = [p["price"] for p in price_data]
            timestamps = [p["timestamp"] for p in price_data]
            
            if len(prices) >= 2:
                metrics.current_price = prices[-1]
                
                # Calculate price changes
                hour_ago = time.time() - 3600
                day_ago = time.time() - 86400
                
                price_1h_ago = self._get_price_at_time(price_data, hour_ago)
                price_24h_ago = self._get_price_at_time(price_data, day_ago)
                
                if price_1h_ago:
                    metrics.price_change_1h = (metrics.current_price - price_1h_ago) / price_1h_ago
                if price_24h_ago:
                    metrics.price_change_24h = (metrics.current_price - price_24h_ago) / price_24h_ago
                    
                # Calculate volatility
                if len(prices) >= 20:
                    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                    metrics.price_volatility = np.std(returns) * np.sqrt(24)  # 24-hour volatility
                    
        except Exception as exc:
            logger.debug(f"Price metrics analysis failed: {exc}")
            
    async def _analyze_market_depth(self, metrics: PoolMetrics):
        """Analyze market depth and liquidity distribution."""
        try:
            # Get orderbook data
            orderbook = await self._fetch_orderbook(metrics.pool_address)
            if not orderbook:
                return
                
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            if not bids or not asks:
                return
                
            mid_price = (bids[0][0] + asks[0][0]) / 2
            metrics.bid_ask_spread = (asks[0][0] - bids[0][0]) / mid_price
            
            # Calculate depth at different levels
            metrics.depth_2pct_bid = self._calculate_depth(bids, mid_price, -0.02)
            metrics.depth_2pct_ask = self._calculate_depth(asks, mid_price, 0.02)
            metrics.depth_5pct_bid = self._calculate_depth(bids, mid_price, -0.05)
            metrics.depth_5pct_ask = self._calculate_depth(asks, mid_price, 0.05)
            
            # Calculate depth imbalance
            total_bid_depth = metrics.depth_2pct_bid
            total_ask_depth = metrics.depth_2pct_ask
            if total_bid_depth + total_ask_depth > 0:
                metrics.depth_imbalance = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth)
                
            # Calculate slippage for different trade sizes
            metrics.slippage_0_1pct = self._calculate_slippage(asks, mid_price, 0.001)
            metrics.slippage_1pct = self._calculate_slippage(asks, mid_price, 0.01)
            metrics.slippage_5pct = self._calculate_slippage(asks, mid_price, 0.05)
            
            # Price impact estimation
            metrics.price_impact_1pct = self._estimate_price_impact(orderbook, 0.01)
            
        except Exception as exc:
            logger.debug(f"Market depth analysis failed: {exc}")
            
    async def _analyze_trader_patterns(self, metrics: PoolMetrics):
        """Analyze trader behavior and patterns."""
        try:
            # Get trader data
            trader_data = await self._fetch_trader_data(metrics.pool_address)
            if not trader_data:
                return
                
            traders_24h = trader_data.get("unique_traders_24h", [])
            metrics.unique_traders_24h = len(traders_24h)
            
            # Analyze trader sizes
            if traders_24h:
                volumes = [trader.get("volume", 0) for trader in traders_24h]
                total_volume = sum(volumes)
                
                # Whale dominance (top 10% of traders by volume)
                sorted_volumes = sorted(volumes, reverse=True)
                top_10pct_count = max(1, len(sorted_volumes) // 10)
                whale_volume = sum(sorted_volumes[:top_10pct_count])
                metrics.whale_dominance = whale_volume / total_volume if total_volume > 0 else 0
                
                # New vs returning traders
                new_traders = [t for t in traders_24h if t.get("is_new", True)]
                metrics.new_trader_ratio = len(new_traders) / len(traders_24h)
                
                # Trader retention (simplified)
                returning_traders = [t for t in traders_24h if not t.get("is_new", True)]
                metrics.trader_retention = len(returning_traders) / len(traders_24h)
                
        except Exception as exc:
            logger.debug(f"Trader pattern analysis failed: {exc}")
            
    async def _calculate_technical_health(self, metrics: PoolMetrics):
        """Calculate technical health indicators."""
        try:
            # Pool health score based on multiple factors
            health_factors = []
            
            # Liquidity adequacy
            if metrics.total_liquidity_usd >= 50000:
                health_factors.append(1.0)
            elif metrics.total_liquidity_usd >= 10000:
                health_factors.append(0.7)
            else:
                health_factors.append(0.3)
                
            # Trading activity
            if metrics.trade_count_24h >= 100:
                health_factors.append(1.0)
            elif metrics.trade_count_24h >= 20:
                health_factors.append(0.6)
            else:
                health_factors.append(0.2)
                
            # Spread quality
            if metrics.bid_ask_spread <= 0.01:  # 1%
                health_factors.append(1.0)
            elif metrics.bid_ask_spread <= 0.02:  # 2%
                health_factors.append(0.7)
            else:
                health_factors.append(0.3)
                
            # Trader diversity
            if metrics.unique_traders_24h >= 50:
                health_factors.append(1.0)
            elif metrics.unique_traders_24h >= 10:
                health_factors.append(0.6)
            else:
                health_factors.append(0.2)
                
            metrics.pool_health_score = np.mean(health_factors)
            
            # Liquidity quality
            quality_factors = [
                1.0 - min(metrics.liquidity_concentration, 1.0),  # Lower concentration = better
                min(metrics.depth_2pct_bid / 1000, 1.0),  # Normalized depth
                min(metrics.depth_2pct_ask / 1000, 1.0),
                1.0 - min(abs(metrics.depth_imbalance), 1.0)  # Lower imbalance = better
            ]
            metrics.liquidity_quality = np.mean(quality_factors)
            
            # Market efficiency
            efficiency_factors = [
                1.0 - min(metrics.bid_ask_spread / 0.05, 1.0),  # Normalized spread
                1.0 - min(metrics.slippage_1pct / 0.05, 1.0),   # Normalized slippage
                min(metrics.volume_24h / 100000, 1.0)          # Normalized volume
            ]
            metrics.market_efficiency = np.mean(efficiency_factors)
            
            # Stability score
            stability_factors = [
                1.0 - min(abs(metrics.price_change_1h) / 0.1, 1.0),   # Price stability
                1.0 - min(metrics.price_volatility / 0.5, 1.0),       # Volatility stability
                1.0 - min(metrics.whale_dominance, 1.0)               # Trading stability
            ]
            metrics.stability_score = np.mean(stability_factors)
            
        except Exception as exc:
            logger.debug(f"Technical health calculation failed: {exc}")
            
    async def _assess_risk_indicators(self, metrics: PoolMetrics):
        """Assess various risk indicators."""
        try:
            risk_flags = []
            
            # High concentration risk
            if metrics.liquidity_concentration > 0.8:
                risk_flags.append("high_liquidity_concentration")
                
            # Whale dominance risk
            if metrics.whale_dominance > 0.7:
                risk_flags.append("whale_dominated_trading")
                
            # Low trader diversity
            if metrics.unique_traders_24h < 5:
                risk_flags.append("low_trader_diversity")
                
            # High slippage risk
            if metrics.slippage_1pct > 0.1:  # >10% slippage for 1% trade
                risk_flags.append("high_slippage")
                
            # Extreme price volatility
            if metrics.price_volatility > 1.0:  # >100% daily volatility
                risk_flags.append("extreme_volatility")
                
            # Low trading activity
            if metrics.trade_count_24h < 5:
                risk_flags.append("low_trading_activity")
                
            metrics.rug_pull_indicators = risk_flags
            
            # Calculate manipulation risk
            manipulation_factors = [
                metrics.liquidity_concentration,
                metrics.whale_dominance,
                1.0 - min(metrics.unique_traders_24h / 50, 1.0),
                min(abs(metrics.depth_imbalance), 1.0)
            ]
            metrics.manipulation_risk = np.mean(manipulation_factors)
            
            # Suspicious activity score
            suspicion_factors = [
                len(risk_flags) / 7.0,  # Normalized flag count
                metrics.manipulation_risk,
                1.0 - metrics.trader_retention
            ]
            metrics.suspicious_activity = np.mean(suspicion_factors)
            
        except Exception as exc:
            logger.debug(f"Risk assessment failed: {exc}")
            
    def _calculate_quality_scores(self, metrics: PoolMetrics):
        """Calculate final quality and viability scores."""
        try:
            # Sniping viability score
            viability_factors = {
                "liquidity_depth": metrics.liquidity_quality,
                "trading_activity": min(metrics.volume_24h / 100000, 1.0),
                "price_stability": metrics.stability_score,
                "market_efficiency": metrics.market_efficiency,
                "trader_diversity": min(metrics.unique_traders_24h / 50, 1.0),
                "risk_factors": metrics.manipulation_risk
            }
            
            weighted_viability = 0.0
            for factor, weight in self.quality_weights.items():
                weighted_viability += viability_factors.get(factor, 0) * weight
                
            metrics.sniping_viability = max(0.0, min(1.0, weighted_viability))
            
            # Entry difficulty (lower is better)
            entry_factors = [
                metrics.slippage_1pct / 0.1,  # Normalized slippage
                metrics.bid_ask_spread / 0.05,  # Normalized spread
                metrics.whale_dominance,  # Whale competition
                1.0 - metrics.liquidity_quality  # Liquidity issues
            ]
            metrics.entry_difficulty = min(np.mean(entry_factors), 1.0)
            
            # Exit liquidity (ease of selling)
            exit_factors = [
                metrics.liquidity_quality,
                min(metrics.depth_2pct_bid / 5000, 1.0),  # Bid depth
                metrics.market_efficiency,
                1.0 - metrics.manipulation_risk
            ]
            metrics.exit_liquidity = np.mean(exit_factors)
            
        except Exception as exc:
            logger.error(f"Quality score calculation failed: {exc}")
            
    def _meets_quality_criteria(self, metrics: PoolMetrics) -> bool:
        """Check if pool meets minimum quality criteria."""
        criteria = [
            metrics.total_liquidity_usd >= self.min_liquidity_usd,
            metrics.slippage_1pct <= self.max_slippage_threshold,
            metrics.pool_health_score >= self.min_health_score,
            metrics.sniping_viability >= 0.5,
            len(metrics.rug_pull_indicators) <= 2,
            metrics.manipulation_risk <= 0.7
        ]
        
        return all(criteria)
        
    # Data fetching helper methods
    
    async def _fetch_detailed_pool_data(self, pool_address: str) -> Optional[Dict]:
        """Fetch detailed pool information."""
        try:
            # This would integrate with Raydium/Jupiter APIs
            # Placeholder implementation
            return {
                "reserves": {"token": 1000000, "sol": 100},
                "lp_positions": [
                    {"amount": 500, "owner": "addr1"},
                    {"amount": 300, "owner": "addr2"},
                    {"amount": 200, "owner": "addr3"}
                ]
            }
        except Exception as exc:
            logger.debug(f"Failed to fetch pool data: {exc}")
            return None
            
    async def _fetch_trading_data(self, pool_address: str) -> Optional[Dict]:
        """Fetch trading activity data."""
        try:
            # This would integrate with transaction APIs
            # Placeholder implementation
            current_time = time.time()
            return {
                "trades_24h": [
                    {"volume_usd": 1000, "timestamp": current_time - i * 3600}
                    for i in range(24)
                ],
                "trades_1h": [
                    {"volume_usd": 500, "timestamp": current_time - i * 300}
                    for i in range(12)
                ]
            }
        except Exception as exc:
            logger.debug(f"Failed to fetch trading data: {exc}")
            return None
            
    async def _fetch_price_history(self, token_mint: str) -> List[Dict]:
        """Fetch price history."""
        try:
            # This would integrate with price APIs
            # Placeholder implementation
            current_time = time.time()
            return [
                {"price": 1.0 + i * 0.01, "timestamp": current_time - i * 300}
                for i in range(288)  # 24 hours of 5-minute data
            ]
        except Exception as exc:
            logger.debug(f"Failed to fetch price history: {exc}")
            return []
            
    async def _fetch_orderbook(self, pool_address: str) -> Optional[Dict]:
        """Fetch orderbook data."""
        try:
            # This would integrate with DEX APIs
            # Placeholder implementation
            return {
                "bids": [[0.99, 1000], [0.98, 2000], [0.97, 3000]],
                "asks": [[1.01, 1000], [1.02, 2000], [1.03, 3000]]
            }
        except Exception as exc:
            logger.debug(f"Failed to fetch orderbook: {exc}")
            return None
            
    async def _fetch_trader_data(self, pool_address: str) -> Optional[Dict]:
        """Fetch trader analytics."""
        try:
            # This would integrate with analytics APIs
            # Placeholder implementation
            return {
                "unique_traders_24h": [
                    {"volume": 10000, "is_new": False},
                    {"volume": 5000, "is_new": True},
                    {"volume": 2000, "is_new": True}
                ]
            }
        except Exception as exc:
            logger.debug(f"Failed to fetch trader data: {exc}")
            return None
            
    # Helper calculation methods
    
    def _calculate_gini_coefficient(self, amounts: List[float]) -> float:
        """Calculate Gini coefficient for concentration measurement."""
        if not amounts or all(a == 0 for a in amounts):
            return 0.0
            
        amounts = sorted([a for a in amounts if a > 0])
        n = len(amounts)
        if n <= 1:
            return 0.0
            
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * amounts)) / (n * np.sum(amounts)) - (n + 1) / n
        
    def _get_price_at_time(self, price_data: List[Dict], target_time: float) -> Optional[float]:
        """Get price closest to target time."""
        if not price_data:
            return None
            
        closest = min(price_data, key=lambda x: abs(x["timestamp"] - target_time))
        return closest["price"]
        
    def _calculate_depth(self, orders: List[List[float]], mid_price: float, threshold: float) -> float:
        """Calculate liquidity depth within threshold."""
        target_price = mid_price * (1 + threshold)
        depth = 0.0
        
        for price, size in orders:
            if threshold > 0:  # Ask side
                if price <= target_price:
                    depth += size
                else:
                    break
            else:  # Bid side
                if price >= target_price:
                    depth += size
                else:
                    break
                    
        return depth
        
    def _calculate_slippage(self, asks: List[List[float]], mid_price: float, trade_size_pct: float) -> float:
        """Calculate slippage for a given trade size."""
        if not asks:
            return 1.0  # 100% slippage if no liquidity
            
        trade_value = mid_price * trade_size_pct
        executed_value = 0.0
        executed_volume = 0.0
        
        for price, size in asks:
            if executed_value >= trade_value:
                break
                
            take_size = min(size, (trade_value - executed_value) / price)
            executed_volume += take_size
            executed_value += take_size * price
            
        if executed_volume == 0:
            return 1.0
            
        avg_execution_price = executed_value / executed_volume
        return (avg_execution_price - mid_price) / mid_price
        
    def _estimate_price_impact(self, orderbook: Dict, trade_size_pct: float) -> float:
        """Estimate price impact of a trade."""
        asks = orderbook.get("asks", [])
        if not asks:
            return 1.0
            
        return self._calculate_slippage(asks, asks[0][0], trade_size_pct)
        
    def get_pool_metrics(self, pool_address: str) -> Optional[PoolMetrics]:
        """Get cached metrics for a pool."""
        return self.analyzed_pools.get(pool_address)
        
    def get_statistics(self) -> Dict:
        """Get analyzer statistics."""
        total_analyzed = self.stats["pools_analyzed"] + self.stats["high_quality_pools"]
        stats = self.stats.copy()
        stats["quality_rate"] = (
            self.stats["high_quality_pools"] / max(total_analyzed, 1)
        )
        stats["success_rate"] = (
            1.0 - self.stats["analysis_failures"] / max(total_analyzed, 1)
        )
        return stats
