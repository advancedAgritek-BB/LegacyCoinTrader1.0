"""
Momentum and Volume Spike Detection System

This module provides sophisticated momentum analysis and volume spike detection
specifically designed for memecoin trading, with real-time pattern recognition
and predictive capabilities.
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque, Set, Any
from collections import deque
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

try:
    from .watcher import NewPoolEvent
except Exception:  # pragma: no cover - allow import without aiohttp
    NewPoolEvent = object  # type: ignore
from ..utils.telemetry import telemetry

logger = logging.getLogger(__name__)


@dataclass
class VolumeSpike:
    """Represents a detected volume spike."""
    
    timestamp: float
    volume: float
    baseline_volume: float
    spike_factor: float  # volume / baseline
    duration: float  # seconds
    confidence: float  # 0-1
    accompanying_price_move: float  # price change during spike
    spike_type: str  # "accumulation", "breakout", "dump", "irregular"


@dataclass
class MomentumSignal:
    """Represents a momentum signal."""
    
    signal_type: str  # "acceleration", "breakout", "reversal", "continuation"
    strength: float  # 0-1
    direction: str  # "bullish", "bearish", "neutral"
    timeframe: str  # "1m", "5m", "15m", "1h"
    confidence: float  # 0-1
    timestamp: float
    supporting_indicators: List[str] = field(default_factory=list)
    price_targets: List[float] = field(default_factory=list)


@dataclass
class TokenMomentum:
    """Comprehensive momentum analysis for a token."""
    
    token_mint: str
    token_symbol: str
    
    # Price Momentum
    price_momentum_1m: float
    price_momentum_5m: float
    price_momentum_15m: float
    price_momentum_1h: float
    
    # Volume Analysis
    volume_trend: float  # -1 to 1
    volume_acceleration: float
    avg_volume_1h: float
    volume_spike_count: int
    largest_spike_factor: float
    
    # Technical Indicators
    rsi_momentum: float
    macd_momentum: float
    stochastic_momentum: float
    bollinger_squeeze: bool
    
    # Pattern Recognition
    breakout_probability: float
    reversal_probability: float
    continuation_probability: float
    
    # Velocity Metrics
    price_velocity: float  # Rate of price change
    volume_velocity: float  # Rate of volume change
    momentum_divergence: float  # Price vs volume divergence
    
    # Predictive Metrics
    next_move_probability: float  # Probability of significant move
    expected_direction: str  # "up", "down", "sideways"
    time_to_move: float  # Expected seconds to next move
    
    # Quality Indicators
    momentum_quality: float  # How clean/strong the momentum is
    sustainability_score: float  # How likely momentum is to continue
    
    volume_spikes: List[VolumeSpike] = field(default_factory=list)
    momentum_signals: List[MomentumSignal] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class MomentumDetector:
    """
    Advanced momentum and volume spike detection system.
    
    Features:
    - Real-time volume spike detection
    - Multi-timeframe momentum analysis
    - Pattern recognition for breakouts and reversals
    - Predictive momentum modeling
    - Quality scoring for momentum signals
    - Integration with price and volume feeds
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.momentum_config = config.get("momentum_detector", {})
        
        # Detection parameters
        self.volume_spike_threshold = self.momentum_config.get("volume_spike_threshold", 3.0)
        self.momentum_lookback = self.momentum_config.get("momentum_lookback", 100)
        self.min_spike_duration = self.momentum_config.get("min_spike_duration", 30)
        
        # Data storage
        self.price_data: Dict[str, deque] = {}  # token -> price history
        self.volume_data: Dict[str, deque] = {}  # token -> volume history
        self.token_momentum: Dict[str, TokenMomentum] = {}
        
        # Real-time monitoring
        self.monitoring_tokens: Set[str] = set()
        self.update_intervals = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600
        }
        
        # Pattern recognition
        self.patterns = self._initialize_patterns()
        
        # Background tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Statistics
        self.stats = {
            "volume_spikes_detected": 0,
            "momentum_signals_generated": 0,
            "successful_predictions": 0,
            "false_signals": 0,
            "tokens_monitored": 0
        }
        
    async def start(self):
        """Start the momentum detection system."""
        self.running = True
        
        # Start monitoring tasks
        self._start_monitoring_tasks()
        
        logger.info("Momentum detector started")
        
    async def stop(self):
        """Stop the momentum detection system."""
        self.running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        logger.info("Momentum detector stopped")
        
    def add_token_for_monitoring(self, token_mint: str, token_symbol: str):
        """Add a token for momentum monitoring."""
        self.monitoring_tokens.add(token_mint)
        
        # Initialize data structures
        if token_mint not in self.price_data:
            self.price_data[token_mint] = deque(maxlen=1000)
            self.volume_data[token_mint] = deque(maxlen=1000)
            
        # Initialize momentum analysis
        self.token_momentum[token_mint] = TokenMomentum(
            token_mint=token_mint,
            token_symbol=token_symbol,
            price_momentum_1m=0.0,
            price_momentum_5m=0.0,
            price_momentum_15m=0.0,
            price_momentum_1h=0.0,
            volume_trend=0.0,
            volume_acceleration=0.0,
            avg_volume_1h=0.0,
            volume_spike_count=0,
            largest_spike_factor=0.0,
            rsi_momentum=0.0,
            macd_momentum=0.0,
            stochastic_momentum=0.0,
            bollinger_squeeze=False,
            breakout_probability=0.0,
            reversal_probability=0.0,
            continuation_probability=0.0,
            price_velocity=0.0,
            volume_velocity=0.0,
            momentum_divergence=0.0,
            next_move_probability=0.0,
            expected_direction="sideways",
            time_to_move=0.0,
            momentum_quality=0.0,
            sustainability_score=0.0
        )
        
        self.stats["tokens_monitored"] += 1
        logger.info(f"Added {token_symbol} for momentum monitoring")
        
    async def update_token_data(self, token_mint: str, price: float, volume: float, timestamp: float):
        """Update price and volume data for a token."""
        if token_mint not in self.monitoring_tokens:
            return
            
        # Add new data points
        self.price_data[token_mint].append((timestamp, price))
        self.volume_data[token_mint].append((timestamp, volume))
        
        # Check for volume spikes
        await self._check_volume_spike(token_mint)
        
        # Update momentum analysis
        await self._update_momentum_analysis(token_mint)
        
    async def _check_volume_spike(self, token_mint: str):
        """Check for volume spikes in real-time."""
        volume_history = self.volume_data[token_mint]
        
        if len(volume_history) < 20:  # Need minimum history
            return
            
        # Get recent data
        recent_volumes = [vol for _, vol in list(volume_history)[-20:]]
        current_volume = recent_volumes[-1]
        
        # Calculate baseline (exclude current volume)
        baseline_volumes = recent_volumes[:-1]
        baseline_mean = np.mean(baseline_volumes)
        baseline_std = np.std(baseline_volumes)
        
        # Detect spike
        if baseline_mean > 0:
            spike_factor = current_volume / baseline_mean
            z_score = (current_volume - baseline_mean) / max(baseline_std, baseline_mean * 0.1)
            
            if spike_factor >= self.volume_spike_threshold and z_score > 2.0:
                await self._process_volume_spike(token_mint, current_volume, baseline_mean, spike_factor)
                
    async def _process_volume_spike(self, token_mint: str, volume: float, baseline: float, spike_factor: float):
        """Process a detected volume spike."""
        try:
            price_history = self.price_data[token_mint]
            
            # Calculate accompanying price move
            if len(price_history) >= 2:
                current_price = price_history[-1][1]
                previous_price = price_history[-2][1]
                price_move = (current_price - previous_price) / previous_price
            else:
                price_move = 0.0
                
            # Classify spike type
            spike_type = self._classify_volume_spike(spike_factor, price_move)
            
            # Calculate confidence
            confidence = min(1.0, (spike_factor - self.volume_spike_threshold) / self.volume_spike_threshold)
            
            # Create spike record
            spike = VolumeSpike(
                timestamp=time.time(),
                volume=volume,
                baseline_volume=baseline,
                spike_factor=spike_factor,
                duration=0.0,  # Will be updated as spike continues
                confidence=confidence,
                accompanying_price_move=price_move,
                spike_type=spike_type
            )
            
            # Add to momentum analysis
            momentum = self.token_momentum[token_mint]
            momentum.volume_spikes.append(spike)
            momentum.volume_spike_count += 1
            momentum.largest_spike_factor = max(momentum.largest_spike_factor, spike_factor)
            
            # Generate momentum signal if significant
            if spike_factor > 5.0 or abs(price_move) > 0.05:
                await self._generate_momentum_signal(token_mint, spike)
                
            self.stats["volume_spikes_detected"] += 1
            telemetry.inc("momentum_detector.volume_spikes")
            telemetry.gauge("momentum_detector.spike_factor", spike_factor)
            
            logger.info(
                f"Volume spike detected for {momentum.token_symbol}: "
                f"{spike_factor:.1f}x baseline, price move: {price_move:.2%}, "
                f"type: {spike_type}"
            )
            
        except Exception as exc:
            logger.error(f"Volume spike processing failed: {exc}")
            
    async def _update_momentum_analysis(self, token_mint: str):
        """Update comprehensive momentum analysis for a token."""
        try:
            momentum = self.token_momentum[token_mint]
            price_history = self.price_data[token_mint]
            volume_history = self.volume_data[token_mint]
            
            if len(price_history) < 60:  # Need minimum history
                return
                
            # Convert to arrays for analysis
            timestamps = [t for t, _ in price_history]
            prices = [p for _, p in price_history]
            volumes = [v for _, v in volume_history]
            
            # Calculate multi-timeframe momentum
            momentum.price_momentum_1m = self._calculate_momentum(prices, 1)
            momentum.price_momentum_5m = self._calculate_momentum(prices, 5)
            momentum.price_momentum_15m = self._calculate_momentum(prices, 15)
            momentum.price_momentum_1h = self._calculate_momentum(prices, 60)
            
            # Volume analysis
            momentum.volume_trend = self._calculate_volume_trend(volumes)
            momentum.volume_acceleration = self._calculate_volume_acceleration(volumes)
            momentum.avg_volume_1h = np.mean(volumes[-60:]) if len(volumes) >= 60 else np.mean(volumes)
            
            # Technical indicators
            momentum.rsi_momentum = self._calculate_rsi_momentum(prices)
            momentum.macd_momentum = self._calculate_macd_momentum(prices)
            momentum.stochastic_momentum = self._calculate_stochastic_momentum(prices)
            momentum.bollinger_squeeze = self._detect_bollinger_squeeze(prices)
            
            # Velocity calculations
            momentum.price_velocity = self._calculate_price_velocity(prices, timestamps)
            momentum.volume_velocity = self._calculate_volume_velocity(volumes, timestamps)
            momentum.momentum_divergence = self._calculate_momentum_divergence(momentum)
            
            # Pattern recognition
            momentum.breakout_probability = self._calculate_breakout_probability(momentum)
            momentum.reversal_probability = self._calculate_reversal_probability(momentum)
            momentum.continuation_probability = self._calculate_continuation_probability(momentum)
            
            # Predictive analysis
            await self._update_predictive_metrics(momentum)
            
            # Quality scoring
            momentum.momentum_quality = self._calculate_momentum_quality(momentum)
            momentum.sustainability_score = self._calculate_sustainability_score(momentum)
            
            momentum.last_updated = time.time()
            
        except Exception as exc:
            logger.error(f"Momentum analysis update failed: {exc}")
            
    def _calculate_momentum(self, prices: List[float], periods: int) -> float:
        """Calculate price momentum over specified periods."""
        if len(prices) < periods + 1:
            return 0.0
            
        current_price = prices[-1]
        past_price = prices[-(periods + 1)]
        
        return (current_price - past_price) / past_price
        
    def _calculate_volume_trend(self, volumes: List[float]) -> float:
        """Calculate volume trend using linear regression."""
        if len(volumes) < 10:
            return 0.0
            
        x = np.arange(len(volumes))
        slope, _, r_value, _, _ = stats.linregress(x, volumes)
        
        # Normalize slope by average volume
        avg_volume = np.mean(volumes)
        if avg_volume > 0:
            normalized_slope = slope / avg_volume
            return np.tanh(normalized_slope * len(volumes))  # Bound between -1 and 1
        return 0.0
        
    def _calculate_volume_acceleration(self, volumes: List[float]) -> float:
        """Calculate volume acceleration (second derivative)."""
        if len(volumes) < 3:
            return 0.0
            
        # Calculate first derivatives (velocity)
        velocities = np.diff(volumes)
        
        # Calculate second derivatives (acceleration)
        accelerations = np.diff(velocities)
        
        # Return recent acceleration normalized by average volume
        if len(accelerations) > 0:
            recent_acceleration = np.mean(accelerations[-3:])  # Average of last 3
            avg_volume = np.mean(volumes)
            if avg_volume > 0:
                return recent_acceleration / avg_volume
                
        return 0.0
        
    def _calculate_rsi_momentum(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI-based momentum."""
        if len(prices) < period + 1:
            return 0.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 1.0 if avg_gain > 0 else 0.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Convert RSI to momentum score (-1 to 1)
        return (rsi - 50) / 50
        
    def _calculate_macd_momentum(self, prices: List[float]) -> float:
        """Calculate MACD-based momentum."""
        if len(prices) < 26:
            return 0.0
            
        # Simple MACD calculation
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        
        if len(ema_12) >= 2 and len(ema_26) >= 2:
            macd_current = ema_12[-1] - ema_26[-1]
            macd_previous = ema_12[-2] - ema_26[-2]
            
            # MACD momentum is the change in MACD
            avg_price = np.mean(prices[-26:])
            if avg_price > 0:
                return (macd_current - macd_previous) / avg_price
                
        return 0.0
        
    def _calculate_stochastic_momentum(self, prices: List[float], period: int = 14) -> float:
        """Calculate Stochastic-based momentum."""
        if len(prices) < period:
            return 0.0
            
        recent_prices = prices[-period:]
        high = max(recent_prices)
        low = min(recent_prices)
        close = recent_prices[-1]
        
        if high == low:
            return 0.0
            
        k_percent = 100 * (close - low) / (high - low)
        
        # Convert to momentum score (-1 to 1)
        return (k_percent - 50) / 50
        
    def _detect_bollinger_squeeze(self, prices: List[float], period: int = 20) -> bool:
        """Detect Bollinger Band squeeze condition."""
        if len(prices) < period:
            return False
            
        recent_prices = prices[-period:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        # Check if standard deviation is unusually low
        if len(prices) >= period * 2:
            historical_stds = []
            for i in range(period, len(prices)):
                window = prices[i-period:i]
                historical_stds.append(np.std(window))
                
            if historical_stds:
                avg_historical_std = np.mean(historical_stds)
                return std_price < avg_historical_std * 0.7  # Current std is 70% of historical
                
        return False
        
    def _calculate_price_velocity(self, prices: List[float], timestamps: List[float]) -> float:
        """Calculate price velocity (price change per unit time)."""
        if len(prices) < 2 or len(timestamps) < 2:
            return 0.0
            
        time_diff = timestamps[-1] - timestamps[-2]
        price_diff = prices[-1] - prices[-2]
        
        if time_diff > 0:
            velocity = price_diff / time_diff
            # Normalize by current price
            if prices[-1] > 0:
                return velocity / prices[-1]
                
        return 0.0
        
    def _calculate_volume_velocity(self, volumes: List[float], timestamps: List[float]) -> float:
        """Calculate volume velocity (volume change per unit time)."""
        if len(volumes) < 2 or len(timestamps) < 2:
            return 0.0
            
        time_diff = timestamps[-1] - timestamps[-2]
        volume_diff = volumes[-1] - volumes[-2]
        
        if time_diff > 0:
            velocity = volume_diff / time_diff
            # Normalize by current volume
            if volumes[-1] > 0:
                return velocity / volumes[-1]
                
        return 0.0
        
    def _calculate_momentum_divergence(self, momentum: TokenMomentum) -> float:
        """Calculate divergence between price and volume momentum."""
        price_momentum = momentum.price_momentum_5m
        volume_momentum = momentum.volume_trend
        
        # Positive divergence: price up, volume down (bearish)
        # Negative divergence: price down, volume up (bullish)
        divergence = price_momentum - volume_momentum
        
        return np.tanh(divergence)  # Bound between -1 and 1
        
    def _calculate_breakout_probability(self, momentum: TokenMomentum) -> float:
        """Calculate probability of an imminent breakout."""
        factors = [
            momentum.bollinger_squeeze,  # Squeeze suggests breakout coming
            min(1.0, momentum.volume_acceleration / 2.0),  # Volume building
            abs(momentum.price_momentum_1m) > 0.02,  # Recent price movement
            momentum.largest_spike_factor > 3.0,  # Recent volume spike
            momentum.momentum_quality > 0.6  # Good quality momentum
        ]
        
        # Weight the factors
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        weighted_score = sum(f * w for f, w in zip(factors, weights))
        
        return min(1.0, weighted_score)
        
    def _calculate_reversal_probability(self, momentum: TokenMomentum) -> float:
        """Calculate probability of trend reversal."""
        factors = [
            abs(momentum.momentum_divergence) > 0.5,  # Strong divergence
            abs(momentum.price_momentum_1h) > 0.2,  # Extended move
            momentum.rsi_momentum > 0.7 or momentum.rsi_momentum < -0.7,  # Overbought/oversold
            momentum.volume_trend < 0 and momentum.price_momentum_5m > 0  # Volume declining on rally
        ]
        
        weights = [0.4, 0.3, 0.2, 0.1]
        weighted_score = sum(f * w for f, w in zip(factors, weights))
        
        return min(1.0, weighted_score)
        
    def _calculate_continuation_probability(self, momentum: TokenMomentum) -> float:
        """Calculate probability of trend continuation."""
        factors = [
            momentum.volume_trend > 0.3,  # Volume supporting trend
            abs(momentum.momentum_divergence) < 0.3,  # No divergence
            momentum.price_velocity * momentum.volume_velocity > 0,  # Aligned momentum
            momentum.sustainability_score > 0.6  # Sustainable momentum
        ]
        
        weights = [0.3, 0.25, 0.25, 0.2]
        weighted_score = sum(f * w for f, w in zip(factors, weights))
        
        return min(1.0, weighted_score)
        
    async def _update_predictive_metrics(self, momentum: TokenMomentum):
        """Update predictive metrics for next price moves."""
        # Probability of significant move (>5% in next hour)
        move_factors = [
            momentum.breakout_probability,
            momentum.reversal_probability,
            min(1.0, momentum.volume_acceleration / 1.0),
            momentum.bollinger_squeeze
        ]
        momentum.next_move_probability = np.mean(move_factors)
        
        # Expected direction
        if momentum.price_momentum_5m > 0.02:
            momentum.expected_direction = "up"
        elif momentum.price_momentum_5m < -0.02:
            momentum.expected_direction = "down"
        else:
            momentum.expected_direction = "sideways"
            
        # Time to move (rough estimate in seconds)
        if momentum.next_move_probability > 0.7:
            momentum.time_to_move = 300  # 5 minutes
        elif momentum.next_move_probability > 0.5:
            momentum.time_to_move = 900  # 15 minutes
        else:
            momentum.time_to_move = 3600  # 1 hour
            
    def _calculate_momentum_quality(self, momentum: TokenMomentum) -> float:
        """Calculate overall quality of momentum signals."""
        quality_factors = [
            1.0 - abs(momentum.momentum_divergence),  # Low divergence is good
            min(1.0, momentum.volume_trend),  # Positive volume trend
            1.0 - min(1.0, abs(momentum.rsi_momentum)),  # Not extreme RSI
            min(1.0, len(momentum.volume_spikes) / 5.0)  # Some volume activity
        ]
        
        return np.mean(quality_factors)
        
    def _calculate_sustainability_score(self, momentum: TokenMomentum) -> float:
        """Calculate how sustainable the current momentum is."""
        sustainability_factors = [
            momentum.volume_trend > 0.2,  # Volume supporting
            abs(momentum.momentum_divergence) < 0.4,  # No major divergence  
            momentum.momentum_quality > 0.5,  # Good quality
            momentum.price_velocity > 0 if momentum.expected_direction == "up" else momentum.price_velocity < 0  # Velocity aligned
        ]
        
        weights = [0.3, 0.3, 0.2, 0.2]
        weighted_score = sum(f * w for f, w in zip(sustainability_factors, weights))
        
        return min(1.0, weighted_score)
        
    async def _generate_momentum_signal(self, token_mint: str, volume_spike: VolumeSpike):
        """Generate momentum signal based on volume spike and other factors."""
        momentum = self.token_momentum[token_mint]
        
        # Determine signal type
        if volume_spike.spike_factor > 5.0 and momentum.breakout_probability > 0.6:
            signal_type = "breakout"
            strength = min(1.0, volume_spike.spike_factor / 10.0)
        elif abs(momentum.momentum_divergence) > 0.5:
            signal_type = "reversal"
            strength = abs(momentum.momentum_divergence)
        elif momentum.continuation_probability > 0.7:
            signal_type = "continuation"
            strength = momentum.continuation_probability
        else:
            signal_type = "acceleration"
            strength = min(1.0, volume_spike.spike_factor / 5.0)
            
        # Determine direction
        if momentum.price_momentum_1m > 0.01:
            direction = "bullish"
        elif momentum.price_momentum_1m < -0.01:
            direction = "bearish"
        else:
            direction = "neutral"
            
        # Calculate confidence
        confidence = min(1.0, (
            strength + 
            momentum.momentum_quality + 
            (1.0 - abs(momentum.momentum_divergence))
        ) / 3.0)
        
        # Create signal
        signal = MomentumSignal(
            signal_type=signal_type,
            strength=strength,
            direction=direction,
            timeframe="5m",
            confidence=confidence,
            timestamp=time.time(),
            supporting_indicators=[
                f"volume_spike_{volume_spike.spike_factor:.1f}x",
                f"momentum_quality_{momentum.momentum_quality:.2f}",
                f"breakout_prob_{momentum.breakout_probability:.2f}"
            ]
        )
        
        momentum.momentum_signals.append(signal)
        self.stats["momentum_signals_generated"] += 1
        
        telemetry.inc("momentum_detector.signals_generated")
        telemetry.gauge("momentum_detector.signal_strength", strength)
        
        logger.info(
            f"Momentum signal generated for {momentum.token_symbol}: "
            f"{signal_type} {direction} (strength: {strength:.2f}, confidence: {confidence:.2f})"
        )
        
    def _classify_volume_spike(self, spike_factor: float, price_move: float) -> str:
        """Classify the type of volume spike."""
        if abs(price_move) < 0.01:  # < 1% price move
            return "accumulation"  # Volume without price movement
        elif price_move > 0.03:  # > 3% up
            return "breakout"  # Volume with strong upward price movement
        elif price_move < -0.03:  # > 3% down
            return "dump"  # Volume with strong downward price movement
        else:
            return "irregular"  # Moderate price movement with volume
            
    def _ema(self, values: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(values) < period:
            return values
            
        alpha = 2.0 / (period + 1)
        ema_values = [values[0]]  # Start with first value
        
        for i in range(1, len(values)):
            ema_value = alpha * values[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema_value)
            
        return ema_values
        
    def _initialize_patterns(self) -> Dict:
        """Initialize pattern recognition templates."""
        return {
            "cup_and_handle": {
                "volume_pattern": "declining_then_spike",
                "price_pattern": "u_shape_then_breakout",
                "duration": 3600  # 1 hour typical
            },
            "accumulation": {
                "volume_pattern": "increasing_baseline",
                "price_pattern": "sideways_tight_range",
                "duration": 1800  # 30 minutes typical
            },
            "breakout": {
                "volume_pattern": "explosive_spike",
                "price_pattern": "range_break",
                "duration": 300  # 5 minutes typical
            }
        }
        
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        # Periodic momentum updates
        task = asyncio.create_task(self._periodic_momentum_updates())
        self.monitoring_tasks.append(task)
        
    async def _periodic_momentum_updates(self):
        """Periodically update momentum analysis for all monitored tokens."""
        while self.running:
            try:
                for token_mint in list(self.monitoring_tokens):
                    await self._update_momentum_analysis(token_mint)
                    
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as exc:
                logger.error(f"Periodic momentum update error: {exc}")
                await asyncio.sleep(60)
                
    def get_token_momentum(self, token_mint: str) -> Optional[TokenMomentum]:
        """Get momentum analysis for a specific token."""
        return self.token_momentum.get(token_mint)
        
    def get_top_momentum_tokens(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get tokens with highest momentum scores."""
        momentum_scores = []
        
        for token_mint, momentum in self.token_momentum.items():
            # Calculate composite momentum score
            score = (
                abs(momentum.price_momentum_5m) * 0.3 +
                momentum.volume_trend * 0.2 +
                momentum.breakout_probability * 0.2 +
                momentum.momentum_quality * 0.15 +
                momentum.sustainability_score * 0.15
            )
            momentum_scores.append((token_mint, score))
            
        momentum_scores.sort(key=lambda x: x[1], reverse=True)
        return momentum_scores[:limit]
        
    def get_statistics(self) -> Dict:
        """Get detector statistics."""
        stats = self.stats.copy()
        
        if self.stats["momentum_signals_generated"] > 0:
            stats["signal_accuracy"] = (
                self.stats["successful_predictions"] / 
                self.stats["momentum_signals_generated"]
            )
        else:
            stats["signal_accuracy"] = 0.0
            
        return stats


def calculate_momentum_score(pool_data: Dict[str, Any]) -> float:
    """Compute a simple momentum score in [0,1] based on volume and price change.

    Expects keys: 'volume_5m', 'price_history'. Robust to missing/invalid data.
    """
    try:
        volume_5m = float(pool_data.get("volume_5m", 0) or 0)
        prices = pool_data.get("price_history") or []
        if not isinstance(prices, (list, tuple)) or len(prices) < 2:
            return 0.0
        try:
            price_change = (float(prices[-1]) - float(prices[0])) / max(abs(float(prices[0])), 1e-9)
        except Exception:
            return 0.0
        # Normalize heuristically
        vol_component = min(max(volume_5m / 100000.0, 0.0), 1.0)
        price_component = min(max(price_change / 0.2, 0.0), 1.0)
        score = 0.6 * price_component + 0.4 * vol_component
        return float(min(max(score, 0.0), 1.0))
    except Exception:
        return 0.0


def detect_volume_spike(current_volume: float, historical_volumes: List[float]) -> bool:
    """Return True if current_volume is a significant spike over history.

    A spike is detected if there's insufficient history (>=1) and the ratio to
    average is >= 2x.
    """
    try:
        if not historical_volumes:
            return False
        if len(historical_volumes) == 1:
            return True
        avg = float(sum(historical_volumes)) / max(len(historical_volumes), 1)
        if avg <= 0:
            return False
        return float(current_volume) / avg >= 2.0
    except Exception:
        return False


def analyze_price_action(price_history: List[float]) -> Dict[str, Any]:
    """Analyze trend direction and strength from price history.

    Returns dict with keys: 'trend' in {'bullish','bearish','sideways','unknown'} and 'strength' in [0,1].
    """
    try:
        if not price_history or len(price_history) < 2:
            return {"trend": "unknown", "strength": 0.0}
        start = float(price_history[0])
        end = float(price_history[-1])
        if start == 0:
            return {"trend": "unknown", "strength": 0.0}
        change = (end - start) / abs(start)
        # Strength based on absolute change; 20% -> strength 1.0
        strength = min(max(abs(change) / 0.2, 0.0), 1.0)
        if abs(change) < 0.01:
            return {"trend": "sideways", "strength": 0.0}
        return {"trend": "bullish" if change > 0 else "bearish", "strength": strength}
    except Exception:
        return {"trend": "unknown", "strength": 0.0}
