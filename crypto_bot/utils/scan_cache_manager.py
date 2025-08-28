"""
Enhanced Scan Cache Manager for continuous strategy review and trade execution.

This module provides persistent caching of scan results with intelligent
review cycles for strategy fit and trade execution opportunities.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from collections import defaultdict, deque
import threading
import pickle

import pandas as pd
import numpy as np
from cachetools import TTLCache, LRUCache

from .logger import LOG_DIR, setup_logger
from .telemetry import telemetry

logger = setup_logger(__name__, LOG_DIR / "scan_cache.log")

# Cache file paths
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
SCAN_CACHE_FILE = CACHE_DIR / "scan_results_cache.json"
STRATEGY_CACHE_FILE = CACHE_DIR / "strategy_fit_cache.json"
EXECUTION_CACHE_FILE = CACHE_DIR / "execution_opportunities.json"

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ScanResult:
    """Represents a cached scan result with metadata."""
    
    symbol: str
    timestamp: float
    data: Dict[str, Any]  # OHLCV data, indicators, etc.
    score: float
    regime: str
    strategy_fit: Dict[str, float]  # Strategy -> fit score mapping
    market_conditions: Dict[str, Any]  # Volatility, volume, sentiment, etc.
    last_review: float
    review_count: int = 0
    execution_attempts: int = 0
    last_execution_attempt: Optional[float] = None
    status: str = "active"  # active, executed, expired, blocked
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyFit:
    """Represents strategy fit analysis for a symbol."""
    
    symbol: str
    strategy: str
    fit_score: float
    confidence: float
    last_analysis: float
    market_regime: str
    conditions_met: List[str]
    conditions_failed: List[str]
    recommendation: str  # STRONG_BUY, BUY, HOLD, AVOID, STRONG_SELL


@dataclass
class ExecutionOpportunity:
    """Represents a trade execution opportunity."""
    
    symbol: str
    strategy: str
    direction: str  # long, short
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward_ratio: float
    market_conditions: Dict[str, Any]
    timestamp: float
    status: str = "pending"  # pending, executed, expired, cancelled
    execution_metadata: Dict[str, Any] = field(default_factory=dict)


class ScanCacheManager:
    """
    Manages persistent caching of scan results with continuous review.
    
    Features:
    - Persistent storage of scan results
    - Continuous strategy fit analysis
    - Execution opportunity tracking
    - Intelligent cache refresh and cleanup
    - Integration with existing strategy routing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_config = config.get("scan_cache", {})
        
        # Cache settings
        self.max_cache_size = self.cache_config.get("max_cache_size", 1000)
        self.review_interval = self.cache_config.get("review_interval_minutes", 15)
        self.max_age_hours = self.cache_config.get("max_age_hours", 24)
        self.min_score_threshold = self.cache_config.get("min_score_threshold", 0.3)
        
        # In-memory caches
        self.scan_cache: Dict[str, ScanResult] = {}
        self.strategy_fit_cache: Dict[str, StrategyFit] = {}
        self.execution_cache: Dict[str, ExecutionOpportunity] = {}
        
        # TTL caches for performance
        self.performance_cache = TTLCache(maxsize=1000, ttl=300)  # 5 min TTL
        self.regime_cache = TTLCache(maxsize=500, ttl=1800)  # 30 min TTL
        
        # Review queue and locks
        self.review_queue: deque[str] = deque()
        self.cache_lock = threading.RLock()
        self.review_lock = threading.Lock()
        
        # Background tasks
        self.review_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Load existing cache
        self._load_cache()
        
        # Start background tasks
        self.start()
    
    def start(self):
        """Start background review and cleanup tasks."""
        if self.running:
            return
            
        self.running = True
        
        # Start background tasks
        loop = asyncio.get_event_loop()
        if loop.is_running():
            self.review_task = asyncio.create_task(self._review_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        else:
            # For testing or non-async contexts
            threading.Thread(target=self._start_background_tasks, daemon=True).start()
    
    def stop(self):
        """Stop background tasks."""
        self.running = False
        if self.review_task:
            self.review_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
    
    def _start_background_tasks(self):
        """Start background tasks in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._review_loop())
        except asyncio.CancelledError:
            pass
        finally:
            loop.close()
    
    async def _review_loop(self):
        """Continuous loop for reviewing cached scan results."""
        while self.running:
            try:
                await self._review_batch()
                await asyncio.sleep(self.review_interval * 60)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in review loop: {exc}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_loop(self):
        """Continuous loop for cleaning up expired cache entries."""
        while self.running:
            try:
                await self._cleanup_expired()
                await asyncio.sleep(3600)  # Cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in cleanup loop: {exc}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _review_batch(self):
        """Review a batch of cached scan results for strategy fit."""
        with self.review_lock:
            if not self.review_queue:
                return
            
            batch_size = min(10, len(self.review_queue))
            batch = [self.review_queue.popleft() for _ in range(batch_size)]
        
        logger.info(f"Reviewing batch of {len(batch)} scan results")
        
        for symbol in batch:
            try:
                await self._review_symbol(symbol)
            except Exception as exc:
                logger.error(f"Failed to review {symbol}: {exc}")
                # Re-queue for later review
                self.review_queue.append(symbol)
    
    async def _review_symbol(self, symbol: str):
        """Review a single symbol for strategy fit and execution opportunities."""
        scan_result = self.scan_cache.get(symbol)
        if not scan_result:
            return
        
        # Check if review is needed
        now = time.time()
        if now - scan_result.last_review < self.review_interval * 60:
            return
        
        # Update review timestamp
        scan_result.last_review = now
        scan_result.review_count += 1
        
        # Analyze strategy fit
        await self._analyze_strategy_fit(scan_result)
        
        # Check for execution opportunities
        await self._check_execution_opportunities(scan_result)
        
        # Update cache
        self._update_scan_cache(symbol, scan_result)
        
        logger.debug(f"Reviewed {symbol} (review #{scan_result.review_count})")
    
    async def _analyze_strategy_fit(self, scan_result: ScanResult):
        """Analyze strategy fit for a scan result."""
        try:
            # Get current market data
            current_data = scan_result.data
            if not current_data or not isinstance(current_data, dict):
                return
            
            # Analyze with different strategies
            strategies = self._get_available_strategies()
            
            for strategy_name in strategies:
                try:
                    fit_score, confidence, conditions = await self._evaluate_strategy_fit(
                        strategy_name, scan_result
                    )
                    
                    # Create or update strategy fit record
                    fit_key = f"{scan_result.symbol}_{strategy_name}"
                    strategy_fit = StrategyFit(
                        symbol=scan_result.symbol,
                        strategy=strategy_name,
                        fit_score=fit_score,
                        confidence=confidence,
                        last_analysis=time.time(),
                        market_regime=scan_result.regime,
                        conditions_met=conditions.get("met", []),
                        conditions_failed=conditions.get("failed", []),
                        recommendation=self._get_recommendation(fit_score, confidence)
                    )
                    
                    self.strategy_fit_cache[fit_key] = strategy_fit
                    
                    # Update scan result
                    scan_result.strategy_fit[strategy_name] = fit_score
                    
                except Exception as exc:
                    logger.debug(f"Failed to analyze {strategy_name} for {scan_result.symbol}: {exc}")
            
        except Exception as exc:
            logger.error(f"Failed to analyze strategy fit for {scan_result.symbol}: {exc}")
    
    async def _evaluate_strategy_fit(self, strategy_name: str, scan_result: ScanResult) -> Tuple[float, float, Dict[str, List[str]]]:
        """Evaluate how well a strategy fits the current market conditions."""
        try:
            # This would integrate with your existing strategy evaluation system
            # For now, we'll use a simplified scoring approach
            
            base_score = scan_result.score
            regime_match = self._calculate_regime_match(strategy_name, scan_result.regime)
            volatility_score = self._calculate_volatility_score(scan_result.market_conditions)
            volume_score = self._calculate_volume_score(scan_result.market_conditions)
            
            # Combine scores
            fit_score = (
                base_score * 0.4 +
                regime_match * 0.3 +
                volatility_score * 0.2 +
                volume_score * 0.1
            )
            
            # Calculate confidence based on data quality and consistency
            confidence = min(1.0, scan_result.review_count * 0.1 + 0.5)
            
            # Determine conditions met/failed
            conditions = {
                "met": [],
                "failed": []
            }
            
            if base_score >= self.min_score_threshold:
                conditions["met"].append("minimum_score")
            else:
                conditions["failed"].append("minimum_score")
            
            if regime_match >= 0.7:
                conditions["met"].append("regime_match")
            else:
                conditions["failed"].append("regime_match")
            
            if volatility_score >= 0.6:
                conditions["met"].append("volatility_suitable")
            else:
                conditions["failed"].append("volatility_suitable")
            
            return fit_score, confidence, conditions
            
        except Exception as exc:
            logger.error(f"Strategy fit evaluation failed for {strategy_name}: {exc}")
            return 0.0, 0.0, {"met": [], "failed": ["evaluation_error"]}
    
    def _calculate_regime_match(self, strategy_name: str, regime: str) -> float:
        """Calculate how well a strategy matches the current market regime."""
        # Strategy-regime compatibility matrix
        regime_compatibility = {
            "trend_bot": {"trending": 0.9, "ranging": 0.3, "volatile": 0.7},
            "grid_bot": {"trending": 0.2, "ranging": 0.9, "volatile": 0.4},
            "mean_bot": {"trending": 0.3, "ranging": 0.8, "volatile": 0.6},
            "breakout_bot": {"trending": 0.8, "ranging": 0.6, "volatile": 0.9},
            "sniper_bot": {"trending": 0.7, "ranging": 0.5, "volatile": 0.8},
            "dex_scalper": {"trending": 0.6, "ranging": 0.7, "volatile": 0.9},
        }
        
        return regime_compatibility.get(strategy_name, {}).get(regime, 0.5)
    
    def _calculate_volatility_score(self, market_conditions: Dict[str, Any]) -> float:
        """Calculate volatility suitability score."""
        atr = market_conditions.get("atr", 0)
        atr_percent = market_conditions.get("atr_percent", 0)
        
        if atr_percent > 0:
            # Prefer moderate volatility (not too low, not too high)
            if 0.02 <= atr_percent <= 0.08:
                return 0.9
            elif 0.01 <= atr_percent <= 0.12:
                return 0.7
            else:
                return 0.3
        
        return 0.5
    
    def _calculate_volume_score(self, market_conditions: Dict[str, Any]) -> float:
        """Calculate volume suitability score."""
        volume = market_conditions.get("volume", 0)
        volume_ma = market_conditions.get("volume_ma", 1)
        
        if volume > 0 and volume_ma > 0:
            volume_ratio = volume / volume_ma
            if 0.8 <= volume_ratio <= 2.0:
                return 0.9
            elif 0.5 <= volume_ratio <= 3.0:
                return 0.7
            else:
                return 0.4
        
        return 0.5
    
    def _get_recommendation(self, fit_score: float, confidence: float) -> str:
        """Get trading recommendation based on fit score and confidence."""
        if fit_score >= 0.8 and confidence >= 0.7:
            return "STRONG_BUY"
        elif fit_score >= 0.6 and confidence >= 0.6:
            return "BUY"
        elif fit_score >= 0.4 and confidence >= 0.5:
            return "HOLD"
        elif fit_score < 0.3 or confidence < 0.4:
            return "AVOID"
        else:
            return "NEUTRAL"
    
    async def _check_execution_opportunities(self, scan_result: ScanResult):
        """Check for trade execution opportunities."""
        try:
            # Find best strategy fit
            best_strategy = None
            best_score = 0.0
            
            for strategy_name, fit_score in scan_result.strategy_fit.items():
                if fit_score > best_score and fit_score >= 0.7:
                    best_score = fit_score
                    best_strategy = strategy_name
            
            if not best_strategy:
                return
            
            # Check if conditions are right for execution
            if not self._should_execute(scan_result, best_strategy):
                return
            
            # Create execution opportunity
            opportunity = await self._create_execution_opportunity(
                scan_result, best_strategy, best_score
            )
            
            if opportunity:
                self.execution_cache[opportunity.symbol] = opportunity
                logger.info(f"Created execution opportunity for {opportunity.symbol} via {best_strategy}")
            
        except Exception as exc:
            logger.error(f"Failed to check execution opportunities for {scan_result.symbol}: {exc}")
    
    def _should_execute(self, scan_result: ScanResult, strategy: str) -> bool:
        """Determine if conditions are right for trade execution."""
        # Check if already attempted recently
        if scan_result.last_execution_attempt:
            time_since_attempt = time.time() - scan_result.last_execution_attempt
            if time_since_attempt < 3600:  # 1 hour cooldown
                return False
        
        # Check execution count
        if scan_result.execution_attempts >= 3:  # Max 3 attempts
            return False
        
        # Check market conditions
        conditions = scan_result.market_conditions
        if conditions.get("spread_pct", 0) > 0.5:  # Spread too high
            return False
        
        if conditions.get("volume_ratio", 0) < 0.5:  # Volume too low
            return False
        
        return True
    
    async def _create_execution_opportunity(self, scan_result: ScanResult, strategy: str, fit_score: float) -> Optional[ExecutionOpportunity]:
        """Create an execution opportunity for a scan result."""
        try:
            # Get current price and calculate levels
            current_price = scan_result.data.get("close", 0)
            if not current_price:
                return None
            
            # Calculate position sizing and risk levels
            atr = scan_result.market_conditions.get("atr", 0)
            if not atr:
                return None
            
            # Risk management parameters
            risk_per_trade = self.config.get("risk", {}).get("risk_per_trade", 0.02)
            stop_loss_atr = self.config.get("risk", {}).get("stop_loss_atr_mult", 2.0)
            take_profit_atr = self.config.get("risk", {}).get("take_profit_atr_mult", 4.0)
            
            # Calculate levels
            stop_loss = current_price - (atr * stop_loss_atr)
            take_profit = current_price + (atr * take_profit_atr)
            
            # Calculate position size (simplified)
            account_balance = self.config.get("account_balance", 10000)
            risk_amount = account_balance * risk_per_trade
            position_size = risk_amount / (current_price - stop_loss)
            
            # Calculate risk/reward ratio
            risk_reward_ratio = (take_profit - current_price) / (current_price - stop_loss)
            
            # Determine direction (simplified - would integrate with your signal system)
            direction = "long"  # Default to long for now
            
            opportunity = ExecutionOpportunity(
                symbol=scan_result.symbol,
                strategy=strategy,
                direction=direction,
                confidence=fit_score,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_reward_ratio=risk_reward_ratio,
                market_conditions=scan_result.market_conditions,
                timestamp=time.time()
            )
            
            return opportunity
            
        except Exception as exc:
            logger.error(f"Failed to create execution opportunity: {exc}")
            return None
    
    def add_scan_result(self, symbol: str, data: Dict[str, Any], score: float, regime: str, market_conditions: Dict[str, Any]):
        """Add a new scan result to the cache."""
        with self.cache_lock:
            # Check if symbol already exists
            existing = self.scan_cache.get(symbol)
            
            scan_result = ScanResult(
                symbol=symbol,
                timestamp=time.time(),
                data=data,
                score=score,
                regime=regime,
                strategy_fit={},
                market_conditions=market_conditions,
                last_review=time.time(),
                review_count=0,
                execution_attempts=0
            )
            
            self.scan_cache[symbol] = scan_result
            
            # Add to review queue
            if symbol not in self.review_queue:
                self.review_queue.append(symbol)
            
            # Maintain cache size
            if len(self.scan_cache) > self.max_cache_size:
                self._evict_oldest()
            
            logger.debug(f"Added scan result for {symbol} (score: {score:.3f})")
    
    def get_scan_result(self, symbol: str) -> Optional[ScanResult]:
        """Get a cached scan result."""
        return self.scan_cache.get(symbol)
    
    def get_strategy_fit(self, symbol: str, strategy: str) -> Optional[StrategyFit]:
        """Get strategy fit analysis for a symbol."""
        key = f"{symbol}_{strategy}"
        return self.strategy_fit_cache.get(key)
    
    def get_execution_opportunities(self, min_confidence: float = 0.7) -> List[ExecutionOpportunity]:
        """Get all execution opportunities above a confidence threshold."""
        opportunities = []
        for opportunity in self.execution_cache.values():
            if opportunity.confidence >= min_confidence and opportunity.status == "pending":
                opportunities.append(opportunity)
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        return opportunities
    
    def mark_execution_attempted(self, symbol: str):
        """Mark that an execution was attempted for a symbol."""
        scan_result = self.scan_cache.get(symbol)
        if scan_result:
            scan_result.execution_attempts += 1
            scan_result.last_execution_attempt = time.time()
            self._update_scan_cache(symbol, scan_result)
    
    def mark_execution_completed(self, symbol: str, success: bool = True):
        """Mark that an execution was completed for a symbol."""
        opportunity = self.execution_cache.get(symbol)
        if opportunity:
            if success:
                opportunity.status = "executed"
                opportunity.execution_metadata["completed_at"] = time.time()
            else:
                opportunity.status = "expired"
                opportunity.execution_metadata["failed_at"] = time.time()
    
    def _evict_oldest(self):
        """Evict oldest cache entries to maintain size limit."""
        if len(self.scan_cache) <= self.max_cache_size:
            return
        
        # Sort by timestamp and remove oldest
        sorted_items = sorted(
            self.scan_cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        items_to_remove = len(sorted_items) - self.max_cache_size
        for i in range(items_to_remove):
            symbol, _ = sorted_items[i]
            del self.scan_cache[symbol]
            
            # Also remove related caches
            self._remove_related_caches(symbol)
    
    def _remove_related_caches(self, symbol: str):
        """Remove related cache entries for a symbol."""
        # Remove strategy fit entries
        keys_to_remove = [k for k in self.strategy_fit_cache.keys() if k.startswith(f"{symbol}_")]
        for key in keys_to_remove:
            del self.strategy_fit_cache[key]
        
        # Remove execution opportunity
        if symbol in self.execution_cache:
            del self.execution_cache[symbol]
    
    async def _cleanup_expired(self):
        """Clean up expired cache entries."""
        now = time.time()
        max_age_seconds = self.max_age_hours * 3600
        
        expired_symbols = []
        
        for symbol, scan_result in self.scan_cache.items():
            if now - scan_result.timestamp > max_age_seconds:
                expired_symbols.append(symbol)
        
        if expired_symbols:
            logger.info(f"Cleaning up {len(expired_symbols)} expired scan results")
            
            for symbol in expired_symbols:
                with self.cache_lock:
                    del self.scan_cache[symbol]
                    self._remove_related_caches(symbol)
    
    def _get_available_strategies(self) -> List[str]:
        """Get list of available trading strategies."""
        return [
            "trend_bot",
            "grid_bot", 
            "mean_bot",
            "breakout_bot",
            "sniper_bot",
            "dex_scalper",
            "sniper_solana",
            "meme_wave_bot",
            "momentum_bot",
            "cross_chain_arb_bot"
        ]
    
    def _update_scan_cache(self, symbol: str, scan_result: ScanResult):
        """Update scan cache and persist to disk."""
        self.scan_cache[symbol] = scan_result
        self._persist_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            if SCAN_CACHE_FILE.exists():
                with open(SCAN_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    for symbol, item in data.items():
                        scan_result = ScanResult(**item)
                        self.scan_cache[symbol] = scan_result
                        # Add to review queue
                        self.review_queue.append(symbol)
                
                logger.info(f"Loaded {len(self.scan_cache)} scan results from cache")
        except Exception as exc:
            logger.warning(f"Failed to load scan cache: {exc}")
    
    def _persist_cache(self):
        """Persist cache to disk."""
        try:
            # Convert to serializable format
            cache_data = {}
            for symbol, scan_result in self.scan_cache.items():
                cache_data[symbol] = asdict(scan_result)
            
            with open(SCAN_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
                
        except Exception as exc:
            logger.error(f"Failed to persist scan cache: {exc}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            return {
                "scan_results": len(self.scan_cache),
                "strategy_fits": len(self.strategy_fit_cache),
                "execution_opportunities": len(self.execution_cache),
                "review_queue_size": len(self.review_queue),
                "cache_size_limit": self.max_cache_size,
                "oldest_entry": min([sr.timestamp for sr in self.scan_cache.values()]) if self.scan_cache else 0,
                "newest_entry": max([sr.timestamp for sr in self.scan_cache.values()]) if self.scan_cache else 0
            }
    
    def clear_cache(self):
        """Clear all caches."""
        with self.cache_lock:
            self.scan_cache.clear()
            self.strategy_fit_cache.clear()
            self.execution_cache.clear()
            self.review_queue.clear()
            
            # Remove cache files
            for cache_file in [SCAN_CACHE_FILE, STRATEGY_CACHE_FILE, EXECUTION_CACHE_FILE]:
                if cache_file.exists():
                    cache_file.unlink()
            
            logger.info("All caches cleared")


# Global instance
_scan_cache_manager: Optional[ScanCacheManager] = None


def get_scan_cache_manager(config: Dict[str, Any]) -> ScanCacheManager:
    """Get or create the global scan cache manager instance."""
    global _scan_cache_manager
    
    if _scan_cache_manager is None:
        _scan_cache_manager = ScanCacheManager(config)
    
    return _scan_cache_manager


def get_scan_cache_stats() -> Dict[str, Any]:
    """Get cache statistics if manager exists."""
    if _scan_cache_manager:
        return _scan_cache_manager.get_cache_stats()
    return {"error": "Scan cache manager not initialized"}
