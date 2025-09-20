"""Adaptive cache management for trading bot data."""

from collections import defaultdict, deque
from typing import Dict, Any, Optional, List, Union
import time
import asyncio
from dataclasses import dataclass, field
import logging

from .logger import setup_logger


@dataclass
class ScanResult:
    """Represents a scan result with metadata."""
    symbol: str
    data: Any
    timestamp: float
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    data: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0


class AdaptiveCacheManager:
    """Intelligent cache management with adaptive sizing based on usage patterns."""
    
    def __init__(
        self,
        initial_size: int = 1000,
        max_size: int = 10000,
        min_size: int = 100,
        hit_rate_window: int = 100,
        eviction_policy: str = "lru",
        enable_compression: bool = False
    ):
        """
        Initialize the adaptive cache manager.
        
        Args:
            initial_size: Initial cache size for each cache type
            max_size: Maximum cache size allowed
            min_size: Minimum cache size allowed
            hit_rate_window: Number of accesses to track for hit rate calculation
            eviction_policy: Cache eviction policy ('lru', 'lfu', 'adaptive')
            enable_compression: Whether to enable data compression
        """
        self.initial_size = initial_size
        self.max_size = max_size
        self.min_size = min_size
        self.hit_rate_window = hit_rate_window
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression
        
        # Cache storage
        self.caches: Dict[str, Dict[str, CacheEntry]] = defaultdict(dict)
        
        # Hit rate tracking
        self.hit_rates = defaultdict(lambda: deque(maxlen=hit_rate_window))
        self.access_patterns = defaultdict(int)
        self.total_accesses = defaultdict(int)
        self.total_hits = defaultdict(int)
        
        # Performance metrics
        self.cache_sizes = defaultdict(lambda: initial_size)
        self.eviction_counts = defaultdict(int)
        self.compression_ratios = defaultdict(float)
        
        # Statistics
        self.stats = {
            "total_memory_usage": 0,
            "total_entries": 0,
            "compression_enabled": enable_compression
        }
        
        self.logger = setup_logger("adaptive_cache_manager", "logs/adaptive_cache_manager.log")
        
    def get_cache_size(self, cache_type: str) -> int:
        """
        Get adaptive cache size based on hit rates and usage patterns.
        
        Args:
            cache_type: Type of cache (e.g., 'ohlcv', 'orderbook', 'regime')
            
        Returns:
            Recommended cache size
        """
        hit_rate = self._calculate_hit_rate(cache_type)
        access_frequency = self._calculate_access_frequency(cache_type)
        
        # Base size calculation
        base_size = self.cache_sizes[cache_type]
        
        # Adjust based on hit rate
        if hit_rate > 0.8:  # High hit rate - increase size
            size_multiplier = 1.5
            self.logger.debug(f"High hit rate ({hit_rate:.2f}) for {cache_type}, increasing cache size")
        elif hit_rate < 0.3:  # Low hit rate - decrease size
            size_multiplier = 0.7
            self.logger.debug(f"Low hit rate ({hit_rate:.2f}) for {cache_type}, decreasing cache size")
        else:
            size_multiplier = 1.0
        
        # Adjust based on access frequency
        if access_frequency > 0.8:  # High access frequency - increase size
            size_multiplier *= 1.2
        elif access_frequency < 0.2:  # Low access frequency - decrease size
            size_multiplier *= 0.8
        
        new_size = int(base_size * size_multiplier)
        new_size = max(self.min_size, min(self.max_size, new_size))
        
        # Update cache size if it changed significantly
        if abs(new_size - base_size) / base_size > 0.1:  # 10% change threshold
            self.cache_sizes[cache_type] = new_size
            self.logger.info(f"Adjusted {cache_type} cache size from {base_size} to {new_size}")
        
        return new_size
    
    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """
        Retrieve data from cache with hit tracking.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        self.total_accesses[cache_type] += 1
        
        if key in self.caches[cache_type]:
            # Cache hit
            entry = self.caches[cache_type][key]
            entry.access_count += 1
            entry.last_access = time.time()
            
            self.total_hits[cache_type] += 1
            self.hit_rates[cache_type].append(True)
            
            return entry.data
        else:
            # Cache miss
            self.hit_rates[cache_type].append(False)
            return None
    
    def set(self, cache_type: str, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Store data in cache with adaptive sizing.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds (optional)
        """
        # Check if we need to evict entries
        current_size = len(self.caches[cache_type])
        # Use strict max_size limit, not adaptive size
        max_size = self.max_size

        if current_size >= max_size:
            self._evict_entries(cache_type, current_size - max_size + 1)
        
        # Create cache entry
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            access_count=1,
            last_access=time.time(),
            size_bytes=self._estimate_size(data)
        )
        
        self.caches[cache_type][key] = entry
        self.stats["total_entries"] += 1
        self.stats["total_memory_usage"] += entry.size_bytes
    
    def invalidate(self, cache_type: str, key: str) -> bool:
        """
        Remove a specific entry from cache.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            
        Returns:
            True if entry was found and removed
        """
        if key in self.caches[cache_type]:
            entry = self.caches[cache_type][key]
            self.stats["total_memory_usage"] -= entry.size_bytes
            self.stats["total_entries"] -= 1
            del self.caches[cache_type][key]
            return True
        return False
    
    def clear(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            cache_type: Specific cache type to clear, or None for all
        """
        if cache_type is None:
            # Clear all caches
            for ct in list(self.caches.keys()):
                self.clear(ct)
        else:
            if cache_type in self.caches:
                # Update memory usage
                for entry in self.caches[cache_type].values():
                    self.stats["total_memory_usage"] -= entry.size_bytes
                
                self.stats["total_entries"] -= len(self.caches[cache_type])
                self.caches[cache_type].clear()
                self.logger.info(f"Cleared {cache_type} cache")
    
    def get_stats(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            cache_type: Specific cache type, or None for all
            
        Returns:
            Dictionary with cache statistics
        """
        if cache_type is None:
            # Return aggregate stats
            stats = {
                "total_memory_usage_mb": self.stats["total_memory_usage"] / (1024 * 1024),
                "total_entries": self.stats["total_entries"],
                "cache_types": list(self.caches.keys()),
                "compression_enabled": self.stats["compression_enabled"]
            }
            
            # Aggregate hit rates
            total_hits = sum(self.total_hits.values())
            total_accesses = sum(self.total_accesses.values())
            stats["overall_hit_rate"] = total_hits / max(total_accesses, 1)
            
            return stats
        else:
            # Return specific cache stats
            if cache_type not in self.caches:
                return {}
            
            cache = self.caches[cache_type]
            total_accesses = self.total_accesses.get(cache_type, 0)
            total_hits = self.total_hits.get(cache_type, 0)
            
            return {
                "entries": len(cache),
                "max_size": self.get_cache_size(cache_type),
                "hit_rate": self._calculate_hit_rate(cache_type),
                "access_frequency": self._calculate_access_frequency(cache_type),
                "memory_usage_mb": sum(e.size_bytes for e in cache.values()) / (1024 * 1024),
                "eviction_count": self.eviction_counts.get(cache_type, 0),
                "total_accesses": total_accesses,
                "total_hits": total_hits
            }
    
    def _calculate_hit_rate(self, cache_type: str) -> float:
        """Calculate hit rate for a cache type."""
        if not self.hit_rates[cache_type]:
            return 0.0
        
        hits = sum(1 for hit in self.hit_rates[cache_type] if hit)
        return hits / len(self.hit_rates[cache_type])
    
    def _calculate_access_frequency(self, cache_type: str) -> float:
        """Calculate access frequency for a cache type."""
        total_accesses = self.total_accesses.get(cache_type, 0)
        if total_accesses == 0:
            return 0.0
        
        # Calculate frequency based on recent activity
        recent_window = 300  # 5 minutes
        current_time = time.time()
        
        recent_accesses = 0
        for entry in self.caches[cache_type].values():
            if current_time - entry.last_access < recent_window:
                recent_accesses += entry.access_count
        
        return recent_accesses / max(total_accesses, 1)
    
    def _evict_entries(self, cache_type: str, count: int) -> None:
        """Evict entries based on the configured policy."""
        cache = self.caches[cache_type]
        
        if self.eviction_policy == "lru":
            # Least Recently Used
            entries = sorted(cache.items(), key=lambda x: x[1].last_access)
        elif self.eviction_policy == "lfu":
            # Least Frequently Used
            entries = sorted(cache.items(), key=lambda x: x[1].access_count)
        else:  # adaptive
            # Adaptive: combination of LRU and LFU
            current_time = time.time()
            entries = sorted(
                cache.items(),
                key=lambda x: (x[1].access_count, current_time - x[1].last_access)
            )
        
        # Evict the specified number of entries
        for i in range(min(count, len(entries))):
            key, entry = entries[i]
            self.stats["total_memory_usage"] -= entry.size_bytes
            self.stats["total_entries"] -= 1
            del cache[key]
        
        self.eviction_counts[cache_type] += count
        self.logger.debug(f"Evicted {count} entries from {cache_type} cache")
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        try:
            import sys
            return sys.getsizeof(data)
        except:
            # Fallback estimation
            return 1024  # Default 1KB estimate
    
    async def cleanup_expired_entries(self, ttl: int = 3600) -> None:
        """
        Remove expired entries from all caches.
        
        Args:
            ttl: Time to live in seconds
        """
        current_time = time.time()
        expired_count = 0
        
        for cache_type, cache in self.caches.items():
            expired_keys = []
            for key, entry in cache.items():
                if current_time - entry.timestamp > ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = cache[key]
                self.stats["total_memory_usage"] -= entry.size_bytes
                self.stats["total_entries"] -= 1
                del cache[key]
                expired_count += 1
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired cache entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        cache_stats = self.stats.copy()

        # Add cache-specific statistics
        per_cache_counts = {cache_type: len(cache) for cache_type, cache in self.caches.items()}
        cache_stats.update({
            "cache_types": list(self.caches.keys()),
            "total_caches": len(self.caches),
            "hit_rates": dict(self.hit_rates),
            "cache_sizes": dict(self.cache_sizes),
            "eviction_counts": dict(self.eviction_counts),
            "compression_ratios": dict(self.compression_ratios),
            "total_accesses": dict(self.total_accesses),
            "total_hits": dict(self.total_hits),
            "entries_per_cache": per_cache_counts,
        })

        # Provide friendly aliases expected by status tooling
        cache_stats.setdefault("scan_results", per_cache_counts.get("scan_results", 0))
        cache_stats.setdefault("strategy_fits", per_cache_counts.get("strategy_fits", 0))
        cache_stats.setdefault("execution_opportunities", per_cache_counts.get("execution_opportunities", 0))
        cache_stats.setdefault("review_queue_size", per_cache_counts.get("review_queue", 0))
        cache_stats.setdefault("cache_size_limit", self.max_size)

        # Calculate overall hit rate
        total_accesses = sum(self.total_accesses.values())
        total_hits = sum(self.total_hits.values())
        overall_hit_rate = (total_hits / total_accesses * 100) if total_accesses > 0 else 0.0
        cache_stats["overall_hit_rate_pct"] = overall_hit_rate

        return cache_stats

    def get_execution_opportunities(self, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get cached execution opportunities above confidence threshold.

        Args:
            min_confidence: Minimum confidence score for opportunities

        Returns:
            List of execution opportunity dictionaries
        """
        opportunities = []

        # Scan through all cached results to find opportunities
        for cache_type, cache in self.caches.items():
            for key, entry in cache.items():
                data = entry.data if isinstance(entry.data, dict) else {}
                if not data:
                    continue

                confidence = float(data.get('confidence', 0) or 0)
                if confidence < min_confidence:
                    continue

                symbol = str(data.get('symbol') or key)
                strategy = str(data.get('strategy') or 'enhanced_solana_scanner')
                direction = str(data.get('direction') or 'long').lower()

                # Derive pricing information with sensible fallbacks
                price = float(data.get('entry_price') or data.get('price') or 0)
                if price <= 0:
                    continue  # Skip malformed entries

                atr = float(data.get('atr') or 0)

                stop_loss = data.get('stop_loss')
                take_profit = data.get('take_profit')

                if stop_loss is None or stop_loss <= 0:
                    if atr > 0 and atr < price:
                        stop_loss = max(price - (2 * atr), 0.0001)
                    else:
                        stop_loss = price * 0.97

                if take_profit is None or take_profit <= 0:
                    if atr > 0 and atr < price * 10:
                        take_profit = price + (3 * atr)
                    else:
                        take_profit = price * 1.05

                risk_reward = data.get('risk_reward_ratio')
                if not risk_reward or risk_reward <= 0:
                    risk = price - stop_loss
                    reward = take_profit - price
                    risk_reward = (reward / risk) if risk > 0 else 0
                    if risk_reward <= 0:
                        risk_reward = 2.0

                position_size = float(data.get('position_size') or 0.02)
                status = str(data.get('status') or 'pending')

                opportunity = {
                    'symbol': symbol,
                    'strategy': strategy,
                    'direction': direction,
                    'confidence': confidence,
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward_ratio': float(risk_reward),
                    'position_size': position_size,
                    'timestamp': entry.timestamp,
                    'status': status,
                    'cache_key': key,
                    'cache_type': cache_type,
                    'raw': data,
                }

                opportunities.append(opportunity)

        # Sort by confidence descending
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities

    def add_scan_result(self, symbol: str, data: Dict[str, Any], score: float,
                       regime: str, market_conditions: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a scan result to the cache.

        Args:
            symbol: Trading symbol
            data: Scan result data
            score: Quality/confidence score
            regime: Market regime classification
            market_conditions: Additional market condition data
        """
        # Prepare the data structure
        scan_data = data.copy()
        scan_data.update({
            'symbol': symbol,
            'score': score,
            'regime': regime,
            'confidence': score  # Map score to confidence for opportunities
        })

        if market_conditions:
            scan_data.update(market_conditions)

        # Cache with appropriate TTL (12 hours for scan results)
        price = float(scan_data.get('entry_price') or scan_data.get('price') or 0)

        atr_value = scan_data.get('atr')
        if atr_value is None and market_conditions:
            atr_value = market_conditions.get('atr')
        atr = float(atr_value or 0)

        if price > 0:
            scan_data.setdefault('entry_price', price)
            if atr > 0 and atr < price:
                scan_data.setdefault('stop_loss', max(price - (2 * atr), 0.0001))
                scan_data.setdefault('take_profit', price + (3 * atr))

        scan_data.setdefault('risk_reward_ratio', (scan_data.get('risk_reward_ratio') or 0.0))
        scan_data.setdefault('position_size', (scan_data.get('position_size') or 0.02))
        scan_data.setdefault('strategy', (scan_data.get('strategy') or 'enhanced_solana_scanner'))
        scan_data.setdefault('direction', (scan_data.get('direction') or 'long'))
        scan_data.setdefault('status', (scan_data.get('status') or 'pending'))

        self.set("scan_results", symbol, scan_data, ttl=43200)

    def get_scan_result(self, symbol: str) -> Optional[CacheEntry]:
        """Return cached scan result entry for ``symbol`` if present."""

        cache = self.caches.get("scan_results")
        self.total_accesses["scan_results"] += 1

        if not cache or symbol not in cache:
            self.hit_rates["scan_results"].append(False)
            return None

        entry = cache[symbol]
        entry.access_count += 1
        entry.last_access = time.time()
        self.total_hits["scan_results"] += 1
        self.hit_rates["scan_results"].append(True)
        return entry

    def mark_execution_attempted(self, opportunity: Union[Dict[str, Any], str],
                                  default_cache_type: str = "scan_results") -> None:
        """Mark ``opportunity`` as attempted in the cache."""

        if isinstance(opportunity, dict):
            cache_type = opportunity.get('cache_type', default_cache_type)
            key = opportunity.get('cache_key') or opportunity.get('symbol')
        else:
            cache_type = default_cache_type
            key = opportunity

        if not key:
            return

        cache = self.caches.get(cache_type)
        if not cache or key not in cache:
            return

        entry = cache[key]
        if isinstance(entry.data, dict):
            entry.data['status'] = 'attempted'
        entry.last_access = time.time()

    def get_scan_results(self, min_score: float = 0.0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get cached scan results above minimum score.

        Args:
            min_score: Minimum score threshold
            limit: Maximum number of results to return

        Returns:
            List of scan result dictionaries
        """
        results = []

        if "scan_results" in self.caches:
            for key, entry in self.caches["scan_results"].items():
                if hasattr(entry.data, 'get') and 'score' in entry.data:
                    score = entry.data.get('score', 0)
                    if score >= min_score:
                        result = entry.data.copy()
                        result['cache_timestamp'] = entry.timestamp
                        results.append(result)

        # Sort by score descending and apply limit
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        if limit:
            results = results[:limit]

        return results

    async def clear_expired_entries(self, ttl: int = 3600) -> None:
        """
        Remove expired entries from all caches.

        Args:
            ttl: Time to live in seconds
        """
        current_time = time.time()
        expired_count = 0

        for cache_type, cache in self.caches.items():
            expired_keys = []
            for key, entry in cache.items():
                if current_time - entry.timestamp > ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                entry = cache[key]
                self.stats["total_memory_usage"] -= entry.size_bytes
                self.stats["total_entries"] -= 1
                del cache[key]
                expired_count += 1

        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired cache entries")


# Global cache manager instance
_global_cache_manager: Optional[AdaptiveCacheManager] = None


def get_scan_cache_manager(config: Optional[Dict[str, Any]] = None) -> AdaptiveCacheManager:
    """
    Get or create the global scan cache manager instance.
    Alias for get_cache_manager for backward compatibility.
    
    Args:
        config: Optional configuration dictionary (currently unused, kept for compatibility)
        
    Returns:
        AdaptiveCacheManager instance
    """
    return get_cache_manager()


def get_cache_manager() -> AdaptiveCacheManager:
    """
    Get or create the global cache manager instance.
    
    Returns:
        AdaptiveCacheManager instance
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = AdaptiveCacheManager()
    return _global_cache_manager


def configure_cache_manager(
    initial_size: Optional[int] = None,
    max_size: Optional[int] = None,
    min_size: Optional[int] = None,
    hit_rate_window: Optional[int] = None,
    eviction_policy: Optional[str] = None,
    enable_compression: Optional[bool] = None
) -> None:
    """
    Configure the global cache manager with new settings.
    
    Args:
        initial_size: Initial cache size for each cache type
        max_size: Maximum cache size allowed
        min_size: Minimum cache size allowed
        hit_rate_window: Number of accesses to track for hit rate calculation
        eviction_policy: Cache eviction policy ('lru', 'lfu', 'adaptive')
        enable_compression: Whether to enable data compression
    """
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = AdaptiveCacheManager()
    
    if initial_size is not None:
        _global_cache_manager.initial_size = initial_size
    if max_size is not None:
        _global_cache_manager.max_size = max_size
    if min_size is not None:
        _global_cache_manager.min_size = min_size
    if hit_rate_window is not None:
        _global_cache_manager.hit_rate_window = hit_rate_window
    if eviction_policy is not None:
        _global_cache_manager.eviction_policy = eviction_policy
    if enable_compression is not None:
        _global_cache_manager.enable_compression = enable_compression
