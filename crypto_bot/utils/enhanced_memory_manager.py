"""
Enhanced Memory Manager for LegacyCoinTrader

Addresses memory leak prevention, cache management, and performance optimization
with proper testing and validation.
"""

import gc
import time
import threading
import weakref
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import psutil
import os
import logging
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from contextlib import contextmanager
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    value: Any
    timestamp: float
    size_bytes: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    memory_threshold: float = 0.8  # 80% memory usage threshold
    gc_threshold: float = 0.7      # 70% triggers GC
    cache_size_limit_mb: int = 500
    model_cleanup_interval: int = 300  # 5 minutes
    cache_cleanup_interval: int = 600  # 10 minutes
    max_cache_entries: int = 1000
    cache_ttl_seconds: int = 3600  # 1 hour
    enable_background_cleanup: bool = True
    aggressive_cleanup_threshold: float = 0.9  # 90% triggers aggressive cleanup


class ManagedCache:
    """A memory-managed cache with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_size_bytes = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with access tracking."""
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                self._remove_entry(key)
                return None
            
            # Update access stats
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            
            return entry.value
    
    def put(self, key: str, value: Any, size_bytes: Optional[int] = None) -> None:
        """Put value in cache with size tracking."""
        with self._lock:
            # Estimate size if not provided
            if size_bytes is None:
                size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            # Check if we need to evict
            while (len(self.cache) >= self.max_size or 
                   self.total_size_bytes + size_bytes > self.max_size * 1024 * 1024):  # MB limit
                self._evict_oldest()
            
            # Add new entry
            self.cache[key] = entry
            self.total_size_bytes += size_bytes
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.total_size_bytes -= entry.size_bytes
            del self.cache[key]
    
    def _evict_oldest(self) -> None:
        """Evict the oldest entry (LRU)."""
        if self.cache:
            oldest_key = next(iter(self.cache))
            self._remove_entry(oldest_key)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        try:
            if isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(v) for v in value.values())
            else:
                return len(str(value)) * 8  # Rough estimate
        except:
            return 1024  # Default 1KB estimate
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                k for k, v in self.cache.items()
                if current_time - v.timestamp > self.ttl_seconds
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "total_size_mb": self.total_size_bytes / (1024 * 1024),
                "hit_rate": self._calculate_hit_rate(),
                "oldest_entry_age": self._get_oldest_entry_age()
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        return total_accesses / max(len(self.cache), 1)
    
    def _get_oldest_entry_age(self) -> float:
        """Get age of oldest entry in seconds."""
        if not self.cache:
            return 0.0
        oldest_timestamp = min(entry.timestamp for entry in self.cache.values())
        return time.time() - oldest_timestamp


class EnhancedMemoryManager:
    """
    Enhanced memory management system with proactive monitoring and cleanup.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        
        # Caches
        self.caches: Dict[str, ManagedCache] = {}
        
        # ML models tracking
        self.ml_models: Dict[str, Dict[str, Any]] = {}
        
        # Memory monitoring
        self.memory_history: deque = deque(maxlen=100)
        self.last_cleanup = time.time()
        
        # Background cleanup
        self.cleanup_thread = None
        self.running = False
        
        # Performance tracking
        self.cleanup_count = 0
        self.memory_pressure_events = 0
        
        logger.info("Enhanced memory manager initialized")
    
    def register_cache(self, name: str, max_size: int = 1000, ttl_seconds: int = 3600) -> ManagedCache:
        """Register a new managed cache."""
        cache = ManagedCache(max_size=max_size, ttl_seconds=ttl_seconds)
        self.caches[name] = cache
        logger.debug(f"Registered cache: {name}")
        return cache
    
    def get_cache(self, name: str) -> Optional[ManagedCache]:
        """Get a registered cache."""
        return self.caches.get(name)
    
    def register_ml_model(self, name: str, model: Any, size_estimate_mb: int = 100) -> None:
        """Register an ML model for memory management."""
        self.ml_models[name] = {
            "model": weakref.ref(model),  # Use weak reference
            "created": time.time(),
            "last_used": time.time(),
            "size_estimate_mb": size_estimate_mb
        }
        logger.debug(f"Registered ML model: {name}")
    
    def update_ml_model_usage(self, name: str) -> None:
        """Update last used time for ML model."""
        if name in self.ml_models:
            self.ml_models[name]["last_used"] = time.time()
    
    def unregister_ml_model(self, name: str) -> None:
        """Unregister an ML model."""
        if name in self.ml_models:
            del self.ml_models[name]
            logger.debug(f"Unregistered ML model: {name}")
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            memory = psutil.virtual_memory()
            usage_ratio = memory.percent / 100.0
            return usage_ratio > self.config.memory_threshold
        except Exception as e:
            logger.error(f"Memory pressure check failed: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            
            # Calculate cache statistics
            total_cache_size = sum(cache.total_size_bytes for cache in self.caches.values())
            total_cache_entries = sum(len(cache.cache) for cache in self.caches.values())
            
            # Calculate ML model statistics
            total_ml_size = sum(info["size_estimate_mb"] for info in self.ml_models.values())
            
            stats = {
                "system_memory_percent": memory.percent,
                "system_memory_used_mb": memory.used / (1024 * 1024),
                "system_memory_total_mb": memory.total / (1024 * 1024),
                "process_memory_mb": process.memory_info().rss / (1024 * 1024),
                "caches_count": len(self.caches),
                "total_cache_size_mb": total_cache_size / (1024 * 1024),
                "total_cache_entries": total_cache_entries,
                "ml_models_count": len(self.ml_models),
                "ml_models_total_size_mb": total_ml_size,
                "gc_objects": len(gc.get_objects()),
                "cleanup_count": self.cleanup_count,
                "memory_pressure_events": self.memory_pressure_events,
                "timestamp": time.time()
            }
            
            # Add cache-specific stats
            stats["caches"] = {
                name: cache.get_stats() for name, cache in self.caches.items()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform all memory maintenance tasks."""
        current_time = time.time()
        maintenance_results = {
            "timestamp": current_time,
            "cache_cleanup": 0,
            "ml_cleanup": 0,
            "gc_collected": 0,
            "memory_pressure": False
        }
        
        try:
            # Check for memory pressure
            if self.check_memory_pressure():
                maintenance_results["memory_pressure"] = True
                self.memory_pressure_events += 1
                self._handle_memory_pressure(maintenance_results)
            
            # Regular cache cleanup
            if current_time - self.last_cleanup > self.config.cache_cleanup_interval:
                maintenance_results["cache_cleanup"] = self._cleanup_caches()
                self.last_cleanup = current_time
            
            # ML model cleanup
            if current_time - self.last_cleanup > self.config.model_cleanup_interval:
                maintenance_results["ml_cleanup"] = self._cleanup_ml_models()
            
            # Force garbage collection if needed
            if self.should_force_gc():
                maintenance_results["gc_collected"] = self.force_garbage_collection()
            
        except Exception as e:
            logger.error(f"Memory maintenance failed: {e}")
            maintenance_results["error"] = str(e)
        
        # Update memory history
        try:
            self.memory_history.append(self.get_memory_stats())
        except Exception as e:
            logger.error(f"Failed to update memory history: {e}")
        
        self.cleanup_count += 1
        return maintenance_results
    
    def _handle_memory_pressure(self, results: Dict[str, Any]) -> None:
        """Handle memory pressure situation."""
        logger.warning("Memory pressure detected, performing emergency cleanup")
        
        # Aggressive cache cleanup
        results["cache_cleanup"] = self._cleanup_caches(aggressive=True)
        
        # Force garbage collection
        results["gc_collected"] = self.force_garbage_collection()
        
        # Emergency ML model cleanup
        results["ml_cleanup"] = self._cleanup_ml_models(aggressive=True)
    
    def _cleanup_caches(self, aggressive: bool = False) -> int:
        """Clean up expired cache entries."""
        total_cleaned = 0
        
        for name, cache in self.caches.items():
            try:
                cleaned = cache.cleanup_expired()
                total_cleaned += cleaned
                
                if aggressive and len(cache.cache) > cache.max_size // 2:
                    # Remove half of the cache entries
                    keys_to_remove = list(cache.cache.keys())[:len(cache.cache) // 2]
                    for key in keys_to_remove:
                        cache._remove_entry(key)
                    total_cleaned += len(keys_to_remove)
                    
            except Exception as e:
                logger.error(f"Cache cleanup failed for {name}: {e}")
        
        if total_cleaned > 0:
            logger.info(f"Cleaned up {total_cleaned} cache entries")
        
        return total_cleaned
    
    def _cleanup_ml_models(self, aggressive: bool = False) -> int:
        """Clean up unused ML models."""
        current_time = time.time()
        cleanup_age = 1800 if aggressive else 3600  # 30 min vs 1 hour
        
        models_to_cleanup = []
        for name, info in self.ml_models.items():
            if current_time - info["last_used"] > cleanup_age:
                models_to_cleanup.append(name)
        
        # If aggressive, remove half of the models
        if aggressive and len(models_to_cleanup) < len(self.ml_models) // 2:
            all_models = list(self.ml_models.keys())
            models_to_cleanup.extend(all_models[:len(all_models) // 2])
        
        for name in models_to_cleanup:
            try:
                self.unregister_ml_model(name)
                logger.info(f"Cleaned up ML model: {name}")
            except Exception as e:
                logger.error(f"ML model cleanup failed for {name}: {e}")
        
        return len(models_to_cleanup)
    
    def should_force_gc(self) -> bool:
        """Check if garbage collection should be forced."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0 > self.config.gc_threshold
        except:
            return False
    
    def force_garbage_collection(self) -> int:
        """Force garbage collection and return collected objects."""
        try:
            before = len(gc.get_objects())
            collected = gc.collect()
            after = len(gc.get_objects())
            
            logger.info(f"Garbage collection: {collected} objects collected")
            return collected
            
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
            return 0
    
    def start_background_cleanup(self) -> None:
        """Start background memory cleanup thread."""
        if self.config.enable_background_cleanup and not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(
                target=self._background_cleanup_loop,
                daemon=True,
                name="EnhancedMemoryCleanup"
            )
            self.cleanup_thread.start()
            logger.info("Background memory cleanup started")
    
    def stop_background_cleanup(self) -> None:
        """Stop background memory cleanup thread."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        logger.info("Background memory cleanup stopped")
    
    def _background_cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.running:
            try:
                self.perform_maintenance()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
                time.sleep(30)  # Wait before retry
    
    @contextmanager
    def memory_monitoring(self, operation_name: str):
        """Context manager for monitoring memory usage during operations."""
        start_stats = self.get_memory_stats()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_stats = self.get_memory_stats()
            
            duration = end_time - start_time
            memory_delta = end_stats.get("process_memory_mb", 0) - start_stats.get("process_memory_mb", 0)
            
            logger.debug(f"Operation {operation_name}: {duration:.2f}s, memory delta: {memory_delta:.2f}MB")
            
            # Alert if memory usage increased significantly
            if memory_delta > 100:  # More than 100MB increase
                logger.warning(f"Large memory increase in {operation_name}: {memory_delta:.2f}MB")


# Global instance
_enhanced_memory_manager = None
_memory_manager_lock = threading.Lock()


def get_enhanced_memory_manager(config: Optional[MemoryConfig] = None) -> EnhancedMemoryManager:
    """Get or create global enhanced memory manager instance."""
    global _enhanced_memory_manager
    if _enhanced_memory_manager is None:
        with _memory_manager_lock:
            if _enhanced_memory_manager is None:
                _enhanced_memory_manager = EnhancedMemoryManager(config)
                _enhanced_memory_manager.start_background_cleanup()
    return _enhanced_memory_manager


def reset_enhanced_memory_manager() -> None:
    """Reset the global enhanced memory manager instance."""
    global _enhanced_memory_manager
    with _memory_manager_lock:
        if _enhanced_memory_manager:
            _enhanced_memory_manager.stop_background_cleanup()
        _enhanced_memory_manager = None
    logger.info("Enhanced memory manager global instance reset")
