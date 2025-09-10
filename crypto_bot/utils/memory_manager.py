"""
Production Memory Management for LegacyCoinTrader

Handles memory leaks in ML models, caches, and long-running processes.
"""

import gc
import time
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
import psutil
import os

from .logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "memory_manager.log")


class ProductionMemoryManager:
    """
    Production-grade memory management system.

    Features:
    - ML model cleanup (LSTM, regime classifiers)
    - Cache size optimization
    - Memory pressure detection
    - Automatic garbage collection
    - Memory usage monitoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # Memory thresholds
        self.memory_threshold = self.config.get("memory_threshold", 0.8)
        self.gc_threshold = self.config.get("gc_threshold", 0.7)
        self.cache_size_limit_mb = self.config.get("cache_size_limit_mb", 500)

        # ML model tracking
        self.ml_models = {}
        self.model_cleanup_interval = self.config.get("model_cleanup_interval", 300)  # 5 minutes
        self.last_model_cleanup = time.time()

        # Cache tracking
        self.caches = {}
        self.cache_cleanup_interval = self.config.get("cache_cleanup_interval", 600)  # 10 minutes
        self.last_cache_cleanup = time.time()

        # Monitoring
        self.memory_stats_history = []
        self.stats_history_size = self.config.get("stats_history_size", 100)

        # Background cleanup thread
        self.cleanup_thread = None
        self.running = False

        logger.info("Production memory manager initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default memory management configuration."""
        return {
            "memory_threshold": 0.8,  # 80% memory usage threshold
            "gc_threshold": 0.7,      # 70% triggers GC
            "cache_size_limit_mb": 500,
            "model_cleanup_interval": 300,
            "cache_cleanup_interval": 600,
            "stats_history_size": 100,
            "enable_background_cleanup": True
        }

    def start_background_cleanup(self):
        """Start background memory cleanup thread."""
        if self.config.get("enable_background_cleanup", True) and not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(
                target=self._background_cleanup_loop,
                daemon=True,
                name="MemoryCleanup"
            )
            self.cleanup_thread.start()
            logger.info("Background memory cleanup started")

    def stop_background_cleanup(self):
        """Stop background memory cleanup thread."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        logger.info("Background memory cleanup stopped")

    def _background_cleanup_loop(self):
        """Background cleanup loop."""
        while self.running:
            try:
                self.perform_maintenance()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
                time.sleep(30)  # Wait before retry

    def perform_maintenance(self):
        """Perform all memory maintenance tasks."""
        current_time = time.time()

        # ML model cleanup
        if current_time - self.last_model_cleanup > self.model_cleanup_interval:
            self.cleanup_ml_models()
            self.last_model_cleanup = current_time

        # Cache cleanup
        if current_time - self.last_cache_cleanup > self.cache_cleanup_interval:
            self.optimize_caches()
            self.last_cache_cleanup = current_time

        # Memory pressure check
        if self.check_memory_pressure():
            self.handle_memory_pressure()

        # Force GC if needed
        if self.should_force_gc():
            self.force_garbage_collection()

    def register_ml_model(self, name: str, model: Any):
        """Register an ML model for memory management."""
        self.ml_models[name] = {
            "model": model,
            "created": time.time(),
            "last_used": time.time(),
            "size_estimate": self._estimate_model_size(model)
        }
        logger.debug(f"Registered ML model: {name}")

    def unregister_ml_model(self, name: str):
        """Unregister an ML model."""
        if name in self.ml_models:
            del self.ml_models[name]
            logger.debug(f"Unregistered ML model: {name}")

    def update_ml_model_usage(self, name: str):
        """Update last used time for ML model."""
        if name in self.ml_models:
            self.ml_models[name]["last_used"] = time.time()

    def cleanup_ml_models(self):
        """Clean up unused ML models."""
        current_time = time.time()
        cleanup_age = self.config.get("model_cleanup_age", 3600)  # 1 hour

        models_to_cleanup = []
        for name, info in self.ml_models.items():
            if current_time - info["last_used"] > cleanup_age:
                models_to_cleanup.append(name)

        for name in models_to_cleanup:
            try:
                self._cleanup_ml_model(name)
                logger.info(f"Cleaned up ML model: {name}")
            except Exception as e:
                logger.error(f"Failed to cleanup ML model {name}: {e}")

    def _cleanup_ml_model(self, name: str):
        """Clean up a specific ML model."""
        if name not in self.ml_models:
            return

        info = self.ml_models[name]
        model = info["model"]

        # Clear model references
        if hasattr(model, 'model'):
            model.model = None
        if hasattr(model, 'scaler'):
            model.scaler = None
        if hasattr(model, 'feature_names'):
            model.feature_names = None

        # Force cleanup
        del self.ml_models[name]
        del model

    def _estimate_model_size(self, model: Any) -> int:
        """Estimate memory size of ML model."""
        try:
            # Rough estimation based on model attributes
            size = 0
            if hasattr(model, 'model') and model.model is not None:
                size += 1000000  # Assume ~1MB for sklearn models
            if hasattr(model, 'scaler') and model.scaler is not None:
                size += 100000  # Assume ~100KB for scalers
            return size
        except:
            return 1000000  # Default 1MB estimate

    def register_cache(self, name: str, cache: Any):
        """Register a cache for memory management."""
        self.caches[name] = {
            "cache": cache,
            "created": time.time(),
            "last_used": time.time()
        }
        logger.debug(f"Registered cache: {name}")

    def optimize_caches(self):
        """Optimize cache sizes."""
        for name, info in self.caches.items():
            try:
                self._optimize_cache(name, info["cache"])
            except Exception as e:
                logger.error(f"Failed to optimize cache {name}: {e}")

    def _optimize_cache(self, name: str, cache: Any):
        """Optimize a specific cache."""
        # Reduce cache size by 20% if it's too large
        if hasattr(cache, 'maxlen') and cache.maxlen > 100:
            current_size = len(cache)
            if current_size > 0:
                new_maxlen = max(100, int(cache.maxlen * 0.8))
                # Create new cache with reduced size
                from collections import deque
                if isinstance(cache, deque):
                    new_cache = deque(list(cache)[-new_maxlen:], maxlen=new_maxlen)
                    # Replace the cache (this is a bit hacky but works for our use case)
                    cache.clear()
                    cache.extend(new_cache)
                    cache.maxlen = new_maxlen

        logger.debug(f"Optimized cache {name}")

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            memory = psutil.virtual_memory()
            usage_ratio = memory.percent / 100.0
            return usage_ratio > self.memory_threshold
        except Exception as e:
            logger.error(f"Memory pressure check failed: {e}")
            return False

    def handle_memory_pressure(self):
        """Handle memory pressure situation."""
        logger.warning("Memory pressure detected, performing emergency cleanup")

        # Force garbage collection
        collected = self.force_garbage_collection()
        logger.info(f"Emergency GC collected {collected} objects")

        # Clean up oldest ML models
        self._emergency_ml_cleanup()

        # Optimize all caches aggressively
        for name, info in self.caches.items():
            try:
                cache = info["cache"]
                if hasattr(cache, 'maxlen'):
                    # Reduce to minimum size
                    cache.maxlen = max(50, int(cache.maxlen * 0.5))
                    # Keep only most recent items
                    if len(cache) > cache.maxlen:
                        excess = len(cache) - cache.maxlen
                        for _ in range(excess):
                            cache.popleft()
            except Exception as e:
                logger.error(f"Emergency cache cleanup failed for {name}: {e}")

    def _emergency_ml_cleanup(self):
        """Emergency cleanup of ML models."""
        # Remove oldest models first
        sorted_models = sorted(
            self.ml_models.items(),
            key=lambda x: x[1]["last_used"]
        )

        # Remove half of the models
        models_to_remove = sorted_models[:len(sorted_models)//2]

        for name, _ in models_to_remove:
            try:
                self._cleanup_ml_model(name)
                logger.warning(f"Emergency cleanup: removed ML model {name}")
            except Exception as e:
                logger.error(f"Emergency ML cleanup failed for {name}: {e}")

    def should_force_gc(self) -> bool:
        """Check if garbage collection should be forced."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0 > self.gc_threshold
        except:
            return False

    def force_garbage_collection(self) -> int:
        """Force garbage collection and return collected objects."""
        try:
            before = len(gc.get_objects())
            collected = gc.collect()
            after = len(gc.get_objects())

            # Log memory stats
            memory = psutil.virtual_memory()
            logger.info(".1f")

            # Store in history
            self.memory_stats_history.append({
                "timestamp": time.time(),
                "memory_percent": memory.percent,
                "collected_objects": collected,
                "objects_before": before,
                "objects_after": after
            })

            # Keep only recent history
            if len(self.memory_stats_history) > self.stats_history_size:
                self.memory_stats_history = self.memory_stats_history[-self.stats_history_size:]

            return collected

        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
            return 0

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())

            stats = {
                "system_memory_percent": memory.percent,
                "system_memory_used_mb": memory.used / 1024 / 1024,
                "system_memory_total_mb": memory.total / 1024 / 1024,
                "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                "ml_models_count": len(self.ml_models),
                "caches_count": len(self.caches),
                "gc_objects": len(gc.get_objects()),
                "timestamp": time.time()
            }

            # Add ML model sizes
            total_ml_size = sum(info["size_estimate"] for info in self.ml_models.values())
            stats["ml_models_total_size_mb"] = total_ml_size / 1024 / 1024

            return stats

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }

    def get_ml_models_status(self) -> Dict[str, Any]:
        """Get status of registered ML models."""
        current_time = time.time()

        models_info = {}
        for name, info in self.ml_models.items():
            models_info[name] = {
                "age_hours": (current_time - info["created"]) / 3600,
                "last_used_hours": (current_time - info["last_used"]) / 3600,
                "size_mb": info["size_estimate"] / 1024 / 1024
            }

        return {
            "total_models": len(self.ml_models),
            "models": models_info,
            "cleanup_interval_hours": self.model_cleanup_interval / 3600
        }


# Global memory manager instance
_memory_manager = None

def get_memory_manager(config: Optional[Dict[str, Any]] = None) -> ProductionMemoryManager:
    """Get or create global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ProductionMemoryManager(config)
        _memory_manager.start_background_cleanup()
    return _memory_manager
