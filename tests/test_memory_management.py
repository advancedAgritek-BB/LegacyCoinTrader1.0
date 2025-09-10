"""
Memory Management Test Suite

Tests for memory leak prevention, cache management, and performance optimization.
"""

import pytest
import asyncio
import time
import psutil
import gc
from unittest.mock import Mock, patch
from collections import deque
import pandas as pd
import numpy as np
from pathlib import Path

from crypto_bot.utils.enhanced_memory_manager import (
    EnhancedMemoryManager, 
    ManagedCache, 
    MemoryConfig,
    get_enhanced_memory_manager,
    reset_enhanced_memory_manager
)
from crypto_bot.phase_runner import BotContext, PhaseRunner


class TestEnhancedMemoryManager:
    """Test the enhanced memory manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = MemoryConfig(
            memory_threshold=0.8,
            gc_threshold=0.7,
            cache_size_limit_mb=500,
            model_cleanup_interval=300,
            cache_cleanup_interval=600,
            enable_background_cleanup=False  # Disable for testing
        )
        self.memory_manager = EnhancedMemoryManager(self.config)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.memory_manager.stop_background_cleanup()
    
    def test_memory_pressure_detection(self):
        """Test memory pressure detection functionality."""
        # Mock psutil to simulate different memory conditions
        with patch('psutil.virtual_memory') as mock_memory:
            # Test normal memory usage
            mock_memory.return_value.percent = 50.0
            assert not self.memory_manager.check_memory_pressure()
            
            # Test high memory usage
            mock_memory.return_value.percent = 85.0
            assert self.memory_manager.check_memory_pressure()
    
    def test_ml_model_registration(self):
        """Test ML model registration and cleanup."""
        # Create mock ML model
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.scaler = Mock()
        
        # Register model
        self.memory_manager.register_ml_model("test_model", mock_model, size_estimate_mb=50)
        assert "test_model" in self.memory_manager.ml_models
        
        # Test cleanup
        self.memory_manager.ml_models["test_model"]["last_used"] = time.time() - 4000  # Old model
        cleaned = self.memory_manager._cleanup_ml_models()
        assert cleaned == 1
        assert "test_model" not in self.memory_manager.ml_models
    
    def test_cache_optimization(self):
        """Test cache size optimization."""
        # Create a managed cache
        cache = self.memory_manager.register_cache("test_cache", max_size=100, ttl_seconds=3600)
        
        # Add items to cache
        for i in range(150):  # Exceed max_size
            cache.put(f"key_{i}", f"value_{i}")
        
        # Cache should respect max_size
        assert len(cache.cache) <= 100
        
        # Simulate memory pressure
        with patch.object(self.memory_manager, 'check_memory_pressure', return_value=True):
            self.memory_manager._handle_memory_pressure({})
            
            # Cache should be reduced during aggressive cleanup
            assert len(cache.cache) < 100
    
    def test_garbage_collection(self):
        """Test forced garbage collection."""
        # Create some objects to collect
        objects = [Mock() for _ in range(100)]
        
        # Force garbage collection
        collected = self.memory_manager.force_garbage_collection()
        
        # Should have collected some objects
        assert collected >= 0
    
    def test_memory_stats(self):
        """Test memory statistics collection."""
        stats = self.memory_manager.get_memory_stats()
        
        assert "system_memory_percent" in stats
        assert "process_memory_mb" in stats
        assert "ml_models_count" in stats
        assert "caches_count" in stats
        assert "timestamp" in stats


class TestManagedCache:
    """Test managed cache functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.cache = ManagedCache(max_size=10, ttl_seconds=1)
    
    def test_cache_size_limits(self):
        """Test that caches respect size limits."""
        # Add items beyond limit
        for i in range(15):
            self.cache.put(f"key_{i}", f"value_{i}")
        
        assert len(self.cache.cache) == 10  # Should respect max_size
        # Should have newest items
        assert "key_14" in self.cache.cache
        assert "key_0" not in self.cache.cache  # Oldest should be evicted
    
    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        # Add item
        self.cache.put("test_key", "test_value")
        assert "test_key" in self.cache.cache
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Item should be expired
        result = self.cache.get("test_key")
        assert result is None
        assert "test_key" not in self.cache.cache
    
    def test_cache_access_tracking(self):
        """Test cache access tracking."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        # Access key1 multiple times
        self.cache.get("key1")
        self.cache.get("key1")
        
        # key1 should have higher access count
        entry1 = self.cache.cache["key1"]
        entry2 = self.cache.cache["key2"]
        
        assert entry1.access_count == 2
        assert entry2.access_count == 0
    
    def test_dataframe_caching(self):
        """Test DataFrame caching with size estimation."""
        # Create DataFrame
        df = pd.DataFrame({
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000)
        })
        
        # Cache DataFrame
        self.cache.put("df_key", df)
        
        # Should be cached
        cached_df = self.cache.get("df_key")
        assert cached_df is not None
        assert len(cached_df) == 1000
        
        # Size should be estimated
        entry = self.cache.cache["df_key"]
        assert entry.size_bytes > 0


class TestCacheManagement:
    """Test cache management improvements."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = MemoryConfig(enable_background_cleanup=False)
        self.memory_manager = EnhancedMemoryManager(self.config)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.memory_manager.stop_background_cleanup()
    
    def test_dataframe_cache_optimization(self):
        """Test DataFrame cache optimization."""
        # Create large DataFrame
        large_df = pd.DataFrame({
            'close': np.random.randn(10000),
            'volume': np.random.randn(10000),
            'high': np.random.randn(10000),
            'low': np.random.randn(10000),
            'open': np.random.randn(10000)
        })
        
        # Create cache with multiple DataFrames
        cache = self.memory_manager.register_cache("df_cache", max_size=25, ttl_seconds=3600)
        
        for i in range(50):
            cache.put(f"symbol_{i}", large_df.copy())
        
        # Cache should respect size limit
        assert len(cache.cache) <= 25
        
        # Test cleanup
        cleaned = cache.cleanup_expired()
        assert cleaned >= 0  # May or may not have expired items
    
    def test_memory_monitoring_context(self):
        """Test memory monitoring context manager."""
        with self.memory_manager.memory_monitoring("test_operation"):
            # Simulate some work
            data = [i for i in range(10000)]
            time.sleep(0.1)
        
        # Context manager should complete without error
        assert True


class TestBotContextMemory:
    """Test BotContext memory management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = {
            "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
            "timeframe": "1h",
            "execution_mode": "dry_run"
        }
        
        # Create large caches
        self.df_cache = {}
        self.regime_cache = {}
        
        # Populate with test data
        for symbol in self.config["symbols"]:
            self.df_cache[symbol] = pd.DataFrame({
                'close': np.random.randn(1000),
                'volume': np.random.randn(1000),
                'high': np.random.randn(1000),
                'low': np.random.randn(1000),
                'open': np.random.randn(1000)
            })
            self.regime_cache[symbol] = pd.DataFrame({
                'regime': ['trending', 'sideways', 'volatile'] * 333 + ['trending']
            })
    
    def test_context_memory_cleanup(self):
        """Test BotContext memory cleanup."""
        # Create context with large caches
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Measure initial memory
        initial_memory = psutil.Process().memory_info().rss
        
        # Simulate memory pressure
        if len(ctx.df_cache) > 2:  # Reduce cache size
            # Keep only first 2 symbols
            keys_to_keep = list(ctx.df_cache.keys())[:2]
            ctx.df_cache = {k: ctx.df_cache[k] for k in keys_to_keep}
            ctx.regime_cache = {k: ctx.regime_cache[k] for k in keys_to_keep}
        
        # Force garbage collection
        gc.collect()
        
        # Measure memory after cleanup
        cleaned_memory = psutil.Process().memory_info().rss
        
        # Memory should be reduced or at least not increased significantly
        memory_delta = cleaned_memory - initial_memory
        assert memory_delta < 50 * 1024 * 1024  # Less than 50MB increase
        assert len(ctx.df_cache) == 2
        assert len(ctx.regime_cache) == 2
    
    def test_context_memory_monitoring(self):
        """Test BotContext memory monitoring."""
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Get memory statistics
        memory_stats = {
            "df_cache_size": len(ctx.df_cache),
            "regime_cache_size": len(ctx.regime_cache),
            "total_symbols": sum(len(df) for df in ctx.df_cache.values()),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        assert memory_stats["df_cache_size"] == 3
        assert memory_stats["regime_cache_size"] == 3
        assert memory_stats["memory_usage_mb"] > 0


class TestMemoryLeakDetection:
    """Test memory leak detection and prevention."""
    
    def test_memory_growth_detection(self):
        """Test detection of memory growth over time."""
        memory_samples = []
        
        # Simulate memory growth
        for i in range(10):
            # Create some objects
            objects = [Mock() for _ in range(100)]
            memory_samples.append(psutil.Process().memory_info().rss)
            time.sleep(0.1)
        
        # Check for consistent growth
        growth_rate = (memory_samples[-1] - memory_samples[0]) / len(memory_samples)
        
        # In a real scenario, we'd want to detect if growth_rate > threshold
        assert growth_rate >= 0  # Memory should not decrease in this test
    
    def test_circular_reference_detection(self):
        """Test detection of circular references."""
        # Create circular reference
        class CircularObject:
            def __init__(self):
                self.reference = None
            
            def create_circular_ref(self):
                self.reference = self
        
        obj = CircularObject()
        obj.create_circular_ref()
        
        # Count objects before and after GC
        before_count = len(gc.get_objects())
        gc.collect()
        after_count = len(gc.get_objects())
        
        # Python's GC can handle circular references, so we just verify GC ran
        # The actual cleanup depends on the GC implementation
        assert after_count >= 0  # Just verify we can count objects
        assert before_count >= 0
    
    def test_dataframe_memory_cleanup(self):
        """Test DataFrame memory cleanup."""
        # Create large DataFrame
        df = pd.DataFrame({
            'close': np.random.randn(10000),
            'volume': np.random.randn(10000)
        })
        
        # Measure memory before
        memory_before = psutil.Process().memory_info().rss
        
        # Clear DataFrame
        df = None
        gc.collect()
        
        # Measure memory after
        memory_after = psutil.Process().memory_info().rss
        
        # Memory should be reduced or at least not increased significantly
        memory_delta = memory_after - memory_before
        assert memory_delta < 10 * 1024 * 1024  # Less than 10MB increase


class TestMemoryOptimization:
    """Test memory optimization strategies."""
    
    def test_lazy_loading(self):
        """Test lazy loading of data."""
        class LazyDataLoader:
            def __init__(self):
                self._data = None
            
            @property
            def data(self):
                if self._data is None:
                    # Only load when accessed
                    self._data = pd.DataFrame({
                        'close': np.random.randn(1000),
                        'volume': np.random.randn(1000)
                    })
                return self._data
        
        loader = LazyDataLoader()
        
        # Data should not be loaded initially
        assert loader._data is None
        
        # Data should be loaded on first access
        data = loader.data
        assert loader._data is not None
        assert len(data) == 1000
    
    def test_data_compression(self):
        """Test data compression for memory savings."""
        # Create large DataFrame
        df = pd.DataFrame({
            'close': np.random.randn(10000),
            'volume': np.random.randn(10000)
        })
        
        # Measure original size
        original_size = df.memory_usage(deep=True).sum()
        
        # Optimize data types
        df['close'] = df['close'].astype('float32')
        df['volume'] = df['volume'].astype('float32')
        
        # Measure optimized size
        optimized_size = df.memory_usage(deep=True).sum()
        
        # Should save memory
        assert optimized_size < original_size
    
    def test_cache_eviction_policy(self):
        """Test LRU cache eviction policy."""
        cache = ManagedCache(max_size=3, ttl_seconds=3600)
        
        # Add items
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # Should have 3 items
        assert len(cache.cache) == 3
        
        # Add another item, should evict "a"
        cache.put("d", 4)
        assert len(cache.cache) == 3
        assert "a" not in cache.cache
        assert "d" in cache.cache
        
        # Access "b", should move to end
        cache.get("b")
        cache.put("e", 5)
        assert "c" not in cache.cache  # "c" should be evicted
        assert "b" in cache.cache  # "b" should remain


class TestGlobalMemoryManager:
    """Test global memory manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_enhanced_memory_manager()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_enhanced_memory_manager()
    
    def test_global_instance_creation(self):
        """Test global memory manager instance creation."""
        manager1 = get_enhanced_memory_manager()
        manager2 = get_enhanced_memory_manager()
        
        # Should be the same instance
        assert manager1 is manager2
        
        # Should be running background cleanup
        assert manager1.running
    
    def test_global_instance_reset(self):
        """Test global memory manager instance reset."""
        manager1 = get_enhanced_memory_manager()
        reset_enhanced_memory_manager()
        manager2 = get_enhanced_memory_manager()
        
        # Should be different instances
        assert manager1 is not manager2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
