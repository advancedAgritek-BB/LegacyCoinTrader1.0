"""Integration tests for performance enhancements."""

import pytest
import asyncio
import time
import psutil
from unittest.mock import Mock, patch, MagicMock
from collections import deque
import tempfile
import json
from pathlib import Path

from crypto_bot.main import MemoryManager
from crypto_bot.utils.market_loader import AdaptiveRateLimiter, get_rate_limiter, configure_rate_limiter
from crypto_bot.utils.scan_cache_manager import AdaptiveCacheManager, get_cache_manager, configure_cache_manager
from crypto_bot.utils.telemetry import PerformanceMonitor, performance_monitor
from crypto_bot.utils.database import DatabaseManager


class TestMemoryManager:
    """Test MemoryManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory_manager = MemoryManager(memory_threshold=0.8)
    
    @patch('crypto_bot.main.psutil.virtual_memory')
    def test_check_memory_pressure_normal(self, mock_virtual_memory):
        """Test memory pressure check under normal conditions."""
        # Mock normal memory usage (60%)
        mock_virtual_memory.return_value.percent = 60.0
        
        result = self.memory_manager.check_memory_pressure()
        assert result is False
    
    @patch('crypto_bot.main.psutil.virtual_memory')
    def test_check_memory_pressure_high(self, mock_virtual_memory):
        """Test memory pressure check under high memory usage."""
        # Mock high memory usage (85%)
        mock_virtual_memory.return_value.percent = 85.0
        
        result = self.memory_manager.check_memory_pressure()
        assert result is True
    
    @patch('crypto_bot.main.psutil.virtual_memory')
    def test_optimize_cache_sizes(self, mock_virtual_memory):
        """Test cache size optimization."""
        # Mock high memory usage (90%)
        mock_virtual_memory.return_value.percent = 90.0
        
        # Create test caches
        df_cache = {
            "1h": deque([f"symbol_{i}" for i in range(600)], maxlen=500),
            "4h": deque([f"symbol_{i}" for i in range(400)], maxlen=500)
        }
        regime_cache = {
            "1h": deque([f"regime_{i}" for i in range(300)], maxlen=500)
        }
        
        initial_sizes = {tf: len(cache) for tf, cache in df_cache.items()}
        
        self.memory_manager.optimize_cache_sizes(df_cache, regime_cache)
        
        # Verify cache sizes were reduced
        for tf, cache in df_cache.items():
            assert len(cache) < initial_sizes[tf]
    
    @patch('crypto_bot.main.gc.collect')
    def test_force_garbage_collection(self, mock_gc_collect):
        """Test forced garbage collection."""
        mock_gc_collect.return_value = 100
        
        self.memory_manager.force_garbage_collection()
        mock_gc_collect.assert_called_once()
    
    def test_get_memory_stats(self):
        """Test memory statistics retrieval."""
        stats = self.memory_manager.get_memory_stats()
        
        assert "total_mb" in stats
        assert "available_mb" in stats
        assert "used_mb" in stats
        assert "percent" in stats
        assert "optimization_count" in stats


class TestAdaptiveRateLimiter:
    """Test AdaptiveRateLimiter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rate_limiter = AdaptiveRateLimiter(
            max_requests_per_minute=10,
            base_delay=1.0,
            max_delay=10.0
        )
    
    @patch('time.time')
    async def test_wait_if_needed_normal_rate(self, mock_time):
        """Test rate limiting under normal request rates."""
        # Mock time progression
        mock_time.side_effect = [1000.0, 1001.0, 1002.0]
        
        # First request - should not wait
        start_time = time.time()
        await self.rate_limiter.wait_if_needed()
        end_time = time.time()
        
        # Should not wait significantly
        assert end_time - start_time < 0.1
    
    @patch('time.time')
    async def test_wait_if_needed_high_rate(self, mock_time):
        """Test rate limiting under high request rates."""
        # Mock rapid requests
        times = [1000.0 + i * 0.1 for i in range(15)]  # 15 requests in 1.4 seconds
        mock_time.side_effect = times
        
        # Make multiple rapid requests
        for i in range(10):
            await self.rate_limiter.wait_if_needed()
        
        # The 11th request should trigger rate limiting
        start_time = time.time()
        await self.rate_limiter.wait_if_needed()
        end_time = time.time()
        
        # Should have waited due to rate limiting
        assert end_time - start_time > 0.5
    
    def test_record_error_exponential_backoff(self):
        """Test exponential backoff on errors."""
        initial_delay = self.rate_limiter.current_delay
        
        # Record multiple errors
        for i in range(3):
            self.rate_limiter.record_error()
        
        # Delay should increase exponentially
        assert self.rate_limiter.current_delay > initial_delay
        assert self.rate_limiter.consecutive_errors == 3
    
    def test_record_success_recovery(self):
        """Test delay recovery on successful requests."""
        # First create some errors to increase delay
        self.rate_limiter.record_error()
        self.rate_limiter.record_error()
        high_delay = self.rate_limiter.current_delay
        
        # Record success
        self.rate_limiter.record_success()
        
        # Delay should decrease
        assert self.rate_limiter.current_delay < high_delay
        assert self.rate_limiter.consecutive_errors == 0
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        # Record some activity
        self.rate_limiter.record_error()
        self.rate_limiter.record_success()
        
        stats = self.rate_limiter.get_stats()
        
        assert "total_requests" in stats
        assert "total_errors" in stats
        assert "error_rate" in stats
        assert "consecutive_errors" in stats
        assert "current_delay" in stats


class TestAdaptiveCacheManager:
    """Test AdaptiveCacheManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = AdaptiveCacheManager(
            initial_size=100,
            max_size=1000,
            min_size=50
        )
    
    def test_get_cache_size_high_hit_rate(self):
        """Test cache size increase for high hit rates."""
        cache_type = "ohlcv"
        
        # Simulate high hit rate
        for i in range(100):
            self.cache_manager.hit_rates[cache_type].append(True)
        
        size = self.cache_manager.get_cache_size(cache_type)
        assert size > self.cache_manager.initial_size
    
    def test_get_cache_size_low_hit_rate(self):
        """Test cache size decrease for low hit rates."""
        cache_type = "regime"
        
        # Simulate low hit rate
        for i in range(100):
            self.cache_manager.hit_rates[cache_type].append(False)
        
        size = self.cache_manager.get_cache_size(cache_type)
        assert size < self.cache_manager.initial_size
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        cache_type = "test"
        
        # Test set and get
        self.cache_manager.set(cache_type, "key1", "value1")
        result = self.cache_manager.get(cache_type, "key1")
        assert result == "value1"
        
        # Test cache hit tracking
        stats = self.cache_manager.get_stats(cache_type)
        assert stats["total_hits"] == 1
        assert stats["total_accesses"] == 1
    
    def test_cache_eviction(self):
        """Test cache eviction when size limit is reached."""
        cache_type = "test"
        
        # Fill cache beyond limit
        for i in range(150):
            self.cache_manager.set(cache_type, f"key_{i}", f"value_{i}")
        
        # Verify cache size is within limits
        stats = self.cache_manager.get_stats(cache_type)
        assert stats["entries"] <= self.cache_manager.get_cache_size(cache_type)
    
    def test_cache_invalidation(self):
        """Test cache entry invalidation."""
        cache_type = "test"
        
        self.cache_manager.set(cache_type, "key1", "value1")
        assert self.cache_manager.get(cache_type, "key1") == "value1"
        
        # Invalidate entry
        assert self.cache_manager.invalidate(cache_type, "key1") is True
        assert self.cache_manager.get(cache_type, "key1") is None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache_type = "test"
        
        # Add some entries
        self.cache_manager.set(cache_type, "key1", "value1")
        self.cache_manager.set(cache_type, "key2", "value2")
        
        # Clear cache
        self.cache_manager.clear(cache_type)
        
        # Verify cache is empty
        assert self.cache_manager.get(cache_type, "key1") is None
        assert self.cache_manager.get(cache_type, "key2") is None


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(
            memory_threshold=85.0,
            error_rate_threshold=10.0,
            response_time_threshold=5.0,
            cache_hit_rate_threshold=50.0
        )
    
    def test_record_metric(self):
        """Test metric recording."""
        # Record various metrics
        self.monitor.record_metric("memory", 75.5)
        self.monitor.record_metric("response_time", 2.5)
        self.monitor.record_metric("cache_hit", 85.0, cache_type="ohlcv")
        self.monitor.record_metric("error", 5.0)
        
        # Verify metrics were recorded
        assert len(self.monitor.memory_usage) == 1
        assert len(self.monitor.api_response_times) == 1
        assert len(self.monitor.cache_hit_rates["ohlcv"]) == 1
        assert len(self.monitor.error_rates) == 1
    
    def test_get_performance_report(self):
        """Test performance report generation."""
        # Record some metrics
        self.monitor.record_metric("memory", 80.0)
        self.monitor.record_metric("response_time", 3.0)
        self.monitor.record_metric("cache_hit", 70.0, cache_type="ohlcv")
        
        report = self.monitor.get_performance_report()
        
        assert "timestamp" in report
        assert "memory" in report
        assert "api_response_times" in report
        assert "cache_performance" in report
        assert "alerts" in report
        assert "summary" in report
    
    def test_alert_conditions_memory(self):
        """Test memory usage alerts."""
        # Record high memory usage
        self.monitor.record_metric("memory", 90.0)
        
        alerts = self.monitor.get_alert_conditions()
        
        # Should have memory alert
        memory_alerts = [a for a in alerts if a["type"] == "high_memory_usage"]
        assert len(memory_alerts) > 0
        assert memory_alerts[0]["severity"] == "warning"
    
    def test_alert_conditions_error_rate(self):
        """Test error rate alerts."""
        # Record high error rate
        self.monitor.record_metric("error", 15.0)
        
        alerts = self.monitor.get_alert_conditions()
        
        # Should have error rate alert
        error_alerts = [a for a in alerts if a["type"] == "high_error_rate"]
        assert len(error_alerts) > 0
        assert error_alerts[0]["severity"] == "error"
    
    def test_alert_conditions_response_time(self):
        """Test response time alerts."""
        # Record slow response time
        self.monitor.record_metric("response_time", 6.0)
        
        alerts = self.monitor.get_alert_conditions()
        
        # Should have response time alert
        response_alerts = [a for a in alerts if a["type"] == "slow_response_time"]
        assert len(response_alerts) > 0
        assert response_alerts[0]["severity"] == "warning"
    
    def test_export_metrics_json(self):
        """Test metrics export to JSON."""
        # Record some metrics
        self.monitor.record_metric("memory", 75.0)
        self.monitor.record_metric("response_time", 2.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.monitor.export_metrics("json", temp_path)
            
            # Verify file was created and contains valid JSON
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "memory" in data
            assert "api_response_times" in data
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_export_metrics_csv(self):
        """Test metrics export to CSV."""
        # Record some metrics
        self.monitor.record_metric("memory", 75.0)
        self.monitor.record_metric("response_time", 2.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            self.monitor.export_metrics("csv", temp_path)
            
            # Verify file was created
            assert Path(temp_path).exists()
            assert Path(temp_path).stat().st_size > 0
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_password",
            "min_size": 2,
            "max_size": 5,
            "command_timeout": 10
        }
    
    @pytest.mark.asyncio
    async def test_database_manager_initialization(self):
        """Test database manager initialization."""
        # This test would require a mock database or test database
        # For now, we'll test the configuration parsing
        db_manager = DatabaseManager(**self.db_config)
        
        assert db_manager.host == "localhost"
        assert db_manager.port == 5432
        assert db_manager.database == "test_db"
        assert db_manager.min_size == 2
        assert db_manager.max_size == 5
        assert db_manager.command_timeout == 10
    
    @pytest.mark.asyncio
    async def test_connection_pool_management(self):
        """Test connection pool management."""
        # Mock asyncpg connection pool
        with patch('crypto_bot.utils.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = Mock()
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager(**self.db_config)
            await db_manager.initialize()
            
            mock_create_pool.assert_called_once()
            assert db_manager.pool == mock_pool
    
    @pytest.mark.asyncio
    async def test_connection_context_manager(self):
        """Test connection context manager."""
        # Mock connection pool and connection
        mock_pool = Mock()
        mock_connection = Mock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        
        db_manager = DatabaseManager(**self.db_config)
        db_manager.pool = mock_pool
        
        async with db_manager.get_connection() as conn:
            assert conn == mock_connection
        
        mock_pool.acquire.assert_called_once()


class TestPerformanceIntegration:
    """Integration tests for performance components working together."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory_manager = MemoryManager()
        self.rate_limiter = AdaptiveRateLimiter()
        self.cache_manager = AdaptiveCacheManager()
        self.performance_monitor = PerformanceMonitor()
    
    def test_memory_pressure_integration(self):
        """Test memory pressure affecting multiple components."""
        with patch('crypto_bot.main.psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.percent = 90.0
            
            # Simulate memory pressure
            if self.memory_manager.check_memory_pressure():
                # Should trigger cache optimization
                df_cache = {"1h": deque([f"symbol_{i}" for i in range(600)])}
                regime_cache = {"1h": deque([f"regime_{i}" for i in range(400)])}
                
                self.memory_manager.optimize_cache_sizes(df_cache, regime_cache)
                
                # Should record memory metric
                self.performance_monitor.record_metric("memory", 90.0)
                
                # Should generate alert
                alerts = self.performance_monitor.get_alert_conditions()
                assert len(alerts) > 0
    
    def test_rate_limiting_integration(self):
        """Test rate limiting integration with performance monitoring."""
        # Simulate high request rate
        for i in range(15):
            asyncio.run(self.rate_limiter.wait_if_needed())
        
        # Record some errors
        self.rate_limiter.record_error()
        self.rate_limiter.record_error()
        
        # Should affect performance metrics
        self.performance_monitor.record_metric("error", 13.3)  # 2/15 * 100
        
        # Should generate alert
        alerts = self.performance_monitor.get_alert_conditions()
        error_alerts = [a for a in alerts if a["type"] == "high_error_rate"]
        assert len(error_alerts) > 0
    
    def test_cache_performance_integration(self):
        """Test cache performance integration."""
        cache_type = "ohlcv"
        
        # Simulate cache usage patterns
        for i in range(100):
            if i < 80:  # 80% hit rate
                self.cache_manager.get(cache_type, f"key_{i}")
                self.cache_manager.set(cache_type, f"key_{i}", f"value_{i}")
            else:  # 20% miss rate
                self.cache_manager.get(cache_type, f"missing_key_{i}")
        
        # Check cache size adjustment
        size = self.cache_manager.get_cache_size(cache_type)
        assert size > self.cache_manager.initial_size  # Should increase due to high hit rate
        
        # Record cache performance metrics
        stats = self.cache_manager.get_stats(cache_type)
        hit_rate = stats["hit_rate"]
        self.performance_monitor.record_metric("cache_hit", hit_rate, cache_type=cache_type)
        
        # Should not generate cache hit rate alert (hit rate > 50%)
        alerts = self.performance_monitor.get_alert_conditions()
        cache_alerts = [a for a in alerts if a["type"] == "low_cache_hit_rate"]
        assert len(cache_alerts) == 0


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for the enhancements."""
    
    def test_memory_manager_performance(self, benchmark):
        """Benchmark memory manager operations."""
        memory_manager = MemoryManager()
        
        def check_memory_pressure():
            return memory_manager.check_memory_pressure()
        
        result = benchmark(check_memory_pressure)
        assert isinstance(result, bool)
    
    def test_rate_limiter_performance(self, benchmark):
        """Benchmark rate limiter operations."""
        rate_limiter = AdaptiveRateLimiter()
        
        def record_metrics():
            rate_limiter.record_success()
            rate_limiter.record_error()
            return rate_limiter.get_current_delay()
        
        result = benchmark(record_metrics)
        assert isinstance(result, float)
    
    def test_cache_manager_performance(self, benchmark):
        """Benchmark cache manager operations."""
        cache_manager = AdaptiveCacheManager()
        
        def cache_operations():
            cache_manager.set("test", "key1", "value1")
            cache_manager.get("test", "key1")
            return cache_manager.get_cache_size("test")
        
        result = benchmark(cache_operations)
        assert isinstance(result, int)
    
    def test_performance_monitor_metrics(self, benchmark):
        """Benchmark performance monitor operations."""
        monitor = PerformanceMonitor()
        
        def record_metrics():
            monitor.record_metric("memory", 75.0)
            monitor.record_metric("response_time", 2.0)
            return monitor.get_performance_report()
        
        result = benchmark(record_metrics)
        assert isinstance(result, dict)
        assert "memory" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
