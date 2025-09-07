"""
End-to-End Test for Enhanced Memory Management System

This test verifies that the entire system works correctly with the new memory management.
"""

import pytest
import asyncio
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from crypto_bot.phase_runner import BotContext, PhaseRunner
from crypto_bot.utils.enhanced_memory_manager import get_enhanced_memory_manager, reset_enhanced_memory_manager


class TestEndToEndMemoryManagement:
    """End-to-end test for the enhanced memory management system."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_enhanced_memory_manager()
        
        self.config = {
            "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
            "timeframe": "1h",
            "execution_mode": "dry_run",
            "memory_threshold": 0.8,
            "gc_threshold": 0.7,
            "cache_size_limit_mb": 100,  # Small limit for testing
            "enable_background_cleanup": False
        }
        
        # Create test data
        self.df_cache = {}
        self.regime_cache = {}
        
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
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_enhanced_memory_manager()
    
    @pytest.mark.asyncio
    async def test_complete_trading_cycle_with_memory_management(self):
        """Test a complete trading cycle with memory management."""
        
        # Create BotContext with memory management
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Verify memory manager is initialized
        assert ctx.memory_manager is not None
        assert len(ctx.managed_caches) > 0
        
        # Create trading phases
        async def data_collection_phase(ctx):
            """Simulate data collection phase."""
            # Add new data to cache
            df_cache = ctx.get_managed_cache("df_cache")
            new_data = pd.DataFrame({
                'close': np.random.randn(100),
                'volume': np.random.randn(100)
            })
            df_cache.put("NEW/USD", new_data)
            await asyncio.sleep(0.1)
        
        async def analysis_phase(ctx):
            """Simulate analysis phase."""
            # Register ML model
            mock_model = Mock()
            ctx.register_ml_model("test_model", mock_model, size_estimate_mb=10)
            ctx.update_ml_model_usage("test_model")
            await asyncio.sleep(0.1)
        
        async def trading_phase(ctx):
            """Simulate trading phase."""
            # Simulate trading decisions
            positions = {"BTC/USD": {"size": 0.1, "entry_price": 50000}}
            ctx.positions = positions
            await asyncio.sleep(0.1)
        
        # Create phase runner
        runner = PhaseRunner([data_collection_phase, analysis_phase, trading_phase])
        
        # Run phases
        timings = await runner.run(ctx)
        
        # Verify all phases completed
        assert "data_collection_phase" in timings
        assert "analysis_phase" in timings
        assert "trading_phase" in timings
        
        # Verify memory management worked
        memory_stats = ctx.get_memory_stats()
        assert memory_stats["caches_count"] > 0
        assert memory_stats["ml_models_count"] > 0
        
        # Verify cache functionality
        df_cache = ctx.get_managed_cache("df_cache")
        assert df_cache.get("NEW/USD") is not None
        
        # Verify ML model registration
        assert "test_model" in ctx.memory_manager.ml_models
        
        # Verify positions
        assert len(ctx.positions) > 0
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test memory pressure handling during trading."""
        
        # Create context
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Create memory-intensive phase
        async def memory_intensive_phase(ctx):
            """Phase that uses a lot of memory."""
            # Create large data structures
            large_data = []
            for i in range(100):
                large_data.append(pd.DataFrame({
                    'data': np.random.randn(1000)
                }))
            
            # Add to cache
            df_cache = ctx.get_managed_cache("df_cache")
            for i, df in enumerate(large_data):
                df_cache.put(f"large_data_{i}", df)
            
            await asyncio.sleep(0.1)
            return large_data
        
        # Mock memory pressure
        with patch.object(ctx.memory_manager, 'check_memory_pressure', return_value=True):
            runner = PhaseRunner([memory_intensive_phase])
            timings = await runner.run(ctx)
            
            # Should complete without error
            assert "memory_intensive_phase" in timings
    
    def test_memory_cleanup_effectiveness(self):
        """Test that memory cleanup is effective."""
        
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Get initial memory stats
        initial_stats = ctx.get_memory_stats()
        
        # Perform memory-intensive operations
        df_cache = ctx.get_managed_cache("df_cache")
        for i in range(50):
            large_df = pd.DataFrame({
                'data': np.random.randn(1000)
            })
            df_cache.put(f"test_data_{i}", large_df)
        
        # Perform maintenance
        maintenance_results = ctx.perform_memory_maintenance()
        
        # Get final memory stats
        final_stats = ctx.get_memory_stats()
        
        # Verify maintenance was performed
        assert "timestamp" in maintenance_results
        assert "cache_cleanup" in maintenance_results
        
        # Verify cache size is within limits
        cache_stats = final_stats["caches"]["df_cache"]
        assert cache_stats["size"] <= 1000  # max_cache_entries
    
    def test_backward_compatibility(self):
        """Test that existing code still works."""
        
        # Create context with memory management
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Test original cache access still works
        assert "BTC/USD" in ctx.df_cache
        assert len(ctx.df_cache["BTC/USD"]) == 1000
        
        # Test original position access
        ctx.positions["BTC/USD"] = {"size": 0.1, "entry_price": 50000}
        assert len(ctx.positions) == 1
        
        # Test original methods still work
        assert ctx.get_position_count() == 1
        assert "BTC/USD" in ctx.get_position_symbols()
    
    def test_memory_manager_global_instance(self):
        """Test global memory manager instance management."""
        
        # Get global instance
        manager1 = get_enhanced_memory_manager()
        manager2 = get_enhanced_memory_manager()
        
        # Should be the same instance
        assert manager1 is manager2
        
        # Reset global instance
        reset_enhanced_memory_manager()
        
        # Get new instance
        manager3 = get_enhanced_memory_manager()
        
        # Should be different
        assert manager1 is not manager3


class TestProductionReadiness:
    """Test production readiness of the memory management system."""
    
    def test_configuration_validation(self):
        """Test that configuration is properly validated."""
        
        # Test with valid config
        valid_config = {
            "memory_threshold": 0.8,
            "gc_threshold": 0.7,
            "cache_size_limit_mb": 500,
            "enable_background_cleanup": False
        }
        
        ctx = BotContext(
            positions={},
            df_cache={},
            regime_cache={},
            config=valid_config
        )
        
        assert ctx.memory_manager is not None
        
        # Test with invalid config (should still work with defaults)
        invalid_config = {
            "memory_threshold": "invalid",
            "gc_threshold": "invalid"
        }
        
        ctx2 = BotContext(
            positions={},
            df_cache={},
            regime_cache={},
            config=invalid_config
        )
        
        # Should still initialize with default values
        assert ctx2.memory_manager is not None
    
    def test_error_handling(self):
        """Test error handling in memory management."""
        
        ctx = BotContext(
            positions={},
            df_cache={},
            regime_cache={},
            config={"enable_background_cleanup": False}
        )
        
        # Test memory stats with errors
        with patch('psutil.virtual_memory', side_effect=Exception("Test error")):
            stats = ctx.get_memory_stats()
            assert "error" in stats
        
        # Test maintenance with errors
        with patch.object(ctx.memory_manager, 'check_memory_pressure', side_effect=Exception("Test error")):
            results = ctx.perform_memory_maintenance()
            assert "timestamp" in results
    
    def test_performance_characteristics(self):
        """Test performance characteristics of memory management."""
        
        ctx = BotContext(
            positions={},
            df_cache={},
            regime_cache={},
            config={"enable_background_cleanup": False}
        )
        
        # Test memory stats performance
        start_time = time.time()
        stats = ctx.get_memory_stats()
        stats_time = time.time() - start_time
        
        # Should be fast (< 100ms)
        assert stats_time < 0.1
        
        # Test maintenance performance
        start_time = time.time()
        maintenance = ctx.perform_memory_maintenance()
        maintenance_time = time.time() - start_time
        
        # Should be fast (< 100ms)
        assert maintenance_time < 0.1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
