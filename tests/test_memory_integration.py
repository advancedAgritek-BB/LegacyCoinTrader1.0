"""
Test enhanced memory manager integration with BotContext and PhaseRunner.
"""

import pytest
import asyncio
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from crypto_bot.phase_runner import BotContext, PhaseRunner
from crypto_bot.utils.enhanced_memory_manager import EnhancedMemoryManager, MemoryConfig


class TestBotContextMemoryIntegration:
    """Test BotContext integration with enhanced memory manager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = {
            "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
            "timeframe": "1h",
            "execution_mode": "dry_run",
            "memory_threshold": 0.8,
            "gc_threshold": 0.7,
            "cache_size_limit_mb": 500,
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
    
    def test_bot_context_memory_manager_initialization(self):
        """Test that BotContext initializes memory manager correctly."""
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Memory manager should be initialized
        assert ctx.memory_manager is not None
        assert isinstance(ctx.memory_manager, EnhancedMemoryManager)
        
        # Managed caches should be created
        assert "df_cache" in ctx.managed_caches
        assert "regime_cache" in ctx.managed_caches
        
        # Original caches should still exist for backward compatibility
        assert len(ctx.df_cache) == 3
        assert len(ctx.regime_cache) == 3
    
    def test_managed_cache_functionality(self):
        """Test that managed caches work correctly."""
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Get managed caches
        df_cache = ctx.get_managed_cache("df_cache")
        regime_cache = ctx.get_managed_cache("regime_cache")
        
        assert df_cache is not None
        assert regime_cache is not None
        
        # Test cache operations
        cached_df = df_cache.get("BTC/USD")
        assert cached_df is not None
        assert len(cached_df) == 1000
        
        # Test adding new data
        new_df = pd.DataFrame({'close': np.random.randn(500)})
        df_cache.put("NEW/USD", new_df)
        
        retrieved_df = df_cache.get("NEW/USD")
        assert retrieved_df is not None
        assert len(retrieved_df) == 500
    
    def test_ml_model_registration(self):
        """Test ML model registration with memory manager."""
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Create mock ML model
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.scaler = Mock()
        
        # Register model
        ctx.register_ml_model("test_model", mock_model, size_estimate_mb=50)
        
        # Verify registration
        assert "test_model" in ctx.memory_manager.ml_models
        
        # Update usage
        ctx.update_ml_model_usage("test_model")
        
        # Verify usage was updated
        model_info = ctx.memory_manager.ml_models["test_model"]
        assert model_info["last_used"] > time.time() - 1  # Should be recent
    
    def test_memory_maintenance(self):
        """Test memory maintenance functionality."""
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Perform maintenance
        maintenance_results = ctx.perform_memory_maintenance()
        
        # Should return results
        assert "timestamp" in maintenance_results
        assert "cache_cleanup" in maintenance_results
        assert "ml_cleanup" in maintenance_results
        assert "gc_collected" in maintenance_results
    
    def test_memory_stats(self):
        """Test memory statistics collection."""
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache=self.regime_cache,
            config=self.config
        )
        
        # Get memory stats
        stats = ctx.get_memory_stats()
        
        # Should contain expected fields
        assert "system_memory_percent" in stats
        assert "process_memory_mb" in stats
        assert "caches_count" in stats
        assert "ml_models_count" in stats
        assert "timestamp" in stats
        
        # Should have cache-specific stats
        assert "caches" in stats
        assert "df_cache" in stats["caches"]
        assert "regime_cache" in stats["caches"]


class TestPhaseRunnerMemoryIntegration:
    """Test PhaseRunner integration with memory monitoring."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = {
            "symbols": ["BTC/USD"],
            "timeframe": "1h",
            "execution_mode": "dry_run",
            "enable_background_cleanup": False
        }
        
        self.df_cache = {
            "BTC/USD": pd.DataFrame({
                'close': np.random.randn(100),
                'volume': np.random.randn(100)
            })
        }
    
    @pytest.mark.asyncio
    async def test_phase_runner_memory_monitoring(self):
        """Test that PhaseRunner monitors memory during phase execution."""
        # Create a test phase
        async def test_phase(ctx):
            # Simulate some work
            await asyncio.sleep(0.1)
            # Create some data
            data = [i for i in range(10000)]
            return data
        
        # Create context with memory manager
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache={},
            config=self.config
        )
        
        # Create phase runner
        runner = PhaseRunner([test_phase])
        
        # Run phases
        timings = await runner.run(ctx)
        
        # Should have timing information
        assert "test_phase" in timings
        assert timings["test_phase"] > 0
        
        # Memory manager should be available
        assert ctx.memory_manager is not None
    
    @pytest.mark.asyncio
    async def test_phase_runner_memory_pressure_handling(self):
        """Test that PhaseRunner handles memory pressure correctly."""
        # Create a phase that might cause memory pressure
        async def memory_intensive_phase(ctx):
            # Create large data structures
            large_data = []
            for i in range(1000):
                large_data.append(pd.DataFrame({
                    'data': np.random.randn(1000)
                }))
            return large_data
        
        # Create context first
        ctx = BotContext(
            positions={},
            df_cache=self.df_cache,
            regime_cache={},
            config=self.config
        )
        
        # Mock memory pressure
        with patch.object(ctx.memory_manager, 'check_memory_pressure', return_value=True):
            runner = PhaseRunner([memory_intensive_phase])
            
            # Run phases
            timings = await runner.run(ctx)
            
            # Should complete without error
            assert "memory_intensive_phase" in timings


class TestMemoryManagerBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_existing_cache_access(self):
        """Test that existing cache access still works."""
        ctx = BotContext(
            positions={},
            df_cache={"BTC/USD": pd.DataFrame({'close': [1, 2, 3]})},
            regime_cache={},
            config={"enable_background_cleanup": False}
        )
        
        # Original cache access should still work
        assert "BTC/USD" in ctx.df_cache
        assert len(ctx.df_cache["BTC/USD"]) == 3
        
        # Managed cache should also have the data
        df_cache = ctx.get_managed_cache("df_cache")
        cached_df = df_cache.get("BTC/USD")
        assert cached_df is not None
        assert len(cached_df) == 3
    
    def test_memory_manager_fallback(self):
        """Test fallback when memory manager fails to initialize."""
        # Create config that will cause memory manager to fail
        problematic_config = {
            "symbols": ["BTC/USD"],
            "timeframe": "1h",
            "execution_mode": "dry_run",
            "memory_threshold": "invalid"  # This should cause an error
        }
        
        # Should not crash, just log warning
        with patch('crypto_bot.phase_runner.logger') as mock_logger:
            ctx = BotContext(
                positions={},
                df_cache={},
                regime_cache={},
                config=problematic_config
            )
            
            # Memory manager should still be initialized (it handles invalid config gracefully)
            assert ctx.memory_manager is not None
            
            # Other functionality should still work
            assert ctx.get_position_count() == 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
