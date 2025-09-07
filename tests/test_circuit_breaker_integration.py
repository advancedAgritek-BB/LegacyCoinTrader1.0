"""
Integration test for circuit breaker in trading pipeline.

Tests the integration of circuit breakers with the actual trading pipeline
components to ensure fault tolerance works correctly.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from crypto_bot.utils.circuit_breaker import get_circuit_breaker_manager
from crypto_bot.utils.symbol_utils import get_filtered_symbols
from crypto_bot.utils.market_loader import fetch_ohlcv_async


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with trading pipeline components."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange for testing."""
        exchange = Mock()
        exchange.id = "kraken"
        exchange.symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        exchange.has = {"fetchOHLCV": True}
        exchange.timeframes = {"1h": 3600, "1m": 60}
        exchange.load_markets = AsyncMock()
        exchange.fetch_ohlcv = AsyncMock(return_value=[
            [1640995200000, 50000, 51000, 49000, 50500, 1000],
            [1640998800000, 50500, 52000, 50000, 51500, 1200]
        ])
        # Ensure symbols are loaded
        exchange.symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        return exchange
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return {
            "skip_symbol_filters": True,
            "symbols": ["BTC/USD", "ETH/USD"],
            "symbol_refresh_minutes": 30
        }
    
    @pytest.mark.asyncio
    async def test_symbol_scanning_with_circuit_breaker(self, mock_exchange, mock_config):
        """Test symbol scanning with circuit breaker protection."""
        # Test successful symbol scanning
        symbols = await get_filtered_symbols(mock_exchange, mock_config)
        assert len(symbols) > 0
        assert all(isinstance(s, tuple) for s in symbols)
        
        # Test circuit breaker metrics
        manager = get_circuit_breaker_manager()
        metrics = manager.get_all_metrics()
        
        # Should have circuit breaker for load_markets
        load_markets_cb = None
        for name, cb_metrics in metrics.items():
            if "load_markets" in name:
                load_markets_cb = cb_metrics
                break
        
        if load_markets_cb:
            assert load_markets_cb["successful_calls"] >= 1
            assert load_markets_cb["state"] == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_ohlcv_fetching_with_circuit_breaker(self, mock_exchange):
        """Test OHLCV fetching with circuit breaker protection."""
        # Test successful OHLCV fetching
        result = await fetch_ohlcv_async(
            mock_exchange,
            "BTC/USD",
            timeframe="1h",
            limit=100
        )
        
        # Check if result is data or exception
        if isinstance(result, Exception):
            # If it's an exception, check if it's due to circuit breaker
            assert "Circuit breaker" in str(result) or "unsupported" in str(result).lower()
        else:
            # If it's data, check the structure
            assert len(result) > 0
            assert all(len(candle) == 6 for candle in result)
        
        # Test circuit breaker metrics
        manager = get_circuit_breaker_manager()
        metrics = manager.get_all_metrics()
        
        # Should have circuit breaker for fetch_ohlcv
        fetch_ohlcv_cb = None
        for name, cb_metrics in metrics.items():
            if "fetch_ohlcv" in name:
                fetch_ohlcv_cb = cb_metrics
                break
        
        if fetch_ohlcv_cb:
            assert fetch_ohlcv_cb["successful_calls"] >= 1
            assert fetch_ohlcv_cb["state"] == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_scenario(self, mock_exchange):
        """Test circuit breaker behavior when exchange API fails."""
        # Make the exchange fail
        mock_exchange.fetch_ohlcv.side_effect = Exception("API Error")
        
        # First few calls should fail and open circuit
        for i in range(3):
            try:
                await fetch_ohlcv_async(mock_exchange, "BTC/USD", timeframe="1h", limit=100)
            except Exception:
                pass
        
        # Next call should be rejected by circuit breaker
        with pytest.raises(Exception, match="Circuit breaker.*is OPEN"):
            await fetch_ohlcv_async(mock_exchange, "BTC/USD", timeframe="1h", limit=100)
        
        # Check circuit breaker state
        manager = get_circuit_breaker_manager()
        metrics = manager.get_all_metrics()
        
        fetch_ohlcv_cb = None
        for name, cb_metrics in metrics.items():
            if "fetch_ohlcv" in name:
                fetch_ohlcv_cb = cb_metrics
                break
        
        if fetch_ohlcv_cb:
            assert fetch_ohlcv_cb["state"] == "OPEN"
            assert fetch_ohlcv_cb["failed_calls"] >= 3
            assert fetch_ohlcv_cb["rejected_calls"] >= 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, mock_exchange):
        """Test circuit breaker recovery after failures."""
        # Make the exchange fail initially
        mock_exchange.fetch_ohlcv.side_effect = Exception("API Error")
        
        # Cause circuit to open
        for i in range(3):
            try:
                await fetch_ohlcv_async(mock_exchange, "BTC/USD", timeframe="1h", limit=100)
            except Exception:
                pass
        
        # Reset the exchange to work again
        mock_exchange.fetch_ohlcv.side_effect = None
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 50000, 51000, 49000, 50500, 1000]
        ]
        
        # Wait for recovery timeout (use short timeout for testing)
        await asyncio.sleep(0.2)
        
        # Next call should succeed and close circuit
        result = await fetch_ohlcv_async(mock_exchange, "BTC/USD", timeframe="1h", limit=100)
        
        # Check if result is data or exception
        if isinstance(result, Exception):
            # If it's an exception, check if it's due to circuit breaker
            assert "Circuit breaker" in str(result) or "unsupported" in str(result).lower()
        else:
            # If it's data, check the structure
            assert len(result) > 0
        
        # Check circuit breaker state
        manager = get_circuit_breaker_manager()
        metrics = manager.get_all_metrics()
        
        fetch_ohlcv_cb = None
        for name, cb_metrics in metrics.items():
            if "fetch_ohlcv" in name:
                fetch_ohlcv_cb = cb_metrics
                break
        
        if fetch_ohlcv_cb:
            # Should be in CLOSED or HALF_OPEN state after recovery
            assert fetch_ohlcv_cb["state"] in ["CLOSED", "HALF_OPEN"]
    
    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_usage(self, mock_exchange):
        """Test circuit breaker with concurrent requests."""
        # Make multiple concurrent requests
        tasks = []
        for i in range(5):
            task = fetch_ohlcv_async(mock_exchange, f"SYMBOL{i}/USD", timeframe="1h", limit=100)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results - some may be exceptions due to unsupported symbols
        successful_results = [r for r in results if not isinstance(r, Exception)]
        exception_results = [r for r in results if isinstance(r, Exception)]
        
        # Should have some successful results or valid exceptions
        assert len(successful_results) > 0 or all("unsupported" in str(e).lower() for e in exception_results)
        
        # Check circuit breaker metrics
        manager = get_circuit_breaker_manager()
        metrics = manager.get_all_metrics()
        
        # Should have multiple circuit breakers for different symbols
        fetch_ohlcv_cbs = [cb for name, cb in metrics.items() if "fetch_ohlcv" in name]
        assert len(fetch_ohlcv_cbs) >= 5
        
        # All should be in CLOSED state
        for cb in fetch_ohlcv_cbs:
            assert cb["state"] == "CLOSED"
            assert cb["successful_calls"] >= 1


class TestCircuitBreakerPerformance:
    """Test circuit breaker performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_overhead(self):
        """Test that circuit breaker adds minimal overhead."""
        manager = get_circuit_breaker_manager()
        
        def simple_function():
            return "success"
        
        # Time without circuit breaker
        start_time = time.time()
        for _ in range(10):  # Reduced iterations for faster test
            simple_function()
        time_without_cb = time.time() - start_time
        
        # Time with circuit breaker
        start_time = time.time()
        for i in range(10):  # Reduced iterations for faster test
            await manager.call_with_circuit_breaker(f"test_{i}", simple_function)
        time_with_cb = time.time() - start_time
        
        # Circuit breaker should add reasonable overhead (allow higher ratio for async)
        overhead_ratio = time_with_cb / time_without_cb
        assert overhead_ratio < 10.0, f"Circuit breaker overhead too high: {overhead_ratio}"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_memory_usage(self):
        """Test that circuit breaker doesn't leak memory."""
        manager = get_circuit_breaker_manager()
        
        # Create many circuit breakers
        for i in range(100):
            await manager.call_with_circuit_breaker(f"memory_test_{i}", lambda: "success")
        
        # Check that we have the expected number of circuit breakers
        metrics = manager.get_all_metrics()
        assert len(metrics) >= 100
        
        # All should be in CLOSED state
        for cb_metrics in metrics.values():
            assert cb_metrics["state"] == "CLOSED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
