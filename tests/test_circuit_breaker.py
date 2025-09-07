"""
Comprehensive tests for Circuit Breaker implementation.

Tests cover all circuit breaker states, transitions, and edge cases
to ensure robust fault tolerance in the trading pipeline.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from crypto_bot.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitState,
    get_circuit_breaker_manager,
    EXCHANGE_API_CONFIG,
    WEBHOOK_CONFIG,
    DATABASE_CONFIG
)


def raise_exception(msg):
    """Helper function to raise exceptions in lambda expressions."""
    raise Exception(msg)


class TestCircuitBreaker:
    """Test individual circuit breaker functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a fresh circuit breaker for each test."""
        return CircuitBreaker("test_circuit")
    
    @pytest.fixture
    def failing_func(self):
        """Function that always fails."""
        def func():
            raise Exception("Test failure")
        return func
    
    @pytest.fixture
    def succeeding_func(self):
        """Function that always succeeds."""
        def func():
            return "success"
        return func
    
    @pytest.fixture
    def intermittent_func(self):
        """Function that fails intermittently."""
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Intermittent failure")
            return "success"
        return func
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initializes correctly."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.name == "test_circuit"
        assert circuit_breaker.metrics.total_calls == 0
        assert circuit_breaker.metrics.failed_calls == 0
        assert circuit_breaker.metrics.successful_calls == 0
    
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker, succeeding_func):
        """Test successful function execution."""
        result = await circuit_breaker.call(succeeding_func)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.successful_calls == 1
        assert circuit_breaker.metrics.failed_calls == 0
        assert circuit_breaker.metrics.total_calls == 1
    
    @pytest.mark.asyncio
    async def test_failed_call(self, circuit_breaker, failing_func):
        """Test failed function execution."""
        with pytest.raises(Exception, match="Test failure"):
            await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.successful_calls == 0
        assert circuit_breaker.metrics.failed_calls == 1
        assert circuit_breaker.metrics.current_failure_count == 1
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, circuit_breaker, failing_func):
        """Test circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)
        
        # First failure
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == CircuitState.CLOSED
        
        # Second failure - circuit should open
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_rejects_calls_when_open(self, circuit_breaker, failing_func):
        """Test circuit rejects calls when in OPEN state."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        
        # Cause circuit to open
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        # Try to call again - should be rejected
        with pytest.raises(Exception, match="Circuit breaker 'test' is OPEN"):
            await cb.call(failing_func)
        
        assert cb.metrics.rejected_calls == 1
    
    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, circuit_breaker, failing_func):
        """Test circuit transitions to HALF_OPEN after recovery timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)
        
        # Cause circuit to open
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Next call should transition to HALF_OPEN and then fail, reopening the circuit
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        # The circuit should be OPEN again after the failure in HALF_OPEN state
        assert cb.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open_success(self, circuit_breaker):
        """Test circuit transitions to HALF_OPEN and stays there on success."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1, success_threshold=2)
        cb = CircuitBreaker("test", config)
        
        # Cause circuit to open
        with pytest.raises(Exception):
            await cb.call(lambda: raise_exception("fail"))
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Next call should transition to HALF_OPEN and succeed
        result = await cb.call(lambda: "success")
        
        # Should still be in HALF_OPEN state (need 2 successes to close)
        assert cb.state == CircuitState.HALF_OPEN
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_circuit_closes_after_success_in_half_open(self, circuit_breaker):
        """Test circuit closes after successful calls in HALF_OPEN state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1, success_threshold=2)
        cb = CircuitBreaker("test", config)
        
        # Cause circuit to open
        with pytest.raises(Exception):
            await cb.call(lambda: raise_exception("fail"))
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # First success in HALF_OPEN
        result = await cb.call(lambda: "success1")
        assert result == "success1"
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second success should close circuit
        result = await cb.call(lambda: "success2")
        assert result == "success2"
        assert cb.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_reopens_after_failure_in_half_open(self, circuit_breaker):
        """Test circuit reopens after failure in HALF_OPEN state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)
        
        # Cause circuit to open
        with pytest.raises(Exception):
            await cb.call(lambda: raise_exception("fail"))
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Failure in HALF_OPEN should reopen circuit
        with pytest.raises(Exception):
            await cb.call(lambda: raise_exception("fail"))
        
        assert cb.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, circuit_breaker, succeeding_func, failing_func):
        """Test comprehensive metrics collection."""
        # Successful call
        await circuit_breaker.call(succeeding_func)
        
        # Failed call
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)
        
        metrics = circuit_breaker.get_metrics()
        
        assert metrics['successful_calls'] == 1
        assert metrics['failed_calls'] == 1
        assert metrics['total_calls'] == 2
        assert metrics['current_failure_count'] == 1
        assert metrics['last_success_time'] is not None
        assert metrics['last_failure_time'] is not None
        assert len(metrics['state_changes']) >= 0
    
    @pytest.mark.asyncio
    async def test_async_function_support(self, circuit_breaker):
        """Test circuit breaker works with async functions."""
        async def async_func():
            return "async_success"
        
        result = await circuit_breaker.call(async_func)
        assert result == "async_success"
        assert circuit_breaker.metrics.successful_calls == 1
    
    @pytest.mark.asyncio
    async def test_sync_function_support(self, circuit_breaker):
        """Test circuit breaker works with sync functions."""
        def sync_func():
            return "sync_success"
        
        result = await circuit_breaker.call(sync_func)
        assert result == "sync_success"
        assert circuit_breaker.metrics.successful_calls == 1
    
    @pytest.mark.asyncio
    async def test_custom_exception_types(self):
        """Test circuit breaker with custom exception types."""
        class CustomException(Exception):
            pass
        
        config = CircuitBreakerConfig(expected_exception=CustomException)
        cb = CircuitBreaker("test", config)
        
        def func():
            raise CustomException("custom error")
        
        with pytest.raises(CustomException):
            await cb.call(func)
        
        assert cb.metrics.failed_calls == 1
    
    @pytest.mark.asyncio
    async def test_unexpected_exceptions(self, circuit_breaker):
        """Test circuit breaker handles unexpected exceptions."""
        def func():
            raise ValueError("unexpected error")
        
        with pytest.raises(ValueError):
            await circuit_breaker.call(func)
        
        # Should still record as failure
        assert circuit_breaker.metrics.failed_calls == 1
    
    @pytest.mark.asyncio
    async def test_manual_reset(self, circuit_breaker, failing_func):
        """Test manual circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        
        # Cause circuit to open
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Manual reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.metrics.current_failure_count == 0


class TestCircuitBreakerManager:
    """Test circuit breaker manager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh circuit breaker manager."""
        return CircuitBreakerManager()
    
    @pytest.mark.asyncio
    async def test_get_circuit_breaker_creation(self, manager):
        """Test circuit breaker creation through manager."""
        cb = await manager.get_circuit_breaker("test_circuit")
        assert cb.name == "test_circuit"
        assert cb.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_get_circuit_breaker_reuse(self, manager):
        """Test circuit breaker reuse through manager."""
        cb1 = await manager.get_circuit_breaker("test_circuit")
        cb2 = await manager.get_circuit_breaker("test_circuit")
        
        assert cb1 is cb2  # Same instance
    
    @pytest.mark.asyncio
    async def test_call_with_circuit_breaker(self, manager):
        """Test calling functions through manager."""
        def test_func():
            return "success"
        
        result = await manager.call_with_circuit_breaker("test", test_func)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_get_all_metrics(self, manager):
        """Test getting metrics for all circuit breakers."""
        # Create multiple circuit breakers
        await manager.get_circuit_breaker("cb1")
        await manager.get_circuit_breaker("cb2")
        
        metrics = manager.get_all_metrics()
        
        assert "cb1" in metrics
        assert "cb2" in metrics
        assert metrics["cb1"]["name"] == "cb1"
        assert metrics["cb2"]["name"] == "cb2"
    
    @pytest.mark.asyncio
    async def test_reset_all(self, manager):
        """Test resetting all circuit breakers."""
        config = CircuitBreakerConfig(failure_threshold=1)
        
        # Create and open circuit breakers
        cb1 = await manager.get_circuit_breaker("cb1", config)
        cb2 = await manager.get_circuit_breaker("cb2", config)
        
        # Open them
        with pytest.raises(Exception):
            await cb1.call(lambda: raise_exception("fail"))
        with pytest.raises(Exception):
            await cb2.call(lambda: raise_exception("fail"))
        
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN
        
        # Reset all
        await manager.reset_all()
        
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_reset_specific_circuit_breaker(self, manager):
        """Test resetting a specific circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        
        # Create and open circuit breaker
        cb = await manager.get_circuit_breaker("test", config)
        
        # Open it
        with pytest.raises(Exception):
            await cb.call(lambda: raise_exception("fail"))
        
        assert cb.state == CircuitState.OPEN
        
        # Reset specific one
        await manager.reset_circuit_breaker("test")
        
        assert cb.state == CircuitState.CLOSED


class TestGlobalCircuitBreakerManager:
    """Test global circuit breaker manager functionality."""
    
    @pytest.mark.asyncio
    async def test_global_manager_singleton(self):
        """Test global manager is a singleton."""
        manager1 = get_circuit_breaker_manager()
        manager2 = get_circuit_breaker_manager()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_global_manager_functionality(self):
        """Test global manager functionality."""
        manager = get_circuit_breaker_manager()
        
        def test_func():
            return "global_success"
        
        result = await manager.call_with_circuit_breaker("global_test", test_func)
        assert result == "global_success"


class TestPredefinedConfigurations:
    """Test predefined circuit breaker configurations."""
    
    def test_exchange_api_config(self):
        """Test exchange API configuration."""
        assert EXCHANGE_API_CONFIG.failure_threshold == 3
        assert EXCHANGE_API_CONFIG.recovery_timeout == 120.0
        assert EXCHANGE_API_CONFIG.success_threshold == 2
    
    def test_webhook_config(self):
        """Test webhook configuration."""
        assert WEBHOOK_CONFIG.failure_threshold == 5
        assert WEBHOOK_CONFIG.recovery_timeout == 300.0
        assert WEBHOOK_CONFIG.success_threshold == 1
    
    def test_database_config(self):
        """Test database configuration."""
        assert DATABASE_CONFIG.failure_threshold == 2
        assert DATABASE_CONFIG.recovery_timeout == 60.0
        assert DATABASE_CONFIG.success_threshold == 3


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker in realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_exchange_api_failure_scenario(self):
        """Test circuit breaker behavior with exchange API failures."""
        manager = get_circuit_breaker_manager()
        
        # Use a shorter recovery timeout for testing
        test_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.1,  # Short timeout for testing
            expected_exception=Exception,
            success_threshold=1
        )
        
        # Simulate exchange API that fails intermittently
        call_count = 0
        def exchange_api_call():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # First 3 calls fail
                raise Exception("Exchange API error")
            return {"price": 50000.0}
        
        # First 3 calls should fail and open circuit
        for i in range(3):
            with pytest.raises(Exception):
                await manager.call_with_circuit_breaker(
                    "exchange_api", 
                    exchange_api_call, 
                    config=test_config
                )
        
        # Next call should be rejected (circuit open)
        with pytest.raises(Exception, match="Circuit breaker 'exchange_api' is OPEN"):
            await manager.call_with_circuit_breaker(
                "exchange_api", 
                exchange_api_call, 
                config=test_config
            )
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)  # Use short timeout for testing
        
        # Next call should succeed and close circuit
        result = await manager.call_with_circuit_breaker(
            "exchange_api", 
            exchange_api_call, 
            config=test_config
        )
        
        assert result["price"] == 50000.0
    
    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_calls(self):
        """Test circuit breaker with concurrent calls."""
        manager = get_circuit_breaker_manager()
        
        # Function that simulates network delay
        async def delayed_func():
            await asyncio.sleep(0.1)
            return "success"
        
        # Make concurrent calls
        tasks = [
            manager.call_with_circuit_breaker("concurrent_test", delayed_func)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert all(result == "success" for result in results)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retry_logic(self):
        """Test circuit breaker with retry logic."""
        manager = get_circuit_breaker_manager()
        
        call_count = 0
        def unreliable_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return "success"
        
        # Retry logic with circuit breaker
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await manager.call_with_circuit_breaker(
                    "retry_test", 
                    unreliable_func,
                    config=CircuitBreakerConfig(failure_threshold=5)
                )
                assert result == "success"
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
