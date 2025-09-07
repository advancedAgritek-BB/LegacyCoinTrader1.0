"""
Comprehensive tests for Enhanced Error Handling and Retry Logic.

Tests cover all retry strategies, error classification, and edge cases
to ensure robust error handling in the trading pipeline.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from crypto_bot.utils.retry_handler import (
    RetryHandler,
    RetryConfig,
    RetryManager,
    RetryStrategy,
    ErrorSeverity,
    RetryableError,
    NonRetryableError,
    get_retry_manager,
    EXCHANGE_API_RETRY_CONFIG,
    OHLCV_FETCH_RETRY_CONFIG,
    ORDER_PLACEMENT_RETRY_CONFIG,
    WEBHOOK_RETRY_CONFIG
)


class TestRetryHandler:
    """Test individual retry handler functionality."""
    
    @pytest.fixture
    def retry_handler(self):
        """Create a fresh retry handler for each test."""
        return RetryHandler("test_retry")
    
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
            if call_count <= 2:  # First 2 calls fail
                raise Exception("Intermittent failure")
            return "success"
        return func
    
    @pytest.mark.asyncio
    async def test_retry_handler_initialization(self, retry_handler):
        """Test retry handler initializes correctly."""
        assert retry_handler.name == "test_retry"
        assert retry_handler.metrics.total_attempts == 0
        assert retry_handler.metrics.failed_attempts == 0
        assert retry_handler.metrics.successful_attempts == 0
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, retry_handler, succeeding_func):
        """Test successful function execution."""
        result = await retry_handler.execute_with_retry(succeeding_func)
        
        assert result == "success"
        assert retry_handler.metrics.successful_attempts == 1
        assert retry_handler.metrics.failed_attempts == 0
        assert retry_handler.metrics.total_attempts == 1
        assert retry_handler.metrics.retry_count == 0
    
    @pytest.mark.asyncio
    async def test_failed_execution_with_retries(self, retry_handler, failing_func):
        """Test failed execution with retries."""
        config = RetryConfig(max_retries=2)
        handler = RetryHandler("test", config)
        
        with pytest.raises(RetryableError, match="All retries exhausted"):
            await handler.execute_with_retry(failing_func)
        
        assert handler.metrics.failed_attempts == 3  # Initial + 2 retries
        assert handler.metrics.successful_attempts == 0
        assert handler.metrics.total_attempts == 3
        assert handler.metrics.retry_count == 2
    
    @pytest.mark.asyncio
    async def test_intermittent_failure_recovery(self, retry_handler, intermittent_func):
        """Test recovery from intermittent failures."""
        config = RetryConfig(max_retries=3)
        handler = RetryHandler("test", config)
        
        result = await handler.execute_with_retry(intermittent_func)
        
        assert result == "success"
        assert handler.metrics.failed_attempts == 2
        assert handler.metrics.successful_attempts == 1
        assert handler.metrics.total_attempts == 3
        assert handler.metrics.retry_count == 2
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_strategy(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            exponential_base=2.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        handler = RetryHandler("test", config)
        
        # Test delay calculations
        delay1 = handler._calculate_delay(0)
        delay2 = handler._calculate_delay(1)
        delay3 = handler._calculate_delay(2)
        
        # Should be approximately exponential (with jitter)
        assert 0.9 <= delay1 <= 1.1  # ~1 second
        assert 1.8 <= delay2 <= 2.2  # ~2 seconds
        assert 3.6 <= delay3 <= 4.4  # ~4 seconds
    
    @pytest.mark.asyncio
    async def test_linear_backoff_strategy(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            strategy=RetryStrategy.LINEAR_BACKOFF
        )
        handler = RetryHandler("test", config)
        
        # Test delay calculations
        delay1 = handler._calculate_delay(0)
        delay2 = handler._calculate_delay(1)
        delay3 = handler._calculate_delay(2)
        
        # Should be approximately linear (with jitter)
        assert 0.9 <= delay1 <= 1.1  # ~1 second
        assert 1.8 <= delay2 <= 2.2  # ~2 seconds
        assert 2.7 <= delay3 <= 3.3  # ~3 seconds
    
    @pytest.mark.asyncio
    async def test_constant_delay_strategy(self):
        """Test constant delay strategy."""
        config = RetryConfig(
            max_retries=3,
            base_delay=2.0,
            strategy=RetryStrategy.CONSTANT_DELAY
        )
        handler = RetryHandler("test", config)
        
        # All delays should be approximately the same
        delay1 = handler._calculate_delay(0)
        delay2 = handler._calculate_delay(1)
        delay3 = handler._calculate_delay(2)
        
        assert 1.8 <= delay1 <= 2.2
        assert 1.8 <= delay2 <= 2.2
        assert 1.8 <= delay3 <= 2.2
    
    @pytest.mark.asyncio
    async def test_immediate_retry_strategy(self):
        """Test immediate retry strategy."""
        config = RetryConfig(strategy=RetryStrategy.IMMEDIATE_RETRY)
        handler = RetryHandler("test", config)
        
        # All delays should be 0
        delay1 = handler._calculate_delay(0)
        delay2 = handler._calculate_delay(1)
        delay3 = handler._calculate_delay(2)
        
        assert delay1 == 0.0
        assert delay2 == 0.0
        assert delay3 == 0.0
    
    @pytest.mark.asyncio
    async def test_error_severity_classification(self, retry_handler):
        """Test error severity classification."""
        # Network errors should be LOW severity
        network_error = Exception("Connection timeout")
        severity = retry_handler._classify_error_severity(network_error)
        assert severity == ErrorSeverity.LOW
        
        # Auth errors should be CRITICAL severity
        auth_error = Exception("Invalid API key")
        severity = retry_handler._classify_error_severity(auth_error)
        assert severity == ErrorSeverity.CRITICAL
        
        # Validation errors should be HIGH severity
        validation_error = Exception("Invalid request parameters")
        severity = retry_handler._classify_error_severity(validation_error)
        assert severity == ErrorSeverity.HIGH
        
        # Server errors should be MEDIUM severity
        server_error = Exception("Internal server error 500")
        severity = retry_handler._classify_error_severity(server_error)
        assert severity == ErrorSeverity.MEDIUM
    
    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test that critical errors are not retried."""
        config = RetryConfig(max_retries=3)
        handler = RetryHandler("test", config)
        
        def critical_func():
            raise Exception("Invalid API key")
        
        with pytest.raises(NonRetryableError, match="Non-retryable error"):
            await handler.execute_with_retry(critical_func)
        
        # Should only attempt once
        assert handler.metrics.total_attempts == 1
        assert handler.metrics.failed_attempts == 1
        assert handler.metrics.retry_count == 0
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        config = RetryConfig(max_retries=2, timeout=0.1)
        handler = RetryHandler("test", config)
        
        async def slow_func():
            await asyncio.sleep(1.0)
            return "success"
        
        with pytest.raises(RetryableError):
            await handler.execute_with_retry(slow_func)
        
        # Should have attempted retries
        assert handler.metrics.total_attempts == 3
        assert handler.metrics.failed_attempts == 3
    
    @pytest.mark.asyncio
    async def test_context_aware_retry(self):
        """Test context-aware retry decisions."""
        config = RetryConfig(max_retries=3)
        handler = RetryHandler("test", config)
        
        def test_func():
            raise Exception("Network error")
        
        # Test with force_no_retry context
        context = {"force_no_retry": True}
        with pytest.raises(NonRetryableError):
            await handler.execute_with_retry(test_func, context=context)
        
        # Should only attempt once
        assert handler.metrics.total_attempts == 1
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, retry_handler, intermittent_func):
        """Test comprehensive metrics collection."""
        config = RetryConfig(max_retries=3)
        handler = RetryHandler("test", config)
        
        # Execute with some failures and success
        result = await handler.execute_with_retry(intermittent_func)
        
        assert result == "success"
        metrics = handler.get_metrics()
        
        assert metrics['name'] == 'test'
        assert metrics['total_attempts'] == 3
        assert metrics['successful_attempts'] == 1
        assert metrics['failed_attempts'] == 2
        assert metrics['retry_count'] == 2
        assert metrics['success_rate'] == 1/3
        assert metrics['total_time'] > 0
        assert metrics['last_attempt_time'] is not None
        assert metrics['last_success_time'] is not None
        assert metrics['last_failure_time'] is not None
    
    @pytest.mark.asyncio
    async def test_metrics_reset(self, retry_handler, failing_func):
        """Test metrics reset functionality."""
        config = RetryConfig(max_retries=1)
        handler = RetryHandler("test", config)
        
        # Execute to generate some metrics
        try:
            await handler.execute_with_retry(failing_func)
        except RetryableError:
            pass
        
        # Verify metrics exist
        assert handler.metrics.total_attempts > 0
        
        # Reset metrics
        handler.reset_metrics()
        
        # Verify metrics are reset
        assert handler.metrics.total_attempts == 0
        assert handler.metrics.failed_attempts == 0
        assert handler.metrics.successful_attempts == 0
        assert handler.metrics.retry_count == 0


class TestRetryManager:
    """Test retry manager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh retry manager."""
        return RetryManager()
    
    @pytest.mark.asyncio
    async def test_get_retry_handler_creation(self, manager):
        """Test retry handler creation through manager."""
        handler = await manager.get_retry_handler("test_handler")
        assert handler.name == "test_handler"
        assert handler.metrics.total_attempts == 0
    
    @pytest.mark.asyncio
    async def test_get_retry_handler_reuse(self, manager):
        """Test retry handler reuse through manager."""
        handler1 = await manager.get_retry_handler("test_handler")
        handler2 = await manager.get_retry_handler("test_handler")
        
        assert handler1 is handler2  # Same instance
    
    @pytest.mark.asyncio
    async def test_execute_with_retry(self, manager):
        """Test executing functions through manager."""
        def test_func():
            return "success"
        
        result = await manager.execute_with_retry("test", test_func)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_get_all_metrics(self, manager):
        """Test getting metrics for all retry handlers."""
        # Create multiple retry handlers
        await manager.get_retry_handler("handler1")
        await manager.get_retry_handler("handler2")
        
        metrics = manager.get_all_metrics()
        
        assert "handler1" in metrics
        assert "handler2" in metrics
        assert metrics["handler1"]["name"] == "handler1"
        assert metrics["handler2"]["name"] == "handler2"
    
    @pytest.mark.asyncio
    async def test_reset_all_metrics(self, manager):
        """Test resetting all retry handler metrics."""
        # Create and use retry handlers
        handler1 = await manager.get_retry_handler("handler1")
        handler2 = await manager.get_retry_handler("handler2")
        
        # Generate some metrics
        def test_func():
            return "success"
        
        await manager.execute_with_retry("handler1", test_func)
        await manager.execute_with_retry("handler2", test_func)
        
        # Verify metrics exist
        assert handler1.metrics.total_attempts > 0
        assert handler2.metrics.total_attempts > 0
        
        # Reset all
        await manager.reset_all_metrics()
        
        # Verify metrics are reset
        assert handler1.metrics.total_attempts == 0
        assert handler2.metrics.total_attempts == 0


class TestGlobalRetryManager:
    """Test global retry manager functionality."""
    
    @pytest.mark.asyncio
    async def test_global_manager_singleton(self):
        """Test global manager is a singleton."""
        manager1 = get_retry_manager()
        manager2 = get_retry_manager()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_global_manager_functionality(self):
        """Test global manager functionality."""
        manager = get_retry_manager()
        
        def test_func():
            return "global_success"
        
        result = await manager.execute_with_retry("global_test", test_func)
        assert result == "global_success"


class TestPredefinedConfigurations:
    """Test predefined retry configurations."""
    
    def test_exchange_api_retry_config(self):
        """Test exchange API retry configuration."""
        assert EXCHANGE_API_RETRY_CONFIG.max_retries == 3
        assert EXCHANGE_API_RETRY_CONFIG.base_delay == 1.0
        assert EXCHANGE_API_RETRY_CONFIG.max_delay == 30.0
        assert EXCHANGE_API_RETRY_CONFIG.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert EXCHANGE_API_RETRY_CONFIG.timeout == 30.0
    
    def test_ohlcv_fetch_retry_config(self):
        """Test OHLCV fetch retry configuration."""
        assert OHLCV_FETCH_RETRY_CONFIG.max_retries == 5
        assert OHLCV_FETCH_RETRY_CONFIG.base_delay == 2.0
        assert OHLCV_FETCH_RETRY_CONFIG.max_delay == 60.0
        assert OHLCV_FETCH_RETRY_CONFIG.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert OHLCV_FETCH_RETRY_CONFIG.timeout == 60.0
    
    def test_order_placement_retry_config(self):
        """Test order placement retry configuration."""
        assert ORDER_PLACEMENT_RETRY_CONFIG.max_retries == 2
        assert ORDER_PLACEMENT_RETRY_CONFIG.base_delay == 0.5
        assert ORDER_PLACEMENT_RETRY_CONFIG.max_delay == 10.0
        assert ORDER_PLACEMENT_RETRY_CONFIG.strategy == RetryStrategy.LINEAR_BACKOFF
        assert ORDER_PLACEMENT_RETRY_CONFIG.timeout == 15.0
    
    def test_webhook_retry_config(self):
        """Test webhook retry configuration."""
        assert WEBHOOK_RETRY_CONFIG.max_retries == 3
        assert WEBHOOK_RETRY_CONFIG.base_delay == 5.0
        assert WEBHOOK_RETRY_CONFIG.max_delay == 300.0
        assert WEBHOOK_RETRY_CONFIG.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert WEBHOOK_RETRY_CONFIG.timeout == 30.0


class TestRetryHandlerIntegration:
    """Integration tests for retry handler in realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_exchange_api_retry_scenario(self):
        """Test retry handler behavior with exchange API failures."""
        manager = get_retry_manager()
        
        # Simulate exchange API that fails intermittently
        call_count = 0
        def exchange_api_call():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First 2 calls fail
                raise Exception("Exchange API error")
            return {"price": 50000.0}
        
        # Should succeed after retries
        result = await manager.execute_with_retry(
            "exchange_api", 
            exchange_api_call, 
            config=EXCHANGE_API_RETRY_CONFIG
        )
        
        assert result["price"] == 50000.0
        
        # Check metrics
        metrics = manager.get_all_metrics()
        exchange_metrics = metrics.get("exchange_api")
        if exchange_metrics:
            assert exchange_metrics["successful_attempts"] == 1
            assert exchange_metrics["failed_attempts"] == 2
            assert exchange_metrics["retry_count"] == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_retry_handlers(self):
        """Test retry handler with concurrent operations."""
        manager = get_retry_manager()
        
        # Function that simulates network delay
        async def delayed_func():
            await asyncio.sleep(0.1)
            return "success"
        
        # Make concurrent calls
        tasks = [
            manager.execute_with_retry("concurrent_test", delayed_func)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert all(result == "success" for result in results)
        
        # Check metrics
        metrics = manager.get_all_metrics()
        concurrent_metrics = metrics.get("concurrent_test")
        if concurrent_metrics:
            assert concurrent_metrics["successful_attempts"] == 5
            assert concurrent_metrics["total_attempts"] == 5
    
    @pytest.mark.asyncio
    async def test_retry_handler_with_custom_conditions(self):
        """Test retry handler with custom retry conditions."""
        def should_retry_on_network_error(exception):
            return "network" in str(exception).lower()
        
        config = RetryConfig(
            max_retries=2,
            retry_on_conditions=[should_retry_on_network_error]
        )
        
        manager = get_retry_manager()
        
        def network_error_func():
            raise Exception("Network timeout")
        
        def auth_error_func():
            raise Exception("Invalid API key")
        
        # Network error should be retried
        with pytest.raises(RetryableError):
            await manager.execute_with_retry("network_test", network_error_func, config=config)
        
        # Auth error should not be retried
        with pytest.raises(NonRetryableError):
            await manager.execute_with_retry("auth_test", auth_error_func, config=config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
