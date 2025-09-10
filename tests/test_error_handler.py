"""
Test suite for Enhanced Error Handling and Recovery System

Tests the error handling, circuit breaker, retry logic, and recovery mechanisms.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from crypto_bot.utils.error_handler import (
    ErrorHandler, CircuitBreaker, RetryHandler, ErrorContext, ErrorSeverity, ErrorCategory,
    get_global_error_handler, reset_global_error_handler, handle_errors, with_circuit_breaker,
    error_context, async_error_context
)


class TestErrorHandler:
    """Test the main error handler functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_global_error_handler()
        self.error_handler = ErrorHandler()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_global_error_handler()
    
    def test_error_classification(self):
        """Test error classification functionality."""
        # Test network error
        network_error = Exception("Connection timeout")
        context = self.error_handler.classify_error(network_error, "fetch_data", "market_loader")
        
        assert context.category == ErrorCategory.NETWORK
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.operation == "fetch_data"
        assert context.component == "market_loader"
        
        # Test API error
        api_error = Exception("Rate limit exceeded")
        context = self.error_handler.classify_error(api_error, "api_call", "exchange")
        
        assert context.category == ErrorCategory.API
        assert context.severity == ErrorSeverity.MEDIUM
        
        # Test data error
        data_error = Exception("Invalid data format")
        context = self.error_handler.classify_error(data_error, "parse_data", "data_processor")
        
        assert context.category == ErrorCategory.DATA
        assert context.severity == ErrorSeverity.HIGH
        
        # Test exchange error
        exchange_error = Exception("Kraken API error")
        context = self.error_handler.classify_error(exchange_error, "place_order", "exchange")
        
        assert context.category == ErrorCategory.EXCHANGE
        assert context.severity == ErrorSeverity.HIGH
        
        # Test memory error
        memory_error = Exception("Out of memory")
        context = self.error_handler.classify_error(memory_error, "process_data", "memory_manager")
        
        assert context.category == ErrorCategory.MEMORY
        assert context.severity == ErrorSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and recovery."""
        # Create a test error context
        error_context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            original_exception=Exception("Test error")
        )
        
        # Handle the error
        success = await self.error_handler.handle_error(error_context)
        
        # Should have recorded the error
        assert len(self.error_handler.error_history) == 1
        assert self.error_handler.error_counts["test_component"] == 1
        
        # Should have attempted recovery
        assert success is True  # Default recovery actions return True
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and management."""
        # Get circuit breaker
        cb = self.error_handler.get_circuit_breaker("test_component")
        
        assert cb.name == "test_component"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
        
        # Should return the same instance
        cb2 = self.error_handler.get_circuit_breaker("test_component")
        assert cb is cb2
    
    def test_error_stats(self):
        """Test error statistics collection."""
        # Create some test errors
        for i in range(5):
            context = ErrorContext(
                operation=f"operation_{i}",
                component="test_component",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                original_exception=Exception(f"Error {i}")
            )
            self.error_handler.error_history.append(context)
            self.error_handler.error_counts["test_component"] += 1
        
        # Create a circuit breaker to ensure it appears in stats
        self.error_handler.get_circuit_breaker("test_component")
        
        # Get stats
        stats = self.error_handler.get_error_stats()
        
        assert stats["total_errors"] == 5
        assert stats["error_counts"]["test_component"] == 5
        assert "test_component" in stats["circuit_breaker_states"]


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            name="test_cb"
        )
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        # Should start in CLOSED state
        assert self.cb.state == "CLOSED"
        
        # Simulate failures
        for i in range(3):
            self.cb._on_failure()
        
        # Should be OPEN after threshold failures
        assert self.cb.state == "OPEN"
        assert self.cb.failure_count == 3
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should transition to HALF_OPEN when call is made
        assert self.cb._should_attempt_reset() is True
        
        # Simulate a call that transitions to HALF_OPEN
        def test_func():
            return "success"
        
        # This should transition to HALF_OPEN and then to CLOSED on success
        result = self.cb.call(test_func)
        assert result == "success"
        assert self.cb.state == "CLOSED"
        assert self.cb.failure_count == 0
    
    def test_circuit_breaker_call(self):
        """Test circuit breaker function execution."""
        # Test successful call
        def success_func():
            return "success"
        
        result = self.cb.call(success_func)
        assert result == "success"
        assert self.cb.state == "CLOSED"
        assert self.cb.successful_requests == 1
        
        # Test failing call
        def fail_func():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            self.cb.call(fail_func)
        
        assert self.cb.failure_count == 1
        
        # Test circuit opening
        for i in range(2):  # Already 1 failure
            with pytest.raises(Exception):
                self.cb.call(fail_func)
        
        assert self.cb.state == "OPEN"
        
        # Test circuit blocking calls when OPEN
        with pytest.raises(Exception, match="Circuit breaker test_cb is OPEN"):
            self.cb.call(success_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_async_call(self):
        """Test circuit breaker async function execution."""
        # Test successful async call
        async def success_async_func():
            return "success"
        
        result = await self.cb.async_call(success_async_func)
        assert result == "success"
        assert self.cb.state == "CLOSED"
        
        # Test failing async call
        async def fail_async_func():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            await self.cb.async_call(fail_async_func)
        
        assert self.cb.failure_count == 1
    
    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics."""
        # Make some calls
        def success_func():
            return "success"
        
        def fail_func():
            raise Exception("failure")
        
        # Successful calls
        for i in range(5):
            self.cb.call(success_func)
        
        # Failed calls
        for i in range(2):
            with pytest.raises(Exception):
                self.cb.call(fail_func)
        
        # Get stats
        stats = self.cb.get_stats()
        
        assert stats["name"] == "test_cb"
        assert stats["total_requests"] == 7
        assert stats["successful_requests"] == 5
        assert stats["failure_count"] == 2
        assert stats["success_rate"] == pytest.approx(71.43, abs=0.1)


class TestRetryHandler:
    """Test retry handler functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.retry_handler = RetryHandler(
            max_retries=3,
            base_delay=0.1,  # Short delays for testing
            max_delay=1.0,
            jitter=False  # Disable jitter for predictable tests
        )
    
    def test_retry_handler_success(self):
        """Test retry handler with successful call."""
        call_count = 0
        
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = self.retry_handler.call(test_func)
        
        assert result == "success"
        assert call_count == 1  # Should succeed on first try
    
    def test_retry_handler_failure_then_success(self):
        """Test retry handler with failure then success."""
        call_count = 0
        
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = self.retry_handler.call(test_func)
        
        assert result == "success"
        assert call_count == 3  # Should succeed on third try
    
    def test_retry_handler_max_retries_exceeded(self):
        """Test retry handler when max retries are exceeded."""
        def test_func():
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception, match="Persistent failure"):
            self.retry_handler.call(test_func)
    
    @pytest.mark.asyncio
    async def test_retry_handler_async(self):
        """Test retry handler with async functions."""
        call_count = 0
        
        async def test_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = await self.retry_handler.async_call(test_async_func)
        
        assert result == "success"
        assert call_count == 2
    
    def test_retry_handler_delay_calculation(self):
        """Test retry handler delay calculation."""
        # Test exponential backoff with configured base delay of 0.1
        delays = []
        for attempt in range(3):
            delay = self.retry_handler._calculate_delay(attempt)
            delays.append(delay)
        
        # Should be exponential: 0.1, 0.2, 0.4 (no jitter configured)
        assert delays[0] == 0.1
        assert delays[1] == 0.2
        assert delays[2] == 0.4
        
        # Test with jitter enabled
        retry_handler_with_jitter = RetryHandler(base_delay=1.0, jitter=True)
        delays_with_jitter = []
        for attempt in range(3):
            delay = retry_handler_with_jitter._calculate_delay(attempt)
            delays_with_jitter.append(delay)
        
        # Should be exponential: 1.0, 2.0, 4.0 (with jitter)
        assert delays_with_jitter[0] >= 0.5  # With jitter: 50-100% of 1.0
        assert delays_with_jitter[0] <= 1.0
        assert delays_with_jitter[1] >= 1.0   # With jitter: 50-100% of 2.0
        assert delays_with_jitter[1] <= 2.0
        assert delays_with_jitter[2] >= 2.0   # With jitter: 50-100% of 4.0
        assert delays_with_jitter[2] <= 4.0
        
        # Test without jitter
        retry_handler_no_jitter = RetryHandler(base_delay=1.0, jitter=False)
        delays_no_jitter = []
        for attempt in range(3):
            delay = retry_handler_no_jitter._calculate_delay(attempt)
            delays_no_jitter.append(delay)
        
        # Should be exact: 1.0, 2.0, 4.0
        assert delays_no_jitter[0] == 1.0
        assert delays_no_jitter[1] == 2.0
        assert delays_no_jitter[2] == 4.0


class TestDecorators:
    """Test error handling decorators."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_global_error_handler()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_global_error_handler()
    
    def test_handle_errors_decorator(self):
        """Test handle_errors decorator."""
        call_count = 0
        
        @handle_errors("test_operation", "test_component", max_retries=2, fallback_value="fallback")
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 4:  # Fail 3 times, succeed on 4th
                raise Exception("Temporary failure")
            return "success"
        
        # Should use fallback value after max retries
        result = test_func()
        assert result == "fallback"
        assert call_count == 3  # Should have tried 3 times (max_retries + 1)
    
    def test_handle_errors_decorator_success(self):
        """Test handle_errors decorator with successful call."""
        @handle_errors("test_operation", "test_component")
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_handle_errors_decorator_async(self):
        """Test handle_errors decorator with async function."""
        call_count = 0
        
        @handle_errors("test_operation", "test_component", max_retries=1, fallback_value="fallback")
        async def test_async_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Failure")
        
        result = await test_async_func()
        assert result == "fallback"
        assert call_count == 2  # Should have tried 2 times
    
    def test_with_circuit_breaker_decorator(self):
        """Test with_circuit_breaker decorator."""
        @with_circuit_breaker("test_cb", failure_threshold=2, recovery_timeout=0.1)
        def test_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            test_func()
        
        # Second failure should open circuit
        with pytest.raises(Exception):
            test_func()
        
        # Third call should be blocked by circuit breaker
        with pytest.raises(Exception, match="Circuit breaker test_cb is OPEN"):
            test_func()
    
    @pytest.mark.asyncio
    async def test_with_circuit_breaker_decorator_async(self):
        """Test with_circuit_breaker decorator with async function."""
        @with_circuit_breaker("test_cb", failure_threshold=1, recovery_timeout=0.1)
        async def test_async_func():
            raise Exception("Test failure")
        
        # First failure should open circuit
        with pytest.raises(Exception):
            await test_async_func()
        
        # Second call should be blocked
        with pytest.raises(Exception, match="Circuit breaker test_cb is OPEN"):
            await test_async_func()


class TestContextManagers:
    """Test error handling context managers."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_global_error_handler()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_global_error_handler()
    
    def test_error_context_manager(self):
        """Test error_context context manager."""
        with error_context("test_operation", "test_component"):
            # This should not raise an exception
            pass
        
        # Test with exception
        with pytest.raises(Exception):
            with error_context("test_operation", "test_component"):
                raise Exception("Test error")
        
        # Should have recorded the error
        error_handler = get_global_error_handler()
        assert len(error_handler.error_history) == 1
    
    @pytest.mark.asyncio
    async def test_async_error_context_manager(self):
        """Test async_error_context context manager."""
        async with async_error_context("test_operation", "test_component"):
            # This should not raise an exception
            pass
        
        # Test with exception
        with pytest.raises(Exception):
            async with async_error_context("test_operation", "test_component"):
                raise Exception("Test error")
        
        # Should have recorded the error
        error_handler = get_global_error_handler()
        assert len(error_handler.error_history) == 1


class TestGlobalErrorHandler:
    """Test global error handler functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_global_error_handler()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_global_error_handler()
    
    def test_global_error_handler_singleton(self):
        """Test that global error handler is a singleton."""
        handler1 = get_global_error_handler()
        handler2 = get_global_error_handler()
        
        assert handler1 is handler2
    
    def test_global_error_handler_reset(self):
        """Test global error handler reset functionality."""
        handler1 = get_global_error_handler()
        reset_global_error_handler()
        handler2 = get_global_error_handler()
        
        assert handler1 is not handler2
    
    def test_global_error_handler_config(self):
        """Test global error handler with configuration."""
        config = {"circuit_breaker": {"failure_threshold": 10}}
        handler = get_global_error_handler(config)
        
        # Should use config values
        cb = handler.get_circuit_breaker("test")
        assert cb.failure_threshold == 10


class TestIntegration:
    """Integration tests for error handling system."""
    
    def setup_method(self):
        """Set up test environment."""
        reset_global_error_handler()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_global_error_handler()
    
    def test_error_handler_with_circuit_breaker_integration(self):
        """Test integration between error handler and circuit breaker."""
        error_handler = get_global_error_handler()
        
        # Create error context that will trigger circuit breaker
        error_context = ErrorContext(
            operation="test_operation",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SYSTEM,  # Use SYSTEM to avoid recovery actions
            original_exception=Exception("Test error")
        )
        
        # Handle error multiple times to trigger circuit breaker
        for i in range(6):  # More than failure threshold
            asyncio.run(error_handler.handle_error(error_context))
        
        # Circuit breaker should be OPEN
        cb = error_handler.get_circuit_breaker("test_component")
        assert cb.state == "OPEN"
    
    def test_decorator_integration(self):
        """Test integration of decorators with error handling."""
        @handle_errors("test_operation", "test_component", max_retries=1)
        @with_circuit_breaker("test_cb", failure_threshold=2)
        def test_func():
            raise Exception("Test failure")
        
        # Should handle error through decorator chain
        with pytest.raises(Exception):
            test_func()
        
        # Should have recorded error
        error_handler = get_global_error_handler()
        assert len(error_handler.error_history) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
