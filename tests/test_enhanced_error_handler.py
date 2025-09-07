"""
Comprehensive tests for Enhanced Error Handler

This module tests all functionality of the Enhanced Error Handler to ensure
proper error management and recovery mechanisms.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from crypto_bot.utils.enhanced_error_handler import (
    EnhancedErrorHandler,
    ErrorSeverity,
    RecoveryAction,
    ErrorContext,
    RecoveryStrategy,
    SystemHealth,
    retry_on_error,
    circuit_breaker_protected
)

class TestEnhancedErrorHandler:
    """Test suite for Enhanced Error Handler."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler instance for testing."""
        config = {
            'max_error_history': 100
        }
        return EnhancedErrorHandler(config)
    
    @pytest.fixture
    def mock_circuit_breaker(self):
        """Create mock circuit breaker."""
        cb = Mock()
        cb.is_trading_allowed.return_value = True
        cb.record_api_error = Mock()
        cb.record_api_request = Mock()
        return cb
    
    def test_initialization(self, error_handler):
        """Test error handler initialization."""
        assert error_handler.config is not None
        assert len(error_handler.error_history) == 0
        assert len(error_handler.recovery_strategies) > 0
        assert isinstance(error_handler.system_health, SystemHealth)
        assert len(error_handler.error_callbacks) == 0
        assert len(error_handler.recovery_callbacks) == 0
        assert len(error_handler.alert_callbacks) == 0
        assert error_handler.circuit_breaker is None
    
    def test_default_strategies_setup(self, error_handler):
        """Test that default recovery strategies are set up."""
        strategies = error_handler.recovery_strategies
        
        # Check that key strategies exist
        assert 'api' in strategies
        assert 'connection' in strategies
        assert 'data' in strategies
        assert 'system' in strategies
        assert 'trade' in strategies
        
        # Check API strategy
        api_strategy = strategies['api']
        assert api_strategy.severity == ErrorSeverity.MEDIUM
        assert api_strategy.recovery_action == RecoveryAction.RETRY
        assert api_strategy.max_retries == 5
    
    def test_register_circuit_breaker(self, error_handler, mock_circuit_breaker):
        """Test registering circuit breaker."""
        error_handler.register_circuit_breaker(mock_circuit_breaker)
        assert error_handler.circuit_breaker == mock_circuit_breaker
    
    def test_add_recovery_strategy(self, error_handler):
        """Test adding custom recovery strategy."""
        custom_strategy = RecoveryStrategy(
            error_patterns=["custom_error"],
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.ALERT,
            max_retries=2
        )
        
        error_handler.add_recovery_strategy("custom", custom_strategy)
        
        assert "custom" in error_handler.recovery_strategies
        assert error_handler.recovery_strategies["custom"] == custom_strategy
    
    def test_handle_error_basic(self, error_handler):
        """Test basic error handling."""
        error = ValueError("Test error")
        error_context = error_handler.handle_error(error, "test_component", "test_operation")
        
        assert error_context.component == "test_component"
        assert error_context.operation == "test_operation"
        assert error_context.error_type == "ValueError"
        assert error_context.error_message == "Test error"
        assert error_context.severity == ErrorSeverity.MEDIUM  # Default for unknown errors
        assert error_context.recovery_action == RecoveryAction.ALERT  # Default for unknown errors
        assert len(error_handler.error_history) == 1
    
    def test_handle_error_with_context(self, error_handler):
        """Test error handling with additional context."""
        error = ValueError("Test error")
        context = {"user_id": 123, "request_id": "abc"}
        
        error_context = error_handler.handle_error(
            error, "test_component", "test_operation", context
        )
        
        assert error_context.metadata == context
        assert error_context.metadata["user_id"] == 123
        assert error_context.metadata["request_id"] == "abc"
    
    def test_api_error_strategy(self, error_handler):
        """Test API error strategy detection."""
        error = Exception("API rate limit exceeded")
        error_context = error_handler.handle_error(error, "api_client", "fetch_data")
        
        # Should match API strategy
        assert error_context.severity == ErrorSeverity.MEDIUM
        assert error_context.recovery_action == RecoveryAction.RETRY
        assert error_context.max_retries == 5
    
    def test_network_error_strategy(self, error_handler):
        """Test network error strategy detection."""
        error = Exception("Connection timeout")
        error_context = error_handler.handle_error(error, "network", "connect")
        
        # Should match connection strategy
        assert error_context.severity == ErrorSeverity.HIGH
        assert error_context.recovery_action == RecoveryAction.RETRY
        assert error_context.max_retries == 3
        assert error_context.recovery_action == RecoveryAction.RETRY
        assert error_context.max_retries == 3
    
    def test_critical_error_strategy(self, error_handler):
        """Test critical error strategy detection."""
        error = Exception("System memory critical")
        error_context = error_handler.handle_error(error, "system", "monitor")
        
        # Should match system strategy
        assert error_context.severity == ErrorSeverity.CRITICAL
        assert error_context.recovery_action == RecoveryAction.CIRCUIT_BREAK
        assert error_context.max_retries == 0
    
    def test_trading_error_strategy(self, error_handler):
        """Test trading error strategy detection."""
        error = Exception("Trade execution failed")
        error_context = error_handler.handle_error(error, "trading", "execute_order")
        
        # Should match trade strategy
        assert error_context.severity == ErrorSeverity.HIGH
        assert error_context.recovery_action == RecoveryAction.ALERT
        assert error_context.max_retries == 1
    
    def test_system_health_update(self, error_handler):
        """Test system health metrics update."""
        error = ValueError("Test error")
        error_handler.handle_error(error, "test_component", "test_operation")
        
        health = error_handler.system_health
        assert health.total_errors == 1
        assert health.errors_by_severity[ErrorSeverity.MEDIUM] == 1
        assert health.errors_by_component["test_component"] == 1
        assert health.last_error_time is not None
        assert health.uptime_seconds > 0
    
    def test_error_history_limiting(self, error_handler):
        """Test error history size limiting."""
        # Add more errors than the limit
        for i in range(150):
            error = ValueError(f"Error {i}")
            error_handler.handle_error(error, "test", "test")
        
        # Should be limited to max_error_history
        assert len(error_handler.error_history) == 100
    
    def test_get_error_history_filtering(self, error_handler):
        """Test error history filtering."""
        # Add different types of errors
        error_handler.handle_error(ValueError("Error 1"), "component1", "op1")
        error_handler.handle_error(ValueError("Error 2"), "component2", "op2")
        error_handler.handle_error(ValueError("Error 3"), "component1", "op3")
        
        # Filter by component
        component1_errors = error_handler.get_error_history(component="component1")
        assert len(component1_errors) == 2
        
        # Filter by severity
        medium_errors = error_handler.get_error_history(severity=ErrorSeverity.MEDIUM)
        assert len(medium_errors) == 3
        
        # Filter by time
        since = datetime.now() + timedelta(seconds=1)
        recent_errors = error_handler.get_error_history(since=since)
        assert len(recent_errors) == 0
    
    def test_get_error_stats(self, error_handler):
        """Test error statistics."""
        error_handler.handle_error(ValueError("Error 1"), "component1", "op1")
        error_handler.handle_error(ValueError("Error 2"), "component2", "op2")
        
        stats = error_handler.get_error_stats()
        
        assert stats['total_errors'] == 2
        assert stats['errors_by_severity']['medium'] == 2
        assert stats['errors_by_component']['component1'] == 1
        assert stats['errors_by_component']['component2'] == 1
        assert stats['uptime_seconds'] > 0
        assert stats['last_error_time'] is not None
    
    def test_clear_error_history(self, error_handler):
        """Test clearing error history."""
        error_handler.handle_error(ValueError("Test"), "test", "test")
        assert len(error_handler.error_history) == 1
        
        error_handler.clear_error_history()
        assert len(error_handler.error_history) == 0
    
    def test_reset_system_health(self, error_handler):
        """Test resetting system health."""
        error_handler.handle_error(ValueError("Test"), "test", "test")
        assert error_handler.system_health.total_errors == 1
        
        error_handler.reset_system_health()
        assert error_handler.system_health.total_errors == 0
        assert error_handler.system_health.errors_by_severity == {}
        assert error_handler.system_health.errors_by_component == {}
    
    def test_add_callbacks(self, error_handler):
        """Test adding callbacks."""
        error_callback = Mock()
        recovery_callback = Mock()
        alert_callback = Mock()
        
        error_handler.add_error_callback(error_callback)
        error_handler.add_recovery_callback(recovery_callback)
        error_handler.add_alert_callback(alert_callback)
        
        assert len(error_handler.error_callbacks) == 1
        assert len(error_handler.recovery_callbacks) == 1
        assert len(error_handler.alert_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, error_handler, mock_circuit_breaker):
        """Test circuit breaker integration."""
        error_handler.register_circuit_breaker(mock_circuit_breaker)
        
        # Trigger a critical error that should open circuit breaker
        error = Exception("System memory critical")
        error_context = error_handler.handle_error(error, "system", "monitor")
        
        # Wait for recovery action to execute
        await asyncio.sleep(0.1)
        
        # Check that circuit breaker was triggered
        mock_circuit_breaker.record_api_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_alert_sending(self, error_handler):
        """Test alert sending functionality."""
        alert_callback = Mock()
        error_handler.add_alert_callback(alert_callback)
        
        # Trigger a trading error that should send alert
        error = Exception("Trade execution failed")
        error_context = error_handler.handle_error(error, "trading", "execute_order")
        
        # Wait for recovery action to execute
        await asyncio.sleep(0.1)
        
        # Check that alert was sent
        alert_callback.assert_called_once()
        alert_message = alert_callback.call_args[0][0]
        assert alert_message['type'] == 'error_alert'
        assert alert_message['component'] == 'trading'
        assert alert_message['operation'] == 'execute_order'
        assert alert_message['severity'] == 'high'

class TestRetryDecorator:
    """Test suite for retry decorator."""
    
    @pytest.mark.asyncio
    async def test_retry_on_error_success_first_try(self):
        """Test retry decorator when function succeeds on first try."""
        call_count = 0
        
        @retry_on_error(max_retries=3, delay=0.1)
        async def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_function()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_error_success_after_retries(self):
        """Test retry decorator when function succeeds after retries."""
        call_count = 0
        
        @retry_on_error(max_retries=3, delay=0.1)
        async def eventually_successful_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = await eventually_successful_function()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_on_error_max_retries_exceeded(self):
        """Test retry decorator when max retries are exceeded."""
        call_count = 0
        
        @retry_on_error(max_retries=2, delay=0.1)
        async def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")
        
        with pytest.raises(ValueError, match="Persistent error"):
            await always_failing_function()
        
        assert call_count == 3  # Initial call + 2 retries
    
    @pytest.mark.asyncio
    async def test_retry_on_error_exponential_backoff(self):
        """Test retry decorator with exponential backoff."""
        call_times = []
        
        @retry_on_error(max_retries=2, delay=0.1, backoff_multiplier=2.0)
        async def failing_function():
            call_times.append(datetime.now())
            raise ValueError("Error")
        
        start_time = datetime.now()
        
        with pytest.raises(ValueError):
            await failing_function()
        
        # Should have 3 calls total
        assert len(call_times) == 3
        
        # Check that delays increased exponentially
        delay1 = (call_times[1] - call_times[0]).total_seconds()
        delay2 = (call_times[2] - call_times[1]).total_seconds()
        
        # Allow some tolerance for timing
        assert 0.05 <= delay1 <= 0.15
        assert 0.15 <= delay2 <= 0.25  # 2x the first delay

class TestCircuitBreakerDecorator:
    """Test suite for circuit breaker decorator."""
    
    @pytest.fixture
    def mock_circuit_breaker(self):
        """Create mock circuit breaker."""
        cb = Mock()
        cb.is_trading_allowed.return_value = True
        cb.record_api_error = Mock()
        cb.record_api_request = Mock()
        return cb
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_protected_success(self, mock_circuit_breaker):
        """Test circuit breaker decorator with successful execution."""
        call_count = 0
        
        @circuit_breaker_protected(mock_circuit_breaker)
        async def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_function()
        
        assert result == "success"
        assert call_count == 1
        mock_circuit_breaker.record_api_request.assert_called_once()
        mock_circuit_breaker.record_api_error.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_protected_failure(self, mock_circuit_breaker):
        """Test circuit breaker decorator with failed execution."""
        @circuit_breaker_protected(mock_circuit_breaker)
        async def failing_function():
            raise ValueError("Function failed")
        
        with pytest.raises(ValueError, match="Function failed"):
            await failing_function()
        
        mock_circuit_breaker.record_api_error.assert_called_once()
        mock_circuit_breaker.record_api_request.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_protected_circuit_open(self, mock_circuit_breaker):
        """Test circuit breaker decorator when circuit is open."""
        mock_circuit_breaker.is_trading_allowed.return_value = False
        
        @circuit_breaker_protected(mock_circuit_breaker)
        async def any_function():
            return "should not execute"
        
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await any_function()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_protected_no_circuit_breaker(self):
        """Test circuit breaker decorator without circuit breaker."""
        @circuit_breaker_protected(None)
        async def test_function():
            return "success"
        
        result = await test_function()
        assert result == "success"

class TestErrorSeverity:
    """Test suite for ErrorSeverity enum."""
    
    def test_error_severity_values(self):
        """Test error severity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

class TestRecoveryAction:
    """Test suite for RecoveryAction enum."""
    
    def test_recovery_action_values(self):
        """Test recovery action enum values."""
        assert RecoveryAction.RETRY.value == "retry"
        assert RecoveryAction.FALLBACK.value == "fallback"
        assert RecoveryAction.CIRCUIT_BREAK.value == "circuit_break"
        assert RecoveryAction.RESTART.value == "restart"
        assert RecoveryAction.ALERT.value == "alert"
        assert RecoveryAction.IGNORE.value == "ignore"

class TestErrorContext:
    """Test suite for ErrorContext dataclass."""
    
    def test_error_context_creation(self):
        """Test creating an ErrorContext instance."""
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            timestamp=datetime.now(),
            error_type="ValueError",
            error_message="Test error",
            stack_trace="Traceback...",
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.RETRY,
            retry_count=1,
            max_retries=3,
            metadata={"key": "value"}
        )
        
        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert context.error_type == "ValueError"
        assert context.error_message == "Test error"
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.recovery_action == RecoveryAction.RETRY
        assert context.retry_count == 1
        assert context.max_retries == 3
        assert context.metadata["key"] == "value"

class TestRecoveryStrategy:
    """Test suite for RecoveryStrategy dataclass."""
    
    def test_recovery_strategy_creation(self):
        """Test creating a RecoveryStrategy instance."""
        strategy = RecoveryStrategy(
            error_patterns=["test_pattern"],
            severity=ErrorSeverity.HIGH,
            recovery_action=RecoveryAction.ALERT,
            retry_delay=2.0,
            max_retries=5,
            backoff_multiplier=1.5,
            alert_threshold=3
        )
        
        assert strategy.error_patterns == ["test_pattern"]
        assert strategy.severity == ErrorSeverity.HIGH
        assert strategy.recovery_action == RecoveryAction.ALERT
        assert strategy.retry_delay == 2.0
        assert strategy.max_retries == 5
        assert strategy.backoff_multiplier == 1.5
        assert strategy.alert_threshold == 3

class TestSystemHealth:
    """Test suite for SystemHealth dataclass."""
    
    def test_system_health_creation(self):
        """Test creating a SystemHealth instance."""
        health = SystemHealth(
            total_errors=10,
            errors_by_severity={ErrorSeverity.MEDIUM: 5},
            errors_by_component={"api": 3},
            recovery_success_rate=0.8,
            uptime_seconds=3600.0,
            component_status={"api": "healthy"}
        )
        
        assert health.total_errors == 10
        assert health.errors_by_severity[ErrorSeverity.MEDIUM] == 5
        assert health.errors_by_component["api"] == 3
        assert health.recovery_success_rate == 0.8
        assert health.uptime_seconds == 3600.0
        assert health.component_status["api"] == "healthy"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
