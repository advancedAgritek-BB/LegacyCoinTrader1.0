"""
Comprehensive tests for Enhanced Circuit Breaker

This module tests all functionality of the Enhanced Circuit Breaker to ensure
proper risk management and system protection.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from crypto_bot.utils.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerEvent,
    CircuitState
)

class TestEnhancedCircuitBreaker:
    """Test suite for Enhanced Circuit Breaker."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            failure_window_seconds=60,
            recovery_timeout_seconds=120,
            success_threshold=2,
            max_drawdown_percent=5.0,
            max_daily_loss_percent=10.0,
            max_api_error_rate=0.2
        )
    
    @pytest.fixture
    def circuit_breaker(self, config):
        """Create circuit breaker instance for testing."""
        return EnhancedCircuitBreaker(config)
    
    def test_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.total_trades == 0
        assert circuit_breaker.metrics.successful_trades == 0
        assert circuit_breaker.metrics.failed_trades == 0
        assert circuit_breaker.metrics.total_pnl == 0.0
        assert len(circuit_breaker.failure_history) == 0
        assert len(circuit_breaker.success_history) == 0
        assert len(circuit_breaker.event_history) == 0
    
    def test_is_trading_allowed_closed(self, circuit_breaker):
        """Test trading allowed when circuit is closed."""
        circuit_breaker.state = CircuitState.CLOSED
        assert circuit_breaker.is_trading_allowed() == True
    
    def test_is_trading_allowed_open(self, circuit_breaker):
        """Test trading not allowed when circuit is open."""
        circuit_breaker.state = CircuitState.OPEN
        assert circuit_breaker.is_trading_allowed() == False
    
    def test_is_trading_allowed_half_open(self, circuit_breaker):
        """Test trading not allowed when circuit is half-open."""
        circuit_breaker.state = CircuitState.HALF_OPEN
        assert circuit_breaker.is_trading_allowed() == False
    
    def test_record_trade_success(self, circuit_breaker):
        """Test recording a successful trade."""
        circuit_breaker.record_trade(
            symbol='BTC/USD',
            side='buy',
            size=0.1,
            price=50000.0,
            pnl=100.0,
            success=True
        )
        
        assert circuit_breaker.metrics.total_trades == 1
        assert circuit_breaker.metrics.successful_trades == 1
        assert circuit_breaker.metrics.failed_trades == 0
        assert circuit_breaker.metrics.total_pnl == 100.0
        assert circuit_breaker.metrics.current_balance == 100.0
        assert len(circuit_breaker.success_history) == 1
        assert len(circuit_breaker.trade_history) == 1
        
        trade_record = circuit_breaker.trade_history[0]
        assert trade_record['symbol'] == 'BTC/USD'
        assert trade_record['side'] == 'buy'
        assert trade_record['size'] == 0.1
        assert trade_record['price'] == 50000.0
        assert trade_record['pnl'] == 100.0
        assert trade_record['success'] == True
    
    def test_record_trade_failure(self, circuit_breaker):
        """Test recording a failed trade."""
        circuit_breaker.record_trade(
            symbol='ETH/USD',
            side='sell',
            size=1.0,
            price=3000.0,
            pnl=-50.0,
            success=False
        )
        
        assert circuit_breaker.metrics.total_trades == 1
        assert circuit_breaker.metrics.successful_trades == 0
        assert circuit_breaker.metrics.failed_trades == 1
        assert circuit_breaker.metrics.total_pnl == -50.0
        assert circuit_breaker.metrics.current_balance == -50.0
        assert len(circuit_breaker.failure_history) == 1
        assert len(circuit_breaker.trade_history) == 1
        
        trade_record = circuit_breaker.trade_history[0]
        assert trade_record['success'] == False
        assert trade_record['pnl'] == -50.0
    
    def test_record_api_error(self, circuit_breaker):
        """Test recording an API error."""
        circuit_breaker.record_api_error()
        
        assert circuit_breaker.metrics.api_errors == 1
        assert circuit_breaker.metrics.api_requests == 0
        assert circuit_breaker.metrics.last_error_time is not None
    
    def test_record_api_request(self, circuit_breaker):
        """Test recording an API request."""
        circuit_breaker.record_api_request()
        
        assert circuit_breaker.metrics.api_requests == 1
        assert circuit_breaker.metrics.api_errors == 0
    
    def test_update_system_metrics(self, circuit_breaker):
        """Test updating system metrics."""
        circuit_breaker.update_system_metrics(memory_usage=75.5, cpu_usage=60.2)
        
        assert circuit_breaker.metrics.memory_usage == 75.5
        assert circuit_breaker.metrics.cpu_usage == 60.2
    
    def test_update_daily_pnl(self, circuit_breaker):
        """Test updating daily PnL."""
        circuit_breaker.update_daily_pnl(-5.5)
        
        assert circuit_breaker.metrics.daily_loss == -5.5
        assert len(circuit_breaker.daily_pnl_history) == 1
        assert circuit_breaker.daily_pnl_history[0] == -5.5
    
    def test_get_metrics(self, circuit_breaker):
        """Test getting metrics."""
        circuit_breaker.record_trade('BTC/USD', 'buy', 0.1, 50000.0, 100.0, True)
        circuit_breaker.record_api_error()
        
        metrics = circuit_breaker.get_metrics()
        
        assert metrics.total_trades == 1
        assert metrics.successful_trades == 1
        assert metrics.total_pnl == 100.0
        assert metrics.api_errors == 1
    
    def test_get_event_history(self, circuit_breaker):
        """Test getting event history."""
        # Add some events
        event1 = CircuitBreakerEvent(
            event_type="test1",
            timestamp=datetime.now(),
            state=CircuitState.CLOSED,
            reason="test",
            metrics=CircuitBreakerMetrics()
        )
        event2 = CircuitBreakerEvent(
            event_type="test2",
            timestamp=datetime.now(),
            state=CircuitState.OPEN,
            reason="test",
            metrics=CircuitBreakerMetrics()
        )
        
        circuit_breaker.event_history = [event1, event2]
        
        history = circuit_breaker.get_event_history()
        assert len(history) == 2
        assert history[0].event_type == "test1"
        assert history[1].event_type == "test2"
    
    def test_reset(self, circuit_breaker):
        """Test resetting circuit breaker."""
        # Add some data
        circuit_breaker.record_trade('BTC/USD', 'buy', 0.1, 50000.0, 100.0, True)
        circuit_breaker.record_api_error()
        circuit_breaker.state = CircuitState.OPEN
        
        circuit_breaker.reset()
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.total_trades == 0
        assert circuit_breaker.metrics.api_errors == 0
        assert len(circuit_breaker.failure_history) == 0
        assert len(circuit_breaker.success_history) == 0
        assert len(circuit_breaker.event_history) == 0
        assert len(circuit_breaker.trade_history) == 0
    
    def test_export_metrics(self, circuit_breaker):
        """Test exporting metrics."""
        circuit_breaker.record_trade('BTC/USD', 'buy', 0.1, 50000.0, 100.0, True)
        circuit_breaker.record_api_error()
        circuit_breaker.record_api_request()
        
        exported = circuit_breaker.export_metrics()
        
        assert exported['state'] == 'closed'
        assert exported['metrics']['total_trades'] == 1
        assert exported['metrics']['successful_trades'] == 1
        assert exported['metrics']['total_pnl'] == 100.0
        assert exported['metrics']['api_error_rate'] == 1.0  # 1 error / 1 request
        assert 'state_change_time' in exported
        assert 'last_trade_time' in exported
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, circuit_breaker):
        """Test starting and stopping monitoring."""
        # Start monitoring
        await circuit_breaker.start_monitoring()
        assert circuit_breaker.running == True
        assert circuit_breaker.monitoring_task is not None
        
        # Stop monitoring
        await circuit_breaker.stop_monitoring()
        assert circuit_breaker.running == False
        assert circuit_breaker.monitoring_task is None
    
    @pytest.mark.asyncio
    async def test_drawdown_exceeded_opens_circuit(self, circuit_breaker):
        """Test that exceeding drawdown opens the circuit."""
        # Set up initial balance
        circuit_breaker.metrics.current_balance = 1000.0
        circuit_breaker.metrics.peak_balance = 1000.0
        
        # Record a loss that exceeds drawdown threshold
        circuit_breaker.record_trade('BTC/USD', 'sell', 0.1, 50000.0, -60.0, True)
        
        # Trigger drawdown check
        await circuit_breaker._check_trading_performance()
        
        # Circuit should be open due to drawdown
        assert circuit_breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_daily_loss_exceeded_opens_circuit(self, circuit_breaker):
        """Test that exceeding daily loss opens the circuit."""
        # Set daily loss to exceed threshold
        circuit_breaker.metrics.daily_loss = -12.0  # Exceeds 10% threshold
        
        # Trigger daily loss check
        await circuit_breaker._check_trading_performance()
        
        # Circuit should be open due to daily loss
        assert circuit_breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_api_error_rate_opens_circuit(self, circuit_breaker):
        """Test that high API error rate opens the circuit."""
        # Record multiple API errors
        for _ in range(5):
            circuit_breaker.record_api_request()
            circuit_breaker.record_api_error()
        
        # Trigger system health check
        await circuit_breaker._check_system_health()
        
        # Circuit should be open due to high error rate
        assert circuit_breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_consecutive_losses_opens_circuit(self, circuit_breaker):
        """Test that consecutive losses opens the circuit."""
        # Record consecutive losses
        for i in range(4):  # Exceeds failure threshold of 3
            circuit_breaker.record_trade(f'SYMBOL{i}', 'buy', 0.1, 100.0, -10.0, False)
        
        # Trigger risk metrics check
        await circuit_breaker._check_risk_metrics()
        
        # Circuit should be open due to consecutive losses
        assert circuit_breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_recovery_timeout_transitions_to_half_open(self, circuit_breaker):
        """Test that recovery timeout transitions to half-open."""
        # Set circuit to open
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.state_change_time = datetime.now() - timedelta(seconds=130)  # Exceeds 120s timeout
        
        # Trigger state update
        await circuit_breaker._update_state()
        
        # Circuit should transition to half-open
        assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_success_threshold_met_closes_circuit(self, circuit_breaker):
        """Test that meeting success threshold closes the circuit."""
        # Set circuit to half-open
        circuit_breaker.state = CircuitState.HALF_OPEN
        
        # Add recent successes
        recent_time = datetime.now() - timedelta(seconds=30)
        for _ in range(3):  # Exceeds success threshold of 2
            circuit_breaker.success_history.append(recent_time)
        
        # Trigger state update
        await circuit_breaker._update_state()
        
        # Circuit should close
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_state_change_callbacks(self, circuit_breaker):
        """Test state change callbacks."""
        callback_called = False
        callback_args = None
        
        async def test_callback(old_state, new_state, reason, details):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = (old_state, new_state, reason, details)
        
        circuit_breaker.add_state_change_callback(test_callback)
        
        # Trigger circuit open
        await circuit_breaker._trigger_circuit_open("test_reason", "test_details")
        
        # Check callback was called
        assert callback_called == True
        assert callback_args[0] == CircuitState.CLOSED
        assert callback_args[1] == CircuitState.OPEN
        assert callback_args[2] == "test_reason"
        assert callback_args[3] == "test_details"
    
    def test_cleanup_history(self, circuit_breaker):
        """Test cleanup of old history entries."""
        # Add old entries
        old_time = datetime.now() - timedelta(seconds=120)  # Outside window
        recent_time = datetime.now() - timedelta(seconds=30)  # Inside window
        
        circuit_breaker.failure_history = [old_time, recent_time]
        circuit_breaker.success_history = [old_time, recent_time]
        
        # Add many events
        for i in range(150):
            event = CircuitBreakerEvent(
                event_type=f"test{i}",
                timestamp=datetime.now(),
                state=CircuitState.CLOSED,
                reason="test",
                metrics=CircuitBreakerMetrics()
            )
            circuit_breaker.event_history.append(event)
        
        # Add many trades
        for i in range(1500):
            circuit_breaker.trade_history.append({'trade': i})
        
        # Trigger cleanup
        circuit_breaker._cleanup_history()
        
        # Check cleanup
        assert len(circuit_breaker.failure_history) == 1  # Only recent
        assert len(circuit_breaker.success_history) == 1  # Only recent
        assert len(circuit_breaker.event_history) == 100  # Max 100
        assert len(circuit_breaker.trade_history) == 1000  # Max 1000
    
    def test_peak_balance_tracking(self, circuit_breaker):
        """Test peak balance tracking."""
        # Initial balance
        circuit_breaker.metrics.current_balance = 1000.0
        circuit_breaker.metrics.peak_balance = 1000.0
        
        # Record a gain
        circuit_breaker.record_trade('BTC/USD', 'buy', 0.1, 50000.0, 200.0, True)
        
        # Peak should be updated
        assert circuit_breaker.metrics.peak_balance == 1200.0
        
        # Record a loss
        circuit_breaker.record_trade('ETH/USD', 'sell', 1.0, 3000.0, -100.0, True)
        
        # Peak should remain the same
        assert circuit_breaker.metrics.peak_balance == 1200.0
        assert circuit_breaker.metrics.current_balance == 1100.0
    
    def test_drawdown_calculation(self, circuit_breaker):
        """Test drawdown calculation."""
        # Set up peak balance
        circuit_breaker.metrics.peak_balance = 1000.0
        circuit_breaker.metrics.current_balance = 900.0
        
        # Calculate drawdown
        drawdown = (1000.0 - 900.0) / 1000.0 * 100
        assert drawdown == 10.0
        
        # Record a trade that increases drawdown
        circuit_breaker.record_trade('BTC/USD', 'sell', 0.1, 50000.0, -50.0, True)
        
        # Drawdown should be updated
        assert circuit_breaker.metrics.current_drawdown == 15.0  # (1000 - 850) / 1000 * 100
        assert circuit_breaker.metrics.max_drawdown == 15.0

class TestCircuitBreakerConfig:
    """Test suite for CircuitBreakerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.failure_window_seconds == 300
        assert config.recovery_timeout_seconds == 600
        assert config.success_threshold == 3
        assert config.max_drawdown_percent == 10.0
        assert config.max_daily_loss_percent == 15.0
        assert config.max_api_error_rate == 0.1
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            max_drawdown_percent=5.0,
            max_daily_loss_percent=20.0
        )
        
        assert config.failure_threshold == 10
        assert config.max_drawdown_percent == 5.0
        assert config.max_daily_loss_percent == 20.0

class TestCircuitBreakerMetrics:
    """Test suite for CircuitBreakerMetrics."""
    
    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = CircuitBreakerMetrics()
        
        assert metrics.total_trades == 0
        assert metrics.successful_trades == 0
        assert metrics.failed_trades == 0
        assert metrics.total_pnl == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.current_drawdown == 0.0
        assert metrics.peak_balance == 0.0
        assert metrics.current_balance == 0.0
        assert metrics.daily_loss == 0.0
        assert metrics.api_errors == 0
        assert metrics.api_requests == 0
        assert metrics.memory_usage == 0.0
        assert metrics.cpu_usage == 0.0
        assert metrics.last_trade_time is None
        assert metrics.last_error_time is None

class TestCircuitBreakerEvent:
    """Test suite for CircuitBreakerEvent."""
    
    def test_event_creation(self):
        """Test creating a circuit breaker event."""
        metrics = CircuitBreakerMetrics()
        event = CircuitBreakerEvent(
            event_type="test_event",
            timestamp=datetime.now(),
            state=CircuitState.OPEN,
            reason="test_reason",
            metrics=metrics,
            details={"test": "value"}
        )
        
        assert event.event_type == "test_event"
        assert event.state == CircuitState.OPEN
        assert event.reason == "test_reason"
        assert event.metrics == metrics
        assert event.details["test"] == "value"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
