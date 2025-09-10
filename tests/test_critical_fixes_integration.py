"""
Comprehensive Integration Test for Critical Fixes

This module tests the integration of all critical fixes to ensure they work
together properly in the trading pipeline.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from crypto_bot.utils.unified_position_manager import (
    UnifiedPositionManager, 
    PositionConflict, 
    PositionSyncStats
)
from crypto_bot.utils.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    CircuitBreakerConfig,
    CircuitState
)
from crypto_bot.utils.enhanced_error_handler import (
    EnhancedErrorHandler,
    ErrorSeverity,
    RecoveryAction,
    ErrorContext
)

class TestCriticalFixesIntegration:
    """Integration tests for all critical fixes working together."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for test logs."""
        temp_dir = tempfile.mkdtemp()
        log_dir = Path(temp_dir) / "logs"
        log_dir.mkdir()
        yield log_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_trade_manager(self):
        """Create mock TradeManager."""
        tm = Mock()
        tm.get_all_positions.return_value = []
        tm.update_position = Mock()
        tm.close_position = Mock()
        return tm
    
    @pytest.fixture
    def mock_paper_wallet(self):
        """Create mock paper wallet."""
        pw = Mock()
        pw.positions = {}
        return pw
    
    @pytest.fixture
    def circuit_breaker_config(self):
        """Create circuit breaker configuration."""
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
    def circuit_breaker(self, circuit_breaker_config):
        """Create circuit breaker instance."""
        return EnhancedCircuitBreaker(circuit_breaker_config)
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler instance."""
        config = {'max_error_history': 100}
        return EnhancedErrorHandler(config)
    
    @pytest.fixture
    def unified_position_manager(self, mock_trade_manager, mock_paper_wallet, temp_log_dir):
        """Create unified position manager instance."""
        config = {
            'position_sync_interval': 1,
            'max_conflict_history': 10
        }
        
        with patch('crypto_bot.utils.unified_position_manager.Path') as mock_path:
            mock_path.return_value = temp_log_dir / "positions.log"
            
            upm = UnifiedPositionManager(
                trade_manager=mock_trade_manager,
                paper_wallet=mock_paper_wallet,
                config=config
            )
            
            # Clear any existing data
            upm.position_cache.clear()
            upm.paper_wallet.positions.clear()
            upm.trade_manager.get_all_positions.return_value = []
            
            return upm
    
    def test_initialization_integration(self, unified_position_manager, circuit_breaker, error_handler):
        """Test that all components initialize properly together."""
        # Test Unified Position Manager
        assert unified_position_manager.trade_manager is not None
        assert unified_position_manager.paper_wallet is not None
        assert unified_position_manager.config is not None
        
        # Test Circuit Breaker
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.config is not None
        assert circuit_breaker.metrics.total_trades == 0
        
        # Test Error Handler
        assert error_handler.config is not None
        assert len(error_handler.recovery_strategies) > 0
        assert error_handler.system_health.total_errors == 0
        
        # Test integration points
        error_handler.register_circuit_breaker(circuit_breaker)
        assert error_handler.circuit_breaker == circuit_breaker
    
    @pytest.mark.asyncio
    async def test_error_handling_with_circuit_breaker(self, error_handler, circuit_breaker):
        """Test error handling integration with circuit breaker."""
        # Register circuit breaker with error handler
        error_handler.register_circuit_breaker(circuit_breaker)
        
        # Trigger a critical error that should open circuit breaker
        error = Exception("System memory critical")
        error_context = error_handler.handle_error(error, "system", "monitor")
        
        # Wait for recovery action to execute
        await asyncio.sleep(0.1)
        
        # Circuit breaker should be triggered
        assert circuit_breaker.metrics.api_errors == 1
        
        # Error should be recorded in error handler
        assert error_handler.system_health.total_errors == 1
        assert error_context.severity == ErrorSeverity.CRITICAL
        assert error_context.recovery_action == RecoveryAction.CIRCUIT_BREAK
    
    @pytest.mark.asyncio
    async def test_position_sync_with_error_handling(self, unified_position_manager, error_handler):
        """Test position synchronization with error handling."""
        # Mock TradeManager to raise an error
        unified_position_manager.trade_manager.get_all_positions.side_effect = Exception("API error")
        
        # Handle the error
        error = Exception("API error")
        error_context = error_handler.handle_error(error, "trade_manager", "get_positions")
        
        # Position sync should still work (graceful degradation)
        positions = unified_position_manager.get_unified_positions()
        assert isinstance(positions, dict)
        
        # Error should be recorded
        assert error_handler.system_health.total_errors == 1
        assert error_context.severity == ErrorSeverity.MEDIUM
        assert error_context.recovery_action == RecoveryAction.RETRY
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_position_management(self, circuit_breaker, unified_position_manager):
        """Test circuit breaker integration with position management."""
        # Start monitoring
        await circuit_breaker.start_monitoring()
        
        # Record some trades to trigger circuit breaker
        for i in range(4):  # Exceeds failure threshold
            circuit_breaker.record_trade(f'SYMBOL{i}', 'buy', 0.1, 100.0, -10.0, False)
        
        # Wait for monitoring to detect the issue
        await asyncio.sleep(0.2)
        
        # Manually trigger state update to open circuit
        await circuit_breaker._update_state()
        
        # Circuit should be open
        assert circuit_breaker.state == CircuitState.OPEN
        assert not circuit_breaker.is_trading_allowed()
        
        # Position manager should still work (read-only operations)
        positions = unified_position_manager.get_unified_positions()
        assert isinstance(positions, dict)
        
        # Stop monitoring
        await circuit_breaker.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_error_recovery_strategies(self, error_handler, circuit_breaker):
        """Test different error recovery strategies."""
        # Register circuit breaker
        error_handler.register_circuit_breaker(circuit_breaker)
        
        # Test API error (should retry)
        api_error = Exception("API rate limit exceeded")
        api_context = error_handler.handle_error(api_error, "api_client", "fetch_data")
        assert api_context.recovery_action == RecoveryAction.RETRY
        assert api_context.max_retries == 5
        
        # Test network error (should retry with different settings)
        network_error = Exception("Connection timeout")
        network_context = error_handler.handle_error(network_error, "network", "connect")
        assert network_context.recovery_action == RecoveryAction.RETRY
        assert network_context.max_retries == 3
        
        # Test trading error (should alert)
        trading_error = Exception("Trade execution failed")
        trading_context = error_handler.handle_error(trading_error, "trading", "execute_order")
        assert trading_context.recovery_action == RecoveryAction.ALERT
        assert trading_context.max_retries == 1
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, error_handler, circuit_breaker, unified_position_manager):
        """Test comprehensive system health monitoring."""
        # Simulate various system events
        
        # Record API errors
        for _ in range(3):
            error_handler.handle_error(Exception("API error"), "api", "call")
        
        # Record trades
        circuit_breaker.record_trade('BTC/USD', 'buy', 0.1, 50000.0, 100.0, True)
        circuit_breaker.record_trade('ETH/USD', 'sell', 1.0, 3000.0, -50.0, False)
        
        # Update system metrics
        circuit_breaker.update_system_metrics(memory_usage=75.0, cpu_usage=60.0)
        
        # Check system health
        error_stats = error_handler.get_error_stats()
        circuit_metrics = circuit_breaker.get_metrics()
        
        assert error_stats['total_errors'] == 3
        assert circuit_metrics.total_trades == 2
        assert circuit_metrics.successful_trades == 1
        assert circuit_metrics.failed_trades == 1
        assert circuit_metrics.memory_usage == 75.0
        assert circuit_metrics.cpu_usage == 60.0
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, unified_position_manager, error_handler, circuit_breaker):
        """Test graceful degradation when components fail."""
        # Simulate TradeManager failure
        unified_position_manager.trade_manager.get_all_positions.side_effect = Exception("TradeManager down")
        
        # Handle the error
        error_handler.handle_error(Exception("TradeManager down"), "trade_manager", "get_positions")
        
        # System should still function with paper wallet
        unified_position_manager.paper_wallet.positions = {
            'BTC/USD': {
                'symbol': 'BTC/USD',
                'side': 'buy',
                'size': 0.1,
                'entry_price': 50000.0,
                'current_price': 51000.0,
                'pnl': 100.0
            }
        }
        
        positions = unified_position_manager.get_unified_positions()
        assert 'BTC/USD' in positions
        assert positions['BTC/USD']['side'] == 'buy'
    
    @pytest.mark.asyncio
    async def test_alert_integration(self, error_handler):
        """Test alert system integration."""
        alert_received = False
        alert_message = None
        
        async def alert_callback(message):
            nonlocal alert_received, alert_message
            alert_received = True
            alert_message = message
        
        error_handler.add_alert_callback(alert_callback)
        
        # Trigger a trading error that should send alert
        error_handler.handle_error(Exception("Trade execution failed"), "trading", "execute_order")
        
        # Wait for alert to be sent
        await asyncio.sleep(0.1)
        
        assert alert_received
        assert alert_message['type'] == 'error_alert'
        assert alert_message['component'] == 'trading'
        assert alert_message['operation'] == 'execute_order'
        assert alert_message['severity'] == 'high'
    
    @pytest.mark.asyncio
    async def test_recovery_workflow(self, error_handler, circuit_breaker):
        """Test complete recovery workflow."""
        # Start monitoring
        await circuit_breaker.start_monitoring()
        
        # Register circuit breaker
        error_handler.register_circuit_breaker(circuit_breaker)
        
        # Trigger circuit breaker with critical error
        error_handler.handle_error(Exception("System memory critical"), "system", "monitor")
        await asyncio.sleep(0.1)
        
        # Manually open the circuit breaker
        await circuit_breaker._trigger_circuit_open("test", "Manual trigger for testing")
        
        # Circuit should be open
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Simulate recovery by recording successful operations
        circuit_breaker.state = CircuitState.HALF_OPEN
        circuit_breaker.state_change_time = datetime.now()
        
        # Add recent successes
        recent_time = datetime.now() - timedelta(seconds=30)
        for _ in range(3):  # Exceeds success threshold
            circuit_breaker.success_history.append(recent_time)
        
        # Trigger state update
        await circuit_breaker._update_state()
        
        # Circuit should close
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Stop monitoring
        await circuit_breaker.stop_monitoring()
    
    def test_data_consistency_across_components(self, unified_position_manager, circuit_breaker, error_handler):
        """Test data consistency across all components."""
        # Add position data
        position_data = {
            'symbol': 'BTC/USD',
            'side': 'buy',
            'size': 0.1,
            'entry_price': 50000.0,
            'current_price': 51000.0,
            'pnl': 100.0
        }
        
        unified_position_manager.update_position('BTC/USD', position_data)
        
        # Record trade in circuit breaker
        circuit_breaker.record_trade('BTC/USD', 'buy', 0.1, 50000.0, 100.0, True)
        
        # Check consistency
        positions = unified_position_manager.get_unified_positions()
        circuit_metrics = circuit_breaker.get_metrics()
        
        assert 'BTC/USD' in positions
        assert positions['BTC/USD']['side'] == 'buy'
        assert circuit_metrics.total_trades == 1
        assert circuit_metrics.successful_trades == 1
        assert circuit_metrics.total_pnl == 100.0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, unified_position_manager, circuit_breaker, error_handler):
        """Test concurrent operations across all components."""
        import asyncio
        
        # Verify we start with a clean state
        initial_positions = unified_position_manager.get_unified_positions()
        if len(initial_positions) > 0:
            # If there are leftover positions, clear them by updating the cache
            unified_position_manager._update_position_cache({}, {}, {})
        
        # Simulate concurrent operations
        async def concurrent_position_update():
            for i in range(5):
                position_data = {
                    'symbol': f'SYMBOL{i}',
                    'side': 'buy',
                    'size': 0.1,
                    'entry_price': 100.0,
                    'current_price': 110.0,
                    'pnl': 10.0
                }
                unified_position_manager.update_position(f'SYMBOL{i}', position_data)
                await asyncio.sleep(0.01)
        
        async def concurrent_trade_recording():
            for i in range(5):
                circuit_breaker.record_trade(f'SYMBOL{i}', 'buy', 0.1, 100.0, 10.0, True)
                await asyncio.sleep(0.01)
        
        async def concurrent_error_handling():
            for i in range(3):
                error_handler.handle_error(Exception(f"Error {i}"), f"component{i}", f"operation{i}")
                await asyncio.sleep(0.01)
        
        # Run all operations concurrently
        await asyncio.gather(
            concurrent_position_update(),
            concurrent_trade_recording(),
            concurrent_error_handling()
        )
        
        # Check results
        positions = unified_position_manager.get_unified_positions()
        circuit_metrics = circuit_breaker.get_metrics()
        error_stats = error_handler.get_error_stats()
        
        # Only count the positions we added in this test
        test_positions = {k: v for k, v in positions.items() if k.startswith('SYMBOL')}
        assert len(test_positions) == 5
        assert circuit_metrics.total_trades == 5
        assert circuit_metrics.successful_trades == 5
        assert error_stats['total_errors'] == 3
    
    def test_configuration_management(self, unified_position_manager, circuit_breaker, error_handler):
        """Test configuration management across components."""
        # Test Unified Position Manager config
        assert unified_position_manager.sync_interval == 1
        assert unified_position_manager.max_conflict_history == 100  # Default value
        
        # Test Circuit Breaker config
        assert circuit_breaker.config.failure_threshold == 3
        assert circuit_breaker.config.max_drawdown_percent == 5.0
        assert circuit_breaker.config.max_daily_loss_percent == 10.0
        
        # Test Error Handler config
        assert error_handler.max_error_history == 100
        assert len(error_handler.recovery_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_and_telemetry(self, unified_position_manager, circuit_breaker, error_handler):
        """Test monitoring and telemetry across all components."""
        # Generate some activity
        unified_position_manager.update_position('BTC/USD', {
            'symbol': 'BTC/USD',
            'side': 'buy',
            'size': 0.1,
            'entry_price': 50000.0,
            'current_price': 51000.0,
            'pnl': 100.0
        })
        
        circuit_breaker.record_trade('BTC/USD', 'buy', 0.1, 50000.0, 100.0, True)
        error_handler.handle_error(Exception("Test error"), "test", "test")
        
        # Get telemetry data
        position_stats = unified_position_manager.get_sync_stats()
        circuit_metrics = circuit_breaker.export_metrics()
        error_stats = error_handler.get_error_stats()
        
        # Verify telemetry data
        assert position_stats.total_syncs >= 0
        assert circuit_metrics['state'] == 'closed'
        assert circuit_metrics['metrics']['total_trades'] == 1
        assert error_stats['total_errors'] == 1
        assert error_stats['uptime_seconds'] > 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
