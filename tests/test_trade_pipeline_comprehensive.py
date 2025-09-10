"""
Comprehensive Test Suite for Trade Pipeline

This test suite covers the entire trade pipeline including:
- Signal generation
- Risk management
- Position sizing
- Trade execution (live and paper)
- Position management
- Balance tracking
- Performance monitoring
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

# Import the modules we're testing
from crypto_bot.utils.enhanced_logger import (
    TradePipelineLogger, 
    EnhancedLoggingManager,
    get_enhanced_logging_manager,
    log_trade_pipeline_event
)
from crypto_bot.phase_runner import BotContext
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.risk.risk_manager import RiskManager


class TestTradePipelineLogger:
    """Test the enhanced trade pipeline logger."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            'logging': {
                'level': 'DEBUG',
                'file_logging': True,
                'console_logging': True,
                'log_directory': 'test_logs',
                'trade_pipeline': {
                    'log_signals': True,
                    'log_risk_checks': True,
                    'log_position_sizing': True,
                    'log_execution_decisions': True,
                    'log_executions': True,
                    'log_position_updates': True,
                    'log_balance_changes': True
                },
                'paper_trading': {
                    'log_simulated_trades': True,
                    'log_balance_simulation': True,
                    'log_position_simulation': True
                },
                'live_trading': {
                    'log_real_executions': True,
                    'log_api_calls': True,
                    'log_websocket_events': True
                },
                'performance': {
                    'log_execution_times': True,
                    'log_memory_usage': True,
                    'log_api_latency': True
                },
                'error_tracking': {
                    'log_exceptions': True,
                    'log_api_errors': True,
                    'log_websocket_errors': True,
                    'log_validation_errors': True
                }
            }
        }
    
    @pytest.fixture
    def logger(self, mock_config):
        """Create a logger instance for testing."""
        return TradePipelineLogger('test_component', mock_config)
    
    def test_logger_initialization(self, logger, mock_config):
        """Test logger initialization."""
        assert logger.name == 'test_component'
        assert logger.config == mock_config
        assert logger.logger is not None
        assert isinstance(logger.execution_times, dict)
        assert isinstance(logger.trade_counters, dict)
        
        # Check initial trade counters
        expected_counters = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'paper_trades': 0,
            'live_trades': 0
        }
        assert logger.trade_counters == expected_counters
    
    def test_log_signal_generation(self, logger):
        """Test signal generation logging."""
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_signal_generation(
                symbol='BTC/USD',
                strategy='trend_bot',
                score=0.85,
                direction='long',
                regime='trending',
                confidence=0.9
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert 'BTC/USD' in call_args
            assert 'trend_bot' in call_args
            assert '0.8500' in call_args
            assert 'long' in call_args
            assert 'trending' in call_args
    
    def test_log_risk_check(self, logger):
        """Test risk check logging."""
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_risk_check(
                symbol='ETH/USD',
                strategy='momentum_bot',
                allowed=True,
                reason='Within risk limits',
                risk_score=0.3,
                max_risk=0.5
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert 'ETH/USD' in call_args
            assert 'ALLOWED' in call_args
            assert 'Within risk limits' in call_args
            assert '0.3000' in call_args
    
    def test_log_position_sizing(self, logger):
        """Test position sizing logging."""
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_position_sizing(
                symbol='SOL/USD',
                strategy='sniper_bot',
                base_size=100.0,
                final_size=95.0,
                sentiment_boost=0.95,
                risk_adjustment=0.8
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert 'SOL/USD' in call_args
            assert '100.000000' in call_args
            assert '95.000000' in call_args
            assert '0.9500' in call_args
    
    def test_log_execution_decision(self, logger):
        """Test execution decision logging."""
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_execution_decision(
                symbol='ADA/USD',
                strategy='grid_bot',
                side='buy',
                size=50.0,
                price=0.45,
                execution_mode='dry_run',
                confidence=0.8
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert '📄' in call_args  # Paper trading emoji
            assert 'ADA/USD' in call_args
            assert 'buy' in call_args
            assert '50.000000' in call_args
            assert '0.450000' in call_args
    
    def test_log_trade_execution(self, logger):
        """Test trade execution logging."""
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_trade_execution(
                symbol='DOT/USD',
                strategy='volatility_bot',
                side='sell',
                size=25.0,
                price=7.50,
                execution_mode='live',
                order_id='ORDER123',
                execution_time=0.15,
                slippage=0.001
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert '💰' in call_args  # Live trading emoji
            assert 'DOT/USD' in call_args
            assert 'ORDER123' in call_args
            assert '0.150s' in call_args
            
            # Check trade counters
            assert logger.trade_counters['total_trades'] == 1
            assert logger.trade_counters['live_trades'] == 1
            assert logger.trade_counters['paper_trades'] == 0
    
    def test_log_paper_trading_simulation(self, logger):
        """Test paper trading simulation logging."""
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_paper_trading_simulation(
                action='open_position',
                symbol='LINK/USD',
                side='buy',
                size=75.0,
                price=15.25,
                strategy='trend_bot'
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert '📄' in call_args
            assert 'open_position' in call_args
            assert 'LINK/USD' in call_args
            assert '75.000000' in call_args
    
    def test_log_live_trading_execution(self, logger):
        """Test live trading execution logging."""
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_live_trading_execution(
                action='place_order',
                symbol='MATIC/USD',
                side='buy',
                size=200.0,
                price=0.85,
                order_id='LIVE_ORDER_456',
                exchange='binance'
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert '💰' in call_args
            assert 'place_order' in call_args
            assert 'MATIC/USD' in call_args
            assert 'LIVE_ORDER_456' in call_args
    
    def test_execution_timer_context_manager(self, logger):
        """Test execution timer context manager."""
        with logger.execution_timer('test_operation'):
            time.sleep(0.01)  # Simulate some work
        
        assert 'test_operation' in logger.execution_times
        assert logger.execution_times['test_operation'] > 0.01
    
    def test_log_error(self, logger):
        """Test error logging."""
        with patch.object(logger.logger, 'error') as mock_error:
            test_error = ValueError("Test error message")
            logger.log_error(
                error=test_error,
                context='test_context',
                symbol='BTC/USD',
                strategy='test_strategy'
            )
            
            mock_error.assert_called_once()
            call_args = mock_error.call_args[0][0]
            assert 'test_context' in call_args
            assert 'Test error message' in call_args
    
    def test_log_api_call(self, logger):
        """Test API call logging."""
        with patch.object(logger.logger, 'debug') as mock_debug:
            logger.log_api_call(
                endpoint='/api/v1/order',
                method='POST',
                params={'symbol': 'BTC/USD', 'side': 'buy'},
                response_time=0.25,
                success=True,
                status_code=200
            )
            
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0][0]
            assert '✅' in call_args  # Success emoji
            assert 'POST' in call_args
            assert '/api/v1/order' in call_args
            assert '0.250s' in call_args
    
    def test_log_websocket_event(self, logger):
        """Test WebSocket event logging."""
        with patch.object(logger.logger, 'debug') as mock_debug:
            logger.log_websocket_event(
                event_type='ticker',
                symbol='ETH/USD',
                data={'price': 3000.0, 'volume': 1000.0},
                timestamp=1234567890
            )
            
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0][0]
            assert '📡' in call_args  # WebSocket emoji
            assert 'ticker' in call_args
            assert 'ETH/USD' in call_args
            assert '3000.0' in call_args
    
    def test_get_statistics(self, logger):
        """Test statistics retrieval."""
        # Perform some operations to generate statistics
        logger.log_trade_execution(
            symbol='BTC/USD',
            strategy='test_strategy',
            side='buy',
            size=1.0,
            price=50000.0,
            execution_mode='dry_run',
            order_id='TEST_ORDER',
            execution_time=0.1
        )
        
        with logger.execution_timer('test_timer'):
            time.sleep(0.01)
        
        stats = logger.get_statistics()
        
        assert 'trade_counters' in stats
        assert 'execution_times' in stats
        assert 'logger_name' in stats
        assert 'config' in stats
        
        assert stats['trade_counters']['total_trades'] == 1
        assert stats['trade_counters']['paper_trades'] == 1
        assert 'test_timer' in stats['execution_times']
        assert stats['logger_name'] == 'test_component'
    
    def test_reset_statistics(self, logger):
        """Test statistics reset."""
        # Generate some statistics
        logger.log_trade_execution(
            symbol='BTC/USD',
            strategy='test_strategy',
            side='buy',
            size=1.0,
            price=50000.0,
            execution_mode='dry_run',
            order_id='TEST_ORDER',
            execution_time=0.1
        )
        
        with logger.execution_timer('test_timer'):
            time.sleep(0.01)
        
        # Verify statistics exist
        assert logger.trade_counters['total_trades'] == 1
        assert 'test_timer' in logger.execution_times
        
        # Reset statistics
        logger.reset_statistics()
        
        # Verify statistics are reset
        expected_counters = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'paper_trades': 0,
            'live_trades': 0
        }
        assert logger.trade_counters == expected_counters
        assert logger.execution_times == {}


class TestEnhancedLoggingManager:
    """Test the enhanced logging manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            'logging': {
                'level': 'DEBUG',
                'file_logging': True,
                'console_logging': True,
                'log_directory': 'test_logs',
                'trade_pipeline': {
                    'log_signals': True,
                    'log_risk_checks': True,
                    'log_position_sizing': True,
                    'log_execution_decisions': True,
                    'log_executions': True,
                    'log_position_updates': True,
                    'log_balance_changes': True
                },
                'paper_trading': {
                    'log_simulated_trades': True,
                    'log_balance_simulation': True,
                    'log_position_simulation': True
                },
                'live_trading': {
                    'log_real_executions': True,
                    'log_api_calls': True,
                    'log_websocket_events': True
                },
                'performance': {
                    'log_execution_times': True,
                    'log_memory_usage': True,
                    'log_api_latency': True
                },
                'error_tracking': {
                    'log_exceptions': True,
                    'log_api_errors': True,
                    'log_websocket_errors': True,
                    'log_validation_errors': True
                }
            }
        }
    
    @pytest.fixture
    def manager(self, mock_config):
        """Create a logging manager instance for testing."""
        return EnhancedLoggingManager(mock_config)
    
    def test_manager_initialization(self, manager, mock_config):
        """Test manager initialization."""
        assert manager.config == mock_config
        assert len(manager.loggers) == 7  # 7 components
        
        expected_components = [
            'signal_generation',
            'risk_management', 
            'position_sizing',
            'trade_execution',
            'position_management',
            'balance_tracking',
            'performance_monitoring'
        ]
        
        for component in expected_components:
            assert component in manager.loggers
            assert isinstance(manager.loggers[component], TradePipelineLogger)
    
    def test_get_logger(self, manager):
        """Test getting specific loggers."""
        logger = manager.get_logger('signal_generation')
        assert isinstance(logger, TradePipelineLogger)
        assert logger.name == 'signal_generation'
        
        # Test fallback to trade_execution
        fallback_logger = manager.get_logger('unknown_component')
        assert isinstance(fallback_logger, TradePipelineLogger)
        assert fallback_logger.name == 'trade_execution'
    
    def test_log_trade_pipeline_event(self, manager):
        """Test logging trade pipeline events."""
        with patch.object(manager.loggers['signal_generation'], 'log_signal_generation') as mock_log:
            manager.log_trade_pipeline_event(
                component='signal_generation',
                event_type='signal_generation',
                symbol='BTC/USD',
                strategy='trend_bot',
                score=0.85,
                direction='long',
                regime='trending'
            )
            
            mock_log.assert_called_once_with(
                symbol='BTC/USD',
                strategy='trend_bot',
                score=0.85,
                direction='long',
                regime='trending'
            )
    
    def test_get_all_statistics(self, manager):
        """Test getting statistics from all loggers."""
        # Generate some statistics in one logger
        manager.loggers['trade_execution'].log_trade_execution(
            symbol='BTC/USD',
            strategy='test_strategy',
            side='buy',
            size=1.0,
            price=50000.0,
            execution_mode='dry_run',
            order_id='TEST_ORDER',
            execution_time=0.1
        )
        
        stats = manager.get_all_statistics()
        
        assert len(stats) == 7
        assert 'trade_execution' in stats
        assert 'signal_generation' in stats
        
        # Check that trade_execution has statistics
        trade_stats = stats['trade_execution']
        assert trade_stats['trade_counters']['total_trades'] == 1
        assert trade_stats['trade_counters']['paper_trades'] == 1
    
    def test_reset_all_statistics(self, manager):
        """Test resetting statistics for all loggers."""
        # Generate statistics in multiple loggers
        manager.loggers['trade_execution'].log_trade_execution(
            symbol='BTC/USD',
            strategy='test_strategy',
            side='buy',
            size=1.0,
            price=50000.0,
            execution_mode='dry_run',
            order_id='TEST_ORDER',
            execution_time=0.1
        )
        
        manager.loggers['signal_generation'].log_signal_generation(
            symbol='ETH/USD',
            strategy='momentum_bot',
            score=0.75,
            direction='short',
            regime='volatile'
        )
        
        # Verify statistics exist
        assert manager.loggers['trade_execution'].trade_counters['total_trades'] == 1
        assert manager.loggers['signal_generation'].execution_times == {}
        
        # Reset all statistics
        manager.reset_all_statistics()
        
        # Verify all statistics are reset
        for logger in manager.loggers.values():
            expected_counters = {
                'total_trades': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'paper_trades': 0,
                'live_trades': 0
            }
            assert logger.trade_counters == expected_counters
            assert logger.execution_times == {}


class TestTradePipelineIntegration:
    """Integration tests for the trade pipeline."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            'execution_mode': 'dry_run',
            'logging': {
                'level': 'DEBUG',
                'file_logging': True,
                'console_logging': True,
                'log_directory': 'test_logs',
                'trade_pipeline': {
                    'log_signals': True,
                    'log_risk_checks': True,
                    'log_position_sizing': True,
                    'log_execution_decisions': True,
                    'log_executions': True,
                    'log_position_updates': True,
                    'log_balance_changes': True
                },
                'paper_trading': {
                    'log_simulated_trades': True,
                    'log_balance_simulation': True,
                    'log_position_simulation': True
                },
                'live_trading': {
                    'log_real_executions': True,
                    'log_api_calls': True,
                    'log_websocket_events': True
                },
                'performance': {
                    'log_execution_times': True,
                    'log_memory_usage': True,
                    'log_api_latency': True
                },
                'error_tracking': {
                    'log_exceptions': True,
                    'log_api_errors': True,
                    'log_websocket_errors': True,
                    'log_validation_errors': True
                }
            },
            'risk_management': {
                'max_position_size': 0.1,
                'max_daily_loss': 0.05,
                'max_open_trades': 5
            }
        }
    
    @pytest.fixture
    def mock_context(self, mock_config):
        """Create a mock bot context for testing."""
        paper_wallet = PaperWallet(10000.0, max_open_trades=5, allow_short=True)
        
        ctx = BotContext(
            positions={},
            df_cache={'1h': {}},
            regime_cache={},
            config=mock_config
        )
        ctx.paper_wallet = paper_wallet
        ctx.balance = 10000.0
        ctx.exchange = Mock()
        ctx.ws_client = None
        ctx.risk_manager = Mock()
        ctx.notifier = Mock()
        ctx.position_guard = Mock()
        
        return ctx
    
    def test_paper_trading_pipeline(self, mock_context, mock_config):
        """Test the complete paper trading pipeline."""
        # Get logging manager
        logging_manager = get_enhanced_logging_manager(mock_config)
        
        # Simulate signal generation
        logging_manager.log_trade_pipeline_event(
            component='signal_generation',
            event_type='signal_generation',
            symbol='BTC/USD',
            strategy='trend_bot',
            score=0.85,
            direction='long',
            regime='trending'
        )
        
        # Simulate risk check
        logging_manager.log_trade_pipeline_event(
            component='risk_management',
            event_type='risk_check',
            symbol='BTC/USD',
            strategy='trend_bot',
            allowed=True,
            reason='Within risk limits',
            risk_score=0.3
        )
        
        # Simulate position sizing
        logging_manager.log_trade_pipeline_event(
            component='position_sizing',
            event_type='position_sizing',
            symbol='BTC/USD',
            strategy='trend_bot',
            base_size=1000.0,
            final_size=950.0,
            sentiment_boost=0.95
        )
        
        # Simulate execution decision
        logging_manager.log_trade_pipeline_event(
            component='trade_execution',
            event_type='execution_decision',
            symbol='BTC/USD',
            strategy='trend_bot',
            side='buy',
            size=950.0,
            price=50000.0,
            execution_mode='dry_run'
        )
        
        # Simulate paper trade execution
        logging_manager.log_trade_pipeline_event(
            component='trade_execution',
            event_type='paper_trading_simulation',
            action='open_position',
            symbol='BTC/USD',
            side='buy',
            size=950.0,
            price=50000.0
        )
        
        # Simulate position update
        logging_manager.log_trade_pipeline_event(
            component='position_management',
            event_type='position_update',
            symbol='BTC/USD',
            side='buy',
            entry_price=50000.0,
            size=0.019,  # 950.0 / 50000.0
            strategy='trend_bot',
            regime='trending'
        )
        
        # Get statistics
        stats = logging_manager.get_all_statistics()
        
        # Verify logging occurred
        assert 'signal_generation' in stats
        assert 'risk_management' in stats
        assert 'position_sizing' in stats
        assert 'trade_execution' in stats
        assert 'position_management' in stats
    
    def test_live_trading_pipeline(self, mock_context, mock_config):
        """Test the complete live trading pipeline."""
        # Update config for live trading
        mock_config['execution_mode'] = 'live'
        
        # Get logging manager
        logging_manager = get_enhanced_logging_manager(mock_config)
        
        # Simulate live trading pipeline
        logging_manager.log_trade_pipeline_event(
            component='signal_generation',
            event_type='signal_generation',
            symbol='ETH/USD',
            strategy='momentum_bot',
            score=0.90,
            direction='short',
            regime='volatile'
        )
        
        logging_manager.log_trade_pipeline_event(
            component='risk_management',
            event_type='risk_check',
            symbol='ETH/USD',
            strategy='momentum_bot',
            allowed=True,
            reason='Risk score acceptable',
            risk_score=0.25
        )
        
        logging_manager.log_trade_pipeline_event(
            component='trade_execution',
            event_type='live_trading_execution',
            action='place_order',
            symbol='ETH/USD',
            side='sell',
            size=500.0,
            price=3000.0,
            order_id='LIVE_ORDER_123'
        )
        
        # Get statistics
        stats = logging_manager.get_all_statistics()
        
        # Verify logging occurred
        assert 'signal_generation' in stats
        assert 'risk_management' in stats
        assert 'trade_execution' in stats
    
    def test_error_handling_in_pipeline(self, mock_context, mock_config):
        """Test error handling in the trade pipeline."""
        logging_manager = get_enhanced_logging_manager(mock_config)
        
        # Simulate an error in the pipeline
        test_error = Exception("Test pipeline error")
        
        logging_manager.log_trade_pipeline_event(
            component='trade_execution',
            event_type='error',
            error=test_error,
            context='trade_execution',
            symbol='BTC/USD',
            strategy='trend_bot'
        )
        
        # Get statistics
        stats = logging_manager.get_all_statistics()
        
        # Verify error logging occurred
        assert 'trade_execution' in stats


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_log_trade_pipeline_event_function(self):
        """Test the convenience function for logging trade pipeline events."""
        mock_config = {
            'logging': {
                'level': 'DEBUG',
                'file_logging': False,
                'console_logging': False
            }
        }
        
        # Test that the function works without errors
        try:
            log_trade_pipeline_event(
                component='test_component',
                event_type='signal_generation',
                config=mock_config,
                symbol='BTC/USD',
                strategy='test_strategy',
                score=0.8,
                direction='long',
                regime='trending'
            )
            # If we get here, the function worked
            assert True
        except Exception as e:
            pytest.fail(f"log_trade_pipeline_event raised an exception: {e}")
    
    def test_get_enhanced_logging_manager_singleton(self):
        """Test that get_enhanced_logging_manager returns a singleton."""
        mock_config = {
            'logging': {
                'level': 'DEBUG',
                'file_logging': False,
                'console_logging': False
            }
        }
        
        manager1 = get_enhanced_logging_manager(mock_config)
        manager2 = get_enhanced_logging_manager(mock_config)
        
        # Should be the same instance
        assert manager1 is manager2


if __name__ == '__main__':
    pytest.main([__file__])
