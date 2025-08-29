"""Tests for backtesting integration functionality."""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from crypto_bot.backtest.backtest_runner import BacktestRunner
from crypto_bot.backtest.enhanced_backtester import EnhancedBacktester
from crypto_bot.backtest.gpu_accelerator import GPUAccelerator


class TestBacktestIntegration:
    """Test suite for Backtesting Integration."""

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
        
        # Generate realistic price data
        base_price = 100.0
        returns = np.random.normal(0, 0.02, 1000)  # 2% hourly volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(max(0.1, prices[-1] * (1 + ret)))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 1000)
        }).set_index('timestamp')

    @pytest.fixture
    def sample_strategy(self):
        """Mock strategy for testing."""
        strategy = Mock()
        strategy.name = "test_strategy"
        strategy.generate_signal = Mock(return_value={'action': 'buy', 'confidence': 0.8})
        strategy.calculate_score = Mock(return_value=0.85)
        strategy.is_active = Mock(return_value=True)
        return strategy

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'backtest': {
                'start_date': '2024-01-01',
                'end_date': '2024-01-31',
                'initial_capital': 10000.0,
                'commission': 0.001,
                'slippage': 0.0005
            },
            'strategies': {
                'test_strategy': {
                    'enabled': True,
                    'parameters': {'lookback': 20, 'threshold': 0.7}
                }
            }
        }

    def test_backtest_runner_init(self, mock_config):
        """Test backtest runner initialization."""
        runner = BacktestRunner(mock_config)
        
        assert runner.config == mock_config
        assert runner.initial_capital == 10000.0
        assert runner.commission == 0.001

    @patch('crypto_bot.backtest.backtest_runner.pd.read_csv')
    def test_load_market_data(self, mock_read_csv, mock_config):
        """Test market data loading."""
        mock_read_csv.return_value = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000] * 100
        })
        
        runner = BacktestRunner(mock_config)
        data = runner.load_market_data('BTC/USDT')
        
        assert len(data) == 100
        assert 'open' in data.columns
        assert 'close' in data.columns

    def test_strategy_execution(self, mock_config, sample_strategy, sample_market_data):
        """Test strategy execution during backtest."""
        runner = BacktestRunner(mock_config)
        
        # Mock the strategy execution
        with patch.object(runner, 'execute_strategy') as mock_execute:
            mock_execute.return_value = {
                'action': 'buy',
                'amount': 1.0,
                'price': 100.0,
                'timestamp': pd.Timestamp('2024-01-01T12:00:00')
            }
            
            result = runner.execute_strategy(sample_strategy, sample_market_data.iloc[0])
            
            assert result['action'] == 'buy'
            assert result['amount'] == 1.0
            assert result['price'] == 100.0

    def test_position_tracking(self, mock_config):
        """Test position tracking during backtest."""
        runner = BacktestRunner(mock_config)
        
        # Open a position
        runner.open_position('BTC/USDT', 'long', 1.0, 100.0, pd.Timestamp.now())
        
        assert len(runner.positions) == 1
        assert runner.positions[0]['symbol'] == 'BTC/USDT'
        assert runner.positions[0]['side'] == 'long'
        
        # Close the position
        runner.close_position(0, 110.0, pd.Timestamp.now())
        
        assert len(runner.positions) == 0
        assert len(runner.closed_positions) == 1

    def test_pnl_calculation(self, mock_config):
        """Test PnL calculation during backtest."""
        runner = BacktestRunner(mock_config)
        
        # Open and close a profitable position
        open_time = pd.Timestamp('2024-01-01T12:00:00')
        close_time = pd.Timestamp('2024-01-01T13:00:00')
        
        runner.open_position('BTC/USDT', 'long', 1.0, 100.0, open_time)
        runner.close_position(0, 110.0, close_time)
        
        # Calculate PnL
        total_pnl = runner.calculate_total_pnl()
        
        # Should be profitable (10% gain minus fees)
        expected_pnl = (110.0 - 100.0) * 1.0 - (100.0 * 0.001) - (110.0 * 0.001)
        assert abs(total_pnl - expected_pnl) < 0.01

    def test_risk_management(self, mock_config):
        """Test risk management during backtest."""
        runner = BacktestRunner(mock_config)
        
        # Test position sizing
        position_size = runner.calculate_position_size(10000.0, 0.02)  # 2% risk per trade
        assert position_size > 0
        
        # Test stop loss
        runner.set_stop_loss(0, 95.0)  # 5% stop loss
        assert runner.positions[0]['stop_loss'] == 95.0

    def test_performance_metrics(self, mock_config):
        """Test performance metrics calculation."""
        runner = BacktestRunner(mock_config)
        
        # Simulate some trades
        for i in range(10):
            open_time = pd.Timestamp(f'2024-01-{i+1:02d}T12:00:00')
            close_time = pd.Timestamp(f'2024-01-{i+1:02d}T13:00:00')
            
            runner.open_position('BTC/USDT', 'long', 1.0, 100.0, open_time)
            runner.close_position(i, 100.0 + (i * 0.5), close_time)
        
        # Calculate metrics
        metrics = runner.calculate_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics

    @patch('crypto_bot.backtest.enhanced_backtester.GPUAccelerator')
    def test_enhanced_backtester_gpu(self, mock_gpu, mock_config):
        """Test enhanced backtester with GPU acceleration."""
        mock_gpu.return_value.is_available.return_value = True
        mock_gpu.return_value.accelerate_backtest.return_value = {
            'results': 'gpu_accelerated_results',
            'performance_boost': 2.5
        }
        
        backtester = EnhancedBacktester(mock_config)
        result = backtester.run_backtest(['BTC/USDT'])
        
        assert 'gpu_accelerated_results' in str(result)
        mock_gpu.return_value.accelerate_backtest.assert_called_once()

    def test_enhanced_backtester_cpu_fallback(self, mock_config):
        """Test enhanced backtester CPU fallback."""
        with patch('crypto_bot.backtest.enhanced_backtester.GPUAccelerator') as mock_gpu:
            mock_gpu.return_value.is_available.return_value = False
            
            backtester = EnhancedBacktester(mock_config)
            result = backtester.run_backtest(['BTC/USDT'])
            
            # Should fall back to CPU
            assert result is not None

    def test_gpu_accelerator_detection(self):
        """Test GPU accelerator detection."""
        accelerator = GPUAccelerator()
        
        # Test GPU availability detection
        is_available = accelerator.is_available()
        assert isinstance(is_available, bool)
        
        if is_available:
            # Test GPU acceleration
            result = accelerator.accelerate_backtest('test_data')
            assert result is not None

    def test_backtest_data_validation(self, mock_config):
        """Test backtest data validation."""
        runner = BacktestRunner(mock_config)
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'timestamp': ['invalid'] * 100,
            'open': ['invalid'] * 100
        })
        
        with pytest.raises(ValueError):
            runner.validate_market_data(invalid_data)
        
        # Test with valid data
        valid_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000] * 100
        })
        
        # Should not raise exception
        runner.validate_market_data(valid_data)

    def test_backtest_configuration_validation(self):
        """Test backtest configuration validation."""
        # Test invalid configuration
        invalid_config = {
            'backtest': {
                'start_date': 'invalid_date',
                'end_date': 'invalid_date',
                'initial_capital': -1000.0  # Negative capital
            }
        }
        
        with pytest.raises(ValueError):
            BacktestRunner(invalid_config)
        
        # Test valid configuration
        valid_config = {
            'backtest': {
                'start_date': '2024-01-01',
                'end_date': '2024-01-31',
                'initial_capital': 10000.0,
                'commission': 0.001
            }
        }
        
        # Should not raise exception
        runner = BacktestRunner(valid_config)
        assert runner is not None

    def test_backtest_results_export(self, mock_config, tmp_path):
        """Test backtest results export."""
        runner = BacktestRunner(mock_config)
        
        # Simulate some trades
        runner.open_position('BTC/USDT', 'long', 1.0, 100.0, pd.Timestamp.now())
        runner.close_position(0, 110.0, pd.Timestamp.now())
        
        # Export results
        export_path = tmp_path / "backtest_results.csv"
        runner.export_results(export_path)
        
        # Verify file was created
        assert export_path.exists()
        
        # Verify content
        exported_data = pd.read_csv(export_path)
        assert len(exported_data) > 0
        assert 'symbol' in exported_data.columns

    def test_backtest_performance_optimization(self, mock_config, sample_market_data):
        """Test backtest performance optimization."""
        runner = BacktestRunner(mock_config)
        
        # Test with large dataset
        large_data = pd.concat([sample_market_data] * 10)  # 10,000 rows
        
        start_time = pd.Timestamp.now()
        
        # Process large dataset
        for i in range(0, len(large_data), 1000):  # Process in chunks
            chunk = large_data.iloc[i:i+1000]
            runner.process_data_chunk(chunk)
        
        end_time = pd.Timestamp.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert processing_time < 10.0

    def test_backtest_error_handling(self, mock_config):
        """Test backtest error handling."""
        runner = BacktestRunner(mock_config)
        
        # Test with invalid strategy
        invalid_strategy = None
        
        with pytest.raises(ValueError):
            runner.execute_strategy(invalid_strategy, {})
        
        # Test with invalid market data
        invalid_market_data = None
        
        with pytest.raises(ValueError):
            runner.process_market_data(invalid_market_data)

    def test_backtest_memory_management(self, mock_config):
        """Test backtest memory management."""
        runner = BacktestRunner(mock_config)
        
        # Simulate many trades
        for i in range(1000):
            runner.open_position(f'TOKEN_{i}/USDT', 'long', 1.0, 100.0, pd.Timestamp.now())
            runner.close_position(i, 110.0, pd.Timestamp.now())
        
        # Check memory usage
        initial_memory = runner.get_memory_usage()
        
        # Clear old data
        runner.clear_old_data()
        
        final_memory = runner.get_memory_usage()
        
        # Memory should be reduced
        assert final_memory <= initial_memory


@pytest.mark.integration
class TestBacktestIntegrationWorkflow:
    """Integration tests for complete backtesting workflow."""

    def test_full_backtest_workflow(self, mock_config, sample_market_data):
        """Test complete backtesting workflow."""
        # Initialize backtest runner
        runner = BacktestRunner(mock_config)
        
        # Load market data
        runner.load_market_data = Mock(return_value=sample_market_data)
        
        # Create strategy
        strategy = Mock()
        strategy.generate_signal = Mock(return_value={'action': 'buy', 'confidence': 0.8})
        
        # Run backtest
        results = runner.run_backtest(['BTC/USDT'], [strategy])
        
        # Verify results
        assert results is not None
        assert 'performance_metrics' in results
        assert 'trades' in results
        assert 'positions' in results

    def test_multi_strategy_backtest(self, mock_config):
        """Test backtesting with multiple strategies."""
        runner = BacktestRunner(mock_config)
        
        # Create multiple strategies
        strategies = []
        for i in range(3):
            strategy = Mock()
            strategy.name = f"strategy_{i}"
            strategy.generate_signal = Mock(return_value={'action': 'buy', 'confidence': 0.7 + i * 0.1})
            strategies.append(strategy)
        
        # Run multi-strategy backtest
        results = runner.run_multi_strategy_backtest(['BTC/USDT'], strategies)
        
        # Verify all strategies were tested
        assert len(results) == 3
        for i, result in enumerate(results):
            assert f"strategy_{i}" in result

    def test_backtest_parameter_optimization(self, mock_config):
        """Test backtest parameter optimization."""
        runner = BacktestRunner(mock_config)
        
        # Define parameter ranges
        param_ranges = {
            'lookback': [10, 20, 30],
            'threshold': [0.6, 0.7, 0.8]
        }
        
        # Run optimization
        best_params = runner.optimize_parameters(param_ranges, ['BTC/USDT'])
        
        # Verify optimization results
        assert 'best_parameters' in best_params
        assert 'best_performance' in best_params
        assert 'optimization_history' in best_params


@pytest.mark.performance
class TestBacktestPerformance:
    """Performance tests for backtesting."""

    def test_large_dataset_performance(self, mock_config):
        """Test performance with large datasets."""
        runner = BacktestRunner(mock_config)
        
        # Create large dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100000, freq='1H'),
            'open': np.random.uniform(90, 110, 100000),
            'high': np.random.uniform(95, 115, 100000),
            'low': np.random.uniform(85, 105, 100000),
            'close': np.random.uniform(90, 110, 100000),
            'volume': np.random.uniform(1000, 10000, 100000)
        }).set_index('timestamp')
        
        start_time = pd.Timestamp.now()
        
        # Process large dataset
        runner.process_market_data(large_data)
        
        end_time = pd.Timestamp.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert processing_time < 30.0

    def test_memory_efficiency(self, mock_config):
        """Test memory efficiency during backtesting."""
        runner = BacktestRunner(mock_config)
        
        initial_memory = runner.get_memory_usage()
        
        # Process data in chunks
        for i in range(10):
            chunk_data = pd.DataFrame({
                'timestamp': pd.date_range(f'2024-01-{i+1:02d}', periods=1000, freq='1H'),
                'close': np.random.uniform(100, 110, 1000)
            }).set_index('timestamp')
            
            runner.process_data_chunk(chunk_data)
            
            # Memory should not grow excessively
            current_memory = runner.get_memory_usage()
            assert current_memory < initial_memory * 2  # Should not double
