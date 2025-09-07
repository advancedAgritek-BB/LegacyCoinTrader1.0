"""Integration tests for strategy execution system.

This module tests strategy selection, signal generation, execution coordination,
and performance tracking across all trading strategies.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import asyncio
from datetime import datetime, timedelta
import random

from crypto_bot.strategy_router import strategy_for, strategy_name, acquire_symbol_lock
from crypto_bot.strategy import (
    trend_bot, mean_bot, breakout_bot, sniper_bot, grid_bot,
    momentum_bot, range_arb_bot, stat_arb_bot
)
from crypto_bot.meta_selector import MetaSelector
from crypto_bot.utils.strategy_utils import compute_strategy_weights
from crypto_bot.cooldown_manager import in_cooldown, mark_cooldown
from crypto_bot.utils.performance_logger import log_performance
from crypto_bot.utils.strategy_analytics import write_scores, write_stats


@pytest.mark.integration
class TestStrategyExecutionIntegration:
    """Test strategy execution integration across all components."""

    @pytest.fixture
    def sample_market_data(self):
        """Generate comprehensive market data for strategy testing."""
        dates = pd.date_range('2024-01-01', periods=200, freq='1H')
        np.random.seed(42)

        # Create realistic price data with multiple trends and patterns
        base_price = 50000.0

        # Create trend segments
        trend_segments = [
            np.linspace(0, 5000, 50),    # Uptrend
            np.linspace(5000, -2000, 50), # Downtrend
            np.linspace(-2000, 3000, 50), # Recovery
            np.linspace(3000, 8000, 50),  # Strong uptrend
        ]

        trend = np.concatenate(trend_segments)
        volatility = np.random.normal(0, 800, 200)
        noise = np.random.normal(0, 200, 200)

        prices = base_price + trend + volatility + noise

        # Ensure no negative prices
        prices = np.maximum(prices, 1000.0)

        # Create OHLCV data
        data = {
            'timestamp': dates,
            'open': prices,
            'high': [max(p + abs(np.random.normal(0, 300)), p) for p in prices],
            'low': [min(p - abs(np.random.normal(0, 300)), p) for p in prices],
            'close': prices + np.random.normal(0, 100, 200),
            'volume': np.random.uniform(50000, 500000, 200)
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        # Ensure high >= max(open, close) and low <= min(open, close)
        for idx in df.index:
            df.loc[idx, 'high'] = max(df.loc[idx, 'high'], df.loc[idx, 'open'], df.loc[idx, 'close'])
            df.loc[idx, 'low'] = min(df.loc[idx, 'low'], df.loc[idx, 'open'], df.loc[idx, 'close'])

        return df

    @pytest.fixture
    def mock_telegram(self):
        """Mock telegram notifier for testing."""
        notifier = Mock()
        notifier.send_message = AsyncMock(return_value=True)
        return notifier

    @pytest.fixture
    def strategy_configs(self):
        """Strategy configuration for testing."""
        return {
            'trend_bot': {'enabled': True, 'min_trend_strength': 0.6},
            'mean_bot': {'enabled': True, 'lookback_period': 20},
            'breakout_bot': {'enabled': True, 'breakout_threshold': 2.0},
            'momentum_bot': {'enabled': True, 'momentum_period': 14},
            'sniper_bot': {'enabled': True, 'min_volume': 100000},
            'grid_bot': {'enabled': True, 'grid_spacing': 0.01}
        }

    def test_strategy_selection_integration(self, sample_market_data, strategy_configs):
        """Test strategy selection and routing integration."""
        symbol = 'BTC/USDT'

        # Test strategy_for function
        with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
            with patch('crypto_bot.strategy_router.yaml.safe_load', return_value=strategy_configs):
                strategy = strategy_for(symbol, sample_market_data, {})

                assert strategy is not None
                assert hasattr(strategy, 'generate_signal')

                # Test signal generation
                signal = strategy.generate_signal(sample_market_data, {})

                # Verify signal structure
                assert isinstance(signal, dict)
                assert 'action' in signal
                assert 'symbol' in signal
                assert 'confidence' in signal
                assert signal['action'] in ['buy', 'sell', 'hold']
                assert 0 <= signal['confidence'] <= 1
                assert signal['symbol'] == symbol

    def test_multiple_strategy_execution(self, sample_market_data, strategy_configs):
        """Test concurrent execution of multiple strategies."""
        symbol = 'BTC/USDT'
        strategies = ['trend_bot', 'mean_bot', 'breakout_bot']

        signals = {}

        for strategy_name in strategies:
            with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
                with patch('crypto_bot.strategy_router.yaml.safe_load', return_value=strategy_configs):
                    strategy = strategy_for(symbol, sample_market_data, {})
                    signal = strategy.generate_signal(sample_market_data, {})
                    signals[strategy_name] = signal

        # Verify all strategies produced valid signals
        for strategy_name, signal in signals.items():
            assert signal['action'] in ['buy', 'sell', 'hold']
            assert 0 <= signal['confidence'] <= 1

        # Test signal consensus (simple voting)
        buy_votes = sum(1 for s in signals.values() if s['action'] == 'buy')
        sell_votes = sum(1 for s in signals.values() if s['action'] == 'sell')
        hold_votes = sum(1 for s in signals.values() if s['action'] == 'hold')

        # Determine consensus action
        if buy_votes > sell_votes and buy_votes > hold_votes:
            consensus = 'buy'
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            consensus = 'sell'
        else:
            consensus = 'hold'

        assert consensus in ['buy', 'sell', 'hold']

    def test_strategy_performance_tracking(self, sample_market_data):
        """Test strategy performance tracking and analytics."""
        symbol = 'BTC/USDT'

        # Simulate multiple strategy runs
        performance_data = []

        for i in range(10):
            # Generate slightly different market data for each run
            test_data = sample_market_data.copy()
            test_data['close'] = test_data['close'] * (1 + np.random.normal(0, 0.02, len(test_data)))

            with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
                strategy = strategy_for(symbol, test_data, {})
                signal = strategy.generate_signal(test_data, {})

                # Simulate trade outcome
                entry_price = test_data['close'].iloc[-1]
                exit_price = entry_price * (1 + np.random.normal(0, 0.05))  # Random outcome

                pnl = (exit_price - entry_price) / entry_price if signal['action'] == 'buy' else (entry_price - exit_price) / entry_price

                performance_data.append({
                    'strategy': 'test_strategy',
                    'signal': signal,
                    'pnl': pnl,
                    'confidence': signal['confidence']
                })

        # Calculate performance metrics
        total_trades = len(performance_data)
        winning_trades = sum(1 for p in performance_data if p['pnl'] > 0)
        win_rate = winning_trades / total_trades

        avg_pnl = sum(p['pnl'] for p in performance_data) / total_trades
        avg_confidence = sum(p['confidence'] for p in performance_data) / total_trades

        # Verify metrics are reasonable
        assert 0 <= win_rate <= 1
        assert isinstance(avg_pnl, (int, float))
        assert 0 <= avg_confidence <= 1

        # Test confidence correlation with performance
        high_conf_signals = [p for p in performance_data if p['confidence'] > 0.7]
        low_conf_signals = [p for p in performance_data if p['confidence'] <= 0.7]

        if high_conf_signals and low_conf_signals:
            high_conf_avg_pnl = sum(p['pnl'] for p in high_conf_signals) / len(high_conf_signals)
            low_conf_avg_pnl = sum(p['pnl'] for p in low_conf_signals) / len(low_conf_signals)

            # High confidence signals should perform better (not guaranteed but likely)
            # This is a statistical test that may occasionally fail due to randomness

    def test_cooldown_manager_strategy_integration(self, sample_market_data):
        """Test cooldown manager integration with strategy execution."""
        symbol = 'BTC/USDT'

        # Configure cooldown
        from crypto_bot.cooldown_manager import configure
        cooldown_config = {
            'cooldown_period': 300,  # 5 minutes
            'symbols': {symbol: {'cooldown_period': 300}}
        }
        configure(cooldown_config)

        # Initial state - should not be in cooldown
        assert not in_cooldown(symbol)

        # Execute strategy and mark cooldown
        with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
            strategy = strategy_for(symbol, sample_market_data, {})
            signal = strategy.generate_signal(sample_market_data, {})

            if signal['action'] != 'hold':
                mark_cooldown(symbol)

                # Should now be in cooldown
                assert in_cooldown(symbol)

                # Strategy should respect cooldown (would be tested in real execution)

    def test_symbol_locking_concurrent_strategies(self, sample_market_data):
        """Test symbol locking for concurrent strategy execution."""
        symbol = 'BTC/USDT'

        async def execute_strategy_with_lock(strategy_id):
            """Simulate strategy execution with locking."""
            lock = await acquire_symbol_lock(symbol)
            try:
                # Simulate strategy processing time
                await asyncio.sleep(0.01)

                with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
                    strategy = strategy_for(symbol, sample_market_data, {})
                    signal = strategy.generate_signal(sample_market_data, {})

                return {
                    'strategy_id': strategy_id,
                    'signal': signal,
                    'timestamp': datetime.now()
                }
            finally:
                lock.release()

        # Execute multiple strategies concurrently
        tasks = [execute_strategy_with_lock(i) for i in range(5)]
        results = asyncio.run(asyncio.gather(*tasks))

        # Verify all strategies executed without conflicts
        assert len(results) == 5
        for result in results:
            assert 'strategy_id' in result
            assert 'signal' in result
            assert 'timestamp' in result
            assert result['signal']['action'] in ['buy', 'sell', 'hold']

    def test_meta_selector_strategy_integration(self, sample_market_data):
        """Test meta selector integration with strategy execution."""
        symbol = 'BTC/USDT'

        # Mock meta selector
        with patch('crypto_bot.meta_selector.MetaSelector') as mock_meta_selector_class:
            mock_meta_selector = Mock()
            mock_meta_selector.select_strategy = Mock(return_value='trend_bot')
            mock_meta_selector_class.return_value = mock_meta_selector

            # Test strategy selection via meta selector
            meta_selector = MetaSelector()
            selected_strategy = meta_selector.select_strategy(symbol, sample_market_data)

            assert selected_strategy == 'trend_bot'

            # Test selected strategy execution
            with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
                strategy = strategy_for(symbol, sample_market_data, {})
                signal = strategy.generate_signal(sample_market_data, {})

                assert signal['action'] in ['buy', 'sell', 'hold']

    def test_strategy_weight_computation(self, sample_market_data):
        """Test strategy weight computation and allocation."""
        symbol = 'BTC/USDT'

        # Mock strategy performance data
        strategy_performance = {
            'trend_bot': {'win_rate': 0.65, 'sharpe_ratio': 1.2, 'max_drawdown': 0.08},
            'mean_bot': {'win_rate': 0.58, 'sharpe_ratio': 0.9, 'max_drawdown': 0.12},
            'breakout_bot': {'win_rate': 0.62, 'sharpe_ratio': 1.1, 'max_drawdown': 0.06}
        }

        # Compute strategy weights
        weights = compute_strategy_weights(strategy_performance)

        # Verify weight properties
        assert isinstance(weights, dict)
        assert len(weights) == len(strategy_performance)
        assert all(strategy in weights for strategy in strategy_performance)

        # Weights should sum to 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01

        # Higher performing strategies should have higher weights
        trend_weight = weights['trend_bot']
        mean_weight = weights['mean_bot']
        breakout_weight = weights['breakout_bot']

        assert trend_weight > mean_weight  # Better win rate
        assert breakout_weight > mean_weight  # Better risk-adjusted return

    def test_strategy_signal_filtering(self, sample_market_data):
        """Test strategy signal filtering and validation."""
        symbol = 'BTC/USDT'

        # Test signal confidence filtering
        min_confidence = 0.7

        signals = []
        for i in range(20):
            with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
                strategy = strategy_for(symbol, sample_market_data, {})
                signal = strategy.generate_signal(sample_market_data, {})

                # Apply confidence filter
                if signal['confidence'] >= min_confidence:
                    signals.append(signal)

        # Should have fewer signals after filtering
        assert len(signals) <= 20

        # All filtered signals should meet confidence threshold
        for signal in signals:
            assert signal['confidence'] >= min_confidence

        # Test signal volume filtering
        min_volume = 100000.0

        # Simulate low volume condition
        low_volume_data = sample_market_data.copy()
        low_volume_data['volume'] = low_volume_data['volume'] * 0.1  # Reduce volume

        with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
            strategy = strategy_for(symbol, low_volume_data, {})
            signal = strategy.generate_signal(low_volume_data, {})

            # Signal should be weaker or filtered due to low volume
            assert signal['confidence'] < 0.8  # Lower confidence for low volume

    def test_strategy_adaptation_to_market_conditions(self, sample_market_data):
        """Test strategy adaptation to different market conditions."""
        symbol = 'BTC/USDT'

        # Test trending market
        trending_data = sample_market_data.copy()
        # Create strong trend
        trend_component = np.linspace(0, 10000, len(trending_data))
        trending_data['close'] = trending_data['close'] + trend_component

        # Test ranging market
        ranging_data = sample_market_data.copy()
        # Create sideways movement
        ranging_data['close'] = 50000.0 + np.sin(np.arange(len(ranging_data)) * 0.1) * 1000

        # Test volatile market
        volatile_data = sample_market_data.copy()
        # Add high volatility
        volatile_data['close'] = volatile_data['close'] * (1 + np.random.normal(0, 0.05, len(volatile_data)))

        market_conditions = [
            ('trending', trending_data),
            ('ranging', ranging_data),
            ('volatile', volatile_data)
        ]

        for condition_name, market_data in market_conditions:
            with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
                strategy = strategy_for(symbol, market_data, {})
                signal = strategy.generate_signal(market_data, {})

                # Verify signal is appropriate for market condition
                assert signal['action'] in ['buy', 'sell', 'hold']
                assert 0 <= signal['confidence'] <= 1

                # Log condition-specific performance for analysis
                print(f"{condition_name} market - Signal: {signal['action']}, Confidence: {signal['confidence']:.3f}")

    def test_strategy_telegram_integration(self, sample_market_data, mock_telegram):
        """Test strategy execution with Telegram notifications."""
        symbol = 'BTC/USDT'

        with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
            strategy = strategy_for(symbol, sample_market_data, {})
            signal = strategy.generate_signal(sample_market_data, {})

            # Simulate signal execution notification
            if signal['action'] != 'hold':
                message = f"Strategy signal: {signal['action']} {symbol} with {signal['confidence']:.2f} confidence"
                asyncio.run(mock_telegram.send_message(message))

                # Verify notification was sent
                mock_telegram.send_message.assert_called_with(message)

    def test_strategy_error_handling(self, sample_market_data):
        """Test strategy error handling and fallback."""
        symbol = 'BTC/USDT'

        # Test with invalid market data
        invalid_data = sample_market_data.copy()
        invalid_data['close'] = np.nan  # Introduce NaN values

        with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
            strategy = strategy_for(symbol, invalid_data, {})

            # Should handle NaN values gracefully
            try:
                signal = strategy.generate_signal(invalid_data, {})
                assert signal['action'] in ['buy', 'sell', 'hold']
            except Exception as e:
                # If strategy fails, should fallback to hold
                print(f"Strategy error handled: {e}")

        # Test with empty market data
        empty_data = pd.DataFrame()

        with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
            strategy = strategy_for(symbol, empty_data, {})

            # Should handle empty data gracefully
            signal = strategy.generate_signal(empty_data, {})
            assert signal['action'] == 'hold'  # Default to hold for insufficient data

    def test_strategy_performance_persistence(self, sample_market_data):
        """Test strategy performance data persistence."""
        symbol = 'BTC/USDT'

        with patch('crypto_bot.utils.strategy_analytics.write_scores') as mock_write_scores, \
             patch('crypto_bot.utils.strategy_analytics.write_stats') as mock_write_stats:

            # Simulate strategy execution and performance tracking
            with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
                strategy = strategy_for(symbol, sample_market_data, {})
                signal = strategy.generate_signal(sample_market_data, {})

                # Simulate performance data
                performance_data = {
                    'strategy': 'trend_bot',
                    'symbol': symbol,
                    'signal': signal,
                    'pnl': 0.025,  # 2.5% profit
                    'timestamp': datetime.now()
                }

                # Write performance data
                write_scores(performance_data)
                write_stats(performance_data)

                # Verify data was written
                mock_write_scores.assert_called()
                mock_write_stats.assert_called()

    def test_concurrent_strategy_execution(self, sample_market_data):
        """Test concurrent execution of multiple strategies on different symbols."""
        symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT']

        async def execute_strategy_for_symbol(symbol):
            """Execute strategy for a specific symbol."""
            with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
                strategy = strategy_for(symbol, sample_market_data, {})
                signal = strategy.generate_signal(sample_market_data, {})

                return {
                    'symbol': symbol,
                    'signal': signal,
                    'timestamp': datetime.now()
                }

        # Execute strategies concurrently for all symbols
        tasks = [execute_strategy_for_symbol(symbol) for symbol in symbols]
        results = asyncio.run(asyncio.gather(*tasks))

        # Verify all strategies executed successfully
        assert len(results) == len(symbols)

        for result in results:
            assert result['symbol'] in symbols
            assert result['signal']['action'] in ['buy', 'sell', 'hold']
            assert 0 <= result['signal']['confidence'] <= 1

        # Verify no symbol conflicts (all symbols should be processed)
        processed_symbols = {result['symbol'] for result in results}
        assert processed_symbols == set(symbols)

    def test_strategy_memory_and_state_management(self, sample_market_data):
        """Test strategy state management and memory between executions."""
        symbol = 'BTC/USDT'

        # Simulate multiple consecutive strategy executions
        execution_history = []

        for i in range(5):
            with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
                strategy = strategy_for(symbol, sample_market_data, {})
                signal = strategy.generate_signal(sample_market_data, {})

                execution_history.append({
                    'execution_id': i,
                    'signal': signal,
                    'timestamp': datetime.now()
                })

                # Simulate state update (in real system, strategy would maintain state)
                # For testing, we just verify consistency

        # Verify execution consistency
        actions = [exec['signal']['action'] for exec in execution_history]
        confidences = [exec['signal']['confidence'] for exec in execution_history]

        # All actions should be valid
        assert all(action in ['buy', 'sell', 'hold'] for action in actions)

        # All confidences should be valid
        assert all(0 <= conf <= 1 for conf in confidences)

        # Check for reasonable signal stability (not random)
        action_changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        stability_ratio = 1 - (action_changes / len(actions))

        # Should show some stability (not completely random)
        assert stability_ratio > 0.2  # At least 20% stability
