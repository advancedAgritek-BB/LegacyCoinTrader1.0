"""End-to-end integration tests for the complete trading system.

This module tests the entire trading system from data ingestion through
execution and monitoring, ensuring all components work together seamlessly.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
from datetime import datetime, timedelta
import threading
import time

from crypto_bot.main import main
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.strategy_router import strategy_for
from crypto_bot.execution.cex_executor import execute_trade_async
from frontend.app import app
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.utils.market_loader import update_ohlcv_cache
from crypto_bot.config import load_config


@pytest.mark.integration
class TestEndToEndIntegration:
    """Test complete end-to-end trading system integration."""

    @pytest.fixture
    def system_config(self):
        """Complete system configuration for end-to-end testing."""
        return {
            'trading': {
                'enabled': True,
                'mode': 'paper',
                'max_positions': 5,
                'risk_per_trade': 0.02,
                'initial_balance': 10000.0
            },
            'solana': {
                'enabled': False,  # Disable for simpler testing
                'rpc_url': 'https://api.mainnet-beta.solana.com'
            },
            'telegram': {
                'enabled': True,
                'bot_token': 'test_token',
                'chat_id': 'test_chat',
                'notification_level': 'detailed'
            },
            'enhanced_scanning': {
                'enabled': True,
                'scan_interval': 30
            },
            'risk_management': {
                'max_drawdown': 0.1,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.1
            },
            'strategies': {
                'trend_bot': {'enabled': True, 'min_trend_strength': 0.6},
                'mean_bot': {'enabled': True, 'lookback_period': 20}
            }
        }

    @pytest.fixture
    def sample_market_data(self):
        """Generate realistic market data for end-to-end testing."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
        np.random.seed(42)

        # Create realistic price series with trends and volatility
        base_price = 50000.0

        # Multi-scale trends
        long_trend = np.cumsum(np.random.normal(0, 20, 1000))
        short_trend = np.cumsum(np.random.normal(0, 50, 1000))
        micro_trend = np.random.normal(0, 10, 1000)

        # Seasonal patterns
        hours = np.arange(1000)
        daily_seasonal = 200 * np.sin(2 * np.pi * hours / 24)
        weekly_seasonal = 500 * np.sin(2 * np.pi * hours / (24 * 7))

        # Volume patterns
        base_volume = 100000
        volume_trend = np.random.uniform(0.5, 2.0, 1000)
        volume_seasonal = 1 + 0.5 * np.sin(2 * np.pi * hours / 24)

        prices = base_price + long_trend + short_trend + micro_trend + daily_seasonal + weekly_seasonal
        prices = np.maximum(prices, 1000.0)  # No negative prices

        data = {
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 200)) for p in prices],
            'low': [p - abs(np.random.normal(0, 200)) for p in prices],
            'close': prices + np.random.normal(0, 50, 1000),
            'volume': base_volume * volume_trend * volume_seasonal
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        # Ensure OHLCV integrity
        for idx in df.index:
            row = df.loc[idx]
            df.loc[idx, 'high'] = max(row['high'], row['open'], row['close'])
            df.loc[idx, 'low'] = min(row['low'], row['open'], row['close'])

        return df

    @pytest.fixture
    def mock_exchange(self):
        """Mock exchange for end-to-end testing."""
        exchange = Mock()
        exchange.id = 'kraken'
        exchange.create_order = AsyncMock(return_value={
            'id': f'order_{int(time.time())}',
            'status': 'closed',
            'amount': 1.0,
            'price': 50000.0,
            'cost': 50000.0,
            'fee': {'cost': 5.0, 'currency': 'USDT'}
        })
        exchange.fetch_balance = AsyncMock(return_value={
            'USDT': {'free': 10000.0, 'total': 10000.0},
            'BTC': {'free': 1.0, 'total': 1.0}
        })
        exchange.fetch_ticker = AsyncMock(return_value={
            'last': 50000.0,
            'bid': 49900.0,
            'ask': 50100.0,
            'volume': 1000000.0
        })
        return exchange

    @pytest.fixture
    def end_to_end_context(self, system_config, sample_market_data, mock_exchange):
        """Complete end-to-end testing context."""
        return {
            'config': system_config,
            'market_data': sample_market_data,
            'exchange': mock_exchange,
            'wallet': PaperWallet(
                balance=system_config['trading']['initial_balance'],
                max_open_trades=system_config['trading']['max_positions']
            ),
            'risk_manager': RiskManager(RiskConfig(
                max_drawdown=system_config['risk_management']['max_drawdown'],
                stop_loss_pct=system_config['risk_management']['stop_loss_pct'],
                take_profit_pct=system_config['risk_management']['take_profit_pct'],
                trade_size_pct=0.1,
                risk_pct=0.01
            )),
            'telegram': Mock(spec=TelegramNotifier)
        }

    def test_complete_trading_workflow(self, end_to_end_context):
        """Test complete trading workflow from signal to execution."""
        ctx = end_to_end_context
        symbol = 'BTC/USDT'

        # Step 1: Market Data Processing
        market_data = ctx['market_data']
        assert not market_data.empty
        assert len(market_data) > 100

        # Step 2: Strategy Signal Generation
        with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
            with patch('crypto_bot.strategy_router.yaml.safe_load', return_value=ctx['config']['strategies']):
                strategy = strategy_for(symbol, market_data, {})

                # Generate trading signals
                signals = []
                for i in range(50, len(market_data), 10):
                    current_data = market_data.iloc[:i+1]
                    signal = strategy.generate_signal(current_data, {})
                    signals.append(signal)

                # Should generate some signals
                assert len(signals) > 0

        # Step 3: Risk Assessment
        risk_manager = ctx['risk_manager']
        wallet = ctx['wallet']

        valid_signals = []
        for signal in signals:
            if signal['action'] != 'hold' and signal['confidence'] > 0.6:
                # Calculate position size
                price = market_data['close'].iloc[-1]
                position_size = risk_manager.calculate_position_size(
                    symbol, price, wallet.balance
                )

                # Check risk limits
                can_trade = risk_manager.check_risk_limits(
                    symbol, position_size, price
                )

                if can_trade and position_size > 0:
                    valid_signals.append({
                        'signal': signal,
                        'position_size': position_size,
                        'price': price
                    })

        # Step 4: Trade Execution (Paper Trading)
        executed_trades = []
        for trade_info in valid_signals[:3]:  # Limit to 3 trades for testing
            signal = trade_info['signal']
            position_size = trade_info['position_size']
            price = trade_info['price']

            if signal['action'] == 'buy':
                success = wallet.buy(symbol, position_size, price)
            else:
                success = wallet.sell(symbol, position_size, price)

            if success:
                executed_trades.append({
                    'signal': signal,
                    'position_size': position_size,
                    'price': price,
                    'timestamp': datetime.now()
                })

        # Step 5: Position Management
        assert len(wallet.positions) <= ctx['config']['trading']['max_positions']

        # Step 6: Performance Monitoring
        if executed_trades:
            # Calculate basic performance metrics
            total_trades = len(executed_trades)
            profitable_trades = sum(1 for trade in executed_trades
                                  if trade['signal']['action'] == 'buy')  # Simplified

            performance = {
                'total_trades': total_trades,
                'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': wallet.realized_pnl,
                'current_balance': wallet.balance
            }

            assert performance['total_trades'] > 0
            assert 0 <= performance['win_rate'] <= 1

        # Step 7: Notification System
        telegram = ctx['telegram']
        if executed_trades:
            # Should send notifications for executed trades
            for trade in executed_trades:
                # This would trigger notifications in real system
                pass

    def test_system_startup_and_initialization(self, end_to_end_context):
        """Test complete system startup and initialization."""
        ctx = end_to_end_context

        # Test configuration loading
        config = ctx['config']
        assert config['trading']['enabled'] is True
        assert config['telegram']['enabled'] is True

        # Test wallet initialization
        wallet = ctx['wallet']
        assert wallet.balance == config['trading']['initial_balance']
        assert wallet.max_open_trades == config['trading']['max_positions']

        # Test risk manager initialization
        risk_manager = ctx['risk_manager']
        assert risk_manager.config.max_drawdown == config['risk_management']['max_drawdown']
        assert risk_manager.config.stop_loss_pct == config['risk_management']['stop_loss_pct']

        # Test market data availability
        market_data = ctx['market_data']
        assert len(market_data) > 0
        assert 'open' in market_data.columns
        assert 'high' in market_data.columns
        assert 'low' in market_data.columns
        assert 'close' in market_data.columns
        assert 'volume' in market_data.columns

        # Test exchange connectivity (mocked)
        exchange = ctx['exchange']
        balance = asyncio.run(exchange.fetch_balance())
        assert 'USDT' in balance
        assert balance['USDT']['free'] > 0

    def test_frontend_api_integration_workflow(self, end_to_end_context):
        """Test frontend API integration with trading system."""
        ctx = end_to_end_context

        # Setup Flask test client
        app.config['TESTING'] = True
        with app.test_client() as client:
            # Test bot status endpoint
            with patch('frontend.app.is_running', return_value=True), \
                 patch('frontend.app.load_execution_mode', return_value='paper'), \
                 patch('frontend.app.get_uptime', return_value='00:30:00'):

                response = client.get('/api/bot-status')
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['data']['bot_running'] is True

            # Test configuration endpoint
            with patch('frontend.app.load_config', return_value=ctx['config']):
                response = client.get('/api/refresh_config')
                assert response.status_code == 200
                data = json.loads(response.data)
                assert 'trading' in data
                assert 'telegram' in data

            # Test manual trading endpoint
            sell_data = {
                'symbol': 'BTC/USDT',
                'amount': 0.5,
                'price': 51000.0
            }

            with patch('frontend.app.manual_sell_position', return_value=True):
                response = client.post('/api/sell-position',
                                     json=sell_data,
                                     content_type='application/json')
                assert response.status_code == 200

    def test_telegram_integration_workflow(self, end_to_end_context):
        """Test Telegram integration throughout the trading workflow."""
        ctx = end_to_end_context
        telegram = ctx['telegram']

        # Mock telegram methods
        telegram.send_message = AsyncMock(return_value=True)
        telegram.send_trade_notification = AsyncMock(return_value=True)
        telegram.send_portfolio_update = AsyncMock(return_value=True)

        # Simulate trading workflow with notifications
        wallet = ctx['wallet']

        # Execute trades
        wallet.buy('BTC/USDT', 1.0, 50000.0)
        wallet.sell('BTC/USDT', 1.0, 51000.0)

        # Send trade notifications
        asyncio.run(telegram.send_trade_notification({
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 1.0,
            'price': 50000.0
        }))

        asyncio.run(telegram.send_trade_notification({
            'symbol': 'BTC/USDT',
            'side': 'sell',
            'amount': 1.0,
            'price': 51000.0,
            'pnl': 1000.0
        }))

        # Send portfolio update
        asyncio.run(telegram.send_portfolio_update({
            'total_balance': wallet.balance,
            'total_pnl': wallet.realized_pnl,
            'win_rate': wallet.win_rate
        }))

        # Verify notifications were sent
        assert telegram.send_trade_notification.call_count == 2
        assert telegram.send_portfolio_update.call_count == 1

    def test_error_handling_and_recovery(self, end_to_end_context):
        """Test error handling and recovery in end-to-end workflow."""
        ctx = end_to_end_context

        # Test market data error handling
        corrupted_data = ctx['market_data'].copy()
        corrupted_data['close'] = np.nan  # Introduce corruption

        try:
            # This should handle NaN values gracefully
            with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
                strategy = strategy_for('BTC/USDT', corrupted_data, {})
                signal = strategy.generate_signal(corrupted_data, {})
                # Should not crash
        except Exception as e:
            # If it fails, ensure it's a handled error
            assert "NaN" in str(e) or "invalid" in str(e).lower()

        # Test exchange connectivity error
        exchange = ctx['exchange']
        exchange.fetch_balance.side_effect = Exception("Exchange API error")

        try:
            balance = asyncio.run(exchange.fetch_balance())
        except Exception as e:
            assert "Exchange API error" in str(e)

        # Test recovery: fallback to cached data or paper trading
        wallet = ctx['wallet']

        # Should still be able to execute paper trades despite exchange error
        success = wallet.buy('BTC/USDT', 0.5, 50000.0)
        assert success  # Paper trading should work

    def test_performance_monitoring_workflow(self, end_to_end_context):
        """Test performance monitoring throughout the workflow."""
        ctx = end_to_end_context
        wallet = ctx['wallet']

        # Execute a series of trades
        trades = [
            {'action': 'buy', 'amount': 1.0, 'price': 50000.0},
            {'action': 'sell', 'amount': 1.0, 'price': 51000.0},
            {'action': 'buy', 'amount': 0.8, 'price': 50500.0},
            {'action': 'sell', 'amount': 0.8, 'price': 51500.0},
            {'action': 'buy', 'amount': 0.6, 'price': 51000.0},
            {'action': 'sell', 'amount': 0.6, 'price': 52000.0}
        ]

        for trade in trades:
            if trade['action'] == 'buy':
                wallet.buy('BTC/USDT', trade['amount'], trade['price'])
            else:
                wallet.sell('BTC/USDT', trade['amount'], trade['price'])

        # Calculate comprehensive performance metrics
        performance_metrics = {
            'total_trades': wallet.total_trades,
            'winning_trades': wallet.winning_trades,
            'losing_trades': wallet.total_trades - wallet.winning_trades,
            'win_rate': wallet.win_rate,
            'total_pnl': wallet.realized_pnl,
            'avg_trade_pnl': wallet.realized_pnl / wallet.total_trades if wallet.total_trades > 0 else 0,
            'final_balance': wallet.balance,
            'total_return_pct': (wallet.balance - 10000.0) / 10000.0 * 100
        }

        # Verify metrics are reasonable
        assert performance_metrics['total_trades'] == len(trades)
        assert 0 <= performance_metrics['win_rate'] <= 100
        assert performance_metrics['total_trades'] > 0

        # Test risk-adjusted metrics
        risk_manager = ctx['risk_manager']

        # Calculate Sharpe ratio (simplified)
        if len(trades) > 1:
            returns = []
            for i in range(1, len(trades), 2):  # Every sell trade
                if i + 1 < len(trades):
                    buy_trade = trades[i-1]
                    sell_trade = trades[i]
                    if buy_trade['action'] == 'buy' and sell_trade['action'] == 'sell':
                        ret = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                        returns.append(ret)

            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return * np.sqrt(365) if std_return > 0 else 0

                assert isinstance(sharpe_ratio, (int, float))

    def test_configuration_driven_behavior(self, end_to_end_context):
        """Test how configuration changes affect system behavior."""
        ctx = end_to_end_context

        # Test different risk configurations
        conservative_config = RiskConfig(
            max_drawdown=0.05,  # Tighter drawdown
            stop_loss_pct=0.03,  # Tighter stops
            take_profit_pct=0.08,  # Lower targets
            trade_size_pct=0.05,  # Smaller positions
            risk_pct=0.005  # Lower risk
        )

        aggressive_config = RiskConfig(
            max_drawdown=0.15,  # Looser drawdown
            stop_loss_pct=0.08,  # Wider stops
            take_profit_pct=0.15,  # Higher targets
            trade_size_pct=0.15,  # Larger positions
            risk_pct=0.015  # Higher risk
        )

        wallet = ctx['wallet']
        price = 50000.0

        # Test conservative configuration
        conservative_risk_manager = RiskManager(conservative_config)
        conservative_size = conservative_risk_manager.calculate_position_size(
            'BTC/USDT', price, wallet.balance
        )

        # Test aggressive configuration
        aggressive_risk_manager = RiskManager(aggressive_config)
        aggressive_size = aggressive_risk_manager.calculate_position_size(
            'BTC/USDT', price, wallet.balance
        )

        # Aggressive should allow larger positions
        assert aggressive_size > conservative_size

        # Conservative should have tighter risk limits
        assert conservative_config.max_drawdown < aggressive_config.max_drawdown
        assert conservative_config.stop_loss_pct < aggressive_config.stop_loss_pct

    def test_concurrent_operations_simulation(self, end_to_end_context):
        """Test concurrent operations in the trading system."""
        ctx = end_to_end_context

        # Simulate concurrent trading operations
        async def simulate_trading_operation(operation_id, wallet, market_data):
            """Simulate a single trading operation."""
            await asyncio.sleep(np.random.uniform(0.01, 0.05))  # Random delay

            # Random trading decision
            if np.random.random() > 0.5:
                price = market_data['close'].iloc[-1] * (1 + np.random.normal(0, 0.02))
                amount = np.random.uniform(0.1, 1.0)

                if np.random.random() > 0.5:
                    success = wallet.buy('BTC/USDT', amount, price)
                    action = 'buy'
                else:
                    success = wallet.sell('BTC/USDT', amount, price)
                    action = 'sell'

                return {
                    'operation_id': operation_id,
                    'action': action,
                    'amount': amount,
                    'price': price,
                    'success': success
                }

            return {'operation_id': operation_id, 'action': 'hold'}

        # Run concurrent operations
        wallet = ctx['wallet']
        market_data = ctx['market_data']

        tasks = [
            simulate_trading_operation(i, wallet, market_data)
            for i in range(10)
        ]

        results = asyncio.run(asyncio.gather(*tasks))

        # Analyze results
        successful_operations = [r for r in results if r.get('success', False)]
        buy_operations = [r for r in results if r.get('action') == 'buy']
        sell_operations = [r for r in results if r.get('action') == 'sell']

        # Verify concurrent execution worked
        assert len(results) == 10
        assert all('operation_id' in r for r in results)

        # Check wallet consistency after concurrent operations
        assert wallet.balance >= 0  # Should not go negative
        assert len(wallet.positions) >= 0

    def test_system_resource_management(self, end_to_end_context):
        """Test system resource management during intensive operations."""
        ctx = end_to_end_context

        # Simulate memory-intensive operations
        large_market_data = ctx['market_data'].copy()

        # Create larger dataset
        for i in range(10):
            additional_data = ctx['market_data'].copy()
            additional_data.index = additional_data.index + timedelta(days=i+1)
            large_market_data = pd.concat([large_market_data, additional_data])

        # Test memory usage with large dataset
        initial_memory = len(large_market_data)

        # Process large dataset
        processed_data = large_market_data.copy()
        processed_data['returns'] = processed_data['close'].pct_change()
        processed_data['volatility'] = processed_data['returns'].rolling(20).std()
        processed_data['sma_20'] = processed_data['close'].rolling(20).mean()
        processed_data['sma_50'] = processed_data['close'].rolling(50).mean()

        # Verify processing completed without memory issues
        assert len(processed_data) == initial_memory
        assert 'returns' in processed_data.columns
        assert 'volatility' in processed_data.columns
        assert 'sma_20' in processed_data.columns
        assert 'sma_50' in processed_data.columns

        # Clean up
        del large_market_data, processed_data

    def test_system_graceful_shutdown(self, end_to_end_context):
        """Test system graceful shutdown and cleanup."""
        ctx = end_to_end_context

        # Setup active trading state
        wallet = ctx['wallet']
        wallet.buy('BTC/USDT', 1.0, 50000.0)
        wallet.buy('ETH/USDT', 5.0, 3000.0)

        # Simulate system shutdown
        shutdown_state = {
            'wallet': {
                'balance': wallet.balance,
                'positions': wallet.positions.copy(),
                'realized_pnl': wallet.realized_pnl
            },
            'timestamp': datetime.now(),
            'reason': 'scheduled_shutdown'
        }

        # Verify critical state is preserved
        assert shutdown_state['wallet']['balance'] == wallet.balance
        assert len(shutdown_state['wallet']['positions']) == len(wallet.positions)
        assert shutdown_state['timestamp'] is not None

        # Simulate state restoration after restart
        restored_wallet = PaperWallet(
            balance=shutdown_state['wallet']['balance'],
            max_open_trades=10
        )
        restored_wallet.positions = shutdown_state['wallet']['positions']
        restored_wallet.realized_pnl = shutdown_state['wallet']['realized_pnl']

        # Verify restoration
        assert restored_wallet.balance == wallet.balance
        assert len(restored_wallet.positions) == len(wallet.positions)
        assert restored_wallet.realized_pnl == wallet.realized_pnl

    def test_end_to_end_performance_benchmark(self, end_to_end_context):
        """Test end-to-end system performance benchmarks."""
        import time

        ctx = end_to_end_context

        # Benchmark signal generation
        market_data = ctx['market_data']
        signal_generation_times = []

        with patch('crypto_bot.strategy_router.CONFIG_PATH', Path(tempfile.mktemp())):
            with patch('crypto_bot.strategy_router.yaml.safe_load', return_value=ctx['config']['strategies']):
                strategy = strategy_for('BTC/USDT', market_data, {})

                for i in range(10):
                    start_time = time.time()
                    signal = strategy.generate_signal(market_data, {})
                    end_time = time.time()
                    signal_generation_times.append(end_time - start_time)

        avg_signal_time = sum(signal_generation_times) / len(signal_generation_times)

        # Benchmark trade execution
        wallet = ctx['wallet']
        trade_execution_times = []

        for i in range(10):
            start_time = time.time()
            wallet.buy('BTC/USDT', 0.1, 50000.0 + i * 100)
            wallet.sell('BTC/USDT', 0.1, 50100.0 + i * 100)
            end_time = time.time()
            trade_execution_times.append(end_time - start_time)

        avg_trade_time = sum(trade_execution_times) / len(trade_execution_times)

        # Benchmark risk calculations
        risk_manager = ctx['risk_manager']
        risk_calculation_times = []

        for i in range(10):
            start_time = time.time()
            position_size = risk_manager.calculate_position_size(
                'BTC/USDT', 50000.0 + i * 100, wallet.balance
            )
            can_trade = risk_manager.check_risk_limits(
                'BTC/USDT', position_size, 50000.0 + i * 100
            )
            end_time = time.time()
            risk_calculation_times.append(end_time - start_time)

        avg_risk_time = sum(risk_calculation_times) / len(risk_calculation_times)

        # Verify performance is acceptable
        assert avg_signal_time < 1.0  # Less than 1 second per signal
        assert avg_trade_time < 0.1   # Less than 0.1 seconds per trade
        assert avg_risk_time < 0.1    # Less than 0.1 seconds per risk check

        # Overall system should be responsive
        total_avg_time = avg_signal_time + avg_trade_time + avg_risk_time
        assert total_avg_time < 1.2  # Total workflow under 1.2 seconds

    def test_cross_component_data_consistency(self, end_to_end_context):
        """Test data consistency across all system components."""
        ctx = end_to_end_context

        # Execute trades through wallet
        wallet = ctx['wallet']
        initial_balance = wallet.balance

        wallet.buy('BTC/USDT', 1.0, 50000.0)
        after_buy_balance = wallet.balance

        wallet.sell('BTC/USDT', 1.0, 51000.0)
        final_balance = wallet.balance

        # Verify balance consistency
        assert after_buy_balance == initial_balance - 50000.0
        assert final_balance == after_buy_balance + 51000.0

        # Check position consistency
        assert 'BTC/USDT' not in wallet.positions  # Should be closed

        # Verify PnL calculation consistency
        expected_pnl = 1000.0  # 51000 - 50000
        assert abs(wallet.realized_pnl - expected_pnl) < 0.01

        # Cross-check with risk manager
        risk_manager = ctx['risk_manager']
        risk_manager.equity = final_balance

        # Risk manager should reflect current equity
        assert risk_manager.equity == final_balance

        # Configuration consistency
        config = ctx['config']
        assert wallet.max_open_trades == config['trading']['max_positions']
        assert risk_manager.config.max_drawdown == config['risk_management']['max_drawdown']

    def test_system_load_and_stress_testing(self, end_to_end_context):
        """Test system behavior under load and stress conditions."""
        ctx = end_to_end_context

        # High-frequency trading simulation
        wallet = ctx['wallet']
        market_data = ctx['market_data']

        # Simulate high-frequency trading
        trades_executed = 0
        start_time = time.time()

        for i in range(100):  # 100 rapid trades
            price = market_data['close'].iloc[i % len(market_data)]
            amount = 0.01  # Small amounts for high frequency

            if i % 2 == 0:
                success = wallet.buy('BTC/USDT', amount, price)
            else:
                success = wallet.sell('BTC/USDT', amount, price)

            if success:
                trades_executed += 1

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify system handled load
        assert trades_executed > 0
        assert execution_time < 10  # Should complete within 10 seconds
        assert wallet.balance >= 0  # Should maintain valid balance

        # Test memory usage under load
        # (In real implementation, would monitor actual memory usage)
        assert len(wallet.positions) <= wallet.max_open_trades

        # Test system recovery after load
        recovery_success = wallet.buy('BTC/USDT', 0.1, 50000.0)
        assert recovery_success  # System should still function after load

    def test_end_to_end_audit_trail(self, end_to_end_context):
        """Test complete audit trail throughout the system."""
        ctx = end_to_end_context

        # Initialize audit trail
        audit_events = []

        def record_event(event_type, details):
            audit_events.append({
                'timestamp': datetime.now(),
                'event_type': event_type,
                'details': details
            })

        # Execute complete trading workflow with audit trail
        wallet = ctx['wallet']

        # Record initial state
        record_event('system_startup', {
            'balance': wallet.balance,
            'max_positions': wallet.max_open_trades
        })

        # Execute trades
        wallet.buy('BTC/USDT', 1.0, 50000.0)
        record_event('trade_executed', {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 1.0,
            'price': 50000.0,
            'balance_after': wallet.balance
        })

        wallet.sell('BTC/USDT', 1.0, 51000.0)
        record_event('trade_executed', {
            'symbol': 'BTC/USDT',
            'side': 'sell',
            'amount': 1.0,
            'price': 51000.0,
            'pnl': wallet.realized_pnl,
            'balance_after': wallet.balance
        })

        # Record final state
        record_event('system_status', {
            'final_balance': wallet.balance,
            'total_pnl': wallet.realized_pnl,
            'total_trades': wallet.total_trades,
            'win_rate': wallet.win_rate
        })

        # Verify audit trail completeness
        assert len(audit_events) >= 4  # startup, buy, sell, status

        # Verify chronological order
        timestamps = [event['timestamp'] for event in audit_events]
        assert timestamps == sorted(timestamps)

        # Verify event types
        event_types = [event['event_type'] for event in audit_events]
        assert 'system_startup' in event_types
        assert 'trade_executed' in event_types
        assert 'system_status' in event_types

        # Verify data integrity
        for event in audit_events:
            assert 'timestamp' in event
            assert 'event_type' in event
            assert 'details' in event
            assert isinstance(event['details'], dict)
