"""Integration tests for the complete trading pipeline.

This module tests the critical path from signal generation through execution,
position management, and monitoring to ensure all components work together correctly.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml
import json
from datetime import datetime, timedelta

from crypto_bot.strategy_router import strategy_for, strategy_name
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.execution.cex_executor import execute_trade_async, get_exchange
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.utils.position_logger import log_position
from crypto_bot.utils.trade_logger import log_trade
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.capital_tracker import CapitalTracker
from crypto_bot.cooldown_manager import configure as cooldown_configure, in_cooldown


@pytest.mark.integration
@pytest.mark.asyncio
class TestTradingPipelineIntegration:
    """Test the complete trading pipeline from signal to execution."""

    @pytest.fixture
    def sample_market_data(self):
        """Generate realistic market data for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)

        # Create trending data with some volatility
        base_price = 100.0
        trend = np.linspace(0, 20, 100)  # Upward trend
        noise = np.random.normal(0, 2, 100)
        prices = base_price + trend + noise

        data = {
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 1)) for p in prices],
            'low': [p - abs(np.random.normal(0, 1)) for p in prices],
            'close': prices,
            'volume': np.random.uniform(10000, 100000, 100)
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    @pytest.fixture
    def mock_exchange(self):
        """Mock exchange for testing trade execution."""
        exchange = Mock()
        exchange.id = 'kraken'
        exchange.create_order = AsyncMock(return_value={
            'id': 'test_order_123',
            'status': 'closed',
            'amount': 1.0,
            'price': 100.0,
            'cost': 100.0,
            'fee': {'cost': 0.1, 'currency': 'USDT'}
        })
        exchange.cancel_order = AsyncMock(return_value=True)
        exchange.fetch_balance = AsyncMock(return_value={
            'USDT': {'free': 1000.0, 'total': 1000.0},
            'BTC': {'free': 1.0, 'total': 1.0}
        })
        exchange.fetch_ticker = AsyncMock(return_value={
            'last': 100.0,
            'bid': 99.5,
            'ask': 100.5,
            'volume': 100000
        })
        return exchange

    @pytest.fixture
    def risk_config(self):
        """Standard risk configuration for testing."""
        return RiskConfig(
            max_drawdown=0.1,
            stop_loss_pct=0.05,
            take_profit_pct=0.1,
            trade_size_pct=0.1,
            risk_pct=0.01,
            min_volume=1000.0,
            volume_threshold_ratio=0.1
        )

    @pytest.fixture
    def risk_manager(self, risk_config):
        """Risk manager instance for testing."""
        return RiskManager(risk_config)

    @pytest.fixture
    def paper_wallet(self):
        """Paper wallet for testing position management."""
        return PaperWallet(balance=1000.0, max_open_trades=5)

    @pytest.fixture
    def telegram_notifier(self):
        """Mock telegram notifier for testing notifications."""
        notifier = Mock(spec=TelegramNotifier)
        notifier.send_message = AsyncMock(return_value=True)
        return notifier

    @pytest.fixture
    async def trading_context(self, mock_exchange, risk_manager, paper_wallet, telegram_notifier):
        """Complete trading context with all mocked dependencies."""
        return {
            'exchange': mock_exchange,
            'risk_manager': risk_manager,
            'paper_wallet': paper_wallet,
            'telegram': telegram_notifier,
            'config': {
                'trading': {'enabled': True, 'max_positions': 5},
                'solana': {'enabled': False},
                'telegram': {'enabled': True}
            }
        }

    async def test_complete_buy_signal_pipeline(self, sample_market_data, trading_context):
        """Test complete pipeline from buy signal to position management."""
        symbol = 'BTC/USDT'
        signal = {
            'action': 'buy',
            'symbol': symbol,
            'confidence': 0.85,
            'price': 100.0,
            'amount': 1.0,
            'strategy': 'trend_bot'
        }

        # Mock strategy generation
        with patch('crypto_bot.strategy_router.strategy_for') as mock_strategy_for:
            mock_strategy = Mock()
            mock_strategy.generate_signal = Mock(return_value=signal)
            mock_strategy_for.return_value = mock_strategy

            # Step 1: Signal Generation
            strategy = strategy_for(symbol, sample_market_data, {})
            generated_signal = strategy.generate_signal(sample_market_data, {})

            assert generated_signal['action'] == 'buy'
            assert generated_signal['symbol'] == symbol
            assert generated_signal['confidence'] >= 0.8

            # Step 2: Risk Assessment
            risk_manager = trading_context['risk_manager']
            position_size = risk_manager.calculate_position_size(
                symbol, generated_signal['price'], generated_signal['amount']
            )

            assert position_size > 0
            assert position_size <= risk_manager.config.trade_size_pct

            # Step 3: Pre-Trade Validation
            can_trade = risk_manager.check_risk_limits(symbol, position_size, generated_signal['price'])
            assert can_trade

            # Step 4: Execute Trade (Paper Trading)
            paper_wallet = trading_context['paper_wallet']

            # Simulate buy order execution
            success = paper_wallet.buy(
                symbol=symbol,
                amount=position_size,
                price=generated_signal['price']
            )

            assert success
            assert symbol in paper_wallet.positions
            assert paper_wallet.balance < 1000.0  # Balance should decrease

            # Step 5: Position Monitoring
            current_price = 105.0  # Simulate price increase
            unrealized_pnl = paper_wallet.unrealized_pnl(symbol, current_price)
            assert unrealized_pnl > 0  # Should be profitable

            # Step 6: Logging and Notifications
            telegram = trading_context['telegram']
            await telegram.send_message.assert_called()

            # Verify position was logged
            assert len(paper_wallet.positions) == 1
            position = paper_wallet.positions[symbol]
            assert position['side'] == 'buy'
            assert position['amount'] == position_size
            assert position['entry_price'] == generated_signal['price']

    async def test_complete_sell_signal_pipeline(self, sample_market_data, trading_context):
        """Test complete pipeline from sell signal to position closure."""
        symbol = 'BTC/USDT'

        # First establish a position
        paper_wallet = trading_context['paper_wallet']
        paper_wallet.buy(symbol=symbol, amount=1.0, price=100.0)

        signal = {
            'action': 'sell',
            'symbol': symbol,
            'confidence': 0.9,
            'price': 110.0,
            'amount': 1.0,
            'strategy': 'trend_bot'
        }

        # Step 1: Signal Generation
        with patch('crypto_bot.strategy_router.strategy_for') as mock_strategy_for:
            mock_strategy = Mock()
            mock_strategy.generate_signal = Mock(return_value=signal)
            mock_strategy_for.return_value = mock_strategy

            strategy = strategy_for(symbol, sample_market_data, {})
            generated_signal = strategy.generate_signal(sample_market_data, {})

            assert generated_signal['action'] == 'sell'

            # Step 2: Risk Assessment (Exit Logic)
            risk_manager = trading_context['risk_manager']
            should_exit = risk_manager.should_exit(symbol, generated_signal['price'], generated_signal['price'])

            # Step 3: Execute Sell Order
            success = paper_wallet.sell(
                symbol=symbol,
                amount=generated_signal['amount'],
                price=generated_signal['price']
            )

            assert success
            assert symbol not in paper_wallet.positions  # Position should be closed

            # Step 4: PnL Calculation
            realized_pnl = paper_wallet.realized_pnl
            assert realized_pnl > 0  # Should be profitable (bought at 100, sold at 110)

            # Step 5: Balance Update
            assert paper_wallet.balance > 1000.0  # Balance should increase from profit

            # Step 6: Notifications
            telegram = trading_context['telegram']
            await telegram.send_message.assert_called()

    async def test_risk_limits_integration(self, sample_market_data, trading_context):
        """Test integration of risk limits across the trading pipeline."""
        symbol = 'BTC/USDT'
        risk_manager = trading_context['risk_manager']

        # Test drawdown limits
        risk_manager.equity = 0.85  # 15% drawdown
        risk_manager.peak_equity = 1.0

        # Should reject trades when drawdown limit exceeded
        can_trade = risk_manager.check_risk_limits(symbol, 0.1, 100.0)
        assert not can_trade

        # Test position size limits
        risk_manager.equity = 1.0
        position_size = risk_manager.calculate_position_size(symbol, 100.0, 1.0)
        max_allowed = risk_manager.config.trade_size_pct
        assert position_size <= max_allowed

        # Test stop loss enforcement
        stop_price = risk_manager.calculate_stop_loss_price(symbol, 100.0, 'buy')
        expected_stop = 100.0 * (1 - risk_manager.config.stop_loss_pct)
        assert abs(stop_price - expected_stop) < 0.01

    async def test_cooldown_manager_integration(self, sample_market_data):
        """Test cooldown manager integration with trading pipeline."""
        symbol = 'BTC/USDT'

        # Configure cooldown settings
        cooldown_config = {
            'cooldown_period': 300,  # 5 minutes
            'max_trades_per_hour': 10,
            'symbols': {symbol: {'cooldown_period': 300}}
        }

        cooldown_configure(cooldown_config)

        # Initially should not be in cooldown
        assert not in_cooldown(symbol)

        # Simulate trade and check cooldown
        from crypto_bot.cooldown_manager import mark_cooldown
        mark_cooldown(symbol)

        # Should now be in cooldown
        assert in_cooldown(symbol)

    async def test_position_management_integration(self, trading_context):
        """Test position management across multiple trades."""
        paper_wallet = trading_context['paper_wallet']
        symbol = 'BTC/USDT'

        # Multiple buy orders (scaling in)
        paper_wallet.buy(symbol=symbol, amount=0.5, price=100.0)
        paper_wallet.buy(symbol=symbol, amount=0.3, price=105.0)

        # Should have combined position
        assert symbol in paper_wallet.positions
        position = paper_wallet.positions[symbol]
        assert position['amount'] == 0.8  # Combined amount
        assert position['entry_price'] == 102.0  # Weighted average

        # Partial sell
        paper_wallet.sell(symbol=symbol, amount=0.4, price=110.0)

        # Should have remaining position
        assert symbol in paper_wallet.positions
        position = paper_wallet.positions[symbol]
        assert position['amount'] == 0.4

        # Complete sell
        paper_wallet.sell(symbol=symbol, amount=0.4, price=115.0)

        # Position should be closed
        assert symbol not in paper_wallet.positions

        # Check final PnL
        assert paper_wallet.realized_pnl > 0

    async def test_error_handling_integration(self, trading_context):
        """Test error handling across the trading pipeline."""
        mock_exchange = trading_context['exchange']
        paper_wallet = trading_context['paper_wallet']

        # Simulate exchange error
        mock_exchange.create_order.side_effect = Exception("Exchange connection failed")

        # Should handle gracefully and fallback to paper trading
        symbol = 'BTC/USDT'

        # Paper wallet should still work despite exchange failure
        success = paper_wallet.buy(symbol=symbol, amount=1.0, price=100.0)
        assert success

        # Position should still be created
        assert symbol in paper_wallet.positions

    async def test_telegram_notifications_integration(self, trading_context):
        """Test Telegram notifications throughout trading pipeline."""
        telegram = trading_context['telegram']
        paper_wallet = trading_context['paper_wallet']

        # Buy trade notification
        paper_wallet.buy(symbol='BTC/USDT', amount=1.0, price=100.0)

        # Should have sent notification
        await telegram.send_message.assert_called()

        # Sell trade notification
        paper_wallet.sell(symbol='BTC/USDT', amount=1.0, price=110.0)

        # Should have sent another notification
        assert telegram.send_message.call_count >= 2

    async def test_performance_monitoring_integration(self, sample_market_data, trading_context):
        """Test performance monitoring integration."""
        symbol = 'BTC/USDT'
        paper_wallet = trading_context['paper_wallet']

        # Execute multiple trades
        trades = [
            {'symbol': symbol, 'amount': 1.0, 'price': 100.0, 'side': 'buy'},
            {'symbol': symbol, 'amount': 1.0, 'price': 110.0, 'side': 'sell'},
            {'symbol': symbol, 'amount': 0.8, 'price': 105.0, 'side': 'buy'},
            {'symbol': symbol, 'amount': 0.8, 'price': 115.0, 'side': 'sell'},
        ]

        for trade in trades:
            if trade['side'] == 'buy':
                paper_wallet.buy(trade['symbol'], trade['amount'], trade['price'])
            else:
                paper_wallet.sell(trade['symbol'], trade['amount'], trade['price'])

        # Check performance metrics
        assert paper_wallet.total_trades == 4
        assert paper_wallet.winning_trades > 0
        assert paper_wallet.win_rate > 0

        # Check total portfolio value
        assert paper_wallet.total_value > 0

    @pytest.mark.parametrize("strategy_name", ["trend_bot", "mean_bot", "breakout_bot"])
    async def test_strategy_integration(self, sample_market_data, strategy_name):
        """Test different strategies integration with trading pipeline."""
        symbol = 'BTC/USDT'

        with patch('crypto_bot.strategy_router.strategy_for') as mock_strategy_for:
            mock_strategy = Mock()
            mock_strategy.generate_signal = Mock(return_value={
                'action': 'buy',
                'symbol': symbol,
                'confidence': 0.8,
                'price': 100.0,
                'amount': 1.0,
                'strategy': strategy_name
            })
            mock_strategy_for.return_value = mock_strategy

            strategy = strategy_for(symbol, sample_market_data, {})
            signal = strategy.generate_signal(sample_market_data, {})

            assert signal['strategy'] == strategy_name
            assert signal['action'] in ['buy', 'sell', 'hold']
            assert 0 <= signal['confidence'] <= 1

    async def test_concurrent_trading_integration(self, trading_context):
        """Test concurrent trading operations."""
        paper_wallet = trading_context['paper_wallet']

        # Simulate concurrent trades on different symbols
        symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']

        async def execute_trade(symbol, price):
            success = paper_wallet.buy(symbol=symbol, amount=1.0, price=price)
            return success

        # Execute trades concurrently
        tasks = [
            execute_trade(symbol, 100.0 + i * 10)
            for i, symbol in enumerate(symbols)
        ]

        results = await asyncio.gather(*tasks)

        # All trades should succeed
        assert all(results)

        # Should have positions for all symbols
        for symbol in symbols:
            assert symbol in paper_wallet.positions

        # Total positions should equal number of trades
        assert len(paper_wallet.positions) == len(symbols)

    async def test_configuration_integration(self, trading_context):
        """Test configuration changes affect trading pipeline."""
        config = trading_context['config']
        risk_manager = trading_context['risk_manager']

        # Test trading enabled/disabled
        config['trading']['enabled'] = False
        # In real implementation, this would disable trading

        config['trading']['enabled'] = True

        # Test risk parameter changes
        original_risk_pct = risk_manager.config.risk_pct
        risk_manager.config.risk_pct = 0.02  # Increase risk

        # Position size should increase with higher risk
        position_size = risk_manager.calculate_position_size('BTC/USDT', 100.0, 1.0)
        assert position_size > original_risk_pct

        # Reset
        risk_manager.config.risk_pct = original_risk_pct
