"""Integration tests for wallet and position management.

This module tests wallet balance tracking, position management, PnL calculation,
and the interaction between wallet operations and position updates.
"""

import pytest
import json
import tempfile
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.wallet_manager import load_or_create
from crypto_bot.utils.position_logger import log_position
from crypto_bot.utils.trade_logger import log_trade
from crypto_bot.utils.pnl_logger import log_pnl
from crypto_bot.capital_tracker import CapitalTracker


@pytest.mark.integration
class TestWalletManagementIntegration:
    """Test wallet and position management integration."""

    @pytest.fixture
    def paper_wallet(self):
        """Paper wallet instance for testing."""
        return PaperWallet(balance=10000.0, max_open_trades=10)

    @pytest.fixture
    def capital_tracker(self):
        """Capital tracker for portfolio management."""
        return CapitalTracker({'trend_bot': 0.5, 'mean_bot': 0.3, 'breakout_bot': 0.2})

    @pytest.fixture
    def sample_trades(self):
        """Sample trade data for testing."""
        return [
            {
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'amount': 1.0,
                'price': 50000.0,
                'fee': 5.0,
                'timestamp': datetime.now() - timedelta(hours=2)
            },
            {
                'symbol': 'ETH/USDT',
                'side': 'buy',
                'amount': 10.0,
                'price': 3000.0,
                'fee': 3.0,
                'timestamp': datetime.now() - timedelta(hours=1)
            },
            {
                'symbol': 'BTC/USDT',
                'side': 'sell',
                'amount': 0.5,
                'price': 52000.0,
                'fee': 2.6,
                'timestamp': datetime.now() - timedelta(minutes=30)
            }
        ]

    @pytest.fixture
    def mock_exchange(self):
        """Mock exchange for live trading simulation."""
        exchange = Mock()
        exchange.fetch_balance = AsyncMock(return_value={
            'USDT': {'free': 5000.0, 'total': 5000.0, 'used': 0.0},
            'BTC': {'free': 0.5, 'total': 0.5, 'used': 0.0},
            'ETH': {'free': 5.0, 'total': 5.0, 'used': 0.0}
        })
        exchange.create_order = AsyncMock(return_value={
            'id': 'test_order_123',
            'status': 'closed',
            'amount': 1.0,
            'price': 50000.0,
            'cost': 50000.0,
            'fee': {'cost': 5.0, 'currency': 'USDT'}
        })
        return exchange

    def test_paper_wallet_basic_operations(self, paper_wallet):
        """Test basic paper wallet buy/sell operations."""
        initial_balance = paper_wallet.balance

        # Buy operation
        success = paper_wallet.buy('BTC/USDT', 1.0, 50000.0)
        assert success
        assert paper_wallet.balance == initial_balance - 50000.0
        assert 'BTC/USDT' in paper_wallet.positions

        position = paper_wallet.positions['BTC/USDT']
        assert position['amount'] == 1.0
        assert position['entry_price'] == 50000.0
        assert position['side'] == 'buy'

        # Sell operation
        success = paper_wallet.sell('BTC/USDT', 1.0, 52000.0)
        assert success
        assert paper_wallet.balance > initial_balance  # Should be profitable
        assert 'BTC/USDT' not in paper_wallet.positions

        # Check realized PnL
        assert paper_wallet.realized_pnl == 2000.0  # (52000 - 50000) * 1.0

    def test_paper_wallet_partial_positions(self, paper_wallet):
        """Test partial position management."""
        # Buy initial position
        paper_wallet.buy('BTC/USDT', 2.0, 50000.0)

        # Partial sell
        success = paper_wallet.sell('BTC/USDT', 1.0, 52000.0)
        assert success

        # Check remaining position
        assert 'BTC/USDT' in paper_wallet.positions
        position = paper_wallet.positions['BTC/USDT']
        assert position['amount'] == 1.0  # Should have 1.0 remaining

        # Sell remaining position
        success = paper_wallet.sell('BTC/USDT', 1.0, 51000.0)
        assert success
        assert 'BTC/USDT' not in paper_wallet.positions

        # Check total realized PnL
        expected_pnl = (52000.0 - 50000.0) * 1.0 + (51000.0 - 50000.0) * 1.0
        assert paper_wallet.realized_pnl == expected_pnl

    def test_paper_wallet_scaling_positions(self, paper_wallet):
        """Test scaling in/out of positions."""
        # Scale in (multiple buys)
        paper_wallet.buy('BTC/USDT', 1.0, 50000.0)
        paper_wallet.buy('BTC/USDT', 0.5, 51000.0)

        position = paper_wallet.positions['BTC/USDT']
        assert position['amount'] == 1.5

        # Weighted average entry price
        expected_avg_price = (50000.0 * 1.0 + 51000.0 * 0.5) / 1.5
        assert abs(position['entry_price'] - expected_avg_price) < 0.01

        # Scale out (multiple sells)
        paper_wallet.sell('BTC/USDT', 0.5, 52000.0)
        paper_wallet.sell('BTC/USDT', 1.0, 53000.0)

        assert 'BTC/USDT' not in paper_wallet.positions

        # Check realized PnL calculation
        assert paper_wallet.realized_pnl > 0

    def test_paper_wallet_short_positions(self, paper_wallet):
        """Test short position management."""
        paper_wallet.allow_short = True

        # Short sell
        success = paper_wallet.sell('BTC/USDT', 1.0, 50000.0)
        assert success

        position = paper_wallet.positions['BTC/USDT']
        assert position['side'] == 'sell'
        assert position['amount'] == 1.0
        assert position['entry_price'] == 50000.0

        # Cover short (buy back)
        success = paper_wallet.buy('BTC/USDT', 1.0, 48000.0)
        assert success
        assert 'BTC/USDT' not in paper_wallet.positions

        # Check PnL (short profits when price goes down)
        assert paper_wallet.realized_pnl == 2000.0  # (50000 - 48000) * 1.0

    def test_paper_wallet_balance_validation(self, paper_wallet):
        """Test balance validation and insufficient funds handling."""
        # Try to buy more than available balance
        success = paper_wallet.buy('BTC/USDT', 1000.0, 50000.0)  # Would cost 50M
        assert not success  # Should fail
        assert len(paper_wallet.positions) == 0  # No position created
        assert paper_wallet.balance == 10000.0  # Balance unchanged

        # Buy affordable amount
        success = paper_wallet.buy('BTC/USDT', 0.1, 50000.0)
        assert success
        assert paper_wallet.balance == 5000.0  # 10000 - 5000

    def test_paper_wallet_max_positions_limit(self, paper_wallet):
        """Test maximum open positions limit."""
        paper_wallet.max_open_trades = 2

        # Create maximum allowed positions
        paper_wallet.buy('BTC/USDT', 0.1, 50000.0)
        paper_wallet.buy('ETH/USDT', 1.0, 3000.0)

        # Try to create third position (should fail)
        success = paper_wallet.buy('ADA/USDT', 1000.0, 1.0)
        assert not success
        assert len(paper_wallet.positions) == 2

    def test_paper_wallet_pnl_calculation(self, paper_wallet):
        """Test comprehensive PnL calculation."""
        # Create multiple positions with different outcomes
        paper_wallet.buy('BTC/USDT', 1.0, 50000.0)  # Win
        paper_wallet.buy('ETH/USDT', 5.0, 3000.0)   # Loss
        paper_wallet.buy('ADA/USDT', 1000.0, 1.0)   # Win

        # Update prices and calculate unrealized PnL
        current_prices = {
            'BTC/USDT': 52000.0,  # +4%
            'ETH/USDT': 2800.0,   # -6.67%
            'ADA/USDT': 1.1       # +10%
        }

        total_unrealized = 0
        for symbol, position in paper_wallet.positions.items():
            if position['side'] == 'buy':
                current_price = current_prices[symbol]
                pnl = (current_price - position['entry_price']) * position['amount']
                total_unrealized += pnl

        # Verify unrealized PnL calculation
        expected_btc_pnl = (52000.0 - 50000.0) * 1.0
        expected_eth_pnl = (2800.0 - 3000.0) * 5.0
        expected_ada_pnl = (1.1 - 1.0) * 1000.0
        expected_total = expected_btc_pnl + expected_eth_pnl + expected_ada_pnl

        assert abs(total_unrealized - expected_total) < 0.01

    def test_paper_wallet_state_persistence(self, paper_wallet):
        """Test wallet state persistence across sessions."""
        # Create some positions
        paper_wallet.buy('BTC/USDT', 1.0, 50000.0)
        paper_wallet.buy('ETH/USDT', 5.0, 3000.0)
        initial_balance = paper_wallet.balance
        initial_pnl = paper_wallet.realized_pnl

        # Save state
        paper_wallet.save_state()

        # Create new wallet instance (simulate restart)
        new_wallet = PaperWallet(balance=10000.0, max_open_trades=10)
        new_wallet.load_state()

        # Verify state was restored
        assert len(new_wallet.positions) == len(paper_wallet.positions)
        assert new_wallet.balance == initial_balance
        assert new_wallet.realized_pnl == initial_pnl

        for symbol in paper_wallet.positions:
            assert symbol in new_wallet.positions
            original_pos = paper_wallet.positions[symbol]
            restored_pos = new_wallet.positions[symbol]
            assert restored_pos['amount'] == original_pos['amount']
            assert restored_pos['entry_price'] == original_pos['entry_price']

    def test_capital_tracker_integration(self, capital_tracker):
        """Test capital tracker integration with wallet."""
        # Test strategy allocation
        assert capital_tracker.get_allocation('trend_bot') == 0.5
        assert capital_tracker.get_allocation('mean_bot') == 0.3
        assert capital_tracker.get_allocation('breakout_bot') == 0.2

        # Test capital allocation per strategy
        total_capital = 10000.0
        trend_capital = capital_tracker.allocate_capital('trend_bot', total_capital)
        assert trend_capital == 5000.0

        # Test capital usage tracking
        capital_tracker.use_capital('trend_bot', 1000.0)
        remaining = capital_tracker.get_remaining_capital('trend_bot', total_capital)
        assert remaining == 4000.0

    def test_wallet_manager_integration(self):
        """Test wallet manager integration."""
        with patch('crypto_bot.wallet_manager.Path.exists', return_value=False), \
             patch('crypto_bot.wallet_manager.PaperWallet') as mock_wallet_class:

            mock_wallet = Mock()
            mock_wallet_class.return_value = mock_wallet

            # Test wallet creation
            wallet = load_or_create()
            assert wallet is not None
            mock_wallet_class.assert_called_once()

    def test_position_logging_integration(self, paper_wallet):
        """Test position logging integration."""
        with patch('crypto_bot.utils.position_logger.LOG_DIR', Path(tempfile.mkdtemp())), \
             patch('crypto_bot.utils.position_logger.log_position') as mock_log:

            # Create position
            paper_wallet.buy('BTC/USDT', 1.0, 50000.0)

            # Should trigger position logging
            mock_log.assert_called()

            # Verify log call arguments
            call_args = mock_log.call_args
            assert call_args[1]['symbol'] == 'BTC/USDT'
            assert call_args[1]['amount'] == 1.0
            assert call_args[1]['entry_price'] == 50000.0

    def test_trade_logging_integration(self, paper_wallet):
        """Test trade logging integration."""
        with patch('crypto_bot.utils.trade_logger.LOG_DIR', Path(tempfile.mkdtemp())), \
             patch('crypto_bot.utils.trade_logger.log_trade') as mock_log:

            # Execute trade
            paper_wallet.buy('BTC/USDT', 1.0, 50000.0)

            # Should trigger trade logging
            mock_log.assert_called()

    def test_pnl_logging_integration(self, paper_wallet):
        """Test PnL logging integration."""
        with patch('crypto_bot.utils.pnl_logger.LOG_DIR', Path(tempfile.mkdtemp())), \
             patch('crypto_bot.utils.pnl_logger.log_pnl') as mock_log:

            # Execute profitable trade
            paper_wallet.buy('BTC/USDT', 1.0, 50000.0)
            paper_wallet.sell('BTC/USDT', 1.0, 52000.0)

            # Should trigger PnL logging
            mock_log.assert_called()

    def test_multi_asset_portfolio_management(self, paper_wallet):
        """Test multi-asset portfolio management."""
        # Create positions in multiple assets
        assets = [
            ('BTC/USDT', 1.0, 50000.0),
            ('ETH/USDT', 5.0, 3000.0),
            ('ADA/USDT', 1000.0, 1.0),
            ('SOL/USDT', 50.0, 100.0)
        ]

        for symbol, amount, price in assets:
            paper_wallet.buy(symbol, amount, price)

        # Verify portfolio composition
        assert len(paper_wallet.positions) == 4

        # Calculate portfolio weights
        total_value = paper_wallet.total_value
        weights = {}

        for symbol, position in paper_wallet.positions.items():
            if position['side'] == 'buy':
                position_value = position['amount'] * position['entry_price']
                weights[symbol] = position_value / total_value

        # Verify weights sum to 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01

        # Test portfolio rebalancing (simplified)
        target_weights = {symbol: 0.25 for symbol in weights.keys()}

        # This would trigger rebalancing trades in a real system
        for symbol in weights:
            current_weight = weights[symbol]
            target_weight = target_weights[symbol]
            if abs(current_weight - target_weight) > 0.05:  # Rebalance threshold
                # Would execute rebalancing trade here
                pass

    def test_wallet_error_handling(self, paper_wallet):
        """Test wallet error handling and edge cases."""
        # Test invalid symbol
        success = paper_wallet.buy('', 1.0, 50000.0)
        assert not success

        # Test zero/negative amount
        success = paper_wallet.buy('BTC/USDT', 0, 50000.0)
        assert not success

        success = paper_wallet.buy('BTC/USDT', -1.0, 50000.0)
        assert not success

        # Test zero/negative price
        success = paper_wallet.buy('BTC/USDT', 1.0, 0)
        assert not success

        # Test selling without position
        success = paper_wallet.sell('BTC/USDT', 1.0, 50000.0)
        assert not success

        # Test selling more than position
        paper_wallet.buy('BTC/USDT', 1.0, 50000.0)
        success = paper_wallet.sell('BTC/USDT', 2.0, 50000.0)
        assert not success

    def test_wallet_concurrent_operations(self, paper_wallet):
        """Test concurrent wallet operations."""
        import threading

        results = []
        errors = []

        def execute_trade(operation, symbol, amount, price):
            try:
                if operation == 'buy':
                    success = paper_wallet.buy(symbol, amount, price)
                else:
                    success = paper_wallet.sell(symbol, amount, price)
                results.append(success)
            except Exception as e:
                errors.append(str(e))

        # Create concurrent buy operations
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=execute_trade,
                args=('buy', f'BTC{i}/USDT', 0.1, 50000.0)
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(results) == 5
        assert len(errors) == 0
        assert all(results)  # All buys should succeed

        # Check final position count
        assert len(paper_wallet.positions) == 5

    def test_wallet_performance_metrics(self, paper_wallet):
        """Test wallet performance metrics calculation."""
        # Execute a series of trades
        trades = [
            ('buy', 'BTC/USDT', 1.0, 50000.0, 'sell', 52000.0),  # Win
            ('buy', 'ETH/USDT', 5.0, 3000.0, 'sell', 2800.0),   # Loss
            ('buy', 'ADA/USDT', 1000.0, 1.0, 'sell', 1.1),      # Win
            ('buy', 'SOL/USDT', 50.0, 100.0, 'sell', 95.0),     # Loss
        ]

        for trade in trades:
            paper_wallet.buy(trade[1], trade[2], trade[3])
            paper_wallet.sell(trade[1], trade[2], trade[4])

        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade[4] > trade[3])

        expected_win_rate = winning_trades / total_trades * 100

        # Verify metrics
        assert paper_wallet.total_trades == total_trades
        assert paper_wallet.winning_trades == winning_trades
        assert paper_wallet.win_rate == expected_win_rate

        # Verify total realized PnL
        expected_pnl = sum(
            (trade[4] - trade[3]) * trade[2] for trade in trades
        )
        assert abs(paper_wallet.realized_pnl - expected_pnl) < 0.01

    def test_wallet_risk_management_integration(self, paper_wallet):
        """Test wallet integration with risk management."""
        from crypto_bot.risk.risk_manager import RiskManager, RiskConfig

        risk_config = RiskConfig(
            max_drawdown=0.1,
            stop_loss_pct=0.05,
            take_profit_pct=0.1,
            trade_size_pct=0.1,
            risk_pct=0.01
        )

        risk_manager = RiskManager(risk_config)

        # Test position size calculation integration
        symbol = 'BTC/USDT'
        price = 50000.0
        amount = 1.0

        position_size = risk_manager.calculate_position_size(symbol, price, amount)
        max_size = risk_manager.config.trade_size_pct

        assert position_size <= max_size

        # Test wallet balance integration with risk management
        available_balance = paper_wallet.balance
        max_position_value = available_balance * risk_manager.config.trade_size_pct

        # Position size should not exceed risk limits
        assert position_size * price <= max_position_value

    def test_wallet_telegram_integration(self, paper_wallet):
        """Test wallet integration with Telegram notifications."""
        with patch('crypto_bot.utils.telegram.send_message') as mock_send:

            # Execute trade that should trigger notification
            paper_wallet.buy('BTC/USDT', 1.0, 50000.0)

            # Verify notification was sent
            mock_send.assert_called()

            # Check notification content
            call_args = mock_send.call_args
            message = call_args[0][0]
            assert 'BTC/USDT' in message
            assert 'buy' in message or 'BUY' in message
            assert '50000' in message
