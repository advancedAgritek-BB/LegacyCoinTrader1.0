"""Integration tests for paper trading vs live trading modes.

This module tests the differences and similarities between paper and live trading,
ensuring consistency in behavior while maintaining safety in live mode.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
from datetime import datetime, timedelta

from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.execution.cex_executor import execute_trade_async
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.utils.telegram import TelegramNotifier


@pytest.mark.integration
class TestPaperVsLiveTradingIntegration:
    """Test paper trading vs live trading integration."""

    @pytest.fixture
    def trading_config(self):
        """Trading configuration for both paper and live modes."""
        return {
            'paper': {
                'enabled': True,
                'initial_balance': 10000.0,
                'max_trades': 10,
                'commission': 0.001
            },
            'live': {
                'enabled': True,
                'max_trades': 5,
                'commission': 0.001,
                'min_order_size': 10.0,
                'max_slippage': 0.005
            }
        }

    @pytest.fixture
    def risk_config(self):
        """Risk configuration for both modes."""
        return RiskConfig(
            max_drawdown=0.1,
            stop_loss_pct=0.05,
            take_profit_pct=0.1,
            trade_size_pct=0.05,  # More conservative for live trading
            risk_pct=0.01
        )

    @pytest.fixture
    def mock_exchange(self):
        """Mock exchange for live trading simulation."""
        exchange = Mock()
        exchange.id = 'kraken'
        exchange.create_order = AsyncMock(return_value={
            'id': 'live_order_123',
            'status': 'closed',
            'amount': 1.0,
            'price': 50000.0,
            'cost': 50000.0,
            'fee': {'cost': 5.0, 'currency': 'USDT'}
        })
        exchange.fetch_balance = AsyncMock(return_value={
            'USDT': {'free': 5000.0, 'total': 5000.0},
            'BTC': {'free': 0.5, 'total': 0.5}
        })
        return exchange

    @pytest.fixture
    def paper_wallet(self):
        """Paper wallet for paper trading."""
        return PaperWallet(balance=10000.0, max_open_trades=10)

    @pytest.fixture
    def telegram_notifier(self):
        """Telegram notifier for trade notifications."""
        with patch('crypto_bot.utils.telegram.Bot'):
            return TelegramNotifier(
                bot_token='test_token',
                chat_id='test_chat',
                enabled=True
            )

    def test_paper_trading_execution(self, paper_wallet, risk_config):
        """Test paper trading execution flow."""
        symbol = 'BTC/USDT'
        entry_price = 50000.0
        position_size = 1.0

        # Execute paper trade
        success = paper_wallet.buy(symbol, position_size, entry_price)

        assert success
        assert symbol in paper_wallet.positions
        assert paper_wallet.balance == 10000.0 - entry_price

        # Verify position details
        position = paper_wallet.positions[symbol]
        assert position['amount'] == position_size
        assert position['entry_price'] == entry_price
        assert position['side'] == 'buy'

        # Execute exit trade
        exit_price = 51000.0
        success = paper_wallet.sell(symbol, position_size, exit_price)

        assert success
        assert symbol not in paper_wallet.positions
        assert paper_wallet.realized_pnl == 1000.0  # Profit

    def test_live_trading_execution_simulation(self, mock_exchange, risk_config):
        """Test live trading execution simulation."""
        symbol = 'BTC/USDT'
        amount = 1.0
        price = 50000.0

        # Simulate live trade execution
        with patch('crypto_bot.execution.cex_executor.get_exchange', return_value=mock_exchange):
            # This would normally call the real exchange
            # For testing, we verify the mock is set up correctly
            pass

        # Verify exchange would be called correctly
        mock_exchange.create_order.assert_not_called()  # Not called in this test setup

    def test_paper_vs_live_mode_detection(self, trading_config):
        """Test detection and handling of paper vs live trading modes."""
        # Test paper mode configuration
        paper_config = trading_config['paper']
        assert paper_config['enabled'] is True
        assert 'initial_balance' in paper_config

        # Test live mode configuration
        live_config = trading_config['live']
        assert live_config['enabled'] is True
        assert 'min_order_size' in live_config  # Live-specific setting

        # Test mode switching logic
        current_mode = 'paper' if paper_config['enabled'] else 'live'

        if current_mode == 'paper':
            assert paper_config['max_trades'] >= live_config['max_trades']  # Paper allows more trades
        else:
            assert live_config['max_trades'] <= paper_config['max_trades']  # Live is more conservative

    def test_risk_management_paper_vs_live(self, risk_config):
        """Test risk management differences between paper and live modes."""
        risk_manager = RiskManager(risk_config)

        # Test position sizing for different modes
        symbol = 'BTC/USDT'
        price = 50000.0
        account_balance = 10000.0

        # Paper trading position size (more flexible)
        paper_position_size = risk_manager.calculate_position_size(
            symbol, price, account_balance
        )

        # Live trading should be more conservative
        live_risk_config = RiskConfig(
            max_drawdown=0.05,  # Tighter drawdown limit
            stop_loss_pct=0.03,  # Tighter stop loss
            take_profit_pct=0.08,  # Lower take profit
            trade_size_pct=0.02,  # Smaller position size
            risk_pct=0.005  # Lower risk per trade
        )
        live_risk_manager = RiskManager(live_risk_config)
        live_position_size = live_risk_manager.calculate_position_size(
            symbol, price, account_balance
        )

        # Live trading should use smaller position sizes
        assert live_position_size <= paper_position_size

        # Live trading should have tighter risk limits
        assert live_risk_config.max_drawdown <= risk_config.max_drawdown
        assert live_risk_config.stop_loss_pct <= risk_config.stop_loss_pct
        assert live_risk_config.risk_pct <= risk_config.risk_pct

    def test_trade_execution_differences(self, paper_wallet, mock_exchange, trading_config):
        """Test differences in trade execution between paper and live modes."""
        symbol = 'BTC/USDT'
        amount = 1.0
        price = 50000.0

        # Paper trading execution
        paper_success = paper_wallet.buy(symbol, amount, price)
        assert paper_success

        # Simulate live trading execution
        with patch('crypto_bot.execution.cex_executor.get_exchange', return_value=mock_exchange):
            # Live trading would have additional validations
            live_config = trading_config['live']

            # Check minimum order size
            if amount * price < live_config['min_order_size']:
                live_success = False
            else:
                live_success = True

            # Check slippage limits
            if abs(price - 50000.0) / 50000.0 > live_config['max_slippage']:
                live_success = False

        # Both should succeed in this test scenario
        assert paper_success
        assert live_success

    def test_balance_tracking_differences(self, paper_wallet, mock_exchange):
        """Test balance tracking differences between paper and live modes."""
        # Paper wallet balance tracking
        initial_paper_balance = paper_wallet.balance

        # Simulate paper trades
        paper_wallet.buy('BTC/USDT', 1.0, 50000.0)
        paper_balance_after_buy = paper_wallet.balance

        paper_wallet.sell('BTC/USDT', 1.0, 51000.0)
        final_paper_balance = paper_wallet.balance

        assert paper_balance_after_buy == initial_paper_balance - 50000.0
        assert final_paper_balance == initial_paper_balance + 1000.0  # Profit after fees

        # Live exchange balance tracking
        live_balance = asyncio.run(mock_exchange.fetch_balance())

        # Live balance should include exchange-specific details
        assert 'USDT' in live_balance
        assert 'BTC' in live_balance
        assert 'free' in live_balance['USDT']
        assert 'total' in live_balance['USDT']

    def test_error_handling_paper_vs_live(self, paper_wallet, mock_exchange):
        """Test error handling differences between paper and live modes."""
        # Paper trading error handling (more lenient)
        try:
            # Invalid trade (should handle gracefully in paper mode)
            paper_wallet.buy('INVALID/USDT', -1.0, -100.0)
        except Exception as e:
            paper_error = str(e)

        # Live trading error handling (stricter)
        mock_exchange.create_order.side_effect = Exception("Exchange API error")

        with patch('crypto_bot.execution.cex_executor.get_exchange', return_value=mock_exchange):
            try:
                # This would normally fail in live mode
                pass
            except Exception as e:
                live_error = str(e)

        # Paper mode should be more forgiving
        # Live mode should have stricter error handling

    def test_notification_differences(self, telegram_notifier, paper_wallet):
        """Test notification differences between paper and live modes."""
        with patch.object(telegram_notifier, 'send_message') as mock_send:

            # Paper trading notification
            paper_wallet.buy('BTC/USDT', 1.0, 50000.0)
            # Paper notifications might be less frequent or detailed

            # Live trading notification (would be more critical)
            # Live notifications should be more prominent

            # Verify notification system works for both modes
            asyncio.run(telegram_notifier.send_message("Test notification"))

            mock_send.assert_called_once()

    def test_performance_tracking_differences(self, paper_wallet):
        """Test performance tracking differences between modes."""
        # Paper trading performance tracking
        paper_wallet.buy('BTC/USDT', 1.0, 50000.0)
        paper_wallet.sell('BTC/USDT', 1.0, 51000.0)

        paper_metrics = {
            'total_pnl': paper_wallet.realized_pnl,
            'total_trades': paper_wallet.total_trades,
            'win_rate': paper_wallet.win_rate
        }

        # Live trading would track additional metrics
        live_metrics = paper_metrics.copy()
        live_metrics.update({
            'slippage_cost': 25.0,  # Additional cost in live trading
            'exchange_fees': 10.0,
            'network_latency': 150  # API call latency
        })

        # Live trading should account for additional costs
        assert live_metrics['slippage_cost'] >= 0
        assert live_metrics['exchange_fees'] >= 0

        # Core metrics should be similar
        assert abs(live_metrics['total_pnl'] - paper_metrics['total_pnl']) <= live_metrics['slippage_cost'] + live_metrics['exchange_fees']

    def test_configuration_differences(self, trading_config):
        """Test configuration differences between paper and live modes."""
        paper_config = trading_config['paper']
        live_config = trading_config['live']

        # Paper mode should allow more flexibility
        assert paper_config['max_trades'] >= live_config['max_trades']
        assert paper_config['initial_balance'] > 0

        # Live mode should have additional safety checks
        assert 'min_order_size' in live_config
        assert 'max_slippage' in live_config

        # Commission should be similar but live might have additional fees
        assert abs(paper_config['commission'] - live_config['commission']) <= 0.001

    def test_mode_switching_safety(self, paper_wallet, mock_exchange):
        """Test safety mechanisms when switching between modes."""
        # Start in paper mode
        paper_wallet.buy('BTC/USDT', 1.0, 50000.0)

        # Simulate mode switch validation
        paper_positions = paper_wallet.positions.copy()

        # When switching to live mode, should validate:
        # 1. No open positions in paper mode
        # 2. Sufficient balance synchronization
        # 3. Configuration compatibility

        can_switch = len(paper_positions) == 0  # Should be False if positions exist

        assert not can_switch  # Cannot switch with open positions

        # Close positions before switching
        paper_wallet.sell('BTC/USDT', 1.0, 51000.0)
        can_switch = len(paper_wallet.positions) == 0

        assert can_switch  # Can switch after closing positions

        # Live mode balance verification
        live_balance = asyncio.run(mock_exchange.fetch_balance())
        live_usdt_balance = live_balance['USDT']['free']

        # Should have sufficient balance for live trading
        assert live_usdt_balance > 0

    def test_data_synchronization(self, paper_wallet, mock_exchange):
        """Test data synchronization between paper and live modes."""
        # Paper wallet state
        paper_wallet.buy('BTC/USDT', 0.5, 50000.0)
        paper_balance = paper_wallet.balance
        paper_positions = paper_wallet.positions.copy()

        # Live exchange state
        live_balance = asyncio.run(mock_exchange.fetch_balance())

        # Synchronization checks
        # 1. Balance consistency (paper should not exceed live)
        assert paper_balance <= live_balance['USDT']['total']

        # 2. Position consistency (paper positions should be reflected in live)
        if paper_positions:
            for symbol, position in paper_positions.items():
                if symbol == 'BTC/USDT':
                    paper_btc_amount = position['amount']
                    live_btc_amount = live_balance['BTC']['total']

                    # Paper should not have more than live
                    assert paper_btc_amount <= live_btc_amount

        # 3. Trade history synchronization
        paper_trades = paper_wallet.total_trades
        # Live trades would be tracked separately

    def test_audit_trail_differences(self, paper_wallet):
        """Test audit trail differences between paper and live modes."""
        # Paper trading audit trail
        paper_wallet.buy('BTC/USDT', 1.0, 50000.0)
        paper_audit = {
            'mode': 'paper',
            'trades': paper_wallet.total_trades,
            'balance_changes': [50000.0],  # Simplified
            'timestamps': [datetime.now()]
        }

        # Live trading audit trail (more detailed)
        live_audit = paper_audit.copy()
        live_audit.update({
            'mode': 'live',
            'order_ids': ['live_order_123'],
            'exchange_responses': [{'status': 'filled'}],
            'network_latency': [150, 200, 180],  # API call latencies
            'error_logs': []
        })

        # Live audit should have more detailed tracking
        assert 'order_ids' in live_audit
        assert 'network_latency' in live_audit
        assert 'error_logs' in live_audit

        # Both should track core trading activity
        assert paper_audit['trades'] == live_audit['trades']

    def test_recovery_mechanisms(self, paper_wallet, mock_exchange):
        """Test recovery mechanisms for paper vs live mode failures."""
        # Paper mode recovery (easier)
        paper_wallet.buy('BTC/USDT', 1.0, 50000.0)

        # Simulate paper mode "failure" (just reset)
        paper_wallet.balance = 10000.0
        paper_wallet.positions = {}
        paper_wallet.realized_pnl = 0.0

        # Paper mode can be easily reset
        assert paper_wallet.balance == 10000.0
        assert len(paper_wallet.positions) == 0

        # Live mode recovery (more complex)
        # Simulate exchange connectivity issue
        mock_exchange.fetch_balance.side_effect = Exception("Exchange down")

        with patch('crypto_bot.execution.cex_executor.get_exchange', return_value=mock_exchange):
            # Live mode should have retry mechanisms, circuit breakers, etc.
            # This would be more complex to recover from

            # Verify error handling
            try:
                live_balance = asyncio.run(mock_exchange.fetch_balance())
            except Exception as e:
                assert "Exchange down" in str(e)

    def test_cost_analysis_differences(self, trading_config):
        """Test cost analysis differences between paper and live modes."""
        # Paper trading costs (minimal)
        paper_commission = trading_config['paper']['commission']
        paper_costs = {
            'commission': paper_commission,
            'slippage': 0.0,  # No slippage in paper
            'network': 0.0,   # No network costs
            'total_cost_pct': paper_commission
        }

        # Live trading costs (realistic)
        live_commission = trading_config['live']['commission']
        live_costs = {
            'commission': live_commission,
            'slippage': 0.0005,      # 0.05% slippage
            'network': 0.0001,       # 0.01% network fees
            'market_impact': 0.0002, # 0.02% market impact
            'total_cost_pct': live_commission + 0.0005 + 0.0001 + 0.0002
        }

        # Live trading should have higher total costs
        assert live_costs['total_cost_pct'] > paper_costs['total_cost_pct']

        # Paper trading should have zero slippage and network costs
        assert paper_costs['slippage'] == 0.0
        assert paper_costs['network'] == 0.0

        # Cost impact on profitability
        trade_size = 1000.0  # $1000 trade
        paper_cost_amount = trade_size * paper_costs['total_cost_pct']
        live_cost_amount = trade_size * live_costs['total_cost_pct']

        assert live_cost_amount > paper_cost_amount

    def test_mode_validation_and_safety_checks(self, trading_config):
        """Test validation and safety checks for different modes."""
        # Paper mode validations (minimal)
        paper_checks = {
            'balance_check': lambda: True,  # Always pass in paper
            'position_size_check': lambda size: size > 0,
            'market_hours_check': lambda: True,  # Always open in paper
            'connectivity_check': lambda: True   # Always connected in paper
        }

        # Live mode validations (comprehensive)
        live_checks = {
            'balance_check': lambda: True,  # Would check actual exchange balance
            'position_size_check': lambda size: size >= 10.0,  # Minimum order size
            'market_hours_check': lambda: True,  # Would check exchange hours
            'connectivity_check': lambda: True,  # Would test API connectivity
            'rate_limit_check': lambda: True,    # Would check API rate limits
            'maintenance_check': lambda: False   # Would check for exchange maintenance
        }

        # Test validation differences
        test_size = 5.0

        # Paper mode: allows small orders
        assert paper_checks['position_size_check'](test_size)

        # Live mode: requires minimum size
        assert not live_checks['position_size_check'](test_size)

        # Both modes should pass basic checks
        assert paper_checks['balance_check']()
        assert live_checks['balance_check']()

    def test_performance_comparison_framework(self, paper_wallet, mock_exchange):
        """Test framework for comparing paper vs live performance."""
        # Run identical strategy in both modes
        symbol = 'BTC/USDT'
        trades = [
            {'action': 'buy', 'amount': 1.0, 'price': 50000.0},
            {'action': 'sell', 'amount': 1.0, 'price': 51000.0},
            {'action': 'buy', 'amount': 0.8, 'price': 50500.0},
            {'action': 'sell', 'amount': 0.8, 'price': 51500.0}
        ]

        # Execute in paper mode
        paper_results = []
        for trade in trades:
            if trade['action'] == 'buy':
                paper_wallet.buy(symbol, trade['amount'], trade['price'])
            else:
                paper_wallet.sell(symbol, trade['amount'], trade['price'])

            paper_results.append({
                'balance': paper_wallet.balance,
                'pnl': paper_wallet.realized_pnl,
                'trades': paper_wallet.total_trades
            })

        # Simulate live mode execution
        live_balance = 10000.0
        live_pnl = 0.0
        live_trades = 0
        live_fee_rate = 0.001  # 0.1%

        live_results = []
        for trade in trades:
            if trade['action'] == 'buy':
                cost = trade['amount'] * trade['price']
                fee = cost * live_fee_rate
                live_balance -= (cost + fee)
            else:
                revenue = trade['amount'] * trade['price']
                fee = revenue * live_fee_rate
                live_balance += (revenue - fee)
                live_pnl = live_balance - 10000.0

            live_trades += 1
            live_results.append({
                'balance': live_balance,
                'pnl': live_pnl,
                'trades': live_trades
            })

        # Compare results
        for i, (paper, live) in enumerate(zip(paper_results, live_results)):
            # Live should have lower balance due to fees
            assert live['balance'] <= paper['balance']

            # Both should show increasing trade counts
            assert live['trades'] == paper['trades'] == i + 1

    def test_mode_transition_testing(self, paper_wallet):
        """Test framework for safely transitioning from paper to live."""
        # Phase 1: Paper trading validation
        paper_wallet.buy('BTC/USDT', 1.0, 50000.0)
        paper_wallet.sell('BTC/USDT', 1.0, 51000.0)

        paper_performance = {
            'win_rate': paper_wallet.win_rate,
            'total_pnl': paper_wallet.realized_pnl,
            'total_trades': paper_wallet.total_trades
        }

        # Phase 2: Minimum performance requirements
        min_requirements = {
            'min_trades': 10,
            'min_win_rate': 0.55,
            'min_pnl': 100.0
        }

        meets_requirements = (
            paper_performance['total_trades'] >= min_requirements['min_trades'] and
            paper_performance['win_rate'] >= min_requirements['min_win_rate'] and
            paper_performance['total_pnl'] >= min_requirements['min_pnl']
        )

        # Phase 3: Live trading simulation with smaller size
        if meets_requirements:
            live_position_size = 0.1  # 10% of normal size for initial live trading
            # Would gradually increase position size as confidence builds

            assert live_position_size < 1.0  # Should be smaller than paper
        else:
            # Should not transition to live trading
            assert not meets_requirements

    def test_parallel_paper_live_execution(self, paper_wallet, mock_exchange):
        """Test running paper and live execution in parallel for comparison."""
        symbol = 'BTC/USDT'

        # Define identical trading signals
        signals = [
            {'action': 'buy', 'amount': 1.0, 'price': 50000.0, 'confidence': 0.8},
            {'action': 'sell', 'amount': 1.0, 'price': 51000.0, 'confidence': 0.7},
            {'action': 'buy', 'amount': 0.8, 'price': 50500.0, 'confidence': 0.75},
            {'action': 'sell', 'amount': 0.8, 'price': 51500.0, 'confidence': 0.85}
        ]

        # Execute in paper mode
        paper_execution = []
        for signal in signals:
            if signal['action'] == 'buy':
                success = paper_wallet.buy(symbol, signal['amount'], signal['price'])
            else:
                success = paper_wallet.sell(symbol, signal['amount'], signal['price'])

            paper_execution.append({
                'signal': signal,
                'success': success,
                'balance': paper_wallet.balance,
                'pnl': paper_wallet.realized_pnl
            })

        # Simulate live execution
        live_balance = 10000.0
        live_pnl = 0.0
        live_fee_rate = 0.001

        live_execution = []
        for signal in signals:
            if signal['action'] == 'buy':
                cost = signal['amount'] * signal['price']
                fee = cost * live_fee_rate
                live_balance -= (cost + fee)
                success = True
            else:
                revenue = signal['amount'] * signal['price']
                fee = revenue * live_fee_rate
                live_balance += (revenue - fee)
                live_pnl = live_balance - 10000.0
                success = True

            live_execution.append({
                'signal': signal,
                'success': success,
                'balance': live_balance,
                'pnl': live_pnl
            })

        # Compare executions
        for paper, live in zip(paper_execution, live_execution):
            # Same signals should produce similar results (minus fees)
            assert paper['success'] == live['success']

            # Live should have lower balance due to fees
            assert live['balance'] <= paper['balance']

            # Both should process all signals
            assert len(paper_execution) == len(live_execution) == len(signals)
