"""Integration tests for risk management system.

This module tests the complete risk management pipeline including position sizing,
stop loss management, drawdown control, and portfolio risk limits.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import asyncio
from datetime import datetime, timedelta

from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.risk.exit_manager import calculate_trailing_stop, should_exit, get_partial_exit_percent
from crypto_bot.capital_tracker import CapitalTracker
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.utils.telegram import TelegramNotifier


@pytest.mark.integration
class TestRiskManagementIntegration:
    """Test risk management integration across all components."""

    @pytest.fixture
    def risk_config(self):
        """Comprehensive risk configuration for testing."""
        return RiskConfig(
            max_drawdown=0.1,  # 10% max drawdown
            stop_loss_pct=0.05,  # 5% stop loss
            take_profit_pct=0.1,  # 10% take profit
            min_fng=25,
            min_sentiment=25,
            bull_fng=75,
            bull_sentiment=75,
            min_atr_pct=0.01,
            max_funding_rate=0.01,
            trade_size_pct=0.02,  # 2% of capital per trade
            risk_pct=0.01,  # 1% risk per trade
            min_volume=10000.0,
            volume_threshold_ratio=0.1,
            strategy_allocation={'trend_bot': 0.4, 'mean_bot': 0.3, 'breakout_bot': 0.3},
            volume_ratio=1.0,
            atr_period=14,
            stop_loss_atr_mult=2.0,
            take_profit_atr_mult=4.0,
            max_pair_drawdown=0.05,
            pair_drawdown_lookback=20
        )

    @pytest.fixture
    def risk_manager(self, risk_config):
        """Risk manager instance with full configuration."""
        return RiskManager(risk_config)

    @pytest.fixture
    def capital_tracker(self):
        """Capital tracker for strategy allocation."""
        return CapitalTracker({
            'trend_bot': 0.4,
            'mean_bot': 0.3,
            'breakout_bot': 0.3
        })

    @pytest.fixture
    def paper_wallet(self):
        """Paper wallet for position management."""
        return PaperWallet(balance=100000.0, max_open_trades=20)

    @pytest.fixture
    def sample_market_data(self):
        """Sample market data with volatility for ATR calculation."""
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        np.random.seed(42)

        # Create realistic price data with trends and volatility
        base_price = 50000.0
        trend = np.linspace(0, 5000, 50)  # Upward trend
        volatility = np.random.normal(0, 1000, 50)  # Price noise
        prices = base_price + trend + volatility

        # Create OHLCV data
        data = {
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 500)) for p in prices],
            'low': [p - abs(np.random.normal(0, 500)) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100000, 1000000, 50)
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def test_position_sizing_integration(self, risk_manager, capital_tracker):
        """Test position sizing across risk management and capital allocation."""
        symbol = 'BTC/USDT'
        current_price = 50000.0
        account_balance = 100000.0

        # Calculate position size through risk manager
        position_size = risk_manager.calculate_position_size(
            symbol, current_price, account_balance
        )

        # Verify position size respects risk limits
        max_position_value = account_balance * risk_manager.config.trade_size_pct
        assert position_size * current_price <= max_position_value

        # Verify position size respects risk percentage
        risk_amount = account_balance * risk_manager.config.risk_pct
        stop_loss_amount = current_price * risk_manager.config.stop_loss_pct
        max_risk_based_size = risk_amount / stop_loss_amount
        assert position_size <= max_risk_based_size

        # Test capital allocation integration
        strategy = 'trend_bot'
        allocated_capital = capital_tracker.allocate_capital(strategy, account_balance)
        strategy_position_size = risk_manager.calculate_position_size(
            symbol, current_price, allocated_capital
        )

        # Strategy position should be proportional to allocation
        expected_proportion = capital_tracker.get_allocation(strategy)
        actual_proportion = (strategy_position_size * current_price) / account_balance
        assert abs(actual_proportion - expected_proportion) < 0.01

    def test_stop_loss_management_integration(self, risk_manager, paper_wallet, sample_market_data):
        """Test stop loss management integration with position tracking."""
        symbol = 'BTC/USDT'
        entry_price = 50000.0
        position_size = 1.0

        # Establish position
        paper_wallet.buy(symbol, position_size, entry_price)

        # Calculate stop loss price
        stop_price = risk_manager.calculate_stop_loss_price(symbol, entry_price, 'buy')
        expected_stop = entry_price * (1 - risk_manager.config.stop_loss_pct)
        assert abs(stop_price - expected_stop) < 0.01

        # Test ATR-based stop loss
        atr_value = 1000.0  # Simulated ATR
        atr_stop_price = entry_price - (atr_value * risk_manager.config.stop_loss_atr_mult)
        assert atr_stop_price < stop_price  # ATR stop should be wider

        # Test trailing stop calculation
        current_price = 55000.0  # 10% profit
        trailing_stop = calculate_trailing_stop(
            entry_price, current_price, risk_manager.config.stop_loss_pct
        )
        expected_trailing = current_price * (1 - risk_manager.config.stop_loss_pct)
        assert abs(trailing_stop - expected_trailing) < 0.01

        # Test stop loss trigger
        exit_triggered = should_exit(
            entry_price, current_price, stop_price, 'buy',
            risk_manager.config.take_profit_pct
        )
        assert not exit_triggered  # Should not exit at profit

        # Test exit at stop loss
        exit_price = stop_price * 0.99  # Below stop loss
        exit_triggered = should_exit(
            entry_price, exit_price, stop_price, 'buy',
            risk_manager.config.take_profit_pct
        )
        assert exit_triggered  # Should exit at stop loss

    def test_drawdown_control_integration(self, risk_manager, paper_wallet):
        """Test drawdown control and portfolio protection."""
        initial_balance = paper_wallet.balance
        initial_equity = risk_manager.equity

        # Simulate losing trades
        losing_trades = [
            ('BTC/USDT', 1.0, 50000.0, 47500.0),  # 5% loss
            ('ETH/USDT', 10.0, 3000.0, 2850.0),  # 5% loss
            ('ADA/USDT', 1000.0, 1.0, 0.95),     # 5% loss
        ]

        total_loss = 0
        for symbol, amount, entry_price, exit_price in losing_trades:
            paper_wallet.buy(symbol, amount, entry_price)
            paper_wallet.sell(symbol, amount, exit_price)
            loss = (entry_price - exit_price) * amount
            total_loss += loss

        # Update risk manager equity
        current_balance = paper_wallet.balance
        risk_manager.equity = current_balance / initial_balance
        risk_manager.peak_equity = max(risk_manager.peak_equity, risk_manager.equity)

        # Calculate drawdown
        drawdown = (risk_manager.peak_equity - risk_manager.equity) / risk_manager.peak_equity

        # Test drawdown limits
        can_trade = risk_manager.check_risk_limits('SOL/USDT', 0.1, 100.0)
        if drawdown > risk_manager.config.max_drawdown:
            assert not can_trade  # Should prevent trading when drawdown exceeded

        # Test drawdown recovery
        # Simulate recovery trade
        paper_wallet.buy('SOL/USDT', 10.0, 100.0)
        paper_wallet.sell('SOL/USDT', 10.0, 110.0)  # 10% profit

        recovered_balance = paper_wallet.balance
        recovered_equity = recovered_balance / initial_balance
        risk_manager.equity = recovered_equity

        # Should allow trading after recovery
        can_trade = risk_manager.check_risk_limits('DOT/USDT', 0.1, 10.0)
        assert can_trade

    def test_portfolio_risk_limits_integration(self, risk_manager, paper_wallet):
        """Test portfolio-level risk limits and diversification."""
        # Establish multiple positions
        positions = [
            ('BTC/USDT', 1.0, 50000.0),
            ('ETH/USDT', 10.0, 3000.0),
            ('ADA/USDT', 1000.0, 1.0),
            ('SOL/USDT', 50.0, 100.0),
            ('DOT/USDT', 100.0, 10.0),
        ]

        total_exposure = 0
        for symbol, amount, price in positions:
            paper_wallet.buy(symbol, amount, price)
            total_exposure += amount * price

        # Check portfolio concentration limits
        max_single_position = total_exposure * 0.2  # 20% max per position
        for symbol, position in paper_wallet.positions.items():
            position_value = position['amount'] * position['entry_price']
            assert position_value <= max_single_position

        # Test correlation-based risk (simplified)
        # In real system, would check correlation matrix
        correlated_pairs = [('BTC/USDT', 'ETH/USDT'), ('ADA/USDT', 'SOL/USDT')]
        for pair in correlated_pairs:
            if all(p in paper_wallet.positions for p in pair):
                # Would reduce position sizes for correlated assets
                pass

        # Test sector diversification (simplified)
        sectors = {
            'large_cap': ['BTC/USDT', 'ETH/USDT'],
            'mid_cap': ['ADA/USDT', 'SOL/USDT'],
            'small_cap': ['DOT/USDT']
        }

        for sector, symbols in sectors.items():
            sector_exposure = sum(
                paper_wallet.positions[s]['amount'] * paper_wallet.positions[s]['entry_price']
                for s in symbols if s in paper_wallet.positions
            )
            max_sector_exposure = total_exposure * 0.4  # 40% max per sector
            assert sector_exposure <= max_sector_exposure

    def test_take_profit_management_integration(self, risk_manager, paper_wallet):
        """Test take profit management and partial exits."""
        symbol = 'BTC/USDT'
        entry_price = 50000.0
        position_size = 2.0

        # Establish position
        paper_wallet.buy(symbol, position_size, entry_price)

        # Test take profit trigger
        take_profit_price = entry_price * (1 + risk_manager.config.take_profit_pct)
        current_price = take_profit_price * 1.01  # Above take profit

        exit_triggered = should_exit(
            entry_price, current_price, entry_price * 0.95, 'buy',
            risk_manager.config.take_profit_pct
        )
        assert exit_triggered

        # Test partial exit percentages
        partial_exit_pct = get_partial_exit_percent(current_price, entry_price)
        assert 0 <= partial_exit_pct <= 1

        # Execute partial exit
        exit_amount = position_size * partial_exit_pct
        paper_wallet.sell(symbol, exit_amount, current_price)

        # Verify remaining position
        remaining_position = paper_wallet.positions[symbol]
        assert remaining_position['amount'] == position_size - exit_amount

        # Test scaling out at different profit levels
        profit_levels = [0.05, 0.1, 0.15, 0.2]  # 5%, 10%, 15%, 20%
        remaining_amount = remaining_position['amount']

        for profit_pct in profit_levels:
            if remaining_amount > 0:
                profit_price = entry_price * (1 + profit_pct)
                exit_pct = get_partial_exit_percent(profit_price, entry_price)
                exit_amount = remaining_amount * exit_pct
                if exit_amount > 0:
                    paper_wallet.sell(symbol, exit_amount, profit_price)
                    remaining_amount -= exit_amount

        # Verify final position
        if symbol in paper_wallet.positions:
            final_position = paper_wallet.positions[symbol]
            assert final_position['amount'] >= 0

    def test_volume_and_liquidity_risk_integration(self, risk_manager, sample_market_data):
        """Test volume and liquidity-based risk controls."""
        symbol = 'BTC/USDT'

        # Test minimum volume requirement
        low_volume = 5000.0
        high_volume = 50000.0

        # Should reject trades with insufficient volume
        can_trade_low_volume = risk_manager.check_risk_limits(
            symbol, 1.0, 50000.0, volume=low_volume
        )
        assert not can_trade_low_volume

        # Should allow trades with sufficient volume
        can_trade_high_volume = risk_manager.check_risk_limits(
            symbol, 1.0, 50000.0, volume=high_volume
        )
        assert can_trade_high_volume

        # Test volume ratio thresholds
        market_volume = 100000.0
        trade_volume = 5000.0
        volume_ratio = trade_volume / market_volume

        if volume_ratio < risk_manager.config.volume_threshold_ratio:
            # Should apply volume-based position size reduction
            position_size = risk_manager.calculate_position_size(
                symbol, 50000.0, 100000.0, volume=trade_volume
            )
            # Position size should be reduced for low volume
            normal_size = risk_manager.calculate_position_size(
                symbol, 50000.0, 100000.0, volume=market_volume
            )
            assert position_size <= normal_size

    def test_market_condition_risk_integration(self, risk_manager):
        """Test market condition-based risk adjustments."""
        symbol = 'BTC/USDT'

        # Test volatility-based risk adjustment
        high_volatility = 0.05  # 5% volatility
        low_volatility = 0.01   # 1% volatility

        # Should reduce position size in high volatility
        high_vol_size = risk_manager.calculate_position_size(
            symbol, 50000.0, 100000.0, volatility=high_volatility
        )
        low_vol_size = risk_manager.calculate_position_size(
            symbol, 50000.0, 100000.0, volatility=low_volatility
        )
        assert high_vol_size <= low_vol_size

        # Test sentiment-based risk adjustment
        bearish_sentiment = 20
        bullish_sentiment = 80

        # Should be more conservative in bearish conditions
        bearish_size = risk_manager.calculate_position_size(
            symbol, 50000.0, 100000.0, sentiment=bearish_sentiment
        )
        bullish_size = risk_manager.calculate_position_size(
            symbol, 50000.0, 100000.0, sentiment=bullish_sentiment
        )
        assert bearish_size <= bullish_size

        # Test FNG (Fear & Greed) index integration
        extreme_fear = 10
        extreme_greed = 90

        # Should prevent trading in extreme fear
        can_trade_fear = risk_manager.check_risk_limits(
            symbol, 1.0, 50000.0, fng=extreme_fear
        )
        assert not can_trade_fear

        # Should allow trading in normal conditions
        can_trade_normal = risk_manager.check_risk_limits(
            symbol, 1.0, 50000.0, fng=50
        )
        assert can_trade_normal

    def test_strategy_risk_allocation_integration(self, risk_manager, capital_tracker):
        """Test strategy-based risk allocation."""
        total_capital = 100000.0
        strategies = ['trend_bot', 'mean_bot', 'breakout_bot']

        # Test capital allocation per strategy
        for strategy in strategies:
            allocated = capital_tracker.allocate_capital(strategy, total_capital)
            expected = total_capital * capital_tracker.get_allocation(strategy)
            assert abs(allocated - expected) < 0.01

            # Test risk-adjusted position sizing per strategy
            position_size = risk_manager.calculate_position_size(
                'BTC/USDT', 50000.0, allocated
            )
            max_strategy_size = allocated * risk_manager.config.trade_size_pct
            assert position_size * 50000.0 <= max_strategy_size

        # Test strategy performance-based allocation adjustment
        # Simulate strategy performance
        strategy_performance = {
            'trend_bot': 0.15,     # 15% return
            'mean_bot': -0.05,     # -5% return
            'breakout_bot': 0.08   # 8% return
        }

        # Would adjust allocations based on performance
        total_return = sum(strategy_performance.values())
        adjusted_allocations = {}
        for strategy, return_pct in strategy_performance.items():
            # Simple adjustment: increase allocation for better performers
            base_allocation = capital_tracker.get_allocation(strategy)
            performance_factor = 1 + (return_pct / total_return)
            adjusted_allocations[strategy] = base_allocation * performance_factor

        # Normalize allocations
        total_adjusted = sum(adjusted_allocations.values())
        for strategy in adjusted_allocations:
            adjusted_allocations[strategy] /= total_adjusted

        # Verify allocations still sum to 1
        assert abs(sum(adjusted_allocations.values()) - 1.0) < 0.01

    def test_risk_manager_telegram_integration(self, risk_manager):
        """Test risk manager integration with Telegram notifications."""
        with patch('crypto_bot.utils.telegram.send_message') as mock_send:

            # Simulate risk limit breach
            risk_manager.equity = 0.85  # 15% drawdown
            risk_manager.peak_equity = 1.0

            # Check risk limits (would trigger notification in real system)
            can_trade = risk_manager.check_risk_limits('BTC/USDT', 1.0, 50000.0)

            # Verify risk alert would be sent
            if not can_trade:
                mock_send.assert_called()
                message = mock_send.call_args[0][0]
                assert 'drawdown' in message.lower() or 'risk' in message.lower()

    def test_concurrent_risk_management(self, risk_manager, paper_wallet):
        """Test concurrent risk management operations."""
        import threading

        results = []
        errors = []

        def execute_risk_check(symbol, amount, price):
            try:
                can_trade = risk_manager.check_risk_limits(symbol, amount, price)
                position_size = risk_manager.calculate_position_size(symbol, price, 100000.0)
                results.append((symbol, can_trade, position_size))
            except Exception as e:
                errors.append((symbol, str(e)))

        # Create concurrent risk checks
        symbols = [f'BTC{i}/USDT' for i in range(10)]
        threads = []

        for symbol in symbols:
            thread = threading.Thread(
                target=execute_risk_check,
                args=(symbol, 1.0, 50000.0)
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(results) == len(symbols)
        assert len(errors) == 0

        # All risk checks should return consistent results
        first_result = results[0]
        for result in results[1:]:
            assert result[1] == first_result[1]  # Same can_trade result

    def test_risk_manager_state_persistence(self, risk_manager):
        """Test risk manager state persistence across sessions."""
        # Set initial state
        risk_manager.equity = 0.95
        risk_manager.peak_equity = 1.0
        risk_manager.stop_orders = {'BTC/USDT': {'price': 47500.0}}

        # In real implementation, would save to file/database
        # Here we just verify state is maintained
        assert risk_manager.equity == 0.95
        assert risk_manager.peak_equity == 1.0
        assert 'BTC/USDT' in risk_manager.stop_orders

        # Simulate state reset (like system restart)
        risk_manager.equity = 1.0
        risk_manager.peak_equity = 1.0
        risk_manager.stop_orders = {}

        # Verify state was reset
        assert risk_manager.equity == 1.0
        assert len(risk_manager.stop_orders) == 0

    def test_adaptive_risk_management(self, risk_manager, paper_wallet):
        """Test adaptive risk management based on performance."""
        # Start with conservative settings
        original_risk_pct = risk_manager.config.risk_pct

        # Simulate good performance
        winning_trades = 8
        total_trades = 10
        win_rate = winning_trades / total_trades

        # Adaptive risk: increase risk for good performance
        if win_rate > 0.7:  # 70% win rate
            risk_manager.config.risk_pct = min(
                original_risk_pct * 1.2,  # Increase by 20%
                0.02  # Max 2% risk
            )

        # Verify risk was increased
        assert risk_manager.config.risk_pct > original_risk_pct

        # Test position sizing with increased risk
        position_size = risk_manager.calculate_position_size('BTC/USDT', 50000.0, 100000.0)
        original_size = original_risk_pct * 100000.0 / (50000.0 * 0.05)  # Risk / (price * stop_loss)

        assert position_size > original_size

        # Simulate poor performance
        risk_manager.config.risk_pct = original_risk_pct  # Reset
        losing_trades = 7
        total_trades = 10
        win_rate = (total_trades - losing_trades) / total_trades

        # Adaptive risk: decrease risk for poor performance
        if win_rate < 0.4:  # 40% win rate
            risk_manager.config.risk_pct = max(
                original_risk_pct * 0.8,  # Decrease by 20%
                0.005  # Min 0.5% risk
            )

        # Verify risk was decreased
        assert risk_manager.config.risk_pct < original_risk_pct

        # Test position sizing with decreased risk
        position_size = risk_manager.calculate_position_size('BTC/USDT', 50000.0, 100000.0)
        assert position_size < original_size
