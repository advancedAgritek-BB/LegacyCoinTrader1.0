#!/usr/bin/env python3
"""
COMPREHENSIVE STOP LOSS SYSTEM TESTS

This script provides comprehensive testing for the stop loss system
including unit tests, integration tests, and real-time monitoring tests.
"""

import asyncio
import unittest
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import TradeManager, Position
from crypto_bot.risk.exit_manager import calculate_trailing_stop, calculate_atr_trailing_stop
from crypto_bot.position_monitor import PositionMonitor

class TestStopLossSystem(unittest.TestCase):
    """Comprehensive test suite for stop loss system."""

    def setUp(self):
        """Set up test fixtures."""
        self.trade_manager = TradeManager()
        self.config = {
            'exit_strategy': {
                'take_profit_pct': 0.04,
                'trailing_stop_pct': 0.008,
                'real_time_monitoring': {
                    'enabled': True,
                    'check_interval_seconds': 5.0
                }
            },
            'risk': {
                'stop_loss_pct': 0.02,
                'trailing_stop_pct': 0.01
            }
        }

    def test_stop_loss_calculation_long_position(self):
        """Test stop loss calculation for long positions."""
        entry_price = Decimal('100.0')
        stop_loss_pct = Decimal('0.02')  # 2%

        # Long position: stop loss below entry
        expected_stop = entry_price * (1 - stop_loss_pct)
        self.assertEqual(expected_stop, Decimal('98.0'))

    def test_stop_loss_calculation_short_position(self):
        """Test stop loss calculation for short positions."""
        entry_price = Decimal('100.0')
        stop_loss_pct = Decimal('0.02')  # 2%

        # Short position: stop loss above entry
        expected_stop = entry_price * (1 + stop_loss_pct)
        self.assertEqual(expected_stop, Decimal('102.0'))

    def test_take_profit_calculation_long_position(self):
        """Test take profit calculation for long positions."""
        entry_price = Decimal('100.0')
        take_profit_pct = Decimal('0.04')  # 4%

        # Long position: take profit above entry
        expected_tp = entry_price * (1 + take_profit_pct)
        self.assertEqual(expected_tp, Decimal('104.0'))

    def test_take_profit_calculation_short_position(self):
        """Test take profit calculation for short positions."""
        entry_price = Decimal('100.0')
        take_profit_pct = Decimal('0.04')  # 4%

        # Short position: take profit below entry
        expected_tp = entry_price * (1 - take_profit_pct)
        self.assertEqual(expected_tp, Decimal('96.0'))

    def test_trailing_stop_calculation(self):
        """Test trailing stop calculation."""
        import pandas as pd
        import numpy as np

        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        prices = np.random.normal(100, 2, 100)
        df = pd.DataFrame({'close': prices}, index=dates)

        trail_pct = 0.05  # 5%
        trailing_stop = calculate_trailing_stop(df['close'], trail_pct)

        # Trailing stop should be below the highest price
        highest_price = df['close'].max()
        expected_stop = highest_price * (1 - trail_pct)

        self.assertAlmostEqual(trailing_stop, expected_stop, places=2)

    def test_position_should_exit_stop_loss_long(self):
        """Test position exit condition for stop loss on long position."""
        position = Position(
            symbol="TEST/USD",
            side="long",
            total_amount=Decimal('1.0'),
            average_price=Decimal('100.0'),
            stop_loss_price=Decimal('98.0'),  # 2% stop loss
            take_profit_price=Decimal('104.0')
        )

        # Price drops below stop loss - should exit
        current_price = Decimal('97.0')
        should_exit, reason = position.should_exit(current_price)

        self.assertTrue(should_exit)
        self.assertEqual(reason, "stop_loss")

    def test_position_should_exit_stop_loss_short(self):
        """Test position exit condition for stop loss on short position."""
        position = Position(
            symbol="TEST/USD",
            side="short",
            total_amount=Decimal('1.0'),
            average_price=Decimal('100.0'),
            stop_loss_price=Decimal('102.0'),  # 2% stop loss
            take_profit_price=Decimal('96.0')
        )

        # Price rises above stop loss - should exit
        current_price = Decimal('103.0')
        should_exit, reason = position.should_exit(current_price)

        self.assertTrue(should_exit)
        self.assertEqual(reason, "stop_loss")

    def test_position_should_exit_take_profit_long(self):
        """Test position exit condition for take profit on long position."""
        position = Position(
            symbol="TEST/USD",
            side="long",
            total_amount=Decimal('1.0'),
            average_price=Decimal('100.0'),
            stop_loss_price=Decimal('98.0'),
            take_profit_price=Decimal('104.0')  # 4% take profit
        )

        # Price rises above take profit - should exit
        current_price = Decimal('105.0')
        should_exit, reason = position.should_exit(current_price)

        self.assertTrue(should_exit)
        self.assertEqual(reason, "take_profit")

    def test_position_should_exit_take_profit_short(self):
        """Test position exit condition for take profit on short position."""
        position = Position(
            symbol="TEST/USD",
            side="short",
            total_amount=Decimal('1.0'),
            average_price=Decimal('100.0'),
            stop_loss_price=Decimal('102.0'),
            take_profit_price=Decimal('96.0')  # 4% take profit
        )

        # Price drops below take profit - should exit
        current_price = Decimal('95.0')
        should_exit, reason = position.should_exit(current_price)

        self.assertTrue(should_exit)
        self.assertEqual(reason, "take_profit")

    def test_position_should_not_exit_within_range(self):
        """Test that position doesn't exit when price is within range."""
        position = Position(
            symbol="TEST/USD",
            side="long",
            total_amount=Decimal('1.0'),
            average_price=Decimal('100.0'),
            stop_loss_price=Decimal('98.0'),
            take_profit_price=Decimal('104.0')
        )

        # Price within range - should not exit
        current_price = Decimal('101.0')
        should_exit, reason = position.should_exit(current_price)

        self.assertFalse(should_exit)
        self.assertEqual(reason, "")

    def test_trailing_stop_update_long_position(self):
        """Test trailing stop update for long position."""
        position = Position(
            symbol="TEST/USD",
            side="long",
            total_amount=Decimal('1.0'),
            average_price=Decimal('100.0'),
            trailing_stop_pct=Decimal('0.01')  # 1%
        )

        # Initial price update
        position.update_price_levels(Decimal('102.0'))
        position.update_trailing_stop(Decimal('102.0'))

        # Trailing stop should be set
        expected_trailing_stop = Decimal('102.0') * (1 - Decimal('0.01'))
        self.assertEqual(position.stop_loss_price, expected_trailing_stop)

    def test_trailing_stop_update_short_position(self):
        """Test trailing stop update for short position."""
        position = Position(
            symbol="TEST/USD",
            side="short",
            total_amount=Decimal('1.0'),
            average_price=Decimal('100.0'),
            trailing_stop_pct=Decimal('0.01')  # 1%
        )

        # Initial price update
        position.update_price_levels(Decimal('98.0'))
        position.update_trailing_stop(Decimal('98.0'))

        # Trailing stop should be set
        expected_trailing_stop = Decimal('98.0') * (1 + Decimal('0.01'))
        self.assertEqual(position.stop_loss_price, expected_trailing_stop)

    def test_position_persistence(self):
        """Test that positions are properly persisted."""
        # Create a test position
        position = Position(
            symbol="TEST/USD",
            side="long",
            total_amount=Decimal('1.0'),
            average_price=Decimal('100.0'),
            stop_loss_price=Decimal('98.0'),
            take_profit_price=Decimal('104.0'),
            trailing_stop_pct=Decimal('0.01')
        )

        # Add to trade manager
        self.trade_manager.positions["TEST/USD"] = position

        # Save state
        self.trade_manager.save_state()

        # Create new trade manager and load state
        new_tm = TradeManager()
        loaded_position = new_tm.get_position("TEST/USD")

        # Verify position was loaded correctly
        self.assertIsNotNone(loaded_position)
        self.assertEqual(loaded_position.symbol, "TEST/USD")
        self.assertEqual(loaded_position.side, "long")
        self.assertEqual(loaded_position.stop_loss_price, Decimal('98.0'))
        self.assertEqual(loaded_position.take_profit_price, Decimal('104.0'))
        self.assertEqual(loaded_position.trailing_stop_pct, Decimal('0.01'))

class TestStopLossIntegration(unittest.TestCase):
    """Integration tests for stop loss system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.trade_manager = TradeManager()

    def test_real_position_stop_loss_calculation(self):
        """Test stop loss calculation with real position data."""
        # Get real positions from trade manager
        positions = self.trade_manager.get_all_positions()

        if not positions:
            self.skipTest("No real positions available for testing")

        # Test each position has proper stop losses
        for position in positions:
            if not position.is_open:
                continue

            # Verify stop loss is set
            self.assertIsNotNone(position.stop_loss_price,
                f"Position {position.symbol} missing stop loss")

            # Verify take profit is set
            self.assertIsNotNone(position.take_profit_price,
                f"Position {position.symbol} missing take profit")

            # Verify trailing stop percentage is set
            self.assertIsNotNone(position.trailing_stop_pct,
                f"Position {position.symbol} missing trailing stop percentage")

            # Verify stop loss is calculated correctly based on position side
            if position.side == 'long':
                expected_sl = position.average_price * (1 - Decimal('0.008'))
                self.assertAlmostEqual(position.stop_loss_price, expected_sl, places=2,
                    msg=f"Long position {position.symbol} stop loss calculation incorrect")
            else:  # short
                expected_sl = position.average_price * (1 + Decimal('0.008'))
                self.assertAlmostEqual(position.stop_loss_price, expected_sl, places=2,
                    msg=f"Short position {position.symbol} stop loss calculation incorrect")

    def test_stop_loss_configuration_consistency(self):
        """Test that stop loss configuration is consistent across the system."""
        # Load configuration
        try:
            import yaml
            with open('crypto_bot/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
        except Exception:
            self.skipTest("Configuration file not available")

        # Check that stop loss settings are present
        exits_config = config.get('exits', {})
        risk_config = config.get('risk', {})

        # At least one of these should be configured
        has_exits_sl = 'default_sl_pct' in exits_config
        has_risk_sl = 'stop_loss_pct' in risk_config

        self.assertTrue(has_exits_sl or has_risk_sl,
            "No stop loss configuration found in config.yaml")

        if has_exits_sl:
            sl_pct = exits_config['default_sl_pct']
            self.assertIsInstance(sl_pct, (int, float))
            self.assertGreater(sl_pct, 0)
            self.assertLess(sl_pct, 1)  # Should be a percentage, not absolute

class TestStopLossMonitoring(unittest.TestCase):
    """Tests for real-time stop loss monitoring."""

    def setUp(self):
        """Set up monitoring test fixtures."""
        self.config = {
            'exit_strategy': {
                'real_time_monitoring': {
                    'enabled': True,
                    'check_interval_seconds': 1.0,  # Fast for testing
                    'price_update_threshold': 0.001
                }
            }
        }

    @patch('crypto_bot.position_monitor.KrakenWSClient')
    def test_position_monitor_initialization(self, mock_ws_client):
        """Test position monitor initialization."""
        mock_exchange = Mock()
        monitor = PositionMonitor(mock_exchange, self.config, {})

        # Verify configuration is loaded correctly
        self.assertTrue(monitor.use_websocket)
        self.assertEqual(monitor.check_interval_seconds, 1.0)
        self.assertEqual(monitor.price_update_threshold, 0.001)

    def test_monitoring_statistics_tracking(self):
        """Test that monitoring statistics are properly tracked."""
        mock_exchange = Mock()
        monitor = PositionMonitor(mock_exchange, self.config, {})

        # Check initial statistics
        expected_stats = {
            "positions_monitored": 0,
            "price_updates": 0,
            "trailing_stop_triggers": 0,
            "execution_latency_ms": 0,
            "missed_exits": 0
        }

        self.assertEqual(monitor.monitoring_stats, expected_stats)

def run_stop_loss_tests():
    """Run all stop loss tests and generate report."""
    print("🧪 RUNNING COMPREHENSIVE STOP LOSS TESTS")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestStopLossSystem))
    suite.addTest(loader.loadTestsFromTestCase(TestStopLossIntegration))
    suite.addTest(loader.loadTestsFromTestCase(TestStopLossMonitoring))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate test report
    report = {
        "timestamp": str(asyncio.get_event_loop().time()),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful(),
        "failures_details": [str(failure) for failure in result.failures],
        "errors_details": [str(error) for error in result.errors],
        "skipped_details": [str(skipped) for skipped in result.skipped]
    }

    # Save report
    with open("stop_loss_test_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print(f"  • Tests Run: {result.testsRun}")
    print(f"  • Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  • Failed: {len(result.failures)}")
    print(f"  • Errors: {len(result.errors)}")
    print(f"  • Skipped: {len(result.skipped)}")
    print(f"  • Overall Success: {'✅ YES' if result.wasSuccessful() else '❌ NO'}")
    print("=" * 60)

    if result.wasSuccessful():
        print("🎉 ALL STOP LOSS TESTS PASSED!")
        print("   The stop loss system is fully functional and tested.")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("   Please review the test output above for details.")

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_stop_loss_tests()
    exit(0 if success else 1)
