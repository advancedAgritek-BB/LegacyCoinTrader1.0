"""
Tests for the TradeManager - Single Source of Truth for Trades and Positions

This module contains comprehensive tests for the centralized TradeManager system
to ensure consistent trade recording, position management, and PnL calculations.
"""

import unittest
from decimal import Decimal
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

from crypto_bot.utils.trade_manager import (
    TradeManager, Trade, Position, create_trade, get_trade_manager
)


class TestTradeDataModels(unittest.TestCase):
    """Test the Trade and Position data models."""

    def test_trade_creation(self):
        """Test creating a trade with proper decimal handling."""
        trade = Trade(
            id="test-123",
            symbol="BTC/USD",
            side="buy",
            amount=Decimal("1.5"),
            price=Decimal("50000.00"),
            timestamp=datetime.utcnow(),
            strategy="test_strategy",
            exchange="test_exchange",
            fees=Decimal("0.001")
        )

        self.assertEqual(trade.symbol, "BTC/USD")
        self.assertEqual(trade.side, "buy")
        self.assertEqual(trade.amount, Decimal("1.5"))
        self.assertEqual(trade.price, Decimal("50000.00"))
        self.assertEqual(trade.total_value, Decimal("75000.00"))  # 1.5 * 50000
        self.assertEqual(trade.fees, Decimal("0.001"))

    def test_trade_serialization(self):
        """Test trade serialization and deserialization."""
        original_trade = create_trade(
            symbol="ETH/USD",
            side="sell",
            amount=Decimal("2.0"),
            price=Decimal("3000.00"),
            strategy="momentum",
            exchange="cex"
        )

        # Serialize to dict
        trade_dict = original_trade.to_dict()

        # Deserialize from dict
        restored_trade = Trade.from_dict(trade_dict)

        self.assertEqual(original_trade.symbol, restored_trade.symbol)
        self.assertEqual(original_trade.side, restored_trade.side)
        self.assertEqual(original_trade.amount, restored_trade.amount)
        self.assertEqual(original_trade.price, restored_trade.price)
        self.assertEqual(original_trade.strategy, restored_trade.strategy)

    def test_position_pnl_calculation(self):
        """Test position PnL calculations."""
        position = Position(
            symbol="BTC/USD",
            side="long",
            total_amount=Decimal("1.0"),
            average_price=Decimal("50000.00")
        )

        # Test profitable long position
        current_price = Decimal("55000.00")
        pnl, pnl_pct = position.calculate_unrealized_pnl(current_price)

        expected_pnl = Decimal("5000.00")  # (55000 - 50000) * 1.0
        expected_pct = Decimal("10.0")     # 5000 / 50000 * 100

        self.assertEqual(pnl, expected_pnl)
        self.assertEqual(pnl_pct, expected_pct)

        # Test losing short position
        short_position = Position(
            symbol="ETH/USD",
            side="short",
            total_amount=Decimal("2.0"),
            average_price=Decimal("3000.00")
        )

        current_price = Decimal("3200.00")  # Price went up, short is losing
        pnl, pnl_pct = short_position.calculate_unrealized_pnl(current_price)

        expected_pnl = Decimal("-400.00")  # (3000 - 3200) * 2.0
        expected_pct = Decimal("-6.666666666666667")  # -400 / 6000 * 100

        self.assertEqual(pnl, expected_pnl)
        self.assertAlmostEqual(float(pnl_pct), -6.67, places=2)


class TestTradeManager(unittest.TestCase):
    """Test the TradeManager functionality."""

    def setUp(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_trade_manager.json"
        self.manager = TradeManager(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up temporary files."""
        if self.storage_path.exists():
            os.unlink(self.storage_path)

    def test_record_single_trade(self):
        """Test recording a single trade."""
        trade = create_trade(
            symbol="BTC/USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000.00"),
            strategy="test"
        )

        trade_id = self.manager.record_trade(trade)

        # Verify trade was recorded
        self.assertEqual(len(self.manager.trades), 1)
        self.assertEqual(self.manager.trades[0].id, trade_id)

        # Verify position was created
        position = self.manager.get_position("BTC/USD")
        self.assertIsNotNone(position)
        self.assertEqual(position.symbol, "BTC/USD")
        self.assertEqual(position.side, "long")
        self.assertEqual(position.total_amount, Decimal("1.0"))
        self.assertEqual(position.average_price, Decimal("50000.00"))

    def test_multiple_trades_same_symbol(self):
        """Test multiple trades on the same symbol."""
        # First trade - buy
        trade1 = create_trade(
            symbol="ETH/USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("3000.00")
        )
        self.manager.record_trade(trade1)

        # Second trade - buy more at different price
        trade2 = create_trade(
            symbol="ETH/USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("3100.00")
        )
        self.manager.record_trade(trade2)

        # Verify position aggregation
        position = self.manager.get_position("ETH/USD")
        self.assertEqual(position.total_amount, Decimal("2.0"))

        # Average price should be weighted average: (3000*1 + 3100*1) / 2 = 3050
        expected_avg_price = Decimal("3050.00")
        self.assertEqual(position.average_price, expected_avg_price)

    def test_opposite_trades_position_closure(self):
        """Test that opposite trades close positions correctly."""
        # Buy position
        buy_trade = create_trade(
            symbol="BTC/USD",
            side="buy",
            amount=Decimal("2.0"),
            price=Decimal("50000.00")
        )
        self.manager.record_trade(buy_trade)

        # Sell to close half position
        sell_trade = create_trade(
            symbol="BTC/USD",
            side="sell",
            amount=Decimal("1.0"),
            price=Decimal("55000.00")
        )
        self.manager.record_trade(sell_trade)

        # Verify position was reduced
        position = self.manager.get_position("BTC/USD")
        self.assertEqual(position.total_amount, Decimal("1.0"))

        # Verify realized PnL: (55000 - 50000) * 1.0 = 5000
        self.assertEqual(position.realized_pnl, Decimal("5000.00"))

    def test_price_updates_and_pnl(self):
        """Test price updates and PnL calculations."""
        # Create position
        trade = create_trade(
            symbol="BTC/USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000.00")
        )
        self.manager.record_trade(trade)

        # Update price
        self.manager.update_price("BTC/USD", Decimal("55000.00"))

        # Get position and check PnL
        position = self.manager.get_position("BTC/USD")
        pnl, pnl_pct = position.calculate_unrealized_pnl(Decimal("55000.00"))

        self.assertEqual(pnl, Decimal("5000.00"))  # (55000 - 50000) * 1.0
        self.assertEqual(pnl_pct, Decimal("10.0"))  # 5000 / 50000 * 100

    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        # Create multiple positions
        trades = [
            create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00")),
            create_trade("ETH/USD", "buy", Decimal("2.0"), Decimal("3000.00")),
            create_trade("BTC/USD", "sell", Decimal("0.5"), Decimal("55000.00"))  # Partial close
        ]

        for trade in trades:
            self.manager.record_trade(trade)

        # Update prices
        self.manager.update_price("BTC/USD", Decimal("52000.00"))
        self.manager.update_price("ETH/USD", Decimal("3100.00"))

        # Get portfolio summary
        summary = self.manager.get_portfolio_summary()

        self.assertEqual(summary['total_trades'], 3)
        self.assertEqual(summary['open_positions_count'], 2)  # BTC and ETH
        self.assertEqual(summary['closed_positions_count'], 0)

        # Check that positions are included
        self.assertEqual(len(summary['positions']), 2)

        # Find BTC position
        btc_pos = next(p for p in summary['positions'] if p['symbol'] == 'BTC/USD')
        self.assertEqual(float(btc_pos['total_amount']), 0.5)  # 1.0 - 0.5

    def test_state_persistence(self):
        """Test saving and loading state."""
        # Create some trades
        trade1 = create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00"))
        trade2 = create_trade("ETH/USD", "buy", Decimal("2.0"), Decimal("3000.00"))

        self.manager.record_trade(trade1)
        self.manager.record_trade(trade2)

        # Save state
        self.manager.save_state()

        # Create new manager and load state
        new_manager = TradeManager(storage_path=self.storage_path)

        # Verify state was loaded
        self.assertEqual(len(new_manager.trades), 2)
        self.assertEqual(len(new_manager.positions), 2)

        # Verify trade data
        loaded_trade = new_manager.trades[0]
        self.assertEqual(loaded_trade.symbol, "BTC/USD")
        self.assertEqual(loaded_trade.amount, Decimal("1.0"))

    def test_exit_conditions(self):
        """Test position exit condition checking."""
        # Create position with stop loss
        position = Position(
            symbol="BTC/USD",
            side="long",
            total_amount=Decimal("1.0"),
            average_price=Decimal("50000.00"),
            stop_loss_price=Decimal("48000.00")
        )

        # Test stop loss trigger
        should_exit, reason = position.should_exit(Decimal("47000.00"))
        self.assertTrue(should_exit)
        self.assertEqual(reason, "stop_loss")

        # Test take profit trigger
        position.take_profit_price = Decimal("55000.00")
        should_exit, reason = position.should_exit(Decimal("56000.00"))
        self.assertTrue(should_exit)
        self.assertEqual(reason, "take_profit")

        # Test no exit condition
        should_exit, reason = position.should_exit(Decimal("51000.00"))
        self.assertFalse(should_exit)
        self.assertEqual(reason, "")

    def test_trailing_stop(self):
        """Test trailing stop functionality."""
        position = Position(
            symbol="BTC/USD",
            side="long",
            total_amount=Decimal("1.0"),
            average_price=Decimal("50000.00"),
            trailing_stop_pct=Decimal("5.0")  # 5% trailing stop
        )

        # Price moves up
        position.update_price_levels(Decimal("52500.00"))  # +5%
        position.update_trailing_stop(Decimal("52500.00"))

        # Trailing stop should be set to 52500 * (1 - 0.05) = 49875
        expected_stop = Decimal("49875.00")
        self.assertEqual(position.stop_loss_price, expected_stop)


class TestTradeManagerIntegration(unittest.TestCase):
    """Integration tests for TradeManager with other components."""

    def setUp(self):
        """Set up test environment with temporary storage."""
        import tempfile
        import os
        from pathlib import Path

        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_trade_manager_integration.json"
        self.manager = TradeManager(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'storage_path') and self.storage_path.exists():
            os.unlink(self.storage_path)
        if hasattr(self, 'temp_dir'):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_global_instance(self):
        """Test that get_trade_manager returns the same instance."""
        # Clear any existing global instance for clean test
        import crypto_bot.utils.trade_manager as tm_module
        tm_module._trade_manager_instance = None

        manager1 = get_trade_manager()
        manager2 = get_trade_manager()

        self.assertIs(manager1, manager2)

    def test_trade_history_filtering(self):
        """Test filtering trade history by symbol."""
        manager = self.manager

        # Create trades for different symbols
        trades = [
            create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00")),
            create_trade("ETH/USD", "buy", Decimal("2.0"), Decimal("3000.00")),
            create_trade("BTC/USD", "sell", Decimal("0.5"), Decimal("55000.00")),
        ]

        for trade in trades:
            manager.record_trade(trade)

        # Get all trades
        all_trades = manager.get_trade_history()
        self.assertEqual(len(all_trades), 3)

        # Get BTC trades only
        btc_trades = manager.get_trade_history(symbol="BTC/USD")
        self.assertEqual(len(btc_trades), 2)
        for trade in btc_trades:
            self.assertEqual(trade.symbol, "BTC/USD")

    def test_price_cache_updates(self):
        """Test that price cache is updated correctly."""
        manager = self.manager

        # Initially empty
        self.assertEqual(len(manager.price_cache), 0)

        # Update price
        manager.update_price("BTC/USD", Decimal("50000.00"))

        # Verify cache
        self.assertEqual(len(manager.price_cache), 1)
        self.assertEqual(manager.price_cache["BTC/USD"], Decimal("50000.00"))


class TestTradeManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions in TradeManager."""

    def setUp(self):
        """Set up test environment with temporary storage."""
        import tempfile
        import os
        from pathlib import Path

        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_trade_manager_edge_cases.json"
        self.manager = TradeManager(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'storage_path') and self.storage_path.exists():
            os.unlink(self.storage_path)
        if hasattr(self, 'temp_dir'):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_zero_amount_trade(self):
        """Test handling of zero amount trades."""
        trade = create_trade("BTC/USD", "buy", Decimal("0"), Decimal("50000.00"))
        trade_id = self.manager.record_trade(trade)

        # Position should not be created for zero amount
        position = self.manager.get_position("BTC/USD")
        self.assertIsNone(position)

        # Trade should still be recorded
        self.assertEqual(len(self.manager.trades), 1)
        self.assertEqual(self.manager.trades[0].id, trade_id)

    def test_negative_price_trade(self):
        """Test handling of trades with negative prices."""
        with self.assertRaises(Exception):  # Should fail validation
            create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("-50000.00"))

    def test_extremely_large_amounts(self):
        """Test handling of extremely large trade amounts."""
        large_amount = Decimal("999999999999999999999999999999")
        trade = create_trade("BTC/USD", "buy", large_amount, Decimal("50000.00"))

        # Should handle large numbers by capping at reasonable maximum
        # The system caps extremely large numbers to prevent decimal precision issues
        self.assertIsInstance(trade.amount, Decimal)
        self.assertGreater(trade.amount, Decimal("1e20"))  # Should be a very large number
        self.assertEqual(trade.total_value, trade.amount * Decimal("50000.00"))

    def test_partial_position_closure_with_exact_amounts(self):
        """Test partial closure when trade amount exactly matches position amount."""
        # Create position
        buy_trade = create_trade("BTC/USD", "buy", Decimal("2.0"), Decimal("50000.00"))
        self.manager.record_trade(buy_trade)

        # Close exactly half
        sell_trade = create_trade("BTC/USD", "sell", Decimal("1.0"), Decimal("55000.00"))
        self.manager.record_trade(sell_trade)

        position = self.manager.get_position("BTC/USD")
        self.assertEqual(position.total_amount, Decimal("1.0"))
        # Average price should remain unchanged for remaining shares
        self.assertEqual(position.average_price, Decimal("50000.00"))
        self.assertEqual(position.realized_pnl, Decimal("5000.00"))  # (55000 - 50000) * 1.0

    def test_multiple_opposite_trades_complex_scenario(self):
        """Test complex scenario with multiple buys and sells."""
        # Buy 1.0 @ 50000
        trade1 = create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00"))
        self.manager.record_trade(trade1)

        # Buy 2.0 @ 51000 (weighted avg should be 50666.67)
        trade2 = create_trade("BTC/USD", "buy", Decimal("2.0"), Decimal("51000.00"))
        self.manager.record_trade(trade2)

        # Sell 1.5 @ 52000 (should realize profit on 1.5 units)
        trade3 = create_trade("BTC/USD", "sell", Decimal("1.5"), Decimal("52000.00"))
        self.manager.record_trade(trade3)

        position = self.manager.get_position("BTC/USD")
        self.assertEqual(position.total_amount, Decimal("1.5"))  # 3.0 - 1.5

        # Average price should remain the same for remaining shares
        expected_avg = Decimal("50666.66666666666666666666667")
        self.assertAlmostEqual(float(position.average_price), float(expected_avg), places=10)

        # Realized PnL should be correct: (52000 - 50666.67) * 1.5 = 2000.005
        expected_pnl = Decimal("2000.00")
        self.assertAlmostEqual(float(position.realized_pnl), float(expected_pnl), places=1)

    def test_empty_symbol_handling(self):
        """Test handling of trades with empty symbols."""
        trade = create_trade("", "buy", Decimal("1.0"), Decimal("50000.00"))
        trade_id = self.manager.record_trade(trade)

        # Should create position with empty symbol
        position = self.manager.get_position("")
        self.assertIsNotNone(position)
        self.assertEqual(position.symbol, "")

    def test_invalid_side_handling(self):
        """Test handling of trades with invalid sides."""
        # This should work but create unexpected position side
        trade = create_trade("BTC/USD", "invalid_side", Decimal("1.0"), Decimal("50000.00"))
        self.manager.record_trade(trade)

        position = self.manager.get_position("BTC/USD")
        self.assertEqual(position.side, "short")  # Default for non-'buy' side

    def test_concurrent_price_updates(self):
        """Test concurrent price updates for thread safety."""
        import threading
        import time

        results = []
        errors = []

        def update_price_worker(symbol, prices, delay=0):
            try:
                time.sleep(delay)
                for price in prices:
                    self.manager.update_price(symbol, price)
                    results.append(f"{symbol}:{price}")
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads updating different symbols
        thread1 = threading.Thread(target=update_price_worker,
                                  args=("BTC/USD", [Decimal("50000"), Decimal("51000")]))
        thread2 = threading.Thread(target=update_price_worker,
                                  args=("ETH/USD", [Decimal("3000"), Decimal("3100")], 0.01))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Verify no errors occurred
        self.assertEqual(len(errors), 0)

        # Verify price cache was updated
        self.assertIn("BTC/USD", self.manager.price_cache)
        self.assertIn("ETH/USD", self.manager.price_cache)

    def test_callback_error_isolation(self):
        """Test that callback errors don't break the system."""
        def failing_callback(*args):
            raise Exception("Callback failed")

        def working_callback(*args):
            working_callback.called = True

        working_callback.called = False

        # Add both callbacks
        self.manager.add_trade_callback(failing_callback)
        self.manager.add_trade_callback(working_callback)

        # Record a trade - failing callback should be removed, working one should execute
        trade = create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00"))
        self.manager.record_trade(trade)

        # Verify working callback was called
        self.assertTrue(working_callback.called)

        # Verify failing callback was removed
        self.assertNotIn(failing_callback, self.manager.trade_callbacks)
        self.assertIn(working_callback, self.manager.trade_callbacks)

    def test_state_persistence_with_corrupted_data(self):
        """Test state loading with corrupted data."""
        # Create a corrupted state file
        corrupted_state = {
            "trades": [{"invalid": "data"}],
            "positions": "not_a_dict",
            "statistics": {"total_trades": "not_a_number"}
        }

        import json
        with open(self.storage_path, 'w') as f:
            json.dump(corrupted_state, f)

        # Create new manager - should handle corrupted data gracefully
        new_manager = TradeManager(storage_path=self.storage_path)

        # Should start with clean state
        self.assertEqual(len(new_manager.trades), 0)
        self.assertEqual(len(new_manager.positions), 0)
        self.assertEqual(new_manager.total_trades, 0)

    def test_shutdown_with_active_auto_save(self):
        """Test shutdown behavior with active auto-save thread."""
        # Set short auto-save interval for testing
        self.manager.auto_save_interval = 0.1

        # Shutdown should complete successfully
        self.manager.shutdown()

        # Verify auto-save is disabled
        self.assertFalse(self.manager.auto_save_enabled)

    def test_memory_usage_with_many_trades(self):
        """Test memory usage with many trades."""
        # Create many trades
        num_trades = 1000
        for i in range(num_trades):
            trade = create_trade(
                f"BTC/USD_{i}",
                "buy" if i % 2 == 0 else "sell",
                Decimal("1.0"),
                Decimal("50000.00")
            )
            self.manager.record_trade(trade)

        # Verify all trades are stored
        self.assertEqual(len(self.manager.trades), num_trades)
        self.assertEqual(self.manager.total_trades, num_trades)


class TestTradeManagerConcurrency(unittest.TestCase):
    """Test concurrent operations in TradeManager."""

    def setUp(self):
        """Set up test environment."""
        import tempfile
        from pathlib import Path

        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_concurrency.json"
        self.manager = TradeManager(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_trade_recording(self):
        """Test recording trades from multiple threads."""
        import threading
        import time

        results = []
        errors = []

        def record_trades_worker(trades_data, thread_id):
            try:
                for i, (symbol, side, amount, price) in enumerate(trades_data):
                    trade = create_trade(symbol, side, amount, price)
                    trade_id = self.manager.record_trade(trade)
                    results.append(f"thread_{thread_id}_trade_{i}")
                time.sleep(0.01)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(f"thread_{thread_id}: {str(e)}")

        # Create trade data for multiple threads
        trade_data_1 = [
            ("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00")),
            ("ETH/USD", "buy", Decimal("2.0"), Decimal("3000.00")),
            ("BTC/USD", "sell", Decimal("0.5"), Decimal("51000.00"))
        ]

        trade_data_2 = [
            ("LTC/USD", "buy", Decimal("10.0"), Decimal("100.00")),
            ("BTC/USD", "buy", Decimal("0.5"), Decimal("50500.00")),
            ("ETH/USD", "sell", Decimal("1.0"), Decimal("3100.00"))
        ]

        # Start concurrent threads
        thread1 = threading.Thread(target=record_trades_worker, args=(trade_data_1, 1))
        thread2 = threading.Thread(target=record_trades_worker, args=(trade_data_2, 2))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Verify no errors occurred
        self.assertEqual(len(errors), 0)

        # Verify all trades were recorded
        self.assertEqual(len(self.manager.trades), 6)
        self.assertEqual(self.manager.total_trades, 6)

        # Verify positions are consistent
        btc_position = self.manager.get_position("BTC/USD")
        eth_position = self.manager.get_position("ETH/USD")
        ltc_position = self.manager.get_position("LTC/USD")

        self.assertIsNotNone(btc_position)
        self.assertIsNotNone(eth_position)
        self.assertIsNotNone(ltc_position)


class TestTradeManagerSerialization(unittest.TestCase):
    """Test serialization and deserialization functionality."""

    def setUp(self):
        """Set up test environment."""
        import tempfile
        from pathlib import Path

        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_serialization.json"
        self.manager = TradeManager(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trade_serialization_round_trip(self):
        """Test that trade serialization preserves all data."""
        original_trade = create_trade(
            symbol="BTC/USD",
            side="buy",
            amount=Decimal("1.5"),
            price=Decimal("50000.12345678"),
            strategy="test_strategy",
            exchange="test_exchange",
            fees=Decimal("0.001"),
            order_id="order123",
            client_order_id="client456"
        )
        original_trade.metadata = {"test_key": "test_value", "number": 42}

        # Serialize and deserialize
        trade_dict = original_trade.to_dict()
        restored_trade = Trade.from_dict(trade_dict)

        # Verify all fields match
        self.assertEqual(original_trade.id, restored_trade.id)
        self.assertEqual(original_trade.symbol, restored_trade.symbol)
        self.assertEqual(original_trade.side, restored_trade.side)
        self.assertEqual(original_trade.amount, restored_trade.amount)
        self.assertEqual(original_trade.price, restored_trade.price)
        self.assertEqual(original_trade.strategy, restored_trade.strategy)
        self.assertEqual(original_trade.exchange, restored_trade.exchange)
        self.assertEqual(original_trade.fees, restored_trade.fees)
        self.assertEqual(original_trade.status, restored_trade.status)
        self.assertEqual(original_trade.order_id, restored_trade.order_id)
        self.assertEqual(original_trade.client_order_id, restored_trade.client_order_id)
        self.assertEqual(original_trade.metadata, restored_trade.metadata)

    def test_position_serialization_round_trip(self):
        """Test that position serialization preserves all data."""
        # Create a position with some trades
        position = Position(
            symbol="ETH/USD",
            side="long",
            total_amount=Decimal("2.5"),
            average_price=Decimal("3000.50"),
            realized_pnl=Decimal("150.25"),
            fees_paid=Decimal("0.005"),
            stop_loss_price=Decimal("2900.00"),
            take_profit_price=Decimal("3200.00"),
            trailing_stop_pct=Decimal("2.5")
        )

        # Add some trades
        trade1 = create_trade("ETH/USD", "buy", Decimal("1.0"), Decimal("2950.00"))
        trade2 = create_trade("ETH/USD", "buy", Decimal("1.5"), Decimal("3050.00"))
        position.trades = [trade1, trade2]
        position.metadata = {"strategy": "momentum", "risk_level": "medium"}

        # Serialize and deserialize
        position_dict = position.to_dict()
        restored_position = Position.from_dict(position_dict)

        # Verify all fields match
        self.assertEqual(position.symbol, restored_position.symbol)
        self.assertEqual(position.side, restored_position.side)
        self.assertEqual(position.total_amount, restored_position.total_amount)
        self.assertEqual(position.average_price, restored_position.average_price)
        self.assertEqual(position.realized_pnl, restored_position.realized_pnl)
        self.assertEqual(position.fees_paid, restored_position.fees_paid)
        self.assertEqual(position.stop_loss_price, restored_position.stop_loss_price)
        self.assertEqual(position.take_profit_price, restored_position.take_profit_price)
        self.assertEqual(position.trailing_stop_pct, restored_position.trailing_stop_pct)
        self.assertEqual(position.metadata, restored_position.metadata)

        # Verify trades are preserved
        self.assertEqual(len(position.trades), len(restored_position.trades))
        for orig_trade, rest_trade in zip(position.trades, restored_position.trades):
            self.assertEqual(orig_trade.symbol, rest_trade.symbol)
            self.assertEqual(orig_trade.amount, rest_trade.amount)

    def test_state_persistence_with_special_characters(self):
        """Test state persistence with special characters in data."""
        # Create trades with special characters
        special_trade = create_trade(
            symbol="BTC/USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000.00")
        )
        special_trade.metadata = {
            "special_chars": "!@#$%^&*()[]{}|;:,.<>?",
            "unicode": "测试数据 🚀",
            "nested": {"key": "value", "list": [1, 2, 3]}
        }

        self.manager.record_trade(special_trade)
        self.manager.save_state()

        # Load in new manager
        new_manager = TradeManager(storage_path=self.storage_path)

        # Verify data was preserved
        self.assertEqual(len(new_manager.trades), 1)
        loaded_trade = new_manager.trades[0]
        self.assertEqual(loaded_trade.metadata["special_chars"], "!@#$%^&*()[]{}|;:,.<>?")
        self.assertEqual(loaded_trade.metadata["unicode"], "测试数据 🚀")
        self.assertEqual(loaded_trade.metadata["nested"], {"key": "value", "list": [1, 2, 3]})

    def test_empty_state_serialization(self):
        """Test serialization of completely empty state."""
        # Save empty state
        self.manager.save_state()

        # Load in new manager
        new_manager = TradeManager(storage_path=self.storage_path)

        # Verify empty state
        self.assertEqual(len(new_manager.trades), 0)
        self.assertEqual(len(new_manager.positions), 0)
        self.assertEqual(len(new_manager.price_cache), 0)
        self.assertEqual(new_manager.total_trades, 0)


class TestTradeManagerPositionManagement(unittest.TestCase):
    """Test advanced position management features."""

    def setUp(self):
        """Set up test environment."""
        import tempfile
        from pathlib import Path

        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_position_management.json"
        self.manager = TradeManager(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_position_scaling_in(self):
        """Test scaling into a position with multiple entries."""
        # Initial position
        trade1 = create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00"))
        self.manager.record_trade(trade1)

        # Scale in - add to position
        trade2 = create_trade("BTC/USD", "buy", Decimal("2.0"), Decimal("51000.00"))
        self.manager.record_trade(trade2)

        # Scale in again
        trade3 = create_trade("BTC/USD", "buy", Decimal("1.5"), Decimal("52000.00"))
        self.manager.record_trade(trade3)

        position = self.manager.get_position("BTC/USD")
        self.assertEqual(position.total_amount, Decimal("4.5"))

        # Calculate expected weighted average: (1*50000 + 2*51000 + 1.5*52000) / 4.5
        expected_avg = (Decimal("50000") + Decimal("102000") + Decimal("78000")) / Decimal("4.5")
        self.assertAlmostEqual(float(position.average_price), float(expected_avg), places=8)

    def test_position_scaling_out(self):
        """Test scaling out of a position with multiple exits."""
        # Large initial position
        trade1 = create_trade("BTC/USD", "buy", Decimal("10.0"), Decimal("50000.00"))
        self.manager.record_trade(trade1)

        # Scale out - partial exit
        trade2 = create_trade("BTC/USD", "sell", Decimal("3.0"), Decimal("55000.00"))
        self.manager.record_trade(trade2)

        # Scale out again
        trade3 = create_trade("BTC/USD", "sell", Decimal("2.0"), Decimal("56000.00"))
        self.manager.record_trade(trade3)

        position = self.manager.get_position("BTC/USD")
        self.assertEqual(position.total_amount, Decimal("5.0"))
        self.assertEqual(position.average_price, Decimal("50000.00"))  # Should remain unchanged

        # Realized PnL should be correct
        pnl1 = (Decimal("55000") - Decimal("50000")) * Decimal("3.0")  # 150000
        pnl2 = (Decimal("56000") - Decimal("50000")) * Decimal("2.0")  # 120000
        expected_total_pnl = pnl1 + pnl2  # 270000

        self.assertEqual(position.realized_pnl, expected_total_pnl)

    def test_short_position_management(self):
        """Test short position creation and management."""
        # Open short position
        short_trade = create_trade("ETH/USD", "sell", Decimal("5.0"), Decimal("3000.00"))
        self.manager.record_trade(short_trade)

        position = self.manager.get_position("ETH/USD")
        self.assertEqual(position.side, "short")
        self.assertEqual(position.total_amount, Decimal("5.0"))
        self.assertEqual(position.average_price, Decimal("3000.00"))

        # Test PnL calculation for short position
        # Price goes down - short position profits
        pnl_down = position.calculate_unrealized_pnl(Decimal("2800.00"))[0]
        expected_pnl_down = (Decimal("3000") - Decimal("2800")) * Decimal("5.0")  # 1000
        self.assertEqual(pnl_down, expected_pnl_down)

        # Price goes up - short position loses
        pnl_up = position.calculate_unrealized_pnl(Decimal("3200.00"))[0]
        expected_pnl_up = (Decimal("3000") - Decimal("3200")) * Decimal("5.0")  # -1000
        self.assertEqual(pnl_up, expected_pnl_up)

    def test_position_reversal(self):
        """Test position reversal (long to short or vice versa)."""
        # Start with long position
        buy_trade = create_trade("BTC/USD", "buy", Decimal("2.0"), Decimal("50000.00"))
        self.manager.record_trade(buy_trade)

        # Reverse position - sell more than current position
        sell_trade = create_trade("BTC/USD", "sell", Decimal("3.0"), Decimal("48000.00"))
        self.manager.record_trade(sell_trade)

        position = self.manager.get_position("BTC/USD")
        self.assertEqual(position.side, "short")  # Should be short now
        self.assertEqual(position.total_amount, Decimal("1.0"))  # 3.0 - 2.0 = 1.0 short

        # Average price should be the sell price for the remaining position
        self.assertEqual(position.average_price, Decimal("48000.00"))

        # Realized PnL from the 2.0 units closed
        expected_pnl = (Decimal("48000") - Decimal("50000")) * Decimal("2.0")  # -4000
        self.assertEqual(position.realized_pnl, expected_pnl)

    def test_multiple_symbol_portfolio(self):
        """Test managing multiple symbols simultaneously."""
        # Create positions in multiple symbols
        symbols_trades = {
            "BTC/USD": [("buy", Decimal("2.0"), Decimal("50000.00")), ("sell", Decimal("1.0"), Decimal("55000.00"))],
            "ETH/USD": [("buy", Decimal("10.0"), Decimal("3000.00")), ("buy", Decimal("5.0"), Decimal("3100.00"))],
            "LTC/USD": [("sell", Decimal("50.0"), Decimal("100.00")), ("buy", Decimal("20.0"), Decimal("90.00"))]
        }

        for symbol, trades in symbols_trades.items():
            for side, amount, price in trades:
                trade = create_trade(symbol, side, amount, price)
                self.manager.record_trade(trade)

        # Verify all positions exist
        portfolio = self.manager.get_portfolio_summary()
        self.assertEqual(len(portfolio['positions']), 3)

        # Check individual positions
        btc_pos = next(p for p in portfolio['positions'] if p['symbol'] == 'BTC/USD')
        eth_pos = next(p for p in portfolio['positions'] if p['symbol'] == 'ETH/USD')
        ltc_pos = next(p for p in portfolio['positions'] if p['symbol'] == 'LTC/USD')

        self.assertEqual(float(btc_pos['total_amount']), 1.0)  # 2.0 - 1.0
        self.assertEqual(float(eth_pos['total_amount']), 15.0)  # 10.0 + 5.0
        self.assertEqual(float(ltc_pos['total_amount']), 30.0)  # 50.0 - 20.0 (short)

    def test_risk_management_levels(self):
        """Test stop loss and take profit level management."""
        # Create position with risk management levels
        position = Position(
            symbol="BTC/USD",
            side="long",
            total_amount=Decimal("1.0"),
            average_price=Decimal("50000.00"),
            stop_loss_price=Decimal("48000.00"),  # 4% stop loss
            take_profit_price=Decimal("55000.00")  # 10% take profit
        )

        # Test stop loss trigger
        self.assertTrue(position.should_exit(Decimal("47000.00"))[0])  # Below stop loss
        self.assertEqual(position.should_exit(Decimal("47000.00"))[1], "stop_loss")

        # Test take profit trigger
        self.assertTrue(position.should_exit(Decimal("56000.00"))[0])  # Above take profit
        self.assertEqual(position.should_exit(Decimal("56000.00"))[1], "take_profit")

        # Test no trigger
        self.assertFalse(position.should_exit(Decimal("51000.00"))[0])  # Within range

    def test_trailing_stop_advanced(self):
        """Test advanced trailing stop scenarios."""
        position = Position(
            symbol="BTC/USD",
            side="long",
            total_amount=Decimal("1.0"),
            average_price=Decimal("50000.00"),
            trailing_stop_pct=Decimal("5.0")
        )

        # Price moves up and trailing stop adjusts
        position.update_price_levels(Decimal("51000.00"))
        position.update_trailing_stop(Decimal("51000.00"))
        # Stop should be at 51000 * (1 - 0.05) = 48450
        self.assertEqual(position.stop_loss_price, Decimal("48450.00"))

        # Price continues up, stop should trail
        position.update_price_levels(Decimal("53000.00"))
        position.update_trailing_stop(Decimal("53000.00"))
        # Stop should be at 53000 * (1 - 0.05) = 50350
        self.assertEqual(position.stop_loss_price, Decimal("50350.00"))

        # Price drops but not enough to trigger stop
        self.assertFalse(position.should_exit(Decimal("50500.00"))[0])

        # Price drops to stop level
        self.assertTrue(position.should_exit(Decimal("50000.00"))[0])


class TestTradeManagerSystemIntegration(unittest.TestCase):
    """Integration tests with other system components."""

    def setUp(self):
        """Set up test environment."""
        import tempfile
        from pathlib import Path

        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_system_integration.json"
        self.manager = TradeManager(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trade_manager_with_price_feed_simulation(self):
        """Test trade manager with simulated price feed updates."""
        # Create initial position
        trade = create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00"))
        self.manager.record_trade(trade)

        # Simulate price feed updates
        price_updates = [
            Decimal("51000.00"),  # +2%
            Decimal("52000.00"),  # +4%
            Decimal("51500.00"),  # +3%
            Decimal("52500.00"),  # +5%
            Decimal("52200.00"),  # +4.4%
        ]

        for price in price_updates:
            self.manager.update_price("BTC/USD", price)

        # Verify position tracking
        position = self.manager.get_position("BTC/USD")
        self.assertEqual(position.highest_price, Decimal("52500.00"))
        # Lowest price should be the initial trade price (50000.00), not affected by price updates
        self.assertEqual(position.lowest_price, Decimal("50000.00"))

        # Test unrealized PnL calculation
        pnl, pnl_pct = position.calculate_unrealized_pnl(Decimal("52200.00"))
        expected_pnl = (Decimal("52200.00") - Decimal("50000.00")) * Decimal("1.0")
        self.assertEqual(pnl, expected_pnl)

    def test_trade_manager_with_risk_management_workflow(self):
        """Test complete risk management workflow."""
        # Set up position with risk parameters
        position = Position(
            symbol="ETH/USD",
            side="long",
            total_amount=Decimal("10.0"),
            average_price=Decimal("3000.00"),
            stop_loss_price=Decimal("2850.00"),  # 5% stop loss
            take_profit_price=Decimal("3300.00"),  # 10% take profit
            trailing_stop_pct=Decimal("3.0")  # 3% trailing stop
        )

        # Simulate price movement triggering risk management
        price_scenarios = [
            (Decimal("3150.00"), False, ""),  # Price up 5%, no exit signal
            (Decimal("3300.00"), True, "take_profit"),  # Hit take profit
            (Decimal("3250.00"), False, ""),  # Price drops but trailing stop not yet triggered
        ]

        for price, should_exit, reason in price_scenarios:
            position.update_price_levels(price)
            position.update_trailing_stop(price)
            exit_signal, exit_reason = position.should_exit(price)

            self.assertEqual(exit_signal, should_exit, f"Price {price}: expected {should_exit}, got {exit_signal}")
            if should_exit:
                self.assertEqual(exit_reason, reason, f"Price {price}: expected {reason}, got {exit_reason}")

    def test_trade_manager_portfolio_rebalancing_simulation(self):
        """Test portfolio rebalancing simulation."""
        # Create initial portfolio
        initial_trades = [
            ("BTC/USD", "buy", Decimal("2.0"), Decimal("50000.00")),
            ("ETH/USD", "buy", Decimal("10.0"), Decimal("3000.00")),
            ("LTC/USD", "buy", Decimal("50.0"), Decimal("100.00")),
        ]

        for symbol, side, amount, price in initial_trades:
            trade = create_trade(symbol, side, amount, price)
            self.manager.record_trade(trade)

        # Simulate market movement
        price_updates = {
            "BTC/USD": Decimal("52000.00"),  # +4%
            "ETH/USD": Decimal("2850.00"),   # -5%
            "LTC/USD": Decimal("105.00"),   # +5%
        }

        for symbol, price in price_updates.items():
            self.manager.update_price(symbol, price)

        # Check portfolio impact
        summary = self.manager.get_portfolio_summary()

        # Verify positions are updated
        self.assertEqual(len(summary['positions']), 3)

        # Calculate expected total PnL
        btc_pnl = (Decimal("52000.00") - Decimal("50000.00")) * Decimal("2.0")
        eth_pnl = (Decimal("2850.00") - Decimal("3000.00")) * Decimal("10.0")
        ltc_pnl = (Decimal("105.00") - Decimal("100.00")) * Decimal("50.0")
        expected_total_pnl = btc_pnl + eth_pnl + ltc_pnl

        self.assertAlmostEqual(float(summary['total_unrealized_pnl']), float(expected_total_pnl), places=8)

    def test_trade_manager_with_callback_system(self):
        """Test trade manager callback system integration."""
        callback_events = []

        def trade_callback(trade):
            callback_events.append(f"trade_{trade.symbol}_{trade.side}_{trade.amount}")

        def position_callback(position):
            callback_events.append(f"position_{position.symbol}_{position.total_amount}")

        def price_callback(symbol, price):
            callback_events.append(f"price_{symbol}_{price}")

        # Register callbacks
        self.manager.add_trade_callback(trade_callback)
        self.manager.add_position_callback(position_callback)
        self.manager.add_price_callback(price_callback)

        # Perform operations that should trigger callbacks
        trade = create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00"))
        self.manager.record_trade(trade)

        self.manager.update_price("BTC/USD", Decimal("51000.00"))

        # Verify callbacks were triggered (accounting for decimal precision)
        self.assertTrue(any("trade_BTC/USD_buy_1." in event for event in callback_events))
        self.assertTrue(any("position_BTC/USD_1." in event for event in callback_events))
        self.assertIn("price_BTC/USD_51000.00", callback_events)

    def test_trade_manager_shutdown_and_recovery(self):
        """Test shutdown and state recovery functionality."""
        # Create some state
        trades = [
            create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00")),
            create_trade("ETH/USD", "buy", Decimal("2.0"), Decimal("3000.00")),
        ]

        for trade in trades:
            self.manager.record_trade(trade)

        # Shutdown and save state
        self.manager.shutdown()

        # Create new manager and verify state recovery
        recovered_manager = TradeManager(storage_path=self.storage_path)

        # Verify recovered state
        self.assertEqual(len(recovered_manager.trades), 2)
        self.assertEqual(len(recovered_manager.positions), 2)
        self.assertEqual(recovered_manager.total_trades, 2)

        # Verify specific data recovery
        btc_position = recovered_manager.get_position("BTC/USD")
        eth_position = recovered_manager.get_position("ETH/USD")

        self.assertIsNotNone(btc_position)
        self.assertIsNotNone(eth_position)
        self.assertEqual(btc_position.total_amount, Decimal("1.0"))
        self.assertEqual(eth_position.total_amount, Decimal("2.0"))

    def test_trade_manager_memory_efficiency(self):
        """Test memory efficiency with large trade volumes."""
        # Create many trades to test memory handling
        num_trades = 1000
        trades_created = []

        for i in range(num_trades):
            trade = create_trade(
                f"TEST{i:04d}/USD",
                "buy" if i % 2 == 0 else "sell",
                Decimal("1.0"),
                Decimal("100.00")
            )
            self.manager.record_trade(trade)
            trades_created.append(trade.id)

        # Verify all trades were recorded
        self.assertEqual(len(self.manager.trades), num_trades)
        self.assertEqual(self.manager.total_trades, num_trades)

        # Verify trade history retrieval
        all_trades = self.manager.get_trade_history()
        self.assertEqual(len(all_trades), num_trades)

        # Test filtering by symbol
        specific_symbol = "TEST0000/USD"
        symbol_trades = self.manager.get_trade_history(symbol=specific_symbol)
        self.assertEqual(len(symbol_trades), 1)
        self.assertEqual(symbol_trades[0].symbol, specific_symbol)

    def test_trade_manager_error_recovery(self):
        """Test error recovery and resilience."""
        # Test with invalid data handling
        try:
            # This should not crash the system
            invalid_trade = create_trade("", "", Decimal("-1"), Decimal("-100"))
            # The trade creation itself should handle validation
        except ValueError:
            pass  # Expected validation error

        # System should continue to work normally
        valid_trade = create_trade("BTC/USD", "buy", Decimal("1.0"), Decimal("50000.00"))
        trade_id = self.manager.record_trade(valid_trade)

        # Verify normal operation continues
        self.assertIsNotNone(trade_id)
        position = self.manager.get_position("BTC/USD")
        self.assertIsNotNone(position)
        self.assertEqual(position.total_amount, Decimal("1.0"))


if __name__ == '__main__':
    unittest.main()
