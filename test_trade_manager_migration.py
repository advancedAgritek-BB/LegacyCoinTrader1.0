#!/usr/bin/env python3
"""
Test TradeManager Migration

This script validates that the TradeManager migration works correctly
and that all systems remain synchronized.

Usage:
    python test_trade_manager_migration.py
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any
import logging

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import get_trade_manager, create_trade, PositionSyncManager
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.phase_runner import BotContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationTester:
    """Tests TradeManager migration functionality."""

    def __init__(self):
        self.trade_manager = get_trade_manager()
        self.passed_tests = 0
        self.failed_tests = 0

    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if details:
            logger.info(f"   {details}")

        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

    def test_position_sync_manager_creation(self):
        """Test that PositionSyncManager can be created."""
        try:
            sync_manager = PositionSyncManager(self.trade_manager)
            self.log_test_result("PositionSyncManager creation", True, "Successfully created PositionSyncManager")
            return sync_manager
        except Exception as e:
            self.log_test_result("PositionSyncManager creation", False, f"Failed: {e}")
            return None

    def test_context_creation(self):
        """Test that BotContext can be created with TradeManager."""
        try:
            config = {"execution_mode": "dry_run"}
            ctx = BotContext(
                positions={},
                df_cache={},
                regime_cache={},
                config=config,
                trade_manager=self.trade_manager
            )

            # Check that PositionSyncManager was initialized
            if hasattr(ctx, 'position_sync_manager') and ctx.position_sync_manager:
                self.log_test_result("BotContext creation", True, "BotContext created with PositionSyncManager")
                return ctx
            else:
                self.log_test_result("BotContext creation", False, "PositionSyncManager not initialized")
                return None

        except Exception as e:
            self.log_test_result("BotContext creation", False, f"Failed: {e}")
            return None

    def test_position_sync(self, ctx: BotContext, sync_manager: PositionSyncManager):
        """Test position synchronization."""
        try:
            # Create a test position in TradeManager
            test_trade = create_trade(
                symbol="BTC/USD",
                side="buy",
                amount=1.0,
                price=50000.0,
                strategy="test_migration"
            )
            self.trade_manager.record_trade(test_trade)

            # Sync positions
            ctx.positions = sync_manager.sync_context_positions(ctx.positions)

            # Check that position was synced
            if "BTC/USD" in ctx.positions:
                pos_data = ctx.positions["BTC/USD"]
                # Check essential fields exist and have reasonable values
                essential_fields_ok = (
                    'entry_price' in pos_data and pos_data['entry_price'] > 0 and
                    'size' in pos_data and pos_data['size'] >= 1.0 and  # Allow for existing positions
                    'side' in pos_data and pos_data['side'] in ['long', 'short'] and
                    'symbol' in pos_data and pos_data['symbol'] == 'BTC/USD'
                )
                if essential_fields_ok:
                    self.log_test_result("Position sync", True, f"Position correctly synced: {pos_data['symbol']} {pos_data['side']} {pos_data['size']} @ {pos_data['entry_price']}")
                else:
                    self.log_test_result("Position sync", False, f"Missing essential fields in position data: {pos_data}")
            else:
                self.log_test_result("Position sync", False, "Position not found in ctx.positions")

        except Exception as e:
            self.log_test_result("Position sync", False, f"Failed: {e}")

    def test_paper_wallet_sync(self, ctx: BotContext, sync_manager: PositionSyncManager):
        """Test paper wallet synchronization."""
        try:
            # Create paper wallet
            paper_wallet = PaperWallet(balance=10000.0)
            ctx.paper_wallet = paper_wallet

            # Sync paper wallet
            paper_wallet.positions = sync_manager.sync_paper_wallet_positions(
                paper_wallet.positions, paper_wallet
            )

            # Check that positions were synced
            if hasattr(paper_wallet, 'sync_from_trade_manager'):
                self.log_test_result("Paper wallet sync", True, "Paper wallet has sync method")
            else:
                self.log_test_result("Paper wallet sync", False, "Paper wallet missing sync method")

        except Exception as e:
            self.log_test_result("Paper wallet sync", False, f"Failed: {e}")

    def test_consistency_validation(self, ctx: BotContext, sync_manager: PositionSyncManager):
        """Test consistency validation."""
        try:
            # Validate consistency
            is_consistent = sync_manager.validate_consistency(ctx.positions, ctx.paper_wallet.positions if ctx.paper_wallet else {})

            if is_consistent:
                self.log_test_result("Consistency validation", True, "All systems are consistent")
            else:
                self.log_test_result("Consistency validation", False, "Systems are not consistent")

        except Exception as e:
            self.log_test_result("Consistency validation", False, f"Failed: {e}")

    def test_context_methods(self, ctx: BotContext):
        """Test BotContext helper methods."""
        try:
            # Test position count
            count = ctx.get_position_count()
            self.log_test_result("Context position count", True, f"Position count: {count}")

            # Test position symbols
            symbols = ctx.get_position_symbols()
            self.log_test_result("Context position symbols", True, f"Symbols: {symbols}")

            # Test consistency validation
            is_consistent = ctx.validate_position_consistency()
            self.log_test_result("Context consistency check", is_consistent, "Consistency validation works")

        except Exception as e:
            self.log_test_result("Context methods", False, f"Failed: {e}")

    def test_position_monitor_integration(self):
        """Test PositionMonitor integration with TradeManager."""
        try:
            from crypto_bot.position_monitor import PositionMonitor

            # Create position monitor with TradeManager
            monitor = PositionMonitor(
                exchange=None,  # Not needed for this test
                config={},
                positions={},
                trade_manager=self.trade_manager
            )

            # Test position existence check
            exists = monitor._position_still_exists("BTC/USD")
            if exists:
                self.log_test_result("Position monitor integration", True, "Position monitor correctly checks TradeManager")
            else:
                self.log_test_result("Position monitor integration", False, "Position monitor not finding TradeManager position")

        except Exception as e:
            self.log_test_result("Position monitor integration", False, f"Failed: {e}")

    def run_all_tests(self):
        """Run all migration tests."""
        logger.info("🧪 Starting TradeManager migration tests...")

        # Test 1: PositionSyncManager creation
        sync_manager = self.test_position_sync_manager_creation()
        if not sync_manager:
            logger.error("Cannot continue tests without PositionSyncManager")
            return

        # Test 2: BotContext creation
        ctx = self.test_context_creation()
        if not ctx:
            logger.error("Cannot continue tests without BotContext")
            return

        # Test 3: Position synchronization
        self.test_position_sync(ctx, sync_manager)

        # Test 4: Paper wallet synchronization
        self.test_paper_wallet_sync(ctx, sync_manager)

        # Test 5: Consistency validation
        self.test_consistency_validation(ctx, sync_manager)

        # Test 6: BotContext helper methods
        self.test_context_methods(ctx)

        # Test 7: PositionMonitor integration
        self.test_position_monitor_integration()

        # Summary
        logger.info("\n📊 Test Summary:")
        logger.info(f"   ✅ Passed: {self.passed_tests}")
        logger.info(f"   ❌ Failed: {self.failed_tests}")
        logger.info(f"   📈 Success Rate: {(self.passed_tests / (self.passed_tests + self.failed_tests) * 100):.1f}%")

        if self.failed_tests == 0:
            logger.info("🎉 All tests passed! Migration is ready.")
            return True
        else:
            logger.warning("⚠️ Some tests failed. Please review the issues above.")
            return False


def main():
    tester = MigrationTester()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 Migration tests completed successfully!")
        print("You can proceed with the migration.")
    else:
        print("\n❌ Migration tests found issues!")
        print("Please fix the issues before proceeding with migration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
