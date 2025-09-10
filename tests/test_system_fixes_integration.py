"""
Integration tests for all system fixes working together.
"""

import unittest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from crypto_bot.telegram_bot_ui import TelegramBotUI
from crypto_bot.utils.pyth import get_pyth_price, get_price_async
from crypto_bot.utils.wallet_sync_utility import WalletSyncUtility
from crypto_bot.utils.telegram import TelegramNotifier


class TestSystemFixesIntegration(unittest.TestCase):
    """Integration tests for all fixes working together."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "integration_test.log"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('crypto_bot.utils.pyth.requests.get')
    def test_pyth_fallback_integration(self, mock_get):
        """Test Pyth fallback system integration."""
        # First call fails (Pyth), second succeeds (CoinGecko fallback)
        call_count = 0

        def mock_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_resp = Mock()
            if call_count == 1:
                # Pyth fails
                mock_resp.json.return_value = []
            else:
                # CoinGecko succeeds
                mock_resp.raise_for_status.return_value = None
                mock_resp.json.return_value = {"bitcoin": {"usd": 45000.0}}
            return mock_resp

        mock_get.side_effect = mock_response

        # Test synchronous fallback
        price = get_pyth_price("BTC/USD", max_retries=1)
        self.assertIsNotNone(price)
        self.assertGreater(price, 0)

    @patch('crypto_bot.utils.pyth.requests.get')
    async def test_async_price_integration(self, mock_get):
        """Test async price fetching with fallbacks."""
        # Mock successful fallback response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"bitcoin": {"usd": 45000.0}}
        mock_get.return_value = mock_response

        price = await get_price_async("BTC/USD")
        self.assertIsNotNone(price)
        self.assertGreater(price, 0)

    def test_wallet_sync_integration(self):
        """Test wallet synchronization integration."""
        utility = WalletSyncUtility()

        # Create mock context with sync issues
        mock_ctx = Mock()
        mock_paper_wallet = Mock()
        mock_ctx.paper_wallet = mock_paper_wallet

        # Setup mismatched positions
        mock_ctx.positions = {"BTC/USD": {"quantity": 1.0}}
        mock_paper_wallet.positions = {
            "BTC/USD": {"quantity": 1.0},
            "ETH/USD": {"quantity": 2.0}
        }

        # Detect issues
        issues = utility.detect_sync_issues(mock_ctx)
        self.assertTrue(issues["has_issues"])

        # Sync from paper wallet
        success, message = utility._sync_from_paper_wallet(mock_ctx)
        self.assertTrue(success)

        # Verify sync worked
        self.assertEqual(len(mock_ctx.positions), 2)
        self.assertIn("BTC/USD", mock_ctx.positions)
        self.assertIn("ETH/USD", mock_ctx.positions)

    @patch('crypto_bot.telegram_bot_ui.setup_logger')
    @patch('crypto_bot.telegram_bot_ui.BotController')
    def test_telegram_bot_lock_integration(self, mock_controller, mock_logger):
        """Test Telegram bot lock mechanism integration."""
        with patch('crypto_bot.telegram_bot_ui.LOG_DIR', Path(self.temp_dir)):
            # Mock notifier
            notifier = Mock(spec=TelegramNotifier)
            notifier.token = "test_token"
            notifier.chat_id = "123456"
            notifier.enabled = True

            state = {"running": False}

            # Create first bot instance
            bot1 = TelegramBotUI(
                notifier=notifier,
                state=state,
                log_file=self.log_file,
                rotator=None,
                exchange=None,
                wallet="",
                command_cooldown=5,
                paper_wallet=None
            )

            # Verify lock file exists
            lock_file = Path(self.temp_dir) / "telegram_bot.lock"
            self.assertTrue(lock_file.exists())

            # Create second bot instance (should handle gracefully)
            bot2 = TelegramBotUI(
                notifier=notifier,
                state=state,
                log_file=self.log_file,
                rotator=None,
                exchange=None,
                wallet="",
                command_cooldown=5,
                paper_wallet=None
            )

            # Both should exist (improved behavior)
            self.assertIsNotNone(bot1)
            self.assertIsNotNone(bot2)

            # Clean up
            bot1._release_lock()
            bot2._release_lock()

    def test_complete_system_workflow(self):
        """Test complete system workflow with all fixes."""
        # 1. Test wallet sync
        utility = WalletSyncUtility()
        mock_ctx = Mock()
        mock_paper_wallet = Mock()
        mock_ctx.paper_wallet = mock_paper_wallet

        # Setup sync issues
        mock_ctx.positions = {}
        mock_paper_wallet.positions = {"BTC/USD": {"quantity": 1.0}}

        # Fix sync issues
        success, message = utility._sync_from_paper_wallet(mock_ctx)
        self.assertTrue(success)

        # 2. Test price fetching with fallbacks
        with patch('crypto_bot.utils.pyth.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"bitcoin": {"usd": 45000.0}}
            mock_get.return_value = mock_response

            price = get_pyth_price("BTC/USD")
            self.assertIsNotNone(price)

        # 3. Test position consistency after sync
        issues = utility.detect_sync_issues(mock_ctx)
        self.assertFalse(issues["has_issues"])

    def test_error_recovery_integration(self):
        """Test error recovery mechanisms across all fixes."""
        # Test wallet sync error recovery
        utility = WalletSyncUtility()

        # Mock context with errors
        mock_ctx = Mock()
        mock_ctx.positions = None  # This will cause errors

        # Should handle gracefully
        issues = utility.detect_sync_issues(mock_ctx)
        self.assertTrue(issues["has_issues"])

        # Test force reset as recovery
        mock_ctx.positions = {"BTC/USD": {"quantity": 1.0}}
        mock_ctx.paper_wallet = Mock()
        mock_ctx.paper_wallet.positions = {"BTC/USD": {"quantity": 1.0}}
        mock_ctx.balances = {"USD": 10000}

        success, message = utility.force_balance_reset(mock_ctx)
        self.assertTrue(success)
        self.assertEqual(mock_ctx.positions, {})
        self.assertEqual(mock_ctx.balances, {})

    def test_concurrent_operations(self):
        """Test concurrent operations don't interfere."""
        async def test_concurrent():
            # Test concurrent price fetching
            tasks = []
            for symbol in ["BTC/USD", "ETH/USD", "SOL/USD"]:
                with patch('crypto_bot.utils.pyth.requests.get') as mock_get:
                    mock_response = Mock()
                    mock_response.raise_for_status.return_value = None
                    mock_response.json.return_value = {"bitcoin": {"usd": 45000.0}}
                    mock_get.return_value = mock_response

                    tasks.append(get_price_async(symbol))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            for result in results:
                if isinstance(result, Exception):
                    self.fail(f"Concurrent operation failed: {result}")
                else:
                    self.assertIsNotNone(result)

        asyncio.run(test_concurrent())

    def test_system_resilience(self):
        """Test system resilience under various failure conditions."""
        # Test with network failures
        with patch('crypto_bot.utils.pyth.requests.get', side_effect=Exception("Network down")):
            price = get_pyth_price("BTC/USD")
            self.assertIsNone(price)  # Should fail gracefully

        # Test wallet sync with missing components
        utility = WalletSyncUtility()
        mock_ctx = Mock()
        mock_ctx.paper_wallet = None  # No paper wallet

        # Should handle missing components gracefully
        success, message = utility._sync_from_paper_wallet(mock_ctx)
        self.assertFalse(success)
        self.assertIn("No paper wallet", message)

    def test_configuration_compatibility(self):
        """Test that fixes work with existing configuration."""
        # Mock a realistic configuration
        config = {
            "telegram": {
                "enabled": True,
                "token": "test_token",
                "chat_id": "123456"
            },
            "symbols": ["BTC/USD", "ETH/USD"],
            "trading": {
                "max_positions": 5,
                "risk_per_trade": 0.01
            }
        }

        # Test that wallet sync works with config
        utility = WalletSyncUtility()
        mock_ctx = Mock()
        mock_paper_wallet = Mock()
        mock_ctx.paper_wallet = mock_paper_wallet
        mock_ctx.positions = {"BTC/USD": {"quantity": 1.0}}
        mock_paper_wallet.positions = {"BTC/USD": {"quantity": 1.0}}

        success, message = utility.validate_sync_status(mock_ctx)
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
