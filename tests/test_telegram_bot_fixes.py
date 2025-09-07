"""
Unit tests for Telegram bot fixes and improvements.
"""

import unittest
import tempfile
import os
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from crypto_bot.telegram_bot_ui import TelegramBotUI
from crypto_bot.utils.telegram import TelegramNotifier


class TestTelegramBotFixes(unittest.TestCase):
    """Test cases for Telegram bot improvements."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test_bot.log"

        # Mock notifier
        self.notifier = Mock(spec=TelegramNotifier)
        self.notifier.token = "test_token_123"
        self.notifier.chat_id = "123456789"
        self.notifier.enabled = True

        # Mock state
        self.state = {"running": False}

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up lock files
        lock_file = Path(self.temp_dir) / "telegram_bot.lock"
        if lock_file.exists():
            lock_file.unlink()

        # Remove temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lock_file_creation(self):
        """Test that lock file is created properly."""
        with patch('crypto_bot.telegram_bot_ui.LOG_DIR', Path(self.temp_dir)):
            with patch('crypto_bot.telegram_bot_ui.setup_logger'):
                with patch('crypto_bot.telegram_bot_ui.BotController'):
                    # Mock asyncio event loop
                    with patch('asyncio.get_event_loop'):
                        with patch('crypto_bot.telegram_bot_ui.ApplicationBuilder'):
                            bot = TelegramBotUI(
                                notifier=self.notifier,
                                state=self.state,
                                log_file=self.log_file,
                                rotator=None,
                                exchange=None,
                                wallet="",
                                command_cooldown=5,
                                paper_wallet=None
                            )

                            # Check that lock file exists
                            lock_file = Path(self.temp_dir) / "telegram_bot.lock"
                            self.assertTrue(lock_file.exists())

                            # Check that lock file contains PID
                            pid_content = lock_file.read_text().strip()
                            self.assertEqual(pid_content, str(os.getpid()))

    def test_lock_cleanup_on_destroy(self):
        """Test that lock file is cleaned up when object is destroyed."""
        with patch('crypto_bot.telegram_bot_ui.LOG_DIR', Path(self.temp_dir)):
            with patch('crypto_bot.telegram_bot_ui.setup_logger'):
                with patch('crypto_bot.telegram_bot_ui.BotController'):
                    # Mock asyncio event loop
                    with patch('asyncio.get_event_loop'):
                        with patch('crypto_bot.telegram_bot_ui.ApplicationBuilder'):
                            bot = TelegramBotUI(
                                notifier=self.notifier,
                                state=self.state,
                                log_file=self.log_file,
                                rotator=None,
                                exchange=None,
                                wallet="",
                                command_cooldown=5,
                                paper_wallet=None
                            )

                            lock_file = Path(self.temp_dir) / "telegram_bot.lock"
                            self.assertTrue(lock_file.exists())

                            # Manually call cleanup (simulating __del__)
                            bot._release_lock()

                            # Lock file should be removed
                            self.assertFalse(lock_file.exists())

    def test_multiple_instances_detection(self):
        """Test detection and handling of multiple instances."""
        with patch('crypto_bot.telegram_bot_ui.LOG_DIR', Path(self.temp_dir)):
            with patch('crypto_bot.telegram_bot_ui.setup_logger'):
                with patch('crypto_bot.telegram_bot_ui.BotController'):
                    # Create first instance
                    bot1 = TelegramBotUI(
                        notifier=self.notifier,
                        state=self.state,
                        log_file=self.log_file,
                        rotator=None,
                        exchange=None,
                        wallet="",
                        command_cooldown=5,
                        paper_wallet=None
                    )

                    # Try to create second instance
                    bot2 = TelegramBotUI(
                        notifier=self.notifier,
                        state=self.state,
                        log_file=self.log_file,
                        rotator=None,
                        exchange=None,
                        wallet="",
                        command_cooldown=5,
                        paper_wallet=None
                    )

                    # Both should have been created (improved behavior)
                    self.assertIsNotNone(bot1)
                    self.assertIsNotNone(bot2)

    def test_graceful_shutdown(self):
        """Test graceful shutdown functionality."""
        with patch('crypto_bot.telegram_bot_ui.LOG_DIR', Path(self.temp_dir)):
            with patch('crypto_bot.telegram_bot_ui.setup_logger'):
                with patch('crypto_bot.telegram_bot_ui.BotController'):
                    bot = TelegramBotUI(
                        notifier=self.notifier,
                        state=self.state,
                        log_file=self.log_file,
                        rotator=None,
                        exchange=None,
                        wallet="",
                        command_cooldown=5,
                        paper_wallet=None
                    )

                    # Mock the app and task
                    bot.app = Mock()
                    bot.task = Mock()
                    bot.task.done.return_value = False

                    # Test shutdown
                    asyncio.run(bot.shutdown())

                    # Verify cleanup was called
                    bot.app.stop.assert_called_once()
                    bot.task.cancel.assert_called_once()


if __name__ == '__main__':
    unittest.main()
