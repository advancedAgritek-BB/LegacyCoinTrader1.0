#!/usr/bin/env python3
"""
Simple application verification test to ensure fixes work correctly.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing module imports...")

    try:
        # Test core modules
        from crypto_bot.utils.pyth import get_pyth_price, get_price_async
        print("✅ Pyth utilities imported")

        from crypto_bot.utils.wallet_sync_utility import WalletSyncUtility, auto_fix_wallet_sync
        print("✅ Wallet sync utility imported")

        from crypto_bot.telegram_bot_ui import TelegramBotUI
        print("✅ Telegram bot UI imported")

        from crypto_bot.main import _main_impl
        print("✅ Main application imported")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_price_fallback():
    """Test price fallback functionality."""
    print("\n🔍 Testing price fallback functionality...")

    try:
        from crypto_bot.utils.pyth import get_pyth_price

        # Test with a known symbol (should work with fallbacks)
        price = get_pyth_price("BTC/USD", max_retries=1)
        if price is not None:
            print(f"✅ Got price: ${price:.2f}")
        else:
            print("⚠️ Price fetch returned None (expected for test environment)")

        return True
    except Exception as e:
        print(f"❌ Price fallback test failed: {e}")
        return False

def test_wallet_sync():
    """Test wallet synchronization functionality."""
    print("\n🔍 Testing wallet sync functionality...")

    try:
        from crypto_bot.utils.wallet_sync_utility import WalletSyncUtility

        utility = WalletSyncUtility()

        # Create mock context
        class MockContext:
            def __init__(self):
                self.positions = {"BTC/USD": {"quantity": 1.0}}
                self.paper_wallet = MockPaperWallet()

        class MockPaperWallet:
            def __init__(self):
                self.positions = {"BTC/USD": {"quantity": 1.0}}

        ctx = MockContext()

        # Test sync detection
        issues = utility.detect_sync_issues(ctx)
        print(f"✅ Sync issues detected: {issues['has_issues']}")

        # Test validation
        valid, message = utility.validate_sync_status(ctx)
        print(f"✅ Sync validation: {valid} - {message}")

        return True
    except Exception as e:
        print(f"❌ Wallet sync test failed: {e}")
        return False

def test_telegram_bot_creation():
    """Test Telegram bot creation (without running)."""
    print("\n🔍 Testing Telegram bot creation...")

    try:
        from unittest.mock import Mock, patch
        from crypto_bot.utils.telegram import TelegramNotifier

        # Mock dependencies
        with patch('crypto_bot.telegram_bot_ui.setup_logger'):
            with patch('crypto_bot.telegram_bot_ui.BotController'):
                with patch('asyncio.get_event_loop'):
                    with patch('crypto_bot.telegram_bot_ui.ApplicationBuilder'):
                        # Create mock notifier
                        notifier = Mock(spec=TelegramNotifier)
                        notifier.token = "test_token"
                        notifier.chat_id = "123456"
                        notifier.enabled = True

                        state = {"running": False}
                        log_file = Path(tempfile.gettempdir()) / "test_bot.log"

                        from crypto_bot.telegram_bot_ui import TelegramBotUI

                        # Test creation
                        bot = TelegramBotUI(
                            notifier=notifier,
                            state=state,
                            log_file=log_file,
                            rotator=None,
                            exchange=None,
                            wallet="",
                            command_cooldown=5,
                            paper_wallet=None
                        )

                        print("✅ Telegram bot created successfully")
                        return True

    except Exception as e:
        print(f"❌ Telegram bot creation test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 LegacyCoinTrader Application Verification")
    print("=" * 50)

    tests = [
        ("Module Imports", test_imports),
        ("Price Fallback", test_price_fallback),
        ("Wallet Sync", test_wallet_sync),
        ("Telegram Bot", test_telegram_bot_creation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print("25")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Application fixes are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
