#!/usr/bin/env python3

"""
Test script to diagnose Telegram bot initialization issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crypto_bot'))

from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.telegram_bot_ui import TelegramBotUI
import yaml
from pathlib import Path

def test_telegram_config():
    """Test Telegram configuration loading."""
    print("=== Testing Telegram Configuration ===")

    # Load config
    config_path = Path("crypto_bot/config.yaml")
    if not config_path.exists():
        print("❌ Config file not found")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    tg_config = config.get("telegram", {})
    print(f"Telegram config: {tg_config}")

    # Check basic settings
    if not tg_config.get("enabled", True):
        print("❌ Telegram is disabled in config")
        return False

    token = tg_config.get("token")
    chat_id = tg_config.get("chat_id")

    print(f"Token configured: {'✅' if token else '❌'}")
    print(f"Chat ID configured: {'✅' if chat_id else '❌'}")

    if not token or not chat_id:
        print("❌ Missing token or chat_id")
        return False

    return True

def test_telegram_notifier():
    """Test TelegramNotifier creation."""
    print("\n=== Testing TelegramNotifier ===")

    config_path = Path("crypto_bot/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    tg_config = config.get("telegram", {})

    try:
        notifier = TelegramNotifier.from_config(tg_config)
        print("✅ TelegramNotifier created successfully")
        print(f"Enabled: {notifier.enabled}")
        print(f"Token length: {len(notifier.token) if notifier.token else 0}")
        print(f"Chat ID: {notifier.chat_id}")
        return notifier
    except Exception as e:
        print(f"❌ Failed to create TelegramNotifier: {e}")
        return None

def test_telegram_bot_ui(notifier):
    """Test TelegramBotUI creation."""
    print("\n=== Testing TelegramBotUI ===")

    try:
        # Mock state and other dependencies
        state = {"running": False}
        log_file = Path("crypto_bot/logs/bot.log")

        bot_ui = TelegramBotUI(
            notifier=notifier,
            state=state,
            log_file=log_file,
            rotator=None,
            exchange=None,
            wallet="",
            command_cooldown=5,
            paper_wallet=None
        )

        print("✅ TelegramBotUI created successfully")
        print(f"Bot token: {bot_ui.token[:10]}..." if bot_ui.token else "No token")
        print(f"Chat ID: {bot_ui.chat_id}")
        return bot_ui

    except Exception as e:
        print(f"❌ Failed to create TelegramBotUI: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("🔍 Telegram Bot Initialization Diagnostic")
    print("=" * 50)

    # Test configuration
    if not test_telegram_config():
        print("\n❌ Configuration issues found. Please check your config.yaml")
        return

    # Test notifier
    notifier = test_telegram_notifier()
    if not notifier:
        print("\n❌ Notifier issues found.")
        return

    # Test bot UI
    bot_ui = test_telegram_bot_ui(notifier)
    if not bot_ui:
        print("\n❌ Bot UI issues found.")
        return

    print("\n✅ All Telegram components initialized successfully!")
    print("\n📝 If the menu still doesn't display, check:")
    print("   1. Bot is running with proper permissions")
    print("   2. Chat ID is correct and bot has access")
    print("   3. Network connectivity to Telegram API")
    print("   4. Check bot logs for runtime errors")

if __name__ == "__main__":
    main()
