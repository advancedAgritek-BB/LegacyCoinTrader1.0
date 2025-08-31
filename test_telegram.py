#!/usr/bin/env python3
"""
Diagnostic script to test Telegram configuration and connectivity.
"""

import sys
import os
import asyncio
import yaml
from pathlib import Path

def test_telegram_config():
    """Test Telegram configuration and connectivity."""
    print("🔧 Testing Telegram Configuration...")
    print("=" * 50)

    # Load config directly
    config_path = Path(__file__).parent / "crypto_bot" / "config.yaml"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        tg_config = config.get('telegram', {})
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    token = tg_config.get('token', '')
    chat_id = tg_config.get('chat_id', '')
    chat_admins = tg_config.get('chat_admins', '')

    print(f"📝 Token: {token[:20]}..." if token else "❌ No token found")
    print(f"📱 Chat ID: {chat_id}" if chat_id else "❌ No chat ID found")
    print(f"👥 Chat Admins: {chat_admins}" if chat_admins else "ℹ️  No chat admins")

    # Validate token format
    if token:
        if ':' not in token or len(token.split(':')[0]) < 5:
            print("⚠️  Token format looks suspicious - should be like '123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11'")
        else:
            print("✅ Token format looks valid")

    # Validate chat ID format
    if chat_id:
        try:
            chat_id_int = int(chat_id)
            if chat_id_int < 0:
                print("✅ Chat ID format looks valid (negative = group)")
            elif len(str(chat_id_int)) >= 9:
                print("✅ Chat ID format looks valid")
            else:
                print("⚠️  Chat ID looks too short - should be 9+ digits")
        except ValueError:
            print("❌ Chat ID should be numeric")

    # Test connectivity
    print("\n🌐 Testing connectivity...")

    # Add the crypto_bot directory to the path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crypto_bot'))

    try:
        from crypto_bot.utils.telegram import send_test_message, check_telegram_health
        health = asyncio.run(check_telegram_health(token, chat_id))
        print(f"📊 Status: {health['status']}")
        if health['response_time']:
            print(f"⏱️  Response time: {health['response_time']:.2f}s")
        if health['error']:
            print(f"❌ Error: {health['error']}")
        if health['recommendations']:
            print("💡 Recommendations:")
            for rec in health['recommendations']:
                print(f"   • {rec}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

    print("\n" + "=" * 50)
    print("🔍 Troubleshooting Tips:")
    print("1. Verify your bot token from @BotFather on Telegram")
    print("2. Get your chat ID from @userinfobot or start a chat with your bot")
    print("3. Make sure the bot is added to the chat/group")
    print("4. Check that the bot has permission to send messages")
    print("5. Verify network connectivity to Telegram API")

if __name__ == "__main__":
    test_telegram_config()
