#!/usr/bin/env python3
"""
Telegram bot connection test script.
Tests the bot's ability to connect and send messages.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any

from crypto_bot.config import load_config as load_bot_config, resolve_config_path

try:
    from telegram import Bot
    from telegram.request import Request
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("❌ python-telegram-bot not installed. Install with: pip install python-telegram-bot")

async def test_telegram_connection(token: str, chat_id: str) -> bool:
    """Test Telegram bot connection and send a test message."""
    if not TELEGRAM_AVAILABLE:
        return False
    
    try:
        # Create bot with proper timeout settings
        bot = Bot(token, request=Request(
            connection_pool_size=8,
            connect_timeout=30.0,
            read_timeout=30.0,
            write_timeout=30.0
        ))
        
        print("🔗 Testing Telegram bot connection...")
        
        # Test getting bot info
        bot_info = await asyncio.wait_for(bot.get_me(), timeout=30.0)
        print(f"✅ Bot connected: @{bot_info.username} (ID: {bot_info.id})")
        
        # Test sending message
        test_message = f"🤖 Bot connection test successful at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        await asyncio.wait_for(
            bot.send_message(chat_id=chat_id, text=test_message),
            timeout=30.0
        )
        print(f"✅ Test message sent to chat {chat_id}")
        
        return True
        
    except asyncio.TimeoutError:
        print("❌ Connection timeout - check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_config_values(config: Dict[str, Any]) -> None:
    """Test configuration values for common issues."""
    print("\n🔍 Checking configuration values...")
    
    telegram = config.get('telegram', {})
    
    # Check required fields
    if not telegram.get('token'):
        print("❌ Missing Telegram token")
        return
    
    if not telegram.get('chat_id'):
        print("❌ Missing Telegram chat ID")
        return
    
    # Check token format
    token = telegram['token']
    if not token or len(token) < 20:
        print("❌ Invalid Telegram token format")
        return
    
    # Check chat ID format
    chat_id = str(telegram['chat_id'])
    if not chat_id.isdigit():
        print("❌ Chat ID should be a numeric value")
        return
    
    print("✅ Configuration values look valid")
    
    # Check optional settings
    timeout = telegram.get('timeout_seconds', 30)
    if timeout < 10:
        print(f"⚠️  timeout_seconds ({timeout}) is quite low, consider increasing to 30+")
    
    if telegram.get('fail_silently', False):
        print("ℹ️  fail_silently is enabled - bot will continue running even if Telegram fails")

def main():
    """Main test function."""
    config_path = resolve_config_path()
    if not Path(config_path).exists():
        print(f"ℹ️ Override configuration not found at {config_path}; using defaults.")

    print("🤖 Telegram Bot Connection Test")
    print("=" * 40)

    config = load_bot_config(config_path)
    if not config:
        print("❌ Failed to load configuration")
        return
    
    telegram = config.get('telegram', {})
    if not telegram.get('enabled', False):
        print("ℹ️  Telegram is disabled in configuration")
        return
    
    # Test configuration values
    test_config_values(config)
    
    # Test connection
    token = telegram.get('token')
    chat_id = telegram.get('chat_id')
    
    if not token or not chat_id:
        print("❌ Cannot test connection - missing token or chat ID")
        return
    
    print(f"\n🚀 Testing connection with token: {token[:10]}...")
    print(f"📱 Chat ID: {chat_id}")
    
    # Run the connection test
    success = asyncio.run(test_telegram_connection(token, chat_id))
    
    if success:
        print("\n🎉 All tests passed! Your Telegram bot is working correctly.")
        print("\n💡 If you're still seeing errors in the main bot:")
        print("  - Check that the bot has been added to the chat")
        print("  - Verify the chat ID is correct")
        print("  - Ensure the bot has permission to send messages")
        print("  - Check your internet connection")
    else:
        print("\n❌ Connection test failed. Check the errors above.")
        print("\n🔧 Troubleshooting steps:")
        print("  1. Verify your bot token is correct")
        print("  2. Ensure the bot is added to the chat")
        print("  3. Check that the chat ID matches the actual chat")
        print("  4. Test with @userinfobot to get your chat ID")
        print("  5. Ensure the bot has permission to send messages")

if __name__ == "__main__":
    main()
