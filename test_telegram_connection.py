#!/usr/bin/env python3

"""
Test script to check Telegram bot connectivity and menu functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crypto_bot'))

from crypto_bot.utils.telegram import send_test_message, check_telegram_health
import yaml
from pathlib import Path

def test_telegram_connectivity():
    """Test basic Telegram connectivity."""
    print("=== Testing Telegram Connectivity ===")

    # Load config
    config_path = Path("crypto_bot/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    tg_config = config.get("telegram", {})
    token = tg_config.get("token")
    chat_id = tg_config.get("chat_id")

    if not token or not chat_id:
        print("❌ Missing token or chat_id")
        return False

    print(f"Testing with token: {token[:10]}...")
    print(f"Testing with chat_id: {chat_id}")

    # Test basic connectivity
    health = check_telegram_health(token, chat_id)
    print(f"Health status: {health['status']}")

    if health['status'] != 'healthy':
        print(f"❌ Health check failed: {health['error']}")
        for rec in health.get('recommendations', []):
            print(f"   💡 {rec}")
        return False

    print(f"✅ Health check passed! Response time: {health.get('response_time', 'N/A')}s")

    # Test sending a message
    success = send_test_message(token, chat_id, "🧪 Connection test from diagnostic script")
    if success:
        print("✅ Test message sent successfully")
    else:
        print("❌ Test message failed")

    return success

def check_bot_process():
    """Check if the main bot process is running and has Telegram enabled."""
    print("\n=== Checking Bot Process ===")

    import subprocess
    try:
        # Check if main bot is running
        result = subprocess.run(
            ['pgrep', '-f', 'crypto_bot.main'],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print("✅ Main bot process is running")
            pid = result.stdout.strip()
            print(f"   PID: {pid}")
            return True
        else:
            print("❌ Main bot process not found")
            print("   Try running: python -m crypto_bot.main")
            return False

    except Exception as e:
        print(f"❌ Error checking bot process: {e}")
        return False

def check_telegram_logs():
    """Check Telegram-related logs for errors."""
    print("\n=== Checking Telegram Logs ===")

    log_files = [
        Path("crypto_bot/logs/telegram_ui.log"),
        Path("crypto_bot/logs/bot.log"),
        Path("bot_output.log"),
        Path("bot_debug.log")
    ]

    recent_errors = []

    for log_file in log_files:
        if log_file.exists():
            print(f"\n📄 Checking {log_file.name}:")
            try:
                # Read last 20 lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-20:]

                telegram_lines = [line for line in lines if 'telegram' in line.lower()]
                error_lines = [line for line in lines if 'error' in line.lower() or 'exception' in line.lower()]

                if telegram_lines:
                    print(f"   Recent Telegram entries: {len(telegram_lines)}")
                    for line in telegram_lines[-3:]:  # Show last 3
                        print(f"   📝 {line.strip()}")
                else:
                    print("   No recent Telegram entries")

                if error_lines:
                    print(f"   Recent errors: {len(error_lines)}")
                    recent_errors.extend(error_lines[-3:])  # Keep last 3 errors

            except Exception as e:
                print(f"   Error reading log: {e}")
        else:
            print(f"   {log_file.name}: File not found")

    if recent_errors:
        print("\n🚨 Recent errors found:")
        for error in recent_errors[-5:]:  # Show last 5
            print(f"   ❌ {error.strip()}")

    return len(recent_errors) == 0

def main():
    """Main diagnostic function."""
    print("🔍 Telegram Connection Diagnostic")
    print("=" * 50)

    # Test connectivity
    connectivity_ok = test_telegram_connectivity()

    # Check bot process
    bot_running = check_bot_process()

    # Check logs
    logs_ok = check_telegram_logs()

    print("\n" + "=" * 50)
    print("📊 Diagnostic Summary:")
    print(f"   Connectivity: {'✅' if connectivity_ok else '❌'}")
    print(f"   Bot Running: {'✅' if bot_running else '❌'}")
    print(f"   Logs Clean: {'✅' if logs_ok else '❌'}")

    if connectivity_ok and bot_running and logs_ok:
        print("\n✅ All systems check out! The issue might be:")
        print("   1. Telegram bot not started within main process")
        print("   2. Menu command not being triggered")
        print("   3. Bot permissions or chat access issues")
        print("\n💡 Try sending /menu to your bot in Telegram")
    else:
        print("\n❌ Issues found that need to be resolved first")

if __name__ == "__main__":
    main()