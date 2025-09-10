#!/usr/bin/env python3
"""
Bot validation script - Run after restart to verify all fixes
"""

import yaml
import time
import subprocess
from pathlib import Path

def validate_bot_state():
    print("🔍 VALIDATING BOT STATE AFTER FIXES:")
    print("=" * 50)

    # Check configuration
    config_file = Path("crypto_bot/config.yaml")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        timeframe = config.get('timeframe')
        regime_timeframes = config.get('regime_timeframes', [])

        print(f"✅ Timeframe: {timeframe}")
        print(f"✅ Regime timeframes: {regime_timeframes}")

        if timeframe and regime_timeframes:
            print("✅ Configuration is correct")
        else:
            print("❌ Configuration issues remain")
    else:
        print("❌ Config file not found")

    # Wait for bot to start
    print("\n⏳ Waiting for bot to start (15 seconds)...")
    time.sleep(15)

    # Check if bot is running
    try:
        result = subprocess.run(['pgrep', '-f', 'python.*bot'], capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Bot is running")
        else:
            print("❌ Bot is not running")
            return
    except Exception as e:
        print(f"⚠️ Could not check if bot is running: {e}")

    # Monitor logs for 30 seconds
    print("\n📊 MONITORING BOT LOGS FOR 30 SECONDS:")
    end_time = time.time() + 30

    success_indicators = [
        "PHASE: fetch_candidates completed",
        "PHASE: analyse_batch - running analysis on",
        "PHASE: analyse_batch completed",
        "Trading cycle completed"
    ]

    found_indicators = set()

    while time.time() < end_time:
        try:
            log_file = Path("crypto_bot/logs/bot.log")
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-20:]  # Check last 20 lines

                for line in lines:
                    for indicator in success_indicators:
                        if indicator in line and indicator not in found_indicators:
                            print(f"✅ Found: {indicator}")
                            found_indicators.add(indicator)
        except Exception as e:
            print(f"⚠️ Error reading logs: {e}")

        time.sleep(2)

    print("\n" + "=" * 50)
    if len(found_indicators) >= 3:
        print("🎉 BOT IS WORKING CORRECTLY!")
        print("All major issues have been resolved.")
    else:
        print("⚠️ SOME ISSUES MAY REMAIN")
        print("Check the bot logs for more details.")

if __name__ == "__main__":
    validate_bot_state()
