#!/usr/bin/env python3
"""
Balance validation script - Run after bot restart to verify balance is correct
"""

import yaml
from pathlib import Path
import time

def validate_balance():
    print("🔍 VALIDATING BOT BALANCE AFTER RESTART:")
    print("=" * 50)

    # Check paper wallet state file
    pw_file = Path("crypto_bot/logs/paper_wallet_state.yaml")
    if pw_file.exists():
        with open(pw_file, 'r') as f:
            state = yaml.safe_load(f)

        balance = state.get('balance', 0)
        positions = state.get('positions', {})

        print(f"File Balance: ${balance:.2f}")
        print(f"Open Positions: {len(positions)}")

        if balance > 0:
            print("✅ File balance is positive")
        else:
            print("❌ File balance is still negative")

    # Wait a moment for bot to start
    print("\n⏳ Waiting for bot to start (10 seconds)...")
    time.sleep(10)

    # Check if bot is running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'python.*bot'], capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Bot is running")
        else:
            print("❌ Bot is not running")
    except Exception as e:
        print(f"⚠️ Could not check if bot is running: {e}")

    print("\n📋 VALIDATION COMPLETE")
    print("If you see negative balance in logs, run this script again after bot restart")

if __name__ == "__main__":
    validate_balance()
