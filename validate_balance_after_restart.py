#!/usr/bin/env python3
"""
Balance validation script - Run after bot restart to verify balance is correct
"""

import time
from pathlib import Path

import yaml

from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger("validate_balance_after_restart", LOG_DIR / "validate_balance_after_restart.log")

def validate_balance():
    logger.info("🔍 VALIDATING BOT BALANCE AFTER RESTART:")
    logger.info("=" * 50)

    # Check paper wallet state file
    pw_file = Path("crypto_bot/logs/paper_wallet_state.yaml")
    if pw_file.exists():
        with open(pw_file, 'r') as f:
            state = yaml.safe_load(f)

        balance = state.get('balance', 0)
        positions = state.get('positions', {})

        logger.info(f"File Balance: ${balance:.2f}")
        logger.info(f"Open Positions: {len(positions)}")

        if balance > 0:
            logger.info("✅ File balance is positive")
        else:
            logger.warning("❌ File balance is still negative")
    else:
        logger.warning(f"Paper wallet state file not found at {pw_file}")

    # Wait a moment for bot to start
    logger.info("⏳ Waiting for bot to start (10 seconds)...")
    time.sleep(10)

    # Check if bot is running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'python.*bot'], capture_output=True, text=True)
        if result.stdout.strip():
            logger.info("✅ Bot is running")
        else:
            logger.warning("❌ Bot is not running")
    except Exception as e:
        logger.error(f"⚠️ Could not check if bot is running: {e}")

    logger.info("📋 VALIDATION COMPLETE")
    logger.info("If you see negative balance in logs, run this script again after bot restart")

if __name__ == "__main__":
    validate_balance()
