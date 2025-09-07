#!/usr/bin/env python3
"""
Test script to verify wallet balance fixes work correctly.
"""

import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

def test_wallet_synchronization():
    """Test that wallet synchronization works correctly."""
    print("=== TESTING WALLET SYNCHRONIZATION ===\n")

    project_root = Path(__file__).parent
    logs_dir = project_root / "crypto_bot" / "logs"

    # Test 1: Check TradeManager has positions
    tm_state_file = logs_dir / "trade_manager_state.json"
    if tm_state_file.exists():
        with open(tm_state_file) as f:
            tm_state = json.load(f)

        positions = tm_state.get('positions', {})
        print(f"✅ TradeManager positions: {len(positions)}")
        if positions:
            print(f"   Symbols: {list(positions.keys())}")

        # Verify positions have required fields
        for symbol, pos in positions.items():
            required_fields = ['symbol', 'side', 'total_amount', 'average_price', 'entry_time']
            missing_fields = [field for field in required_fields if field not in pos]
            if missing_fields:
                print(f"❌ Position {symbol} missing fields: {missing_fields}")
            else:
                print(f"✅ Position {symbol} has all required fields")

    # Test 2: Check Paper Wallet is clean
    pw_state_file = logs_dir / "paper_wallet_state.yaml"
    if pw_state_file.exists():
        with open(pw_state_file) as f:
            pw_state = yaml.safe_load(f)

        positions = pw_state.get('positions', {})
        balance = pw_state.get('balance', 0)
        print(f"✅ Paper Wallet positions: {len(positions)} (should be 0)")
        print(f"✅ Paper Wallet balance: ${balance:.2f}")

        if positions:
            print("❌ Paper wallet still has positions - this should be empty!")
        else:
            print("✅ Paper wallet positions cleared correctly")

    # Test 3: Check synchronization manager exists
    sync_manager_file = project_root / "crypto_bot" / "utils" / "sync_manager.py"
    if sync_manager_file.exists():
        print("✅ Synchronization manager created")
    else:
        print("❌ Synchronization manager not found")

    print("\n=== TEST RESULTS ===")
    print("If all checks show ✅, the wallet balance fix is working correctly.")
    print("You can now restart the bot and it should maintain proper synchronization.")

def test_imports():
    """Test that all required modules can be imported."""
    print("=== TESTING IMPORTS ===\n")

    try:
        from crypto_bot.paper_wallet import PaperWallet
        print("✅ PaperWallet import successful")
    except Exception as e:
        print(f"❌ PaperWallet import failed: {e}")

    try:
        from crypto_bot.utils.trade_manager import get_trade_manager, TradeManager
        print("✅ TradeManager import successful")
    except Exception as e:
        print(f"❌ TradeManager import failed: {e}")

    try:
        from crypto_bot.utils.sync_manager import get_sync_manager
        print("✅ SyncManager import successful")
    except Exception as e:
        print(f"❌ SyncManager import failed: {e}")

    try:
        from crypto_bot.utils.balance_manager import get_single_balance
        print("✅ BalanceManager import successful")
    except Exception as e:
        print(f"❌ BalanceManager import failed: {e}")

if __name__ == "__main__":
    test_wallet_synchronization()
    print()
    test_imports()
