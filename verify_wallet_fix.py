#!/usr/bin/env python3
"""
Verification script to run after restarting the bot.
This ensures the wallet balance fix is working correctly.
"""

import yaml
import json
import time
from pathlib import Path
from datetime import datetime

def verify_wallet_fix():
    """Verify that the wallet balance fix is working correctly."""
    print("🔍 VERIFYING WALLET BALANCE FIX...")
    print("=" * 50)

    project_root = Path(__file__).parent
    logs_dir = project_root / "crypto_bot" / "logs"

    issues_found = []

    # 1. Check TradeManager state
    print("\n1. Checking TradeManager (Single Source of Truth)...")
    tm_state = logs_dir / "trade_manager_state.json"

    if not tm_state.exists():
        issues_found.append("❌ TradeManager state file not found")
        print("❌ TradeManager state file not found")
    else:
        with open(tm_state) as f:
            tm_data = json.load(f)

        positions = tm_data.get('positions', {})
        trades = tm_data.get('trades', [])

        print(f"✅ TradeManager state file exists")
        print(f"   Positions: {len(positions)}")
        print(f"   Trades: {len(trades)}")

        if len(positions) == 0:
            issues_found.append("❌ TradeManager has no positions")
            print("❌ TradeManager has no positions")
        else:
            print("✅ TradeManager has positions")

        # Verify position integrity
        for symbol, pos in positions.items():
            required_fields = ['symbol', 'side', 'total_amount', 'average_price', 'entry_time']
            missing = [f for f in required_fields if f not in pos]
            if missing:
                issues_found.append(f"❌ Position {symbol} missing fields: {missing}")
                print(f"❌ Position {symbol} missing fields: {missing}")

    # 2. Check Paper Wallet state
    print("\n2. Checking Paper Wallet (Balance Tracker)...")
    pw_state = logs_dir / "paper_wallet_state.yaml"

    if not pw_state.exists():
        issues_found.append("❌ Paper wallet state file not found")
        print("❌ Paper wallet state file not found")
    else:
        with open(pw_state) as f:
            pw_data = yaml.safe_load(f)

        balance = pw_data.get('balance', 0)
        positions = pw_data.get('positions', {})

        print(f"✅ Paper wallet state file exists")
        print(f"   Balance: ${balance:.2f}")
        print(f"   Positions: {len(positions)}")

        if len(positions) > 0:
            issues_found.append("❌ Paper wallet still has positions (should be empty)")
            print("❌ Paper wallet still has positions (should be empty)")
        else:
            print("✅ Paper wallet positions cleared correctly")

        if balance <= 0:
            issues_found.append(f"❌ Invalid paper wallet balance: ${balance:.2f}")
            print(f"❌ Invalid paper wallet balance: ${balance:.2f}")

    # 3. Verify architecture consistency
    print("\n3. Verifying Architecture Consistency...")
    tm_positions = len(tm_data.get('positions', {})) if tm_state.exists() else 0
    pw_positions = len(pw_data.get('positions', {})) if pw_state.exists() else 0

    if tm_positions > 0 and pw_positions == 0:
        print("✅ Architecture correct: TradeManager has positions, Paper Wallet clean")
    elif tm_positions == 0 and pw_positions > 0:
        issues_found.append("❌ Architecture issue: TradeManager empty, Paper Wallet has positions")
        print("❌ Architecture issue: TradeManager empty, Paper Wallet has positions")
    elif tm_positions > 0 and pw_positions > 0:
        issues_found.append("❌ Architecture issue: Both systems have positions")
        print("❌ Architecture issue: Both systems have positions")
    else:
        print("ℹ️  No positions found in either system")

    # 4. Test TradeManager loading
    print("\n4. Testing TradeManager Loading...")
    try:
        import sys
        sys.path.append(str(project_root / "crypto_bot"))

        from utils.trade_manager import get_trade_manager
        tm = get_trade_manager()

        loaded_positions = len(tm.positions)
        loaded_trades = len(tm.trades)

        print("✅ TradeManager imports successfully")
        print(f"   Loaded positions: {loaded_positions}")
        print(f"   Loaded trades: {loaded_trades}")

        if loaded_positions != tm_positions:
            issues_found.append(f"❌ TradeManager loaded {loaded_positions} positions but file has {tm_positions}")
            print(f"❌ TradeManager loaded {loaded_positions} positions but file has {tm_positions}")

    except Exception as e:
        issues_found.append(f"❌ TradeManager loading failed: {e}")
        print(f"❌ TradeManager loading failed: {e}")

    # 5. Summary
    print("\n" + "=" * 50)
    print("VERIFICATION RESULTS:")
    print("=" * 50)

    if not issues_found:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Wallet balance fix is working correctly")
        print("✅ TradeManager is single source of truth")
        print("✅ Paper wallet tracks balance only")
        print("✅ No synchronization conflicts")

        # Show portfolio summary
        if tm_state.exists() and pw_state.exists():
            total_position_value = sum(
                pos.get('total_amount', 0) * pos.get('average_price', 0)
                for pos in tm_data.get('positions', {}).values()
            )
            pw_balance = pw_data.get('balance', 0)
            total_portfolio = total_position_value + pw_balance

            print("
📊 PORTFOLIO SUMMARY:"            print(f"   Position Value: ${total_position_value:.2f}")
            print(f"   Cash Balance: ${pw_balance:.2f}")
            print(f"   Total Portfolio: ${total_portfolio:.2f}")

    else:
        print("❌ ISSUES FOUND:")
        for issue in issues_found:
            print(f"   {issue}")

        print("\n🔧 RECOMMENDED ACTIONS:")
        if any("TradeManager" in issue and "empty" in issue for issue in issues_found):
            print("   - Restore positions to TradeManager from backup")
        if any("Paper wallet" in issue and "positions" in issue for issue in issues_found):
            print("   - Clear positions from paper wallet")
        if any("loading failed" in issue for issue in issues_found):
            print("   - Check TradeManager implementation")
        print("   - Restart the bot after fixes")

    return len(issues_found) == 0

if __name__ == "__main__":
    success = verify_wallet_fix()
    exit(0 if success else 1)
