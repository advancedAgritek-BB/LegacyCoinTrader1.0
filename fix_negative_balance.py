#!/usr/bin/env python3
"""Fix negative balance in the paper wallet state file."""

import sys
from pathlib import Path
import yaml

def fix_negative_balance():
    """Fix any negative balance in the paper wallet state file."""

    print("=" * 60)
    print("FIXING NEGATIVE BALANCE IN PAPER WALLET STATE")
    print("=" * 60)

    state_file = Path("crypto_bot/logs/paper_wallet_state.yaml")

    if not state_file.exists():
        print("❌ Paper wallet state file not found")
        return False

    try:
        # Read current state
        with open(state_file, 'r') as f:
            state = yaml.safe_load(f) or {}

        current_balance = state.get('balance', 10000.0)
        print(f"Current balance in state file: ${current_balance:.2f}")

        # Check if balance is negative
        if current_balance < 0:
            print(f"❌ Found negative balance: ${current_balance:.2f}")

            # Fix the balance to 0.0
            state['balance'] = 0.0

            # Save the corrected state
            with open(state_file, 'w') as f:
                yaml.dump(state, f, default_flow_style=False)

            print(f"✅ Corrected balance to: $0.00")
            print("✅ State file updated successfully")

            return True
        else:
            print(f"✅ Balance is already positive: ${current_balance:.2f}")
            return True

    except Exception as e:
        print(f"❌ Error fixing balance: {e}")
        return False

def verify_balance_sources():
    """Verify that all balance sources are consistent."""

    print("\n" + "=" * 50)
    print("VERIFYING BALANCE SOURCE CONSISTENCY")
    print("=" * 50)

    try:
        from crypto_bot.utils.balance_manager import get_single_balance, synchronize_balance_sources, validate_balance_sources

        # Get balance from single source
        single_balance = get_single_balance()
        print(f"Single source balance: ${single_balance:.2f}")

        # Synchronize all sources
        print("Synchronizing all balance sources...")
        synchronize_balance_sources()

        # Validate consistency
        is_valid = validate_balance_sources()
        if is_valid:
            print("✅ All balance sources are consistent")
        else:
            print("❌ Balance sources are inconsistent")

        return is_valid

    except Exception as e:
        print(f"❌ Error verifying balance sources: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Starting balance fix process...")

    # Fix negative balance
    balance_fixed = fix_negative_balance()

    if balance_fixed:
        # Verify and synchronize sources
        sources_valid = verify_balance_sources()

        if sources_valid:
            print("\n🎉 BALANCE FIX COMPLETED SUCCESSFULLY!")
            print("✅ Negative balance corrected")
            print("✅ All sources synchronized")
            print("✅ Single source of truth established")
            sys.exit(0)
        else:
            print("\n⚠️  Balance sources may still be inconsistent")
            sys.exit(1)
    else:
        print("\n❌ Failed to fix negative balance")
        sys.exit(1)
