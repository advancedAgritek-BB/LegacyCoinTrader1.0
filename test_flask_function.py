#!/usr/bin/env python3
"""
Test the get_open_positions function directly by importing it.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_function_directly():
    """Test the get_open_positions function by importing it directly."""
    print("🔍 Testing get_open_positions function directly...")

    try:
        # Import the function
        from frontend.app import get_open_positions
        print("✅ Successfully imported get_open_positions")

        # Call the function
        print("📡 Calling get_open_positions()...")
        positions = get_open_positions()

        print(f"📊 Function returned {len(positions)} positions")

        if positions:
            # Show details of first position
            pos = positions[0]
            print(f"📋 First position details:")
            print(f"   Symbol: {pos.get('symbol')}")
            print(f"   Current Price: {pos.get('current_price')}")
            print(f"   Entry Price: {pos.get('entry_price')}")
            print(f"   PnL: {pos.get('pnl')}")

            # Count positions with null current_price
            null_count = sum(1 for p in positions if p.get('current_price') is None)
            print(f"⚠️ Positions with null current_price: {null_count}/{len(positions)}")

            if null_count == 0:
                print("✅ SUCCESS: All positions have current prices!")
                return True
            else:
                print("❌ ISSUE: Some positions still have null current_price")
                return False
        else:
            print("⚠️ No positions returned")
            return False

    except Exception as e:
        print(f"❌ Error testing function: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_function_directly()
    if success:
        print("\n🎉 SUCCESS: get_open_positions is working correctly!")
    else:
        print("\n❌ FAILURE: get_open_positions has issues")
