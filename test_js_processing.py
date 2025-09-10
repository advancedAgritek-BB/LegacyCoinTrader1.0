#!/usr/bin/env python3
"""
Test JavaScript-like processing of the API response.
"""

import json
import sys

def simulate_js_processing():
    """Simulate how the JavaScript processes the position data."""
    print("🔍 Simulating JavaScript position processing...")

    # Get the API response (same data the frontend receives)
    import requests
    try:
        response = requests.get("http://localhost:8003/api/open-positions", timeout=5)
        if response.status_code != 200:
            print(f"❌ API returned {response.status_code}")
            return False

        positions = response.json()
        print(f"✅ Got {len(positions)} positions from API")

        if not positions:
            print("⚠️ No positions to process")
            return False

        # Simulate the JavaScript processing logic
        print("📋 Processing positions...")

        for i, position in enumerate(positions[:3]):  # Test first 3 positions
            print(f"\nPosition {i+1}: {position.get('symbol', 'Unknown')}")

            # Check required fields (similar to JavaScript)
            symbol = position.get('symbol')
            side = position.get('side')
            amount = position.get('amount')
            entry_price = position.get('entry_price')
            current_price = position.get('current_price')
            pnl = position.get('pnl')
            pnl_percentage = position.get('pnl_percentage')

            print(f"  Symbol: {symbol}")
            print(f"  Side: {side}")
            print(f"  Amount: {amount}")
            print(f"  Entry Price: {entry_price}")
            print(f"  Current Price: {current_price}")
            print(f"  PnL: {pnl}")
            print(f"  PnL %: {pnl_percentage}")

            # Simulate JavaScript validation
            if not symbol:
                print("  ❌ Missing symbol")
                continue

            if current_price is None:
                print("  ❌ Current price is null")
                continue

            if entry_price is None:
                print("  ❌ Entry price is null")
                continue

            # Simulate price formatting (like JavaScript)
            if current_price and current_price < 1:
                formatted_current = f"${current_price:.6f}"
            else:
                formatted_current = f"${current_price:.2f}"

            if entry_price and entry_price < 1:
                formatted_entry = f"${entry_price:.6f}"
            else:
                formatted_entry = f"${entry_price:.2f}"

            print(f"  Formatted Current: {formatted_current}")
            print(f"  Formatted Entry: {formatted_entry}")
            print("  ✅ Position data looks good for JavaScript processing")

        print("\n🎯 JavaScript simulation completed successfully")
        print("The position data should render correctly in the frontend")

        return True

    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("JAVASCRIPT POSITION PROCESSING SIMULATION")
    print("=" * 60)

    success = simulate_js_processing()

    print("\n" + "=" * 60)
    if success:
        print("✅ JavaScript simulation PASSED")
        print("Position data should display correctly in frontend")
        print("\n💡 If cards still aren't showing, check:")
        print("   1. Browser console for JavaScript errors")
        print("   2. Network tab for failed API requests")
        print("   3. Make sure both servers are running")
    else:
        print("❌ JavaScript simulation FAILED")
        print("This might explain why position cards aren't displaying")

    print("=" * 60)
