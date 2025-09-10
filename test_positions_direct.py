#!/usr/bin/env python3

# Test the updated positions endpoint directly
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.api import positions

def test_positions_directly():
    try:
        positions_data = positions()
        print("Positions from updated endpoint:")
        for pos in positions_data:
            if "TREMP" in pos['symbol']:
                print(f"TREMP position from API:")
                print(f"  Symbol: {pos['symbol']}")
                print(f"  Side: {pos['side']}")
                print(f"  Amount: {pos['amount']}")
                print(f"  Entry Price: ${pos['entry_price']:.6f}")
                print(f"  Current Price: ${pos['current_price']:.6f}")
                print(f"  P&L: ${pos['pnl']:.2f}")
                
                # Check if entry price line should be above or below current price
                if pos['entry_price'] > pos['current_price']:
                    print(f"  Entry price is ABOVE current price (should show negative P&L)")
                else:
                    print(f"  Entry price is BELOW current price (should show positive P&L)")
    except Exception as e:
        print(f"Error testing endpoint directly: {e}")

if __name__ == "__main__":
    test_positions_directly()
