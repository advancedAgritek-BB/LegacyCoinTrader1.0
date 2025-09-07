#!/usr/bin/env python3

import requests
import json

def test_positions_endpoint():
    try:
        response = requests.get('http://localhost:5001/positions')
        if response.status_code == 200:
            positions = response.json()
            print("Positions from API:")
            for pos in positions:
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
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error testing endpoint: {e}")

if __name__ == "__main__":
    test_positions_endpoint()
