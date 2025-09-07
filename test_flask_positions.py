#!/usr/bin/env python3

# Test the updated Flask positions endpoint directly
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app context
from frontend.app import app

def test_positions_endpoint():
    with app.test_client() as client:
        try:
            response = client.get('/api/positions')
            if response.status_code == 200:
                positions = response.get_json()
                print("Positions from Flask API:")
                for pos in positions:
                    if "TREMP" in pos['symbol']:
                        print(f"TREMP position from API:")
                        print(f"  Symbol: {pos['symbol']}")
                        print(f"  Side: {pos['side']}")
                        print(f"  Size: {pos['size']}")
                        print(f"  Entry Price: ${pos['entry_price']:.6f}")
                        print(f"  Current Price: ${pos['current_price']:.6f}")
                        print(f"  P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.2f}%)")
                        
                        # Check if entry price line should be above or below current price
                        if pos['entry_price'] > pos['current_price']:
                            print(f"  Entry price is ABOVE current price (should show negative P&L)")
                        else:
                            print(f"  Entry price is BELOW current price (should show positive P&L)")
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.get_data()}")
        except Exception as e:
            print(f"Error testing endpoint: {e}")

if __name__ == "__main__":
    test_positions_endpoint()
