#!/usr/bin/env python3
"""
Test script to manually add a position and test the sell functionality
"""

import json
import time
from pathlib import Path

# Add a test position to the positions.log file
def add_test_position():
    """Add a test position to simulate the bot having an open position"""
    positions_file = Path('crypto_bot/logs/positions.log')
    
    # Create a test position entry
    test_position = f"{time.strftime('%Y-%m-%d %H:%M:%S,000')} - INFO - Active BTC/USDT buy 0.0500 entry 50000.000000 current 50100.000000 pnl $5.00 (positive) balance $10000.00\n"
    
    with open(positions_file, 'a') as f:
        f.write(test_position)
    
    print("Added test position: BTC/USDT buy 0.05 @ $50,000")

# Test the sell functionality
def test_sell_functionality():
    """Test the complete sell workflow"""
    import requests
    
    # Add test position
    add_test_position()
    
    # Wait a moment for the position to be read
    time.sleep(2)
    
    # Check if position appears in frontend
    print("\n1. Checking if position appears in frontend...")
    response = requests.get('http://localhost:8000/api/open-positions')
    positions = response.json()
    print(f"Frontend positions: {positions}")
    
    if positions:
        # Test selling the position
        print("\n2. Testing sell request...")
        sell_response = requests.post('http://localhost:8000/api/sell-position', 
                                    json={'symbol': 'BTC/USDT', 'amount': 0.05})
        print(f"Sell response: {sell_response.json()}")
        
        # Wait for processing
        print("\n3. Waiting for sell processing...")
        time.sleep(5)
        
        # Check if position was removed
        print("\n4. Checking if position was removed...")
        response = requests.get('http://localhost:8000/api/open-positions')
        positions_after = response.json()
        print(f"Positions after sell: {positions_after}")
        
        if not positions_after:
            print("✅ SUCCESS: Position was successfully sold and removed!")
        else:
            print("❌ FAILED: Position was not removed")
    else:
        print("❌ FAILED: No positions found in frontend")

if __name__ == "__main__":
    test_sell_functionality()
