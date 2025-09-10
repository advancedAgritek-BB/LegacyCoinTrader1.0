#!/usr/bin/env python3
"""
Test script to verify market sell functionality
"""

import requests
import json
import time

def test_market_sell_api():
    """Test the market sell API endpoint"""
    
    # Test data
    test_data = {
        'symbol': 'BTC/USD',
        'amount': 0.001
    }
    
    print(f"Testing market sell API with data: {test_data}")
    
    try:
        # Make the API request
        response = requests.post(
            'http://localhost:8000/api/sell-position',
            headers={'Content-Type': 'application/json'},
            json=test_data,
            timeout=10
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response data: {json.dumps(data, indent=2)}")
            
            if data.get('success'):
                print("✅ Market sell API test PASSED")
                return True
            else:
                print(f"❌ Market sell API test FAILED: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Market sell API test FAILED: HTTP {response.status_code}")
            print(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Market sell API test FAILED: Could not connect to server")
        print("Make sure the Flask app is running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Market sell API test FAILED: {e}")
        return False

def test_sell_requests_file():
    """Test if sell requests are being written to the file"""
    
    import os
    from pathlib import Path
    
    # Check if the sell requests file exists
    log_dir = Path('crypto_bot/logs')
    sell_requests_file = log_dir / 'sell_requests.json'
    
    print(f"\nChecking sell requests file: {sell_requests_file}")
    
    if sell_requests_file.exists():
        try:
            with open(sell_requests_file, 'r') as f:
                requests = json.load(f)
            
            print(f"Found {len(requests)} sell requests in file")
            for i, req in enumerate(requests):
                print(f"  Request {i+1}: {req}")
            
            return True
        except Exception as e:
            print(f"Error reading sell requests file: {e}")
            return False
    else:
        print("Sell requests file does not exist")
        return False

if __name__ == '__main__':
    print("Testing Market Sell Functionality")
    print("=" * 40)
    
    # Test 1: API endpoint
    api_success = test_market_sell_api()
    
    # Test 2: Check if requests are being written to file
    file_success = test_sell_requests_file()
    
    print("\n" + "=" * 40)
    if api_success and file_success:
        print("✅ All tests PASSED - Market sell functionality is working")
    else:
        print("❌ Some tests FAILED - Check the issues above")
