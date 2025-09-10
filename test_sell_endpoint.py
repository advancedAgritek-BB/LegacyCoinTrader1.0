#!/usr/bin/env python3
"""
Test script to verify the sell position endpoint functionality.
"""

import requests
import json
import time

def test_sell_position_endpoint():
    """Test the sell position endpoint with mock data."""

    # First, let's check if the endpoint exists by making a test request
    base_url = "http://localhost:8000"

    try:
        # Test 1: Check if endpoint responds
        print("Testing sell position endpoint...")

        # Make a request to the sell position endpoint with test data
        test_data = {
            'symbol': 'BTC/USD',
            'amount': 0.01
        }

        response = requests.post(
            f"{base_url}/api/sell-position",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            print(f"Response data: {json.dumps(result, indent=2)}")

            if result.get('success'):
                print("✅ Sell position endpoint is working correctly!")
                return True
            else:
                print(f"❌ Sell position failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"Response text: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask server. Make sure it's running on port 8000.")
        return False
    except Exception as e:
        print(f"❌ Error testing sell position endpoint: {e}")
        return False

def test_open_positions_endpoint():
    """Test the open positions endpoint to see current positions."""

    base_url = "http://localhost:8000"

    try:
        print("\nTesting open positions endpoint...")

        response = requests.get(f"{base_url}/api/open-positions", timeout=10)

        if response.status_code == 200:
            positions = response.json()
            print(f"Found {len(positions)} open positions")

            if positions:
                print("Sample position data:")
                for i, pos in enumerate(positions[:2]):  # Show first 2 positions
                    print(f"  {i+1}. {pos.get('symbol')} - Entry: ${pos.get('entry_price', 'N/A')}, Current: ${pos.get('current_price', 'N/A')}")
            else:
                print("No open positions found")

            return True
        else:
            print(f"❌ Failed to get positions: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error testing open positions endpoint: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Sell Position Functionality ===\n")

    # Test open positions first
    positions_ok = test_open_positions_endpoint()

    if positions_ok:
        # Test sell position
        sell_ok = test_sell_position_endpoint()

        if sell_ok:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Sell position test failed!")
    else:
        print("\n❌ Cannot test sell functionality without positions data!")
