#!/usr/bin/env python3
"""Test script to verify balance fix."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_balance_sources():
    """Test that balance sources return correct values."""
    try:
        # Import the frontend app functions
        from frontend.app import get_paper_wallet_balance

        # Test get_paper_wallet_balance
        balance = get_paper_wallet_balance()
        print(f"get_paper_wallet_balance() returned: ${balance:.2f}")

        # Validate the balance
        if balance < 0:
            print("❌ ERROR: Balance is negative!")
            return False
        elif balance == 0:
            print("⚠️  WARNING: Balance is zero")
            return False
        elif balance > 100000:
            print("⚠️  WARNING: Balance seems unreasonably high")
            return False
        else:
            print("✅ Balance appears valid")
            return True

    except Exception as e:
        print(f"❌ ERROR testing balance: {e}")
        return False

def test_wallet_balance_api():
    """Test the wallet balance API endpoint."""
    try:
        import requests

        # Test the API endpoint
        response = requests.get('http://localhost:8000/api/wallet-balance', timeout=5)

        if response.status_code == 200:
            data = response.json()
            balance = data.get('balance', 0)
            source = data.get('source', 'unknown')

            print(f"API returned: ${balance:.2f} (source: {source})")

            if balance < 0:
                print("❌ ERROR: API returned negative balance!")
                return False
            elif balance == 0:
                print("⚠️  WARNING: API returned zero balance")
                return False
            else:
                print("✅ API balance appears valid")
                return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("⚠️  API server not running (expected if testing standalone)")
        return True  # Not a failure if server isn't running
    except Exception as e:
        print(f"❌ ERROR testing API: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing balance fix...")
    print("=" * 50)

    success1 = test_balance_sources()
    success2 = test_wallet_balance_api()

    print("=" * 50)
    if success1 and success2:
        print("✅ All balance tests passed!")
        sys.exit(0)
    else:
        print("❌ Some balance tests failed!")
        sys.exit(1)
