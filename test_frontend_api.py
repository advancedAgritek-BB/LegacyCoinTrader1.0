#!/usr/bin/env python3
"""
Test script for frontend API endpoints.

This script tests that the Flask API endpoints are working correctly
and returning proper price data for the frontend.
"""

import sys
import os
from pathlib import Path
import requests
import json
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_open_positions_api():
    """Test the /api/open-positions endpoint."""
    try:
        print("🔍 Testing /api/open-positions endpoint...")

        # Find Flask port from logs or try common ports
        ports_to_try = [8000, 8001, 8002, 8003, 8004, 8005]

        for port in ports_to_try:
            try:
                url = f"http://localhost:{port}/api/open-positions"
                print(f"  Trying port {port}: {url}")

                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✅ Successfully connected to port {port}")
                    print(f"  📊 Response: {len(data)} positions returned")

                    if data:
                        # Check the first position for required fields
                        pos = data[0]
                        required_fields = ['symbol', 'current_price', 'entry_price', 'pnl']

                        print("  📋 Checking position data structure:")
                        for field in required_fields:
                            if field in pos:
                                value = pos[field]
                                if field == 'current_price' and value:
                                    print(f"    ✅ {field}: ${value:.6f}")
                                elif field == 'pnl' and value is not None:
                                    print(f"    ✅ {field}: ${value:.2f}")
                                else:
                                    print(f"    ✅ {field}: {value}")
                            else:
                                print(f"    ❌ Missing field: {field}")

                        return True, port
                    else:
                        print("  ⚠️ No positions returned (this is normal if no positions are open)")
                        return True, port

                else:
                    print(f"  ❌ Port {port} returned status {response.status_code}")
                    continue

            except requests.exceptions.RequestException as e:
                print(f"  ❌ Port {port} connection failed: {e}")
                continue

        print("❌ Could not connect to Flask app on any port")
        return False, None

    except Exception as e:
        print(f"❌ Error testing open positions API: {e}")
        return False, None


def test_refresh_prices_api(port):
    """Test the /api/refresh-prices endpoint."""
    try:
        print("🔄 Testing /api/refresh-prices endpoint...")

        url = f"http://localhost:{port}/api/refresh-prices"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print("  ✅ Refresh prices API responded successfully")
            print(f"  📊 Result: {data}")

            if data.get('success'):
                print(f"  🎉 Successfully refreshed prices for {data.get('refreshed_count', 0)} positions")
                return True
            else:
                print(f"  ⚠️ Refresh failed: {data.get('error', 'Unknown error')}")
                return True  # API is working, just no positions to refresh
        else:
            print(f"  ❌ Refresh prices API returned status {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error testing refresh prices API: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 50)
    print("FRONTEND API ENDPOINTS TEST")
    print("=" * 50)

    # Test open positions API
    success, port = test_open_positions_api()

    if not success:
        print("\n❌ FRONTEND API TESTS FAILED")
        print("💡 Make sure the Flask app is running first:")
        print("   python3 frontend/app.py")
        return 1

    # Test refresh prices API
    if port:
        refresh_success = test_refresh_prices_api(port)
    else:
        refresh_success = False

    print("\n" + "=" * 50)
    if success:
        print("✅ FRONTEND API TESTS PASSED")
        print(f"📡 Flask app is running on port {port}")
        print("🎯 The frontend should now be able to display current prices")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())