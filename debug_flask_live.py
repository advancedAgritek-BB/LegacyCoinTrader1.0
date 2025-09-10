#!/usr/bin/env python3
"""
Live debug script to test Flask API directly while it's running.
"""

import requests
import json
import time

def test_api_endpoint():
    """Test the API endpoint and show detailed debugging."""
    print("🔍 Testing Flask API endpoint live...")

    try:
        url = "http://localhost:8000/api/open-positions"
        print(f"Making request to: {url}")

        start_time = time.time()
        response = requests.get(url, timeout=10)
        end_time = time.time()

        print(f"Response status: {response.status_code}")
        print(f"Response time: {end_time - start_time:.2f} seconds")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ API responded successfully with {len(data)} positions")

            if data:
                print("\n📋 First position details:")
                pos = data[0]
                print(f"   Symbol: {pos.get('symbol')}")
                print(f"   Current Price: {pos.get('current_price')}")
                print(f"   Entry Price: {pos.get('entry_price')}")
                print(f"   Side: {pos.get('side')}")

                # Count positions with null current_price
                null_count = sum(1 for p in data if p.get('current_price') is None)
                print(f"\n⚠️ Positions with null current_price: {null_count}/{len(data)}")

                if null_count > 0:
                    print("\n❌ ISSUE: Some positions still have null current_price")
                    print("This means the price fetching is not working in the live Flask app")
                else:
                    print("\n✅ All positions have current prices - issue resolved!")
            else:
                print("⚠️ No positions returned")
        else:
            print(f"❌ API returned error: {response.status_code}")
            print(f"Response text: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'N/A'}")

def test_basic_flask():
    """Test if basic Flask is working."""
    print("\n🔍 Testing basic Flask connectivity...")

    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Basic Flask is responding")
        else:
            print(f"❌ Basic Flask error: {response.status_code}")
    except Exception as e:
        print(f"❌ Basic Flask connection failed: {e}")

def check_env_vars():
    """Check if required environment variables are set."""
    print("\n🔍 Checking environment variables...")

    import os
    required_vars = ['API_KEY', 'API_SECRET']
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Set (length: {len(value)})")
        else:
            print(f"❌ {var}: Not set")

if __name__ == "__main__":
    print("=" * 50)
    print("FLASK LIVE DEBUGGING")
    print("=" * 50)

    check_env_vars()
    test_basic_flask()
    test_api_endpoint()

    print("\n" + "=" * 50)
    print("DEBUG COMPLETE")
    print("=" * 50)
