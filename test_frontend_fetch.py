#!/usr/bin/env python3
"""
Test the frontend API fetch to see if there are any issues.
"""

import requests
import json
from urllib.parse import urlencode

def test_frontend_api_call():
    """Test the API call that the frontend makes."""
    print("🔍 Testing frontend API call...")

    # This is what the frontend is trying to call
    url = "http://localhost:8003/api/open-positions"
    params = {'_': str(int(time.time() * 1000))}  # Cache buster

    full_url = f"{url}?{urlencode(params)}"
    print(f"Frontend API call: {full_url}")

    try:
        response = requests.get(full_url, timeout=10)

        print(f"Status: {response.status_code}")
        print(f"Response time: {response.elapsed.total_seconds()} seconds")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Received {len(data)} positions")

            # Check data structure
            if data and len(data) > 0:
                sample = data[0]
                required_fields = ['symbol', 'current_price', 'entry_price', 'pnl']
                print("📋 Checking data structure:")

                for field in required_fields:
                    if field in sample:
                        value = sample[field]
                        print(f"  ✅ {field}: {value}")
                    else:
                        print(f"  ❌ Missing: {field}")

                return True, data
            else:
                print("⚠️ No position data received")
                return False, None
        else:
            print(f"❌ API returned error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False, None

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False, None

def test_cors_headers():
    """Test CORS headers."""
    print("\n🔍 Testing CORS headers...")

    try:
        response = requests.options("http://localhost:8003/api/open-positions")
        print(f"CORS preflight status: {response.status_code}")

        if 'Access-Control-Allow-Origin' in response.headers:
            print(f"✅ CORS header present: {response.headers['Access-Control-Allow-Origin']}")
        else:
            print("⚠️ No CORS headers found")

    except Exception as e:
        print(f"CORS test failed: {e}")

if __name__ == "__main__":
    import time

    print("=" * 50)
    print("FRONTEND API FETCH TEST")
    print("=" * 50)

    success, data = test_frontend_api_call()
    test_cors_headers()

    print("\n" + "=" * 50)
    if success:
        print("✅ Frontend API call test PASSED")
        print("The API is working correctly for frontend consumption")
    else:
        print("❌ Frontend API call test FAILED")
        print("This might explain why position cards aren't displaying")

    print("=" * 50)
