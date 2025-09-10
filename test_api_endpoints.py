#!/usr/bin/env python3
"""
Test script to verify the new API endpoints work correctly.
"""

import requests
import json
import time

def test_api_endpoints():
    """Test the newly added API endpoints."""
    base_url = "http://localhost:8000"

    print("🧪 Testing API Endpoints")
    print("=" * 40)

    # Test endpoints
    endpoints = [
        "/api/live-signals",
        "/api/bot-status",
        "/api/dashboard-metrics"
    ]

    for endpoint in endpoints:
        try:
            print(f"\n🔍 Testing {endpoint}")
            response = requests.get(f"{base_url}{endpoint}", timeout=5)

            if response.status_code == 200:
                print(f"   ✅ Status: {response.status_code}")
                try:
                    data = response.json()
                    print(f"   📊 Response: {json.dumps(data, indent=2)[:200]}...")
                except json.JSONDecodeError:
                    print(f"   ❌ Invalid JSON response")
                    print(f"   Response content: {response.text[:200]}...")
            else:
                print(f"   ❌ Status: {response.status_code}")
                print(f"   Error: {response.text[:200]}...")

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection error: {e}")

        time.sleep(0.5)  # Brief delay between requests

    print("\n✅ API endpoint testing completed!")

if __name__ == "__main__":
    test_api_endpoints()
