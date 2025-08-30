#!/usr/bin/env python3
"""Test script to verify the new frontend routes are working."""

import requests
import json
import time

def test_routes():
    """Test the new API configuration and config settings routes."""
    base_url = "http://localhost:8000"
    
    print("Testing new frontend routes...")
    
    # Test API config page
    try:
        response = requests.get(f"{base_url}/api_config", timeout=5)
        if response.status_code == 200:
            print("✅ API Config page accessible")
        else:
            print(f"❌ API Config page failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API Config page error: {e}")
    
    # Test config settings page
    try:
        response = requests.get(f"{base_url}/config_settings", timeout=5)
        if response.status_code == 200:
            print("✅ Config Settings page accessible")
        else:
            print(f"❌ Config Settings page failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Config Settings page error: {e}")
    
    # Test save API config endpoint
    try:
        test_data = {
            "exchange": "kraken",
            "mode": "cex",
            "kraken_api_key": "test_key",
            "kraken_api_secret": "test_secret"
        }
        response = requests.post(
            f"{base_url}/api/save_api_config",
            json=test_data,
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                print("✅ Save API Config endpoint working")
            else:
                print(f"❌ Save API Config failed: {result.get('message')}")
        else:
            print(f"❌ Save API Config endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Save API Config endpoint error: {e}")
    
    # Test save config settings endpoint
    try:
        test_data = {
            "timeframe": "15m",
            "stop_loss_pct": 0.008,
            "take_profit_pct": 0.045,
            "strategy_allocation": {
                "trend_bot": 15,
                "bounce_scalper": 15,
                "grid_bot": 15,
                "sniper_bot": 25,
                "micro_scalp_bot": 30
            }
        }
        response = requests.post(
            f"{base_url}/api/save_config_settings",
            json=test_data,
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                print("✅ Save Config Settings endpoint working")
            else:
                print(f"❌ Save Config Settings failed: {result.get('message')}")
        else:
            print(f"❌ Save Config Settings endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Save Config Settings endpoint error: {e}")
    
    # Test refresh config endpoint
    try:
        response = requests.post(f"{base_url}/api/refresh_config", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                print("✅ Refresh Config endpoint working")
            else:
                print(f"❌ Refresh Config failed: {result.get('message')}")
        else:
            print(f"❌ Refresh Config endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Refresh Config endpoint error: {e}")

if __name__ == "__main__":
    print("Starting frontend route tests...")
    print("Make sure the frontend server is running on http://localhost:8000")
    print()
    
    test_routes()
    
    print("\nTest completed!")
