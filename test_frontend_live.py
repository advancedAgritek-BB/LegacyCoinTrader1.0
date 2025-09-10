#!/usr/bin/env python3
"""
Test script for frontend live dashboard functionality.
This script tests the real-time updates and wallet balance configuration.
"""

import requests
import time
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_WALLET_BALANCE = 15000.0

def test_api_endpoints():
    """Test all the new API endpoints."""
    print("Testing API endpoints...")
    
    # Test paper wallet balance API
    print("\n1. Testing paper wallet balance API...")
    try:
        # Get current balance
        response = requests.get(f"{BASE_URL}/api/paper-wallet-balance")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Current balance: ${data.get('balance', 0):.2f}")
        else:
            print(f"   ✗ Failed to get balance: {response.status_code}")
            return False
        
        # Set new balance
        response = requests.post(
            f"{BASE_URL}/api/paper-wallet-balance",
            json={"balance": TEST_WALLET_BALANCE}
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"   ✓ Balance updated to: ${data.get('balance', 0):.2f}")
            else:
                print(f"   ✗ Failed to update balance: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"   ✗ Failed to update balance: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ✗ Connection failed - make sure the frontend is running")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test live updates API
    print("\n2. Testing live updates API...")
    try:
        response = requests.get(f"{BASE_URL}/api/live-updates")
        if response.status_code == 200:
            data = response.json()
            if 'timestamp' in data and 'bot_status' in data:
                print("   ✓ Live updates API working")
                print(f"   ✓ Bot status: {data['bot_status'].get('mode', 'Unknown')}")
                print(f"   ✓ Paper wallet balance: ${data.get('paper_wallet_balance', 0):.2f}")
            else:
                print("   ✗ Live updates API returned unexpected data structure")
                return False
        else:
            print(f"   ✗ Live updates API failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test dashboard metrics API
    print("\n3. Testing dashboard metrics API...")
    try:
        response = requests.get(f"{BASE_URL}/api/dashboard-metrics")
        if response.status_code == 200:
            data = response.json()
            if 'performance' in data and 'bot_status' in data:
                print("   ✓ Dashboard metrics API working")
                print(f"   ✓ Performance data available: {len(data.get('performance', {}))} metrics")
            else:
                print("   ✗ Dashboard metrics API returned unexpected data structure")
                return False
        else:
            print(f"   ✗ Dashboard metrics API failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    return True

def test_config_files():
    """Test configuration file creation and updates."""
    print("\n4. Testing configuration files...")
    
    # Check if paper wallet config was created
    paper_wallet_config = Path("crypto_bot/paper_wallet_config.yaml")
    if paper_wallet_config.exists():
        print("   ✓ Paper wallet config file exists")
        
        # Check if balance was updated
        try:
            import yaml
            with open(paper_wallet_config) as f:
                config = yaml.safe_load(f)
                balance = config.get('initial_balance', 0)
                if abs(balance - TEST_WALLET_BALANCE) < 0.01:
                    print(f"   ✓ Balance correctly updated to ${balance:.2f}")
                else:
                    print(f"   ✗ Balance not updated correctly: ${balance:.2f}")
                    return False
        except Exception as e:
            print(f"   ✗ Error reading config: {e}")
            return False
    else:
        print("   ✗ Paper wallet config file not created")
        return False
    
    return True

def test_real_time_updates():
    """Test real-time update functionality."""
    print("\n5. Testing real-time updates...")
    
    try:
        # Get initial data
        response1 = requests.get(f"{BASE_URL}/api/live-updates")
        if response1.status_code != 200:
            print("   ✗ Failed to get initial data")
            return False
        
        data1 = response1.json()
        timestamp1 = data1.get('timestamp', 0)
        
        # Wait a moment
        time.sleep(2)
        
        # Get updated data
        response2 = requests.get(f"{BASE_URL}/api/live-updates")
        if response2.status_code != 200:
            print("   ✗ Failed to get updated data")
            return False
        
        data2 = response2.json()
        timestamp2 = data2.get('timestamp', 0)
        
        # Check if timestamp updated
        if timestamp2 > timestamp1:
            print("   ✓ Real-time updates working (timestamps updating)")
        else:
            print("   ✗ Real-time updates not working (timestamps not updating)")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Frontend Live Dashboard Test Suite")
    print("=" * 40)
    
    # Check if frontend is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("✗ Frontend not accessible")
            return False
        print("✓ Frontend is running and accessible")
    except requests.exceptions.ConnectionError:
        print("✗ Frontend not running - start it with: python -m frontend.app")
        return False
    except Exception as e:
        print(f"✗ Error connecting to frontend: {e}")
        return False
    
    # Run tests
    tests = [
        test_api_endpoints,
        test_config_files,
        test_real_time_updates
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n✗ Test failed: {test.__name__}")
            break
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Frontend live dashboard is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
