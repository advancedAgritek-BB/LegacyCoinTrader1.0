#!/usr/bin/env python3
"""
Test script to verify dashboard refresh functionality
"""

import requests
import json
import time
from datetime import datetime

def test_refresh_functionality():
    """Test the dashboard refresh endpoints and data processing."""

    base_url = "http://localhost:5050"

    print("Testing dashboard refresh functionality...")
    print("=" * 50)

    try:
        # Test 1: Get open positions
        print("1. Testing /api/open-positions endpoint...")
        response = requests.get(f"{base_url}/api/open-positions")
        if response.status_code == 200:
            positions = response.json()
            print(f"   ✓ Got {len(positions)} positions")
            if positions:
                sample_pos = positions[0]
                print(f"   Sample position: {sample_pos.get('symbol')} - P&L: {sample_pos.get('pnl', 'N/A')}% (${sample_pos.get('pnl_value', 'N/A')})")
        else:
            print(f"   ✗ Failed with status {response.status_code}")
            return False

        # Test 2: Get portfolio P&L
        print("\n2. Testing /api/portfolio/pnl endpoint...")
        response = requests.get(f"{base_url}/api/portfolio/pnl")
        if response.status_code == 200:
            pnl_data = response.json()
            print(f"   ✓ P&L data: {json.dumps(pnl_data, indent=2)}")
        else:
            print(f"   ✗ Failed with status {response.status_code}")
            return False

        # Test 3: Test refresh endpoint
        print("\n3. Testing /api/refresh-dashboard endpoint...")
        response = requests.post(f"{base_url}/api/refresh-dashboard")
        if response.status_code == 200:
            refresh_result = response.json()
            print(f"   ✓ Refresh result: {json.dumps(refresh_result, indent=2)}")
        else:
            print(f"   ✗ Failed with status {response.status_code}")
            return False

        # Test 4: Verify data consistency
        print("\n4. Verifying data consistency...")
        # Get positions again after refresh
        response = requests.get(f"{base_url}/api/open-positions")
        if response.status_code == 200:
            positions_after = response.json()
            if len(positions_after) == len(positions):
                print(f"   ✓ Position count consistent: {len(positions_after)}")
                if positions_after:
                    sample_pos_after = positions_after[0]
                    sample_pos_before = positions[0]
                    pnl_before = sample_pos_before.get('pnl', 0)
                    pnl_after = sample_pos_after.get('pnl', 0)
                    print(f"   Sample position P&L: {pnl_before}% -> {pnl_after}%")
                    if pnl_before != pnl_after:
                        print("   ⚠ P&L changed - this indicates refresh is working!")
                    else:
                        print("   ℹ P&L unchanged (price may not have moved)")
            else:
                print(f"   ⚠ Position count changed: {len(positions)} -> {len(positions_after)}")
        else:
            print(f"   ✗ Failed to get positions after refresh")
            return False

        print("\n" + "=" * 50)
        print("✓ All tests passed! Dashboard refresh functionality is working.")
        return True

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_refresh_functionality()
    exit(0 if success else 1)
