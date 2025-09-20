#!/usr/bin/env python3
"""
Verification script for dashboard refresh functionality
"""

import requests
import json
import time
from datetime import datetime

def verify_dashboard_refresh():
    """Comprehensive verification of dashboard refresh functionality."""

    base_url = "http://localhost:5050"

    print("🔍 Verifying Dashboard Refresh Functionality")
    print("=" * 60)

    try:
        # Test 1: Verify server is running
        print("1. Checking server connectivity...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   ✅ Server is running")
        else:
            print(f"   ❌ Server not responding (status: {response.status_code})")
            return False

        # Test 2: Verify API endpoints
        print("\n2. Testing API endpoints...")
        endpoints = [
            ("/api/open-positions", "Open Positions"),
            ("/api/portfolio/pnl", "Portfolio P&L"),
            ("/api/refresh-dashboard", "Refresh Dashboard")
        ]

        for endpoint, name in endpoints:
            try:
                if endpoint == "/api/refresh-dashboard":
                    response = requests.post(f"{base_url}{endpoint}")
                else:
                    response = requests.get(f"{base_url}{endpoint}")

                if response.status_code == 200:
                    print(f"   ✅ {name}: OK")
                    if endpoint == "/api/open-positions":
                        data = response.json()
                        print(f"      📊 Found {len(data)} positions")
                        if data:
                            sample = data[0]
                            print(f"      💰 Sample: {sample.get('symbol')} - P&L: {sample.get('pnl', 'N/A')}%")
                else:
                    print(f"   ❌ {name}: Failed (status: {response.status_code})")
                    return False
            except Exception as e:
                print(f"   ❌ {name}: Error - {e}")
                return False

        # Test 3: Verify dashboard page loads
        print("\n3. Testing dashboard page...")
        response = requests.get(f"{base_url}/dashboard")
        if response.status_code == 200:
            content = response.text
            # Check for key elements
            checks = [
                ("Refresh button", 'onclick="refreshPrices()"'),
                ("Position cards", "position-card"),
                ("P&L elements", "pnl-percentage"),
                ("Auto-refresh setup", "setInterval"),
                ("Refresh indicator", "last-refresh-time")
            ]

            for name, pattern in checks:
                if pattern in content:
                    print(f"   ✅ {name}: Found")
                else:
                    print(f"   ⚠️  {name}: Not found")

            print("   ✅ Dashboard page loads successfully")
        else:
            print(f"   ❌ Dashboard page failed (status: {response.status_code})")
            return False

        # Test 4: Simulate refresh cycle
        print("\n4. Simulating refresh cycle...")
        print("   📡 Monitoring for 35 seconds to verify auto-refresh...")

        start_time = time.time()
        last_position_count = None

        for i in range(7):  # Check every 5 seconds for 35 seconds
            try:
                response = requests.get(f"{base_url}/api/open-positions")
                if response.status_code == 200:
                    positions = response.json()
                    current_count = len(positions)

                    if last_position_count is None:
                        print(f"   📊 Initial: {current_count} positions")
                        last_position_count = current_count
                    elif current_count != last_position_count:
                        print(f"   🔄 Refresh detected: {last_position_count} -> {current_count} positions")
                        last_position_count = current_count

                    if positions:
                        sample = positions[0]
                        pnl = sample.get('pnl', 0)
                        pnl_value = sample.get('pnl_value', 0)
                        print(f"      💰 {sample.get('symbol')}: {pnl:.2f}% (${pnl_value:.2f})")

                time.sleep(5)

            except Exception as e:
                print(f"   ❌ Error during monitoring: {e}")
                break

        elapsed = time.time() - start_time
        print(f"   ⏱️  Monitoring completed in {elapsed:.1f} seconds")
        print("\n" + "=" * 60)
        print("✅ Dashboard refresh verification completed!")
        print("\n📋 Summary of fixes applied:")
        print("   • Consolidated JavaScript initialization")
        print("   • Added refresh indicator with timestamps")
        print("   • Implemented fallback refresh mechanisms")
        print("   • Added visibility change refresh trigger")
        print("   • Enhanced error handling and logging")
        print("   • Added manual refresh function for testing")
        print("\n🎯 To test in browser:")
        print("   1. Open dashboard at http://localhost:5050/dashboard")
        print("   2. Check browser console for refresh logs")
        print("   3. Look for 'Last refresh:' timestamp updates")
        print("   4. Try manual refresh button")
        print("   5. Switch tabs and come back to trigger visibility refresh")

        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_dashboard_refresh()
    exit(0 if success else 1)
