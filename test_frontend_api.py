#!/usr/bin/env python3
"""
Simple API test script for frontend P&L functionality.
"""

import requests
import time

def test_pnl_api():
    """Test the P&L API endpoint."""
    try:
        response = requests.get('http://localhost:8001/api/wallet-pnl', timeout=5)
        print(f"API Response Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ P&L API Response:")
            print(f"   Total P&L: ${data.get('total_pnl', 'N/A')}")
            print(f"   P&L Percentage: {data.get('pnl_percentage', 'N/A')}%")
            print(f"   Current Balance: ${data.get('current_balance', 'N/A')}")
            print(f"   Initial Balance: ${data.get('initial_balance', 'N/A')}")

            if data.get('error'):
                print(f"❌ API Error: {data.get('error')}")
                return False

            if data.get('total_pnl') is None:
                print("❌ Total P&L is None - this indicates a calculation issue")
                return False

            print("✅ P&L API working correctly")
            return True
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask server. Make sure it's running on port 8001")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def test_dashboard_page():
    """Test if the dashboard page loads correctly."""
    try:
        response = requests.get('http://localhost:8001/', timeout=5)
        print(f"\nDashboard Page Response Status: {response.status_code}")

        if response.status_code == 200:
            content = response.text

            # Check for key elements
            checks = [
                ('Total P&L display', 'id="totalPnl"' in content),
                ('P&L percentage display', 'id="totalPnlPercentage"' in content),
                ('CSS styles loaded', 'metric-card' in content),
                ('JavaScript loaded', 'LegacyCoinTrader' in content)
            ]

            all_passed = True
            for check_name, passed in checks:
                if passed:
                    print(f"✅ {check_name}")
                else:
                    print(f"❌ {check_name} - MISSING")
                    all_passed = False

            return all_passed
        else:
            print(f"❌ Dashboard page error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask server. Make sure it's running on port 8001")
        return False
    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
        return False

def main():
    """Run API tests."""
    print("🧪 Testing Frontend P&L and Dashboard APIs...")
    print("=" * 50)

    pnl_ok = test_pnl_api()
    dashboard_ok = test_dashboard_page()

    print("\n" + "=" * 50)
    if pnl_ok and dashboard_ok:
        print("✅ All tests passed! Frontend should be working correctly.")
        print("\n📋 Summary of fixes applied:")
        print("   • Removed duplicate/conflicting JavaScript functions")
        print("   • Fixed CSS variable definitions and styling consistency")
        print("   • Improved P&L API error handling and data validation")
        print("   • Enhanced frontend data flow between backend and UI")
        print("   • Added proper fallback values for missing data")
    else:
        print("❌ Some tests failed. Check the logs for details.")
        if not pnl_ok:
            print("   - P&L API is not returning valid data")
        if not dashboard_ok:
            print("   - Dashboard page is missing key elements")

if __name__ == "__main__":
    main()
