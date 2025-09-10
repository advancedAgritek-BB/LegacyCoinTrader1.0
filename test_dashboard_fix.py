#!/usr/bin/env python3
"""Test the dashboard balance fix."""

import requests

def test_dashboard_balance():
    """Test the dashboard balance fix."""

    print("=" * 50)
    print("TESTING DASHBOARD BALANCE FIX")
    print("=" * 50)

    try:
        response = requests.get('http://localhost:8000/api/dashboard-metrics', timeout=10)
        if response.status_code == 200:
            data = response.json()

            print("✅ Dashboard API Response:")
            print(f"  Current Balance: ${data.get('current_balance', 'NOT FOUND')}")
            print(f"  Balance Source: {data.get('balance_source', 'NOT FOUND')}")

            if 'performance' in data and data['performance']:
                perf = data['performance']
                print(f"  Performance Current: ${perf.get('current_balance', 'NOT FOUND')}")
                print(f"  Performance Initial: ${perf.get('initial_balance', 'NOT FOUND')}")
                print(f"  Performance PnL: ${perf.get('total_pnl', 'NOT FOUND')}")

            # Check if balance is reasonable
            current_balance = data.get('current_balance')
            if current_balance is not None:
                if current_balance > 0 and current_balance < 100000:
                    print("✅ Balance appears valid")
                else:
                    print("❌ Balance appears invalid")
            else:
                print("❌ Balance not found in response")

        else:
            print(f"❌ API Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_dashboard_balance()
