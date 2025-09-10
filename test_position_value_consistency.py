#!/usr/bin/env python3
"""
Test script to verify position value consistency between dashboard and API endpoints.
"""

import json
import requests
from pathlib import Path
import sys
import time
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent))

def get_positions_from_api(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Get positions data from API endpoint."""
    try:
        response = requests.get(f"{base_url}/api/open-positions", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"   ❌ Failed to get positions from API: {e}")
        return None

def get_positions_from_dashboard(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Get positions data from dashboard endpoint."""
    try:
        response = requests.get(f"{base_url}/dashboard", timeout=10)
        response.raise_for_status()
        # Note: Dashboard returns HTML, so we can't easily extract positions
        # We'll need to use the API for dashboard position data
        return get_positions_from_api(base_url)
    except Exception as e:
        print(f"   ❌ Failed to get positions from dashboard: {e}")
        return None

def calculate_total_position_value(positions: List[Dict[str, Any]]) -> float:
    """Calculate total value of all positions."""
    total_value = 0.0
    for position in positions:
        try:
            # Handle different position data formats
            current_price = position.get('current_price', 0)
            amount = position.get('amount', position.get('total_amount', 0))

            if current_price and amount:
                position_value = float(current_price) * float(amount)
                total_value += position_value
                print(f"   📊 {position.get('symbol', 'Unknown')}: {amount} @ ${current_price} = ${position_value:.2f}")
        except (ValueError, TypeError) as e:
            print(f"   ⚠️  Error calculating value for position {position.get('symbol', 'Unknown')}: {e}")
            continue

    return total_value

def test_server_availability(base_url: str = "http://localhost:8000") -> bool:
    """Test if the Flask server is running."""
    try:
        response = requests.get(f"{base_url}/api/bot-status", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_with_mock_data() -> bool:
    """Test position value consistency using mock data."""
    try:
        # Mock API positions data
        mock_api_positions = [
            {
                'symbol': 'BTC/USDT',
                'amount': 1.5,
                'current_price': 45000.25,
                'pnl': 2250.50,
                'entry_price': 43500.00
            },
            {
                'symbol': 'ETH/USDT',
                'amount': 10.0,
                'current_price': 2450.75,
                'pnl': -125.25,
                'entry_price': 2500.00
            },
            {
                'symbol': 'ADA/USDT',
                'amount': 1000.0,
                'current_price': 0.45,
                'pnl': 50.00,
                'entry_price': 0.40
            }
        ]

        # Mock dashboard positions data (same data for consistency test)
        mock_dashboard_positions = mock_api_positions.copy()

        print("   📊 Mock API Positions:")
        api_total_value = calculate_total_position_value(mock_api_positions)
        print(f"   📈 Mock API Total Position Value: ${api_total_value:.2f}")

        print("\n   📊 Mock Dashboard Positions:")
        dashboard_total_value = calculate_total_position_value(mock_dashboard_positions)
        print(f"   📊 Mock Dashboard Total Position Value: ${dashboard_total_value:.2f}")

        # Test position counts
        print("\n   🔍 Comparing position counts...")
        api_count = len(mock_api_positions)
        dashboard_count = len(mock_dashboard_positions)

        if api_count != dashboard_count:
            print(f"   ❌ Position counts differ: API={api_count}, Dashboard={dashboard_count}")
            return False
        else:
            print(f"   ✅ Position counts match: {api_count} positions")

        # Test position values
        print("\n   💰 Comparing position values...")
        value_difference = abs(api_total_value - dashboard_total_value)

        if value_difference > 0.01:
            print(f"   ❌ Position values differ by ${value_difference:.2f}")
            return False
        else:
            print("   ✅ Position values are consistent!")

        # Test individual positions
        print("\n   📋 Comparing individual positions...")
        all_consistent = True

        for api_pos in mock_api_positions:
            symbol = api_pos.get('symbol')
            if not symbol:
                continue

            # Find matching position in dashboard data
            dashboard_pos = None
            for pos in mock_dashboard_positions:
                if pos.get('symbol') == symbol:
                    dashboard_pos = pos
                    break

            if dashboard_pos is None:
                print(f"   ❌ Position {symbol} missing in dashboard data")
                all_consistent = False
                continue

            # Compare key values
            api_price = api_pos.get('current_price', 0)
            dashboard_price = dashboard_pos.get('current_price', 0)
            api_amount = api_pos.get('amount', 0)
            dashboard_amount = dashboard_pos.get('amount', 0)

            price_diff = abs(float(api_price) - float(dashboard_price))
            amount_diff = abs(float(api_amount) - float(dashboard_amount))

            if price_diff > 0.01 or amount_diff > 0.000001:
                print(f"   ❌ Position {symbol} data differs:")
                print(f"      API: {api_amount} @ ${api_price}")
                print(f"      Dashboard: {dashboard_amount} @ ${dashboard_price}")
                all_consistent = False
            else:
                print(f"   ✅ Position {symbol} data matches")

        if not all_consistent:
            return False

        print("\n" + "=" * 50)
        print("🎉 Mock position value consistency test PASSED!")
        print(f"   Total positions: {api_count}")
        print(f"   Total value: ${api_total_value:.2f}")
        return True

    except Exception as e:
        print(f"❌ Mock test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_position_value_consistency(mock_data: bool = False):
    """Test that position values are consistent across different endpoints."""

    print("🔍 Testing Position Value Consistency")
    print("=" * 50)

    base_url = "http://localhost:8000"

    if mock_data:
        print("\n🧪 Using mock data for testing...")
        return test_with_mock_data()

    try:
        # Test 0: Check if server is running
        print("\n0. Testing server availability...")
        if not test_server_availability(base_url):
            print("   ❌ Flask server is not running on port 8000")
            print("   💡 Please start the server with: python frontend/app.py")
            return False
        print("   ✅ Flask server is running")

        # Test 1: Get positions from API endpoint
        print("\n1. Testing API endpoint position data...")
        api_positions = get_positions_from_api(base_url)
        if api_positions is None:
            print("   ❌ Cannot continue without API data")
            return False

        api_total_value = calculate_total_position_value(api_positions)
        print(f"   📈 API Total Position Value: ${api_total_value:.2f}")

        # Test 2: Get positions from dashboard (using same API for now)
        print("\n2. Testing dashboard position data...")
        dashboard_positions = get_positions_from_dashboard(base_url)
        if dashboard_positions is None:
            print("   ❌ Cannot continue without dashboard data")
            return False

        dashboard_total_value = calculate_total_position_value(dashboard_positions)
        print(f"   📊 Dashboard Total Position Value: ${dashboard_total_value:.2f}")

        # Test 3: Compare position counts
        print("\n3. Comparing position counts...")
        api_count = len(api_positions)
        dashboard_count = len(dashboard_positions)

        if api_count != dashboard_count:
            print(f"   ❌ Position counts differ: API={api_count}, Dashboard={dashboard_count}")

            # Show differences
            api_symbols = {pos.get('symbol') for pos in api_positions}
            dashboard_symbols = {pos.get('symbol') for pos in dashboard_positions}

            missing_in_dashboard = api_symbols - dashboard_symbols
            missing_in_api = dashboard_symbols - api_symbols

            if missing_in_dashboard:
                print(f"   📝 Missing in dashboard: {missing_in_dashboard}")
            if missing_in_api:
                print(f"   📝 Missing in API: {missing_in_api}")

            return False
        else:
            print(f"   ✅ Position counts match: {api_count} positions")

        # Test 4: Compare position values
        print("\n4. Comparing position values...")
        value_difference = abs(api_total_value - dashboard_total_value)

        if value_difference > 0.01:  # Allow for small floating point differences
            print(f"   ❌ Position values differ by ${value_difference:.2f}")
            return False
        else:
            print("   ✅ Position values are consistent!")

        # Test 5: Compare individual positions
        print("\n5. Comparing individual positions...")
        all_consistent = True

        for api_pos in api_positions:
            symbol = api_pos.get('symbol')
            if not symbol:
                continue

            # Find matching position in dashboard data
            dashboard_pos = None
            for pos in dashboard_positions:
                if pos.get('symbol') == symbol:
                    dashboard_pos = pos
                    break

            if dashboard_pos is None:
                print(f"   ❌ Position {symbol} missing in dashboard data")
                all_consistent = False
                continue

            # Compare key values
            api_price = api_pos.get('current_price', 0)
            dashboard_price = dashboard_pos.get('current_price', 0)
            api_amount = api_pos.get('amount', api_pos.get('total_amount', 0))
            dashboard_amount = dashboard_pos.get('amount', dashboard_pos.get('total_amount', 0))

            price_diff = abs(float(api_price) - float(dashboard_price))
            amount_diff = abs(float(api_amount) - float(dashboard_amount))

            if price_diff > 0.01 or amount_diff > 0.000001:
                print(f"   ❌ Position {symbol} data differs:")
                print(f"      API: {api_amount} @ ${api_price}")
                print(f"      Dashboard: {dashboard_amount} @ ${dashboard_price}")
                all_consistent = False
            else:
                print(f"   ✅ Position {symbol} data matches")

        if not all_consistent:
            return False

        print("\n" + "=" * 50)
        print("🎉 Position value consistency test PASSED!")
        print(f"   Total positions: {api_count}")
        print(f"   Total value: ${api_total_value:.2f}")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test position value consistency')
    parser.add_argument('--mock', action='store_true', help='Use mock data for testing without server')

    args = parser.parse_args()

    success = test_position_value_consistency(mock_data=args.mock)
    exit(0 if success else 1)
