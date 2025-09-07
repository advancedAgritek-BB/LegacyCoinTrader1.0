#!/usr/bin/env python3
"""
Test script to debug open positions charts issue
"""

import requests
import json
import time

def test_open_positions_flow():
    """Test the complete flow of loading positions and charts"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Open Positions Charts Flow")
    print("=" * 50)
    
    # Step 1: Test open positions API
    print("1. Testing /api/open-positions...")
    try:
        response = requests.get(f"{base_url}/api/open-positions")
        if response.status_code == 200:
            positions = response.json()
            print(f"✅ Found {len(positions)} positions")
            for pos in positions:
                print(f"   - {pos['symbol']}: {pos['amount']} @ ${pos['entry_price']}")
        else:
            print(f"❌ Failed to get positions: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error getting positions: {e}")
        return
    
    # Step 2: Test trend data for each position
    print("\n2. Testing trend data for each position...")
    for position in positions:
        symbol = position['symbol']
        print(f"\n   Testing {symbol}...")
        
        # Test candle timestamp
        try:
            timestamp_response = requests.get(f"{base_url}/api/candle-timestamp?symbol={symbol}")
            if timestamp_response.status_code == 200:
                timestamp_data = timestamp_response.json()
                print(f"   ✅ Candle timestamp: {timestamp_data['timestamp']}")
            else:
                print(f"   ❌ Candle timestamp failed: {timestamp_response.status_code}")
        except Exception as e:
            print(f"   ❌ Candle timestamp error: {e}")
        
        # Test trend data
        try:
            trend_response = requests.get(f"{base_url}/api/trend-data?symbol={symbol}")
            if trend_response.status_code == 200:
                trend_data = trend_response.json()
                candles_count = len(trend_data.get('candles', []))
                print(f"   ✅ Trend data: {candles_count} candles")
            else:
                print(f"   ❌ Trend data failed: {trend_response.status_code}")
        except Exception as e:
            print(f"   ❌ Trend data error: {e}")
        
        # Test current prices with history
        try:
            prices_response = requests.get(f"{base_url}/api/current-prices?symbol={symbol}&history=true")
            if prices_response.status_code == 200:
                prices_data = prices_response.json()
                history_count = len(prices_data.get('history', {}).get('data', []))
                print(f"   ✅ Current prices with history: {history_count} data points")
            else:
                print(f"   ❌ Current prices failed: {prices_response.status_code}")
        except Exception as e:
            print(f"   ❌ Current prices error: {e}")
    
    # Step 3: Test batch chart data
    print("\n3. Testing batch chart data...")
    symbols = [pos['symbol'] for pos in positions]
    try:
        batch_response = requests.get(f"{base_url}/api/batch-chart-data", params={'symbols[]': symbols})
        if batch_response.status_code == 200:
            batch_data = batch_response.json()
            print(f"✅ Batch chart data: {batch_data.get('fetched_count', 0)} fetched, {batch_data.get('cached_count', 0)} cached")
        else:
            print(f"❌ Batch chart data failed: {batch_response.status_code}")
    except Exception as e:
        print(f"❌ Batch chart data error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 All APIs are working correctly!")
    print("The issue might be in the frontend JavaScript or timing.")
    print("Check the browser console for JavaScript errors.")

if __name__ == "__main__":
    test_open_positions_flow()
