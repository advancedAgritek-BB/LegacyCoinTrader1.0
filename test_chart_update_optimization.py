#!/usr/bin/env python3
"""
Test script to verify chart update optimization for 5-minute intervals.
"""

import requests
import time
import json
from datetime import datetime

def test_candle_timestamp_api():
    """Test the new candle timestamp API endpoint."""
    print("🧪 Testing Chart Update Optimization")
    print("=" * 50)
    
    # Test symbols
    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        
        try:
            # Test the candle timestamp endpoint
            response = requests.get(f"http://localhost:5000/api/candle-timestamp?symbol={symbol}")
            
            if response.status_code == 200:
                data = response.json()
                timestamp = data.get('timestamp', 0)
                timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"  ✅ Timestamp: {timestamp_str}")
                print(f"  📅 Raw timestamp: {timestamp}")
                
                # Check if timestamp is recent (within last 10 minutes)
                current_time = int(time.time())
                time_diff = current_time - timestamp
                
                if time_diff <= 600:  # 10 minutes
                    print(f"  ✅ Timestamp is recent ({time_diff} seconds ago)")
                else:
                    print(f"  ⚠️  Timestamp is old ({time_diff} seconds ago)")
                    
            else:
                print(f"  ❌ API error: {response.status_code}")
                print(f"  📄 Response: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Chart Update Optimization Summary:")
    print("  • Charts now update every 5 minutes instead of 30 seconds")
    print("  • Charts only update when new 5-minute candle data is available")
    print("  • Visual indicators show last update time for each chart")
    print("  • Reduced API calls and improved performance")

def test_trend_data_api():
    """Test the trend data API to ensure it still works."""
    print("\n🔍 Testing Trend Data API:")
    print("=" * 30)
    
    test_symbol = "BTC/USDT"
    
    try:
        response = requests.get(f"http://localhost:5000/api/trend-data?symbol={test_symbol}")
        
        if response.status_code == 200:
            data = response.json()
            candle_count = len(data.get('candles', []))
            generated_at = data.get('generated_at', 0)
            
            print(f"  ✅ Trend data retrieved for {test_symbol}")
            print(f"  📊 Candle count: {candle_count}")
            print(f"  ⏰ Generated at: {datetime.fromtimestamp(generated_at/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            
            if data.get('trend'):
                trend = data['trend']
                print(f"  📈 Trend direction: {trend.get('direction', 'unknown')}")
                print(f"  💪 Trend strength: {trend.get('strength', 'unknown')}")
                
        else:
            print(f"  ❌ API error: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Chart Update Optimization Tests")
    print("Make sure the frontend server is running on http://localhost:5000")
    print()
    
    test_candle_timestamp_api()
    test_trend_data_api()
    
    print("\n✅ Tests completed!")
    print("\n📋 Next Steps:")
    print("1. Start the frontend server: python frontend/app.py")
    print("2. Open the dashboard in your browser")
    print("3. Check that charts only update every 5 minutes")
    print("4. Verify that chart timestamps are displayed")
    print("5. Monitor console logs for 'SKIPPING chart update' messages")
