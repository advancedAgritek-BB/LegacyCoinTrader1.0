#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

# Test the generate_candle_data function directly
def test_candle_data():
    try:
        from frontend.app import generate_candle_data
        
        print("Testing generate_candle_data function...")
        symbol = 'BTC/USD'
        limit = 5
        
        print(f"Fetching {limit} candles for {symbol}...")
        data = generate_candle_data(symbol, limit)
        
        print(f"Result type: {type(data)}")
        print(f"Result length: {len(data) if isinstance(data, list) else 'Not a list'}")
        
        if isinstance(data, list) and len(data) > 0:
            print("✅ Data received successfully")
            print(f"First candle: {data[0]}")
            print(f"Last candle: {data[-1]}")
            return True
        else:
            print("❌ No data received")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_candle_data()
    sys.exit(0 if success else 1)
