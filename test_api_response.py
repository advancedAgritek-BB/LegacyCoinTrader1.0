#!/usr/bin/env python3
"""
Test script to check API response and price fetching.
"""

import requests
import json

def test_positions_api():
    """Test the open positions API."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Origin': 'http://localhost:5000'
        }
        response = requests.get('http://localhost:5000/api/open-positions', headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response: {len(data)} positions")

            for pos in data:
                symbol = pos.get('symbol', 'Unknown')
                entry_price = pos.get('entry_price', 0)
                current_price = pos.get('current_price', 0)
                pnl = pos.get('pnl', 0)
                pnl_pct = pos.get('pnl_percentage', 0)

                print(f"📊 {symbol}: Entry=${entry_price:.4f}, Current=${current_price:.4f}, PnL=${pnl:.2f} ({pnl_pct:.2f}%)")

                # Check if prices are different (indicating live prices)
                if abs(current_price - entry_price) > 0.0001:
                    print(f"   ✅ Live price detected for {symbol}")
                else:
                    print(f"   ⚠️  Same price as entry for {symbol} - might be showing entry price")

        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_positions_api()
