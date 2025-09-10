#!/usr/bin/env python3
"""
Test Current Price Fetching

This script tests the current price fetching for BTC/USD to identify the source of the incorrect price.
"""

import sys
from pathlib import Path

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

import requests
import time

def test_pyth_price():
    """Test Pyth price fetching for BTC/USD."""
    print("🔍 Testing Pyth Price Fetching for BTC/USD")
    print("=" * 50)
    
    try:
        from crypto_bot.utils.pyth import get_pyth_price
        price = get_pyth_price("BTC/USD")
        print(f"Pyth price for BTC/USD: ${price}")
        return price
    except Exception as e:
        print(f"Pyth price fetch failed: {e}")
        return None

def test_kraken_price():
    """Test Kraken price fetching for BTC/USD."""
    print("\n🔍 Testing Kraken Price Fetching for BTC/USD")
    print("=" * 50)
    
    try:
        response = requests.get(
            "https://api.kraken.com/0/public/Ticker?pair=XXBTZUSD",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if 'result' in data and 'XXBTZUSD' in data['result']:
                price = float(data['result']['XXBTZUSD']['c'][0])
                print(f"Kraken price for BTC/USD: ${price}")
                return price
        else:
            print(f"Kraken API returned status code: {response.status_code}")
    except Exception as e:
        print(f"Kraken price fetch failed: {e}")
    
    return None

def test_binance_price():
    """Test Binance price fetching for BTC/USD."""
    print("\n🔍 Testing Binance Price Fetching for BTC/USD")
    print("=" * 50)
    
    try:
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            price = float(data.get('price', 0.0))
            print(f"Binance price for BTC/USD: ${price}")
            return price
        else:
            print(f"Binance API returned status code: {response.status_code}")
    except Exception as e:
        print(f"Binance price fetch failed: {e}")
    
    return None

def test_frontend_price_function():
    """Test the frontend price function."""
    print("\n🔍 Testing Frontend Price Function for BTC/USD")
    print("=" * 50)
    
    try:
        # Import the function from frontend app
        sys.path.insert(0, str(Path(__file__).parent / 'frontend'))
        from app import get_current_price_for_symbol
        
        # Clear any cached prices
        if hasattr(get_current_price_for_symbol, '_price_cache'):
            get_current_price_for_symbol._price_cache = {}
        
        price = get_current_price_for_symbol("BTC/USD")
        print(f"Frontend price function for BTC/USD: ${price}")
        return price
    except Exception as e:
        print(f"Frontend price function failed: {e}")
        return None

def main():
    print("🔍 Testing Current Price Sources for BTC/USD")
    print("=" * 60)
    
    # Test different price sources
    pyth_price = test_pyth_price()
    kraken_price = test_kraken_price()
    binance_price = test_binance_price()
    frontend_price = test_frontend_price_function()
    
    print("\n📊 Price Comparison Summary:")
    print("=" * 60)
    print(f"Pyth Network:     ${pyth_price if pyth_price else 'N/A'}")
    print(f"Kraken:           ${kraken_price if kraken_price else 'N/A'}")
    print(f"Binance:          ${binance_price if binance_price else 'N/A'}")
    print(f"Frontend Function: ${frontend_price if frontend_price else 'N/A'}")
    
    # Check for significant discrepancies
    prices = [p for p in [pyth_price, kraken_price, binance_price, frontend_price] if p and p > 0]
    if prices:
        avg_price = sum(prices) / len(prices)
        print(f"\nAverage Price:     ${avg_price:.2f}")
        
        for i, price in enumerate([pyth_price, kraken_price, binance_price, frontend_price]):
            if price and price > 0:
                diff_pct = abs(price - avg_price) / avg_price * 100
                source = ['Pyth', 'Kraken', 'Binance', 'Frontend'][i]
                if diff_pct > 10:  # More than 10% difference
                    print(f"⚠️  {source} price differs by {diff_pct:.1f}% from average")

if __name__ == "__main__":
    main()
