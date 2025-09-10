#!/usr/bin/env python3
"""
Test environment variable loading directly.
"""

import os
from pathlib import Path

def test_env_loading():
    """Test loading environment variables from .env file."""
    print("🔍 Testing environment variable loading...")

    # Check if .env file exists
    env_file = Path('.env')
    print(f".env file exists: {env_file.exists()}")
    print(f".env file path: {env_file.absolute()}")

    # Try to load with python-dotenv
    try:
        from dotenv import dotenv_values
        print("✅ python-dotenv is available")

        if env_file.exists():
            print("Loading .env file...")
            env_vars = dotenv_values(str(env_file))
            print(f"Loaded {len(env_vars)} variables from .env")

            # Update environment
            os.environ.update(env_vars)

            # Check API keys
            api_key = os.getenv('API_KEY')
            api_secret = os.getenv('API_SECRET')

            if api_key:
                print(f"✅ API_KEY loaded: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")
            else:
                print("❌ API_KEY not found")

            if api_secret:
                print(f"✅ API_SECRET loaded: {api_secret[:8]}...{api_secret[-4:] if len(api_secret) > 12 else api_secret}")
            else:
                print("❌ API_SECRET not found")

            return True
        else:
            print("❌ .env file does not exist")
            return False

    except ImportError:
        print("❌ python-dotenv not available")
        return False

def test_exchange_connection():
    """Test if we can connect to the exchange with loaded credentials."""
    print("\n🔍 Testing exchange connection...")

    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')

    if not api_key or not api_secret:
        print("❌ API credentials not available")
        return False

    try:
        import ccxt
        exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

        print("✅ Exchange object created")

        # Test a simple API call
        ticker = exchange.fetch_ticker('BTC/USD')
        if ticker and ticker.get('last'):
            print(f"✅ Exchange API working - BTC/USD: ${ticker['last']}")
            return True
        else:
            print("❌ Exchange API returned invalid ticker")
            return False

    except Exception as e:
        print(f"❌ Exchange connection failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("ENVIRONMENT LOADING TEST")
    print("=" * 50)

    env_loaded = test_env_loading()
    if env_loaded:
        exchange_working = test_exchange_connection()
        if exchange_working:
            print("\n✅ SUCCESS: Environment and exchange are working!")
        else:
            print("\n❌ ISSUE: Environment loaded but exchange not working")
    else:
        print("\n❌ FAILURE: Environment loading failed")

    print("=" * 50)
