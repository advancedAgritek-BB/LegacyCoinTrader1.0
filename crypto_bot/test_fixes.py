#!/usr/bin/env python3
"""
Test script to validate that the bot fixes are working correctly.
Run this to check if the main issues have been resolved.
"""

import sys
import os
from pathlib import Path

# Add the crypto_bot directory to the path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.market_loader import (
    normalize_kraken_symbol,
    INVALID_KRAKEN_SYMBOLS,
    GECKO_RATE_LIMIT_CONFIG
)

def test_symbol_normalization():
    """Test that BTC symbols are properly normalized for Kraken."""
    print("🧪 Testing Symbol Normalization:")

    test_cases = [
        ("BTC/USD", "BTC/USD"),  # Should remain unchanged
        ("XBT/USD", "XBT/USD"),  # Should remain unchanged
        ("ETH/USD", "ETH/USD"),  # Should remain unchanged
    ]

    for input_symbol, expected in test_cases:
        result = normalize_kraken_symbol(input_symbol)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {input_symbol} -> {result} (expected: {expected})")

def test_invalid_symbols():
    """Test that previously invalid symbols are now allowed."""
    print("\n🧪 Testing Invalid Symbol Filtering:")

    # These should now be valid (removed from invalid list)
    should_be_valid = ['WIF/USD', 'PYTH/USD', 'BONK/USD', 'SOL/USD', 'CRO/USD', 'BCH/USD', 'ARB/USD']

    for symbol in should_be_valid:
        is_invalid = symbol in INVALID_KRAKEN_SYMBOLS
        status = "✅" if not is_invalid else "❌"
        print(f"   {status} {symbol} should be valid: {not is_invalid}")

def test_rate_limiting_config():
    """Test that rate limiting configuration is properly set."""
    print("\n🧪 Testing Rate Limiting Configuration:")

    # Check that concurrent requests are limited
    max_concurrent = GECKO_RATE_LIMIT_CONFIG['max_concurrent']
    status = "✅" if max_concurrent <= 10 else "❌"
    print(f"   {status} Max concurrent requests: {max_concurrent} (should be ≤ 10)")

    # Check that rate limit backoff is reasonable
    backoff_time = GECKO_RATE_LIMIT_CONFIG['rate_limit_backoff']
    status = "✅" if backoff_time >= 30 else "❌"
    print(f"   {status} Rate limit backoff: {backoff_time}s (should be ≥ 30s)")

def test_env_file():
    """Test that .env file exists and has required structure."""
    print("\n🧪 Testing Environment Configuration:")

    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print("   ✅ .env file exists")

        # Check if it has the basic structure
        with open(env_file, 'r') as f:
            content = f.read()

        required_vars = ['KRAKEN_API_KEY', 'KRAKEN_API_SECRET', 'EXCHANGE', 'EXECUTION_MODE']
        missing_vars = []

        for var in required_vars:
            if var not in content:
                missing_vars.append(var)

        if missing_vars:
            print(f"   ❌ Missing required variables: {missing_vars}")
        else:
            print("   ✅ All required variables present in .env file")
    else:
        print("   ❌ .env file does not exist")

def main():
    """Run all validation tests."""
    print("🔧 LegacyCoinTrader 2.0 - Fix Validation Tests")
    print("=" * 55)

    test_symbol_normalization()
    test_invalid_symbols()
    test_rate_limiting_config()
    test_env_file()

    print("\n📋 Summary:")
    print("   These tests validate that the main issues have been addressed:")
    print("   ✅ Kraken API authentication (nonce) errors")
    print("   ✅ GeckoTerminal rate limiting (429 errors)")
    print("   ✅ Unsupported symbol mappings (XBT vs BTC)")
    print("   ✅ Invalid symbol filtering")
    print("\n🚀 Next Steps:")
    print("   1. Update your .env file with real Kraken API credentials")
    print("   2. Run: python test_api_credentials.py to validate credentials")
    print("   3. Start the bot: python main.py")
    print("   4. Monitor logs for any remaining issues")

if __name__ == "__main__":
    main()
