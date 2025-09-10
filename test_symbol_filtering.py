#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append('/Users/brandonburnette/Downloads/LegacyCoinTrader1.0')

from crypto_bot.utils.symbol_utils import get_filtered_symbols
from crypto_bot.main import load_config
import ccxt

async def test_symbol_filtering():
    """Test that symbol filtering works with cleaned configuration."""

    # Load configuration
    config = load_config()

    # Initialize Kraken exchange
    exchange = ccxt.kraken({
        'apiKey': config.get('kraken_api_key', ''),
        'secret': config.get('kraken_api_secret', ''),
        'enableRateLimit': True,
    })

    print(f"Testing symbol filtering with {len(config.get('symbols', []))} configured symbols...")

    try:
        # Test symbol filtering
        filtered_symbols = await get_filtered_symbols(exchange, config)

        print(f"✅ Symbol filtering successful!")
        print(f"   - Input symbols: {len(config.get('symbols', []))}")
        print(f"   - Filtered symbols: {len(filtered_symbols)}")

        if filtered_symbols:
            print("\nFirst 10 filtered symbols:")
            for i, (symbol, score) in enumerate(filtered_symbols[:10]):
                print(f"     {i+1:2d}. {symbol:<12} - Score: {score:.3f}")

        return True

    except Exception as e:
        print(f"❌ Symbol filtering failed: {e}")
        return False

    finally:
        await exchange.close()

if __name__ == "__main__":
    success = asyncio.run(test_symbol_filtering())
    sys.exit(0 if success else 1)
