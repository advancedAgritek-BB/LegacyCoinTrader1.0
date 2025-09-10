#!/usr/bin/env python3
"""
Debug script to test EnhancedOHLCVFetcher directly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.main import load_config
from crypto_bot.utils.enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher

async def debug_fetcher():
    """Debug the EnhancedOHLCVFetcher to see why it's only caching some symbols."""

    print("🔧 Debugging EnhancedOHLCVFetcher")
    print("=" * 50)

    try:
        # Load configuration
        print("\n📄 Loading configuration...")
        config = load_config()
        print(f"✅ Config loaded. Mode: {config.get('mode', 'cex')}")

        # Mock exchange object
        class MockExchange:
            def __init__(self):
                self.id = "kraken"

        exchange = MockExchange()

        # Create fetcher
        print("\n🔧 Creating EnhancedOHLCVFetcher...")
        fetcher = EnhancedOHLCVFetcher(exchange, config)
        print("✅ Fetcher created successfully")

        # Test symbols from the batch
        test_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD']
        timeframe = "1h"
        limit = 100

        print(f"\n🎯 Testing with symbols: {test_symbols}")
        print(f"   Timeframe: {timeframe}, Limit: {limit}")

        # Test classification
        print("\n📊 Testing symbol classification...")
        try:
            cex_symbols, dex_symbols = fetcher._classify_symbols(test_symbols)
            print(f"✅ CEX symbols: {len(cex_symbols)} - {cex_symbols}")
            print(f"✅ DEX symbols: {len(dex_symbols)} - {dex_symbols}")
        except Exception as e:
            print(f"❌ Classification failed: {e}")
            import traceback
            traceback.print_exc()

        # Test fetch
        print("\n📡 Testing fetch_ohlcv_batch...")
        try:
            cex_data, dex_data = await fetcher.fetch_ohlcv_batch(test_symbols, timeframe, limit)
            print(f"✅ CEX data: {len(cex_data)} symbols")
            print(f"✅ DEX data: {len(dex_data)} symbols")
            print(f"   CEX symbols with data: {list(cex_data.keys()) if cex_data else 'None'}")
            print(f"   DEX symbols with data: {list(dex_data.keys()) if dex_data else 'None'}")
        except Exception as e:
            print(f"❌ Fetch failed: {e}")
            import traceback
            traceback.print_exc()

        # Test update_cache
        print("\n💾 Testing update_cache...")
        try:
            cache = {}
            updated_cache = await fetcher.update_cache(cache, test_symbols, timeframe, limit)
            print(f"✅ Cache updated with {len(updated_cache)} symbols")
            print(f"   Cached symbols: {list(updated_cache.keys())}")
        except Exception as e:
            print(f"❌ Cache update failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_fetcher())
