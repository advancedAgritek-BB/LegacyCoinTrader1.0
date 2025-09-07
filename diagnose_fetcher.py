#!/usr/bin/env python3
"""
Comprehensive diagnostic script for OHLCV fetcher issues
"""
import asyncio
import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

async def diagnose_exchange():
    """Diagnose exchange connectivity and configuration."""
    print("🔍 Diagnosing Exchange Connectivity")
    print("=" * 50)

    try:
        from crypto_bot.execution.cex_executor import get_exchange
        from dotenv import dotenv_values

        # Load environment variables
        secrets = dotenv_values('.env')
        if not secrets:
            secrets = dotenv_values('crypto_bot/.env')
        os.environ.update(secrets)

        # Create exchange using the proper initialization method
        config = {
            'exchange': 'kraken',
            'use_websocket': False
        }
        exchange, ws_client = get_exchange(config)

        print(f"Exchange ID: {exchange.id}")
        print(f"Exchange Name: {exchange.name}")
        print(f"API Key configured: {'✅' if exchange.apiKey else '❌'}")
        print(f"API Secret configured: {'✅' if exchange.secret else '❌'}")

        # Test basic connectivity
        print("\n🔗 Testing basic connectivity...")
        try:
            if hasattr(exchange, 'loadMarkets'):
                if asyncio.iscoroutinefunction(exchange.loadMarkets):
                    await exchange.loadMarkets()
                else:
                    exchange.loadMarkets()
            elif hasattr(exchange, 'load_markets'):
                if asyncio.iscoroutinefunction(exchange.load_markets):
                    await exchange.load_markets()
                else:
                    exchange.load_markets()

            if hasattr(exchange, 'markets') and exchange.markets:
                print(f"✅ Markets loaded: {len(exchange.markets)} symbols")
            else:
                print("❌ No markets loaded")
                return False
        except Exception as e:
            print(f"❌ Failed to load markets: {e}")
            return False

        # Test specific symbols
        test_symbols = ['BTC/USD', 'ETH/USD', 'BTC/USDT']
        print("\n🪙 Testing symbol availability...")
        for symbol in test_symbols:
            if symbol in exchange.markets:
                print(f"✅ {symbol} - Available")
            else:
                print(f"❌ {symbol} - Not available")

                # Try alternative formats
                alternatives = [symbol.replace('/', ''), f"{symbol.split('/')[0]}{symbol.split('/')[1]}"]
                for alt in alternatives:
                    if alt in exchange.markets:
                        print(f"   ↳ Alternative {alt} is available")
                        break

        return True

    except Exception as e:
        print(f"❌ Exchange diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_enhanced_fetcher():
    """Test the enhanced OHLCV fetcher specifically."""
    print("\n🔍 Testing Enhanced OHLCV Fetcher")
    print("=" * 50)

    try:
        from crypto_bot.utils.enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher
        from crypto_bot.execution.cex_executor import get_exchange

        # Create exchange using proper initialization
        exchange_config = {
            'exchange': 'kraken',
            'use_websocket': False
        }
        exchange, ws_client = get_exchange(exchange_config)

        # Create fetcher with detailed config
        fetcher_config = {
            "ohlcv_fetcher_timeout": 30,
            "max_concurrent_cex": 2,
            "max_concurrent_dex": 2,
            "min_volume_usd": 1000
        }

        print("Creating EnhancedOHLCVFetcher...")
        fetcher = EnhancedOHLCVFetcher(exchange, fetcher_config)
        print("✅ EnhancedOHLCVFetcher created successfully")

        # Test symbol classification
        test_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']
        print(f"\n📊 Testing symbol classification for: {test_symbols}")
        cex_symbols, dex_symbols = fetcher._classify_symbols(test_symbols)
        print(f"CEX symbols: {cex_symbols}")
        print(f"DEX symbols: {dex_symbols}")

        # Test individual fetch operations
        print("\n🧪 Testing individual CEX fetch...")
        if cex_symbols:
            try:
                from crypto_bot.utils.market_loader import fetch_ohlcv_async
                symbol = cex_symbols[0]
                print(f"Fetching {symbol} with fetch_ohlcv_async...")
                data = await fetch_ohlcv_async(exchange, symbol, '1h', 10)
                if data:
                    print(f"✅ fetch_ohlcv_async returned {len(data)} candles")
                    if len(data) > 0:
                        print(f"   Sample: {data[0]}")
                else:
                    print(f"❌ fetch_ohlcv_async returned None for {symbol}")
            except Exception as e:
                print(f"❌ fetch_ohlcv_async failed: {e}")

        # Test batch fetch
        print("\n📦 Testing batch fetch...")
        try:
            cex_data, dex_data = await fetcher.fetch_ohlcv_batch(test_symbols, "1h", 10)
            # Combine for analysis
            result = {**cex_data, **dex_data}
            print(f"✅ Batch fetch completed: {len(result)} symbols returned ({len(cex_data)} CEX, {len(dex_data)} DEX)")

            for symbol, data in result.items():
                if data:
                    print(f"   {symbol}: {len(data)} candles ✅")
                else:
                    print(f"   {symbol}: No data ❌")

            return len(result) > 0

        except Exception as e:
            print(f"❌ Batch fetch failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Enhanced fetcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_legacy_fetcher():
    """Test the legacy fetcher for comparison."""
    print("\n🔍 Testing Legacy OHLCV Fetcher")
    print("=" * 50)

    try:
        from crypto_bot.utils.market_loader import update_ohlcv_cache
        from crypto_bot.execution.cex_executor import get_exchange

        # Create exchange using proper initialization
        config = {
            'exchange': 'kraken',
            'use_websocket': False
        }
        exchange, ws_client = get_exchange(config)

        # Test legacy fetcher
        symbols = ['BTC/USD']
        tf_cache = {}

        print("Testing legacy update_ohlcv_cache...")
        result = await update_ohlcv_cache(
            exchange, tf_cache, symbols, timeframe='1h', limit=10
        )

        if 'BTC/USD' in result and result['BTC/USD']:
            print(f"✅ Legacy fetcher returned {len(result['BTC/USD'])} candles")
            return True
        else:
            print("❌ Legacy fetcher returned no data")
            return False

    except Exception as e:
        print(f"❌ Legacy fetcher test failed: {e}")
        return False

async def main():
    """Main diagnostic function."""
    print("🚀 OHLCV Fetcher Diagnostic Tool")
    print("=" * 60)

    # Test exchange connectivity
    exchange_ok = await diagnose_exchange()

    if not exchange_ok:
        print("\n❌ Exchange connectivity issues detected. Cannot proceed.")
        return False

    # Test enhanced fetcher
    enhanced_ok = await test_enhanced_fetcher()

    # Test legacy fetcher
    legacy_ok = await test_legacy_fetcher()

    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Exchange Connectivity: {'✅ PASS' if exchange_ok else '❌ FAIL'}")
    print(f"Enhanced Fetcher: {'✅ PASS' if enhanced_ok else '❌ FAIL'}")
    print(f"Legacy Fetcher: {'✅ PASS' if legacy_ok else '❌ FAIL'}")

    if enhanced_ok and legacy_ok:
        print("\n🎉 All fetchers working correctly!")
        return True
    elif legacy_ok:
        print("\n⚠️ Legacy fetcher works, but enhanced fetcher needs investigation.")
        return False
    else:
        print("\n❌ Both fetchers are failing. Check exchange configuration.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Diagnostic interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
