#!/usr/bin/env python3
"""
Debug script to test fetch_ohlcv_async directly
"""
import asyncio
import sys
import os
import logging
sys.path.insert(0, os.path.dirname(__file__))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def debug_ohlcv_fetch():
    """Debug the OHLCV fetching process step by step."""
    print("🔍 Debugging OHLCV Fetch Process")
    print("=" * 50)

    try:
        from crypto_bot.execution.cex_executor import get_exchange
        from crypto_bot.utils.market_loader import fetch_ohlcv_async

        # Create exchange using proper initialization
        config = {
            'exchange': 'kraken',
            'use_websocket': False
        }
        exchange, ws_client = get_exchange(config)

        print(f"Exchange: {exchange.id}")
        print(f"Has fetch_ohlcv: {hasattr(exchange, 'fetch_ohlcv')}")
        print(f"Has fetchOHLCV: {hasattr(exchange, 'fetchOHLCV')}")

        # Test basic exchange functionality
        print("\n🧪 Testing basic exchange operations...")

        # Check if we can make a simple API call
        try:
            # Test with a simple public endpoint first
            if hasattr(exchange, 'fetch_ticker'):
                ticker = await asyncio.to_thread(exchange.fetch_ticker, 'BTC/USD')
                print(f"✅ Ticker fetch successful: {ticker.get('last', 'N/A')}")
            else:
                print("❌ Exchange doesn't have fetch_ticker method")
        except Exception as e:
            print(f"❌ Ticker fetch failed: {e}")

        # Test OHLCV fetch with different parameters
        test_cases = [
            ('BTC/USD', '1h', 10),
            ('BTC/USD', '1m', 5),
            ('ETH/USD', '1h', 10),
        ]

        for symbol, timeframe, limit in test_cases:
            print(f"\n📊 Testing OHLCV fetch: {symbol} {timeframe} limit={limit}")

            try:
                # Test the fetch_ohlcv_async function directly
                print(f"   Calling fetch_ohlcv_async({symbol}, {timeframe}, {limit})...")
                data = await fetch_ohlcv_async(exchange, symbol, timeframe, limit)
                if data:
                    print(f"✅ fetch_ohlcv_async returned {len(data)} candles")
                    if len(data) > 0:
                        print(f"   First candle: {data[0]}")
                        print(f"   Last candle: {data[-1]}")
                else:
                    print(f"❌ fetch_ohlcv_async returned None")
            except Exception as e:
                print(f"❌ fetch_ohlcv_async raised exception: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")

            # Try the raw exchange method
            print("   Trying raw exchange.fetch_ohlcv...")
            try:
                if hasattr(exchange, 'fetch_ohlcv'):
                    if asyncio.iscoroutinefunction(exchange.fetch_ohlcv):
                        raw_data = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    else:
                        raw_data = await asyncio.to_thread(exchange.fetch_ohlcv, symbol, timeframe, limit=limit)

                    if raw_data:
                        print(f"   ✅ Raw exchange returned {len(raw_data)} candles")
                    else:
                        print("   ❌ Raw exchange returned None/empty")
                else:
                    print("   ❌ Exchange doesn't have fetch_ohlcv method")
            except Exception as raw_e:
                print(f"   ❌ Raw exchange failed: {raw_e}")

            except Exception as e:
                print(f"❌ OHLCV fetch failed: {e}")
                import traceback
                traceback.print_exc()

        # Check exchange capabilities
        print("\n🔧 Exchange Capabilities:")
        print(f"  has.fetchOHLCV: {exchange.has.get('fetchOHLCV', False)}")
        print(f"  has.fetchTrades: {exchange.has.get('fetchTrades', False)}")
        print(f"  rateLimit: {getattr(exchange, 'rateLimit', 'N/A')}")
        print(f"  timeout: {getattr(exchange, 'timeout', 'N/A')}")

        # Check timeframes
        if hasattr(exchange, 'timeframes'):
            print(f"  timeframes: {list(exchange.timeframes.keys())[:5]}...")

        return True

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(debug_ohlcv_fetch())
        print(f"\n{'✅' if success else '❌'} Debug completed")
    except KeyboardInterrupt:
        print("\n🛑 Debug interrupted")
    except Exception as e:
        print(f"\n💥 Debug failed: {e}")
        import traceback
        traceback.print_exc()
