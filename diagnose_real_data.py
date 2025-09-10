#!/usr/bin/env python3
"""
Diagnostic script to investigate real data fetching and analysis issues.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.main import load_config
from crypto_bot.utils.market_analyzer import analyze_symbol
from crypto_bot.utils.enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher
from crypto_bot.utils.market_loader import update_multi_tf_ohlcv_cache
from crypto_bot.phase_runner import BotContext
import pandas as pd

async def diagnose_real_data():
    """Diagnose real data fetching and analysis issues."""

    print("🔍 Diagnosing Real Data Fetching Issues")
    print("=" * 50)

    try:
        # Load configuration
        print("\n📄 Loading configuration...")
        config = load_config()
        print(f"✅ Config loaded. Mode: {config.get('mode', 'cex')}")

        # Create a context similar to what the bot uses
        print("\n🔧 Creating bot context...")
        ctx = BotContext(
            positions={},
            df_cache={},
            regime_cache={},
            config=config
        )

        # Create exchange (using a mock that behaves like Kraken)
        class MockKrakenExchange:
            def __init__(self):
                self.id = "kraken"
                self.symbols = None  # Will be populated by load_markets
                self.markets = None
                self.rateLimit = 100

            async def load_markets(self):
                """Mock market loading."""
                self.symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD"]
                self.markets = {
                    "BTC/USD": {"active": True, "precision": {"price": 1, "amount": 8}},
                    "ETH/USD": {"active": True, "precision": {"price": 2, "amount": 8}},
                    "SOL/USD": {"active": True, "precision": {"price": 3, "amount": 8}},
                    "ADA/USD": {"active": True, "precision": {"price": 5, "amount": 8}},
                    "DOT/USD": {"active": True, "precision": {"price": 4, "amount": 8}},
                }
                print("   ✅ Mock markets loaded")

        exchange = MockKrakenExchange()
        await exchange.load_markets()
        ctx.exchange = exchange

        print("✅ Bot context created")

        # Test symbols from recent batch
        test_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD']
        print(f"\n🎯 Testing with symbols: {test_symbols}")

        # Test the actual cache update process
        print("\n📡 Testing real cache update process...")

        try:
            # Use the same cache update function as the bot
            await update_multi_tf_ohlcv_cache(
                exchange=ctx.exchange,
                cache=ctx.df_cache,
                symbols=test_symbols,
                config=ctx.config,
                limit=100  # Same limit as bot
            )

            print("✅ Cache update completed")

            # Check what data we got
            print("\n📊 Cache contents:")
            for tf, tf_cache in ctx.df_cache.items():
                print(f"   Timeframe {tf}: {len(tf_cache)} symbols")
                for symbol, df in tf_cache.items():
                    if df is not None:
                        print(f"     {symbol}: {len(df)} candles, type: {type(df)}")
                        if hasattr(df, 'empty'):
                            print(f"       Empty: {df.empty}")
                        if len(df) > 0:
                            print(f"       Columns: {list(df.columns)}")
                            print(f"       Sample data: {df.head(1).to_dict('records')[0]}")
                    else:
                        print(f"     {symbol}: None")

        except Exception as e:
            print(f"❌ Cache update failed: {e}")
            import traceback
            traceback.print_exc()

        # Test analysis with the fetched data
        print("\n🔬 Testing analysis with fetched data...")

        for symbol in test_symbols[:2]:  # Test first 2 symbols
            print(f"\n   Testing {symbol}...")

            # Create df_map from cache
            df_map = {}
            for tf, tf_cache in ctx.df_cache.items():
                if symbol in tf_cache:
                    df_map[tf] = tf_cache[symbol]

            if not df_map:
                print(f"   ❌ No data available for {symbol}")
                continue

            print(f"   Available timeframes: {list(df_map.keys())}")

            # Check data requirements
            main_tf = config.get("timeframe", "1h")
            lookback = config.get("indicator_lookback", 14) * 2

            main_df = df_map.get(main_tf)
            if main_df is None:
                print(f"   ❌ No data for main timeframe {main_tf}")
                continue

            print(f"   Main timeframe ({main_tf}) candles: {len(main_df)}")
            print(f"   Required minimum: {lookback}")

            if len(main_df) < lookback:
                print(f"   ⚠️  Insufficient data: {len(main_df)} < {lookback}")
            else:
                print(f"   ✅ Sufficient data for analysis")

                # Try analysis
                try:
                    result = await analyze_symbol(
                        symbol=symbol,
                        df_map=df_map,
                        mode="cex",
                        config=config
                    )

                    print(f"   Analysis result: {result.get('skip', 'No skip')} -> {result.get('direction', 'No direction')}")

                except Exception as e:
                    print(f"   ❌ Analysis failed: {e}")

        # Check regime cache
        print("\n🏛️  Regime cache contents:")
        for tf, tf_cache in ctx.regime_cache.items():
            print(f"   Timeframe {tf}: {len(tf_cache)} symbols")
            for symbol in list(tf_cache.keys())[:3]:  # Show first 3
                df = tf_cache[symbol]
                if df is not None:
                    print(f"     {symbol}: {len(df)} candles")
                else:
                    print(f"     {symbol}: None")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_real_data())
