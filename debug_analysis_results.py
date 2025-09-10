#!/usr/bin/env python3
"""
Debug script to check what analysis results are being generated and stored.
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
from crypto_bot.execution.cex_executor import get_exchange
from crypto_bot.utils.market_loader import update_multi_tf_ohlcv_cache
from crypto_bot.phase_runner import BotContext

async def debug_analysis_results():
    """Debug what analysis results are being generated."""

    print("🔍 Debugging Analysis Results")
    print("=" * 50)

    try:
        # Load configuration
        print("\n📄 Loading configuration...")
        config = load_config()
        print(f"✅ Config loaded. Mode: {config.get('mode', 'cex')}")

        # Get real exchange connection
        print("\n🔗 Creating real exchange connection...")
        exchange, ws_client = get_exchange(config)
        print(f"✅ Connected to {exchange.id}")

        # Load markets
        print("\n📊 Loading exchange markets...")
        try:
            if asyncio.iscoroutinefunction(exchange.load_markets):
                await exchange.load_markets()
            else:
                exchange.load_markets()
            print(f"✅ Loaded {len(exchange.symbols)} markets")
        except Exception as e:
            print(f"❌ Failed to load markets: {e}")
            return

        # Create bot context
        print("\n🔧 Creating bot context...")
        ctx = BotContext(
            positions={},
            df_cache={},
            regime_cache={},
            config=config
        )
        ctx.exchange = exchange

        # Test symbols from recent batch
        test_symbols = ['BTC/USD', 'ETH/USD']
        print(f"\n🎯 Testing with symbols: {test_symbols}")

        # Fetch real data
        print("\n📡 Fetching real data...")
        await update_multi_tf_ohlcv_cache(
            exchange=ctx.exchange,
            cache=ctx.df_cache,
            symbols=test_symbols,
            config=ctx.config,
            limit=100
        )

        print("✅ Data fetch completed")

        # Simulate the analysis process
        print("\n🔬 Simulating analysis process...")
        analysis_results = []

        for symbol in test_symbols:
            if '1h' in ctx.df_cache and symbol in ctx.df_cache['1h']:
                df = ctx.df_cache['1h'][symbol]
                print(f"\n   Analyzing {symbol}...")

                # Create df_map from cache
                df_map = {}
                for tf, tf_cache in ctx.df_cache.items():
                    if symbol in tf_cache:
                        df_map[tf] = tf_cache[symbol]

                if not df_map:
                    print(f"   ❌ No data available for {symbol}")
                    continue

                # Analyze the symbol
                try:
                    result = await analyze_symbol(
                        symbol=symbol,
                        df_map=df_map,
                        mode="cex",
                        config=config
                    )

                    print(f"   ✅ Analysis completed: {result}")

                    # Check if result should be actionable
                    skip = result.get("skip")
                    direction = result.get("direction", "MISSING")

                    is_actionable = not skip and direction != "none"
                    print(f"   📊 Actionable: {is_actionable} (skip: {skip}, direction: {direction})")

                    analysis_results.append(result)

                except Exception as e:
                    print(f"   ❌ Analysis failed: {e}")
                    import traceback
                    traceback.print_exc()

        # Simulate the execute_signals filtering
        print("\n🎯 Simulating execute_signals filtering...")
        print(f"   Total analysis results: {len(analysis_results)}")

        # Apply the same filtering as execute_signals
        initial = len(analysis_results)
        filtered_results = [r for r in analysis_results if not r.get("skip") and r.get("direction") != "none"]
        filtered_count = initial - len(filtered_results)

        print(f"   After filtering: {len(filtered_results)} actionable signals")
        print(f"   Filtered out: {filtered_count} signals")

        if filtered_count > 0:
            print("   📋 Filtered signals:")
            for i, result in enumerate(analysis_results):
                skip = result.get("skip")
                direction = result.get("direction", "MISSING")
                if skip or direction == "none":
                    symbol = result.get("symbol", "UNKNOWN")
                    print(f"     - {symbol}: skip={skip}, direction={direction}")

        # Show actionable signals
        if filtered_results:
            print("   ✅ Actionable signals:")
            for result in filtered_results:
                symbol = result.get("symbol", "UNKNOWN")
                direction = result.get("direction", "MISSING")
                score = result.get("score", 0)
                print(f"     - {symbol}: {direction} (score: {score})")
        else:
            print("   ❌ No actionable signals found")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_analysis_results())
