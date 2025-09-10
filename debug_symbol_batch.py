#!/usr/bin/env python3
"""
Debug script to investigate why only some symbols are being analyzed.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.main import load_config
from crypto_bot.execution.cex_executor import get_exchange
from crypto_bot.utils.market_loader import update_multi_tf_ohlcv_cache
from crypto_bot.phase_runner import BotContext

async def debug_symbol_batch():
    """Debug why only some symbols are being analyzed."""

    print("🔍 Debugging Symbol Batch Processing")
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

        # Test symbols from recent batch (from logs)
        test_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD', 'NEAR/USD', 'ATOM/USD']
        print(f"\n🎯 Testing with symbols: {test_symbols}")

        # Fetch real data for these symbols
        print("\n📡 Fetching real data...")
        await update_multi_tf_ohlcv_cache(
            exchange=ctx.exchange,
            cache=ctx.df_cache,
            symbols=test_symbols,
            config=ctx.config,
            limit=100
        )

        print("✅ Data fetch completed")

        # Check what data we have for each symbol
        main_tf = config.get("timeframe", "1h")
        print(f"\n📊 Checking data availability for main timeframe: {main_tf}")

        available_symbols = []
        missing_symbols = []

        for symbol in test_symbols:
            main_df = ctx.df_cache.get(main_tf, {}).get(symbol)
            if main_df is not None:
                candles = len(main_df) if hasattr(main_df, '__len__') else 'N/A'
                available_symbols.append((symbol, candles))
                print(f"   ✅ {symbol}: {candles} candles available")
            else:
                missing_symbols.append(symbol)
                print(f"   ❌ {symbol}: no data available")

        print(f"\n📈 Summary:")
        print(f"   Available symbols: {len(available_symbols)}")
        print(f"   Missing symbols: {len(missing_symbols)}")

        # Simulate the analyse_batch logic
        print("\n🔬 Simulating analyse_batch logic...")
        batch = test_symbols
        tasks_created = 0
        skipped_symbols = []

        for sym in batch:
            main_df = ctx.df_cache.get(main_tf, {}).get(sym)
            if main_df is None:
                print(f"   ❌ Skipping {sym}: no data available for timeframe {main_tf}")
                skipped_symbols.append(sym)
                continue

            # Count available timeframes
            df_map = {}
            for tf, c in ctx.df_cache.items():
                df = c.get(sym)
                if df is not None:
                    df_map[tf] = df

            for tf, cache in ctx.regime_cache.items():
                df = cache.get(sym)
                if df is not None:
                    df_map[tf] = df

            available_tfs = list(df_map.keys())
            print(f"   ✅ {sym}: {len(available_tfs)} timeframes available {available_tfs}")
            tasks_created += 1

        print("\n📋 Simulation Results:")
        print(f"   Batch size: {len(batch)}")
        print(f"   Tasks created: {tasks_created}")
        print(f"   Skipped symbols: {len(skipped_symbols)}")
        print(f"   Expected analysis: {len(batch)} -> {tasks_created} tasks")

        if skipped_symbols:
            print(f"   Symbols skipped: {skipped_symbols}")

        # Check if this matches the log pattern
        if len(batch) != tasks_created:
            print("\n⚠️  MISMATCH DETECTED!")
            print(f"   This explains why we see 'batch size {len(batch)}' but 'running analysis on {tasks_created} tasks'")
        else:
            print("\n✅ No mismatch - all symbols should be processed")
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_symbol_batch())
