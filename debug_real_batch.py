#!/usr/bin/env python3
"""
Debug script to simulate the real batch analysis process.
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

async def debug_real_batch():
    """Debug the real batch analysis process."""

    print("🔍 Debugging Real Batch Analysis")
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

        # Use the actual batch from the logs
        actual_batch = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD', 'UNI/USD', 'AAVE/USD', 'AVAX/USD', 'MATIC/USD', 'SUSHI/USD', 'COMP/USD']
        print(f"\n🎯 Using actual batch from logs: {len(actual_batch)} symbols")
        print(f"   {actual_batch}")

        # Fetch data for this batch
        print("\n📡 Fetching data for actual batch...")
        await update_multi_tf_ohlcv_cache(
            exchange=ctx.exchange,
            cache=ctx.df_cache,
            symbols=actual_batch,
            config=ctx.config,
            limit=100
        )

        print("✅ Data fetch completed")

        # Simulate the exact analyse_batch logic
        print("\n🔬 Simulating exact analyse_batch logic...")
        batch = actual_batch
        main_tf = config.get("timeframe", "15m")
        print(f"   Main timeframe: {main_tf}")

        tasks_created = 0
        skipped_symbols = []

        for sym in batch:
            # Check if we have data for this symbol in the main timeframe
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
            print(f"   ✅ {sym}: {len(available_tfs)} timeframes available")
            tasks_created += 1

        print("\n📋 Analysis Simulation Results:")
        print(f"   Batch size: {len(batch)}")
        print(f"   Tasks created: {tasks_created}")
        print(f"   Skipped symbols: {len(skipped_symbols)}")

        if skipped_symbols:
            print(f"   ❌ Symbols skipped: {skipped_symbols}")
            print("   This explains the mismatch between batch size and analysis tasks!")
        else:
            print("   ✅ All symbols should be processed")

        # Check if this matches recent log patterns
        if len(batch) != tasks_created:
            print("\n⚠️  CONFIRMED: This matches the log pattern!")
            print(f"   Batch size {len(batch)} -> {tasks_created} tasks")
            print("   The issue is that some symbols in the batch don't have data!")

        # Test individual symbol data availability
        print("\n📊 Detailed data check for all symbols:")
        for sym in actual_batch:
            print(f"\n   {sym}:")
            for tf in ['1h', '4h', '1d', '1w']:
                df = ctx.df_cache.get(tf, {}).get(sym)
                if df is not None:
                    print(f"     ✅ {tf}: {len(df)} candles")
                else:
                    print(f"     ❌ {tf}: no data")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_real_batch())
