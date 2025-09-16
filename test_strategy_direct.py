#!/usr/bin/env python3
"""
Direct test of strategies to see what signals they generate.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.main import load_config
from crypto_bot.strategy.market_making_bot import generate_signal as market_making_signal
from crypto_bot.strategy.grid_bot import generate_signal as grid_signal
from crypto_bot.strategy.maker_spread import generate_signal as maker_spread_signal
from crypto_bot.execution.cex_executor import get_exchange
from crypto_bot.utils.market_loader import update_multi_tf_ohlcv_cache
from crypto_bot.phase_runner import BotContext

async def test_strategies_direct():
    """Test strategies directly with real data."""

    print("🧪 Testing Strategies Directly")
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

        # Test symbols
        test_symbols = ['BTC/USD']
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

        # Check what data we got
        for tf, tf_cache in ctx.df_cache.items():
            print(f"   Timeframe {tf}: {len(tf_cache)} symbols")
            for symbol, df in tf_cache.items():
                if df is not None and hasattr(df, 'shape'):
                    print(f"     {symbol}: {len(df)} candles")
                    print(f"       Latest close: {df['close'].iloc[-1]}")
                    print(f"       Latest volume: {df['volume'].iloc[-1] if 'volume' in df.columns else 'N/A'}")

        # Test strategies directly
        print("\n🧪 Testing strategies directly...")

        for symbol in test_symbols:
            if '1h' in ctx.df_cache and symbol in ctx.df_cache['1h']:
                df = ctx.df_cache['1h'][symbol]
                print(f"\n   Testing {symbol} with {len(df)} candles...")

                # Test market_making_bot
                try:
                    score, direction = market_making_signal(df, config)
                    print(f"   📊 Market Making: score={score:.3f}, direction='{direction}'")
                except Exception as e:
                    print(f"   ❌ Market Making failed: {e}")

                # Test grid_bot
                try:
                    score, direction = grid_signal(df, config)
                    print(f"   📊 Grid Bot: score={score:.3f}, direction='{direction}'")
                except Exception as e:
                    print(f"   ❌ Grid Bot failed: {e}")

                # Test maker_spread
                try:
                    score, direction = maker_spread_signal(df, config)
                    assert direction in {"long", "short", "none"}, (
                        f"Unexpected maker spread direction: {direction}"
                    )
                    print(f"   📊 Maker Spread: score={score:.3f}, direction='{direction}'")
                except Exception as e:
                    print(f"   ❌ Maker Spread failed: {e}")

                # Test with different configurations
                print("\n   🔧 Testing with modified config...")

                # Test with very low thresholds
                test_config = config.copy()
                test_config['min_confidence_score'] = 0.0
                test_config['signal_threshold'] = 0.0

                try:
                    score, direction = market_making_signal(df, test_config)
                    print(f"   📊 Market Making (low threshold): score={score:.3f}, direction='{direction}'")
                except Exception as e:
                    print(f"   ❌ Market Making (low threshold) failed: {e}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_strategies_direct())
