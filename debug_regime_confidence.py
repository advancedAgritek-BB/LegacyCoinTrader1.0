#!/usr/bin/env python3
"""
Debug script to investigate why regime confidence is 0.0.
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
from crypto_bot.regime.regime_classifier import classify_regime_async

async def debug_regime_confidence():
    """Debug why regime confidence is 0.0."""

    print("🔍 Debugging Regime Confidence")
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

        # Test regime classification directly
        print("\n🏛️ Testing regime classification...")

        for symbol in test_symbols:
            if '1h' in ctx.df_cache and symbol in ctx.df_cache['1h']:
                df = ctx.df_cache['1h'][symbol]
                print(f"\n   Testing regime for {symbol} ({len(df)} candles)...")

                # Test with different higher timeframes
                for higher_tf in ['4h', '1d', '1w']:
                    if higher_tf in ctx.df_cache and symbol in ctx.df_cache[higher_tf]:
                        higher_df = ctx.df_cache[higher_tf][symbol]
                        print(f"\n   Testing {higher_tf} higher timeframe ({len(higher_df)} candles)...")

                        try:
                            regime, probs = await classify_regime_async(df, higher_df)
                            print(f"     Regime: {regime}")
                            print(f"     Probabilities: {probs}")
                            print(f"     Probabilities type: {type(probs)}")

                            # Check if probs is a dict and calculate confidence
                            if isinstance(probs, dict):
                                # Find the probability for the detected regime
                                regime_prob = probs.get(regime, 0.0)
                                print(f"     Regime probability: {regime_prob}")

                                # Calculate confidence as in market_analyzer.py
                                base_conf = float(regime_prob)
                                print(f"     Base confidence: {base_conf}")

                            elif isinstance(probs, (int, float)):
                                print(f"     Single probability value: {probs}")

                            else:
                                print(f"     Unexpected probabilities format: {probs}")

                        except Exception as e:
                            print(f"     ❌ Regime classification failed: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"     ❌ No {higher_tf} data for {symbol}")

                # Test with None higher_df (like in the analysis)
                print("\n   Testing with None higher_df (analysis scenario)...")
                try:
                    regime, probs = await classify_regime_async(df, None)
                    print(f"     Regime: {regime}")
                    print(f"     Probabilities: {probs}")

                    if isinstance(probs, dict):
                        regime_prob = probs.get(regime, 0.0)
                        print(f"     Regime probability: {regime_prob}")

                except Exception as e:
                    print(f"     ❌ Regime classification with None higher_df failed: {e}")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_regime_confidence())
