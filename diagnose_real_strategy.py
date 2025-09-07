#!/usr/bin/env python3
"""
Diagnostic script to investigate strategy evaluation issues using real exchange data.
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
from crypto_bot.phase_runner import BotContext
from crypto_bot.utils.market_loader import update_multi_tf_ohlcv_cache

async def diagnose_real_strategy():
    """Diagnose strategy evaluation using real exchange data."""

    print("🔍 Diagnosing Strategy Evaluation with Real Data")
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
            # Try to use cached symbols if available
            if hasattr(exchange, 'symbols') and exchange.symbols:
                print(f"✅ Using cached markets: {len(exchange.symbols)} symbols")
            else:
                print("❌ No market data available")
                return

        # Create a context similar to what the bot uses
        print("\n🔧 Creating bot context...")
        ctx = BotContext(
            positions={},
            df_cache={},
            regime_cache={},
            config=config
        )
        ctx.exchange = exchange

        print("✅ Bot context created")

        # Test symbols from recent batch
        test_symbols = ['BTC/USD', 'ETH/USD']
        print(f"\n🎯 Testing with symbols: {test_symbols}")

        # Test the actual cache update process with real exchange
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
            total_candles = 0
            for tf, tf_cache in ctx.df_cache.items():
                print(f"   Timeframe {tf}: {len(tf_cache)} symbols")
                for symbol, df in tf_cache.items():
                    if df is not None and hasattr(df, 'shape'):
                        candles = len(df)
                        total_candles += candles
                        print(f"     {symbol}: {candles} candles")
                        if candles > 0:
                            print(f"       Sample: {df.iloc[0].to_dict()}")
                    else:
                        print(f"     {symbol}: None or invalid")

            print(f"\n📈 Total data fetched: {total_candles} candles across all timeframes")

        except Exception as e:
            print(f"❌ Cache update failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Test analysis with the real fetched data
        print("\n🔬 Testing analysis with real data...")

        for symbol in test_symbols:
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
            print(f"   Data points per timeframe: {[len(df) if hasattr(df, '__len__') else 'N/A' for df in df_map.values()]}")

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
                print(f"     This would cause 'insufficient_history' skip")
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

                    # Check key fields
                    direction = result.get('direction', 'MISSING')
                    score = result.get('score', 'MISSING')
                    regime = result.get('regime', 'MISSING')
                    confidence = result.get('confidence', 'MISSING')
                    skip_reason = result.get('skip', 'None')

                    print(f"     Direction: {direction}")
                    print(f"     Score: {score}")
                    print(f"     Regime: {regime}")
                    print(f"     Confidence: {confidence}")
                    print(f"     Skip reason: {skip_reason}")

                    if direction == 'none':
                        print(f"   ⚠️  Direction is 'none' - investigating why...")

                        # Check regime
                        if regime == 'unknown':
                            print(f"     ❌ Regime is 'unknown' - classification failed")

                        # Check confidence
                        if confidence is not None and confidence < config.get('min_confidence_score', 0.0):
                            print(f"     ❌ Low confidence: {confidence} < {config.get('min_confidence_score', 0.0)}")

                        # Check score
                        if score is not None and score < config.get('signal_threshold', 0.0):
                            print(f"     ❌ Low score: {score} < {config.get('signal_threshold', 0.0)}")

                    else:
                        print(f"   ✅ Valid signal generated: {direction}")

                except Exception as e:
                    print(f"   ❌ Analysis failed: {e}")
                    import traceback
                    traceback.print_exc()

        # Test regime classification
        print("\n🏛️  Testing regime classification with real data...")

        try:
            from crypto_bot.regime.regime_classifier import classify_regime_async

            for symbol in test_symbols[:1]:  # Test just one symbol
                if symbol in ctx.df_cache.get('1h', {}):
                    df = ctx.df_cache['1h'][symbol]
                    higher_df = ctx.df_cache.get('1d', {}).get(symbol)

                    if df is not None and len(df) > 10:  # Need some minimum data
                        regime, probs = await classify_regime_async(df, higher_df)
                        print(f"   {symbol} regime: {regime}")
                        if isinstance(probs, dict):
                            print(f"     Probabilities: {probs}")
                        elif isinstance(probs, (int, float)):
                            print(f"     Confidence: {probs}")
                        else:
                            print(f"     Probabilities type: {type(probs)}")
                    else:
                        print(f"   ❌ Insufficient data for regime classification of {symbol}")
                else:
                    print(f"   ❌ No data for regime classification of {symbol}")

        except Exception as e:
            print(f"   ❌ Regime classification failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_real_strategy())
