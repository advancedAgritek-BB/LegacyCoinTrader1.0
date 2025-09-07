#!/usr/bin/env python3
"""
Diagnostic script to investigate why strategy evaluation returns 'none' direction for all signals.
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
from crypto_bot.phase_runner import BotContext
import pandas as pd

async def diagnose_strategy_none():
    """Diagnose why strategy evaluation returns 'none' direction."""

    print("🔍 Diagnosing Strategy 'None' Direction Issues")
    print("=" * 50)

    try:
        # Load configuration
        print("\n📄 Loading configuration...")
        config = load_config()
        print(f"✅ Config loaded. Mode: {config.get('mode', 'cex')}")

        # Create a simple mock exchange for testing
        class MockExchange:
            def __init__(self):
                self.id = "kraken"
                self.symbols = ["BTC/USD", "ETH/USD"]
                self.markets = {"BTC/USD": {}, "ETH/USD": {}}
                self.rateLimit = 100

        exchange = MockExchange()

        # Create fetcher
        print("\n🔧 Creating EnhancedOHLCVFetcher...")
        fetcher = EnhancedOHLCVFetcher(exchange, config)
        print("✅ Fetcher created successfully")

        # Test symbols from recent logs
        test_symbols = ['BTC/USD', 'ETH/USD']
        timeframe = "1h"

        print(f"\n🎯 Testing with symbols: {test_symbols}")
        print(f"   Main timeframe: {timeframe}")

        # Try to fetch real data for these symbols
        print("\n📡 Attempting to fetch real OHLCV data...")

        # Create mock OHLCV data that resembles real data
        sample_data = [
            [1640995200000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],  # 2 hours ago
            [1640998800000, 50500.0, 51500.0, 50000.0, 51000.0, 150.0],  # 1 hour ago
            [1641002400000, 51000.0, 52000.0, 50500.0, 51500.0, 200.0],  # current
        ]

        # Create dataframes for different timeframes
        df_1h = pd.DataFrame(sample_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_4h = pd.DataFrame([
            [1640980800000, 49000.0, 52000.0, 48500.0, 51500.0, 500.0],  # 4 hours ago
            [1640995200000, 51500.0, 52500.0, 51000.0, 52000.0, 600.0],  # current
        ], columns=["timestamp", "open", "high", "low", "close", "volume"])

        df_1d = pd.DataFrame([
            [1640918400000, 48000.0, 53000.0, 47500.0, 52000.0, 2000.0],  # yesterday
            [1641004800000, 52000.0, 53500.0, 51500.0, 52500.0, 1800.0],  # today
        ], columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Test data availability
        print("\n📊 Testing data availability...")
        for symbol in test_symbols:
            print(f"   {symbol}:")
            print(f"     1h data: {len(df_1h)} candles")
            print(f"     4h data: {len(df_4h)} candles")
            print(f"     1d data: {len(df_1d)} candles")

        # Test individual symbol analysis
        print("\n🔬 Testing individual symbol analysis...")

        for symbol in test_symbols[:1]:  # Test just one symbol first
            print(f"\n   Testing {symbol}...")

            # Create df_map with multiple timeframes
            df_map = {
                "1h": df_1h,
                "4h": df_4h,
                "1d": df_1d
            }

            # Test analysis
            try:
                result = await analyze_symbol(
                    symbol=symbol,
                    df_map=df_map,
                    mode="cex",
                    config=config
                )

                print(f"   ✅ Analysis completed for {symbol}")
                print(f"     Result: {result}")

                # Check key fields
                print(f"     Direction: {result.get('direction', 'MISSING')}")
                print(f"     Score: {result.get('score', 'MISSING')}")
                print(f"     Regime: {result.get('regime', 'MISSING')}")
                print(f"     Confidence: {result.get('confidence', 'MISSING')}")
                print(f"     Skip reason: {result.get('skip', 'None')}")

                if result.get('direction') == 'none':
                    print(f"   ⚠️  Direction is 'none' - investigating why...")

                    # Check if data requirements are met
                    main_tf = config.get("timeframe", "1h")
                    main_df = df_map.get(main_tf)
                    lookback = config.get("indicator_lookback", 14) * 2

                    if main_df is None:
                        print(f"     ❌ No data for main timeframe {main_tf}")
                    elif len(main_df) < lookback:
                        print(f"     ❌ Insufficient data: {len(main_df)} < {lookback} required")
                    else:
                        print(f"     ✅ Data requirements met: {len(main_df)} >= {lookback}")

            except Exception as e:
                print(f"   ❌ Analysis failed for {symbol}: {e}")
                import traceback
                traceback.print_exc()

        # Test regime classification
        print("\n🏛️  Testing regime classification...")

        try:
            from crypto_bot.regime.regime_classifier import classify_regime_async

            for symbol in test_symbols[:1]:
                df = df_map.get("1h")
                higher_df = df_map.get("1d")

                if df is not None:
                    regime, probs = await classify_regime_async(df, higher_df)
                    print(f"   {symbol} regime: {regime}, confidence: {probs}")
                else:
                    print(f"   ❌ No data for regime classification of {symbol}")

        except Exception as e:
            print(f"   ❌ Regime classification failed: {e}")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_strategy_none())
