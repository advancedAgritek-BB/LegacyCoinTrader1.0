#!/usr/bin/env python3
"""
Simple test to debug why dip_hunter strategy isn't generating signals.
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from crypto_bot.strategy.dip_hunter import generate_signal

def test_strategy_with_clear_signal():
    """Test the strategy with data that should clearly generate a signal."""

    print("🔍 === SIMPLE STRATEGY TEST ===\n")

    # Create very clear dip scenario
    dates = pd.date_range('2024-01-01', periods=200, freq='h')

    # Create price series with a clear 5% dip
    prices = []
    base_price = 50000.0

    # Normal trending up
    for i in range(100):
        price = base_price * (1 + i * 0.001)  # Gradual uptrend
        prices.append(price)

    # Sharp 5% dip over 3 bars
    dip_start = prices[-1]
    prices.append(dip_start * 0.98)  # -2%
    prices.append(dip_start * 0.96)  # -4%
    prices.append(dip_start * 0.95)  # -5%

    # Recovery
    for i in range(10):
        prices.append(prices[-1] * 1.002)  # Slow recovery

    # Continue with normal data
    for i in range(200 - len(prices)):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.002)))

    # Create OHLCV with clear oversold conditions
    df = pd.DataFrame({
        'timestamp': dates[:len(prices)],
        'open': prices,
        'close': prices,
        'high': [p * 1.002 for p in prices],
        'low': [p * 0.998 for p in prices],
        'volume': [1000.0] * len(prices)  # High volume for volume spike
    })

    df = df.set_index('timestamp')

    print(f"✅ Created test data with {len(df)} bars")
    print(f"💰 Price range: \${df['close'].min():.2f} - \${df['close'].max():.2f}")
    print(f"📊 Dip at index ~100: {df['close'].iloc[100]:.2f} -> {df['close'].iloc[103]:.2f} ({((df['close'].iloc[103] - df['close'].iloc[100]) / df['close'].iloc[100] * 100):.1f}%)")

    # Test with very permissive parameters
    config = {
        'dip_hunter': {
            'rsi_window': 10,
            'rsi_oversold': 40.0,  # Very permissive
            'dip_pct': 0.01,  # Very small dip requirement
            'dip_bars': 2,  # Short dip window
            'vol_window': 10,
            'vol_mult': 1.1,  # Very low volume requirement
            'adx_window': 10,
            'adx_threshold': 40.0,  # High threshold to allow trending
            'bb_window': 10,
            'ema_trend': 50,
            'ml_weight': 0.0,  # No ML to simplify
            'ema_slow': 10,
            'atr_normalization': False  # Disable for simplicity
        }
    }

    print("\n🔧 Using VERY permissive parameters:")
    for key, value in config['dip_hunter'].items():
        print(f"   {key}: {value}")

    # Test the dip period specifically
    print("\n🎯 Testing the dip period (bars 95-110)...")
    signals_found = 0

    for i in range(95, min(110, len(df))):
        subset = df.iloc[max(0, i-50):i+1]  # 50 bars of context

        if len(subset) < 30:
            continue

        score, direction = generate_signal(subset, symbol='BTC/USD', timeframe='1h', config=config)

        if score > 0 and direction == 'long':
            signals_found += 1
            print(f"🎯 SIGNAL FOUND at bar {i}!")
            print(f"   Score: {score:.3f}")
            print(f"   Price: \${subset['close'].iloc[-1]:.2f}")

            # Show recent price action
            recent = subset.tail(5)
            print("   Recent prices:")
            for j, (idx, row) in enumerate(recent.iterrows()):
                marker = "📈" if j == len(recent)-1 else "   "
                print(f"{marker} {idx.strftime('%m-%d %H:%M')}: \${row['close']:.2f}")
        elif i % 5 == 0:  # Show some debug info
            score, direction = generate_signal(subset, symbol='BTC/USD', timeframe='1h', config=config)
            print(f"   Bar {i}: Score={score:.3f}, Direction={direction}, Price=\${subset['close'].iloc[-1]:.2f}")

    if signals_found == 0:
        print("❌ NO SIGNALS FOUND even with very permissive parameters!")
        print("💡 This indicates a fundamental issue with the strategy logic")

        # Test with just the current bar
        print("\n🔬 Testing just the dip bar...")
        dip_bar = df.iloc[103:104]  # The lowest point of the dip
        score, direction = generate_signal(dip_bar, symbol='BTC/USD', timeframe='1h', config=config)
        print(f"   Single bar result: Score={score:.3f}, Direction={direction}")

        # Test with more context around the dip
        print("\n🔬 Testing 20 bars around dip...")
        dip_context = df.iloc[90:110]
        score, direction = generate_signal(dip_context, symbol='BTC/USD', timeframe='1h', config=config)
        print(f"   20-bar context result: Score={score:.3f}, Direction={direction}")

    else:
        print(f"✅ Found {signals_found} signals - the system works!")

if __name__ == "__main__":
    test_strategy_with_clear_signal()
