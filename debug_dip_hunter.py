#!/usr/bin/env python3
"""
Debug script to analyze why dip_hunter strategy is not generating signals.
"""

import sys
import os
sys.path.insert(0, '.')

import ccxt
import pandas as pd
import numpy as np
from crypto_bot.strategy.dip_hunter import generate_signal
from crypto_bot.utils.indicators import calculate_rsi, calculate_bollinger_bands
from crypto_bot.utils import stats
from dotenv import load_dotenv

load_dotenv('.env.local')

def debug_dip_hunter_signals():
    """Debug the dip hunter strategy with real market data."""

    print("🔍 === DIP HUNTER SIGNAL DEBUG ===\n")

    # Initialize Kraken exchange
    exchange = ccxt.kraken({
        'apiKey': os.getenv('KRAKEN_API_KEY', os.getenv('API_KEY')),
        'secret': os.getenv('KRAKEN_API_SECRET', os.getenv('API_SECRET')),
        'enableRateLimit': True,
    })

    print("📊 Fetching BTC/USD data from Kraken...")
    ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', limit=200)  # Last 200 hours

    if not ohlcv or len(ohlcv) < 100:
        print(f"❌ Insufficient data: {len(ohlcv) if ohlcv else 0} records")
        return

    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    print(f"✅ Loaded {len(df)} OHLCV records")
    print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
    print(f"💰 Current BTC price: \${df['close'].iloc[-1]:.2f}")
    # Test with default parameters
    config = {
        'dip_hunter': {
            'rsi_window': 14,
            'rsi_oversold': 30.0,
            'dip_pct': 0.03,
            'dip_bars': 3,
            'vol_window': 20,
            'vol_mult': 1.5,
            'adx_window': 14,
            'adx_threshold': 25.0,
            'bb_window': 20,
            'ema_trend': 200,
            'ml_weight': 0.5,
            'atr_normalization': True,
            'ema_slow': 20
        }
    }

    # Test the last 50 bars
    print("\n🔬 Testing last 50 bars for dip conditions...\n")

    signals_found = 0
    for i in range(50, len(df)):
        test_df = df.iloc[:i+1]

        # Manual calculation of conditions
        recent = test_df.tail(50)
        if len(recent) < 30:
            continue

        # RSI
        rsi = calculate_rsi(recent['close'], window=14)
        current_rsi = rsi.iloc[-1] if not rsi.empty else None

        # Bollinger Bands
        bb = calculate_bollinger_bands(recent['close'], window=20, num_std=2)
        bb_pct = (recent['close'].iloc[-1] - bb.lower.iloc[-1]) / (bb.upper.iloc[-1] - bb.lower.iloc[-1]) if bb.lower.iloc[-1] != bb.upper.iloc[-1] else 0.5

        # Volume conditions
        vol_ma = recent['volume'].rolling(20).mean().iloc[-1]
        current_vol = recent['volume'].iloc[-1]
        vol_spike = current_vol > vol_ma * 1.5 if vol_ma > 0 else False

        # Dip conditions
        recent_returns = recent['close'].pct_change().tail(3)
        dip_size = recent_returns.sum()
        is_dip = dip_size <= -0.03

        # Simple ADX approximation (simplified)
        high_diff = recent['high'].diff()
        low_diff = recent['low'].diff()
        dm_plus = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
        dm_minus = ((low_diff > high_diff) & (low_diff > 0)) * (-low_diff)

        atr_window = 14
        if len(recent) >= atr_window:
            tr = np.maximum(
                recent['high'] - recent['low'],
                np.maximum(
                    (recent['high'] - recent['close'].shift(1)).abs(),
                    (recent['low'] - recent['close'].shift(1)).abs()
                )
            )
            atr = tr.rolling(atr_window).mean().iloc[-1]

            if atr > 0:
                di_plus = 100 * (dm_plus.rolling(atr_window).mean().iloc[-1] / atr)
                di_minus = 100 * (dm_minus.rolling(atr_window).mean().iloc[-1] / atr)
                dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0
                adx = dx  # Simplified, should be smoothed
                range_bound = adx < 25.0
            else:
                range_bound = True
        else:
            range_bound = True

        # Oversold condition
        bb_condition = bb_pct < 0
        if not bb_condition and is_dip and bb_pct < 0.3:
            bb_condition = True
        oversold = current_rsi < 30.0 and bb_condition if current_rsi is not None else False

        # Check if all conditions met
        all_conditions = is_dip and oversold and vol_spike and range_bound

        if all_conditions:
            signals_found += 1
            print(f"🎯 SIGNAL FOUND at {recent.index[-1]}:")
            print(f"   Dip Size: {dip_size:.4f} ({dip_size*100:.1f}%)")
            print(f"   Current Price: \${recent['close'].iloc[-1]:.2f}")
            print(f"   Volume: {current_vol:.2f} BTC")
            print(f"   Volume Spike: {vol_spike} ({current_vol:.0f} > {vol_ma:.0f} * 1.5)")
            print(f"   Range Bound: {range_bound} (ADX approx: {adx:.1f})")
            print(f"   Oversold: {oversold} (RSI: {current_rsi:.1f}, BB%: {bb_pct:.2f})")
            print()

        # Show some sample conditions for debugging
        if i % 10 == 0 and signals_found == 0:
            print(f"📊 Sample at {recent.index[-1]}:")
            print(f"   Dip Size: {dip_size:.4f} ({dip_size*100:.1f}%)")
            print(f"   Current Price: \${recent['close'].iloc[-1]:.2f}")
            print(f"   Volume: {current_vol:.2f} BTC")
            print(f"   Volume Spike: {vol_spike} ({current_vol:.0f} vs {vol_ma:.0f})")
            print(f"   Range Bound: {range_bound}")
            print(f"   Oversold: {oversold} (RSI: {current_rsi:.1f}, BB%: {bb_pct:.2f})")
            print()

    if signals_found == 0:
        print(f"❌ NO SIGNALS FOUND in the last 50 bars!")
        print("💡 This suggests the market conditions don't meet dip_hunter criteria:")
        print("   - BTC is likely in a strong uptrend")
        print("   - No significant dips (3%+ drops)")
        print("   - RSI not oversold (< 30)")
        print("   - Volume not spiking")
        print("   - ADX indicates trending (not range-bound)")
    else:
        print(f"✅ Found {signals_found} potential signals")

    # Test the actual strategy function
    print("\n🔬 Testing actual generate_signal function...")
    score, direction = generate_signal(df.tail(100), symbol='BTC/USD', timeframe='1h', config=config)
    print(f"   Strategy Result: Score={score:.3f}, Direction={direction}")

if __name__ == "__main__":
    debug_dip_hunter_signals()
