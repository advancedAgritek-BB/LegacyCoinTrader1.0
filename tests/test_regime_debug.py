#!/usr/bin/env python3
"""Debug script to test regime classification and identify why it's returning 100% unknown."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure the project root is on the Python path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_bot.regime.regime_classifier import classify_regime, _classify_core, CONFIG
from crypto_bot.regime.pattern_detector import detect_patterns
from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "test_regime.log")

def create_sample_data():
    """Create sample OHLCV data for testing."""
    # Generate 50 candles of sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='1H')
    base_price = 50000

    # Generate realistic price movements
    price_changes = np.random.normal(0, 0.02, 50)  # 2% volatility
    prices = base_price * (1 + price_changes).cumprod()

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + np.random.uniform(0, 0.01))  # Small random high
        low = price * (1 - np.random.uniform(0, 0.01))   # Small random low
        volume = np.random.uniform(100, 1000)  # Random volume
        data.append({
            'timestamp': int(dates[i].timestamp() * 1000),
            'open': prices[i-1] if i > 0 else price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })

    df = pd.DataFrame(data)
    return df

def test_regime_classification():
    """Test the regime classification pipeline."""
    print("=== Testing Regime Classification ===")

    # Create sample data
    df = create_sample_data()
    print(f"Sample data shape: {df.shape}")
    print(f"Sample data columns: {df.columns.tolist()}")
    print(f"Sample data (last 5 rows):\n{df.tail()}")

    # Test pattern detection
    print("\n=== Testing Pattern Detection ===")
    patterns = detect_patterns(df)
    print(f"Detected patterns: {patterns}")

    # Test core classification
    print("\n=== Testing Core Classification ===")
    regime = _classify_core(df, CONFIG)
    print(f"Core classification result: {regime}")

    # Test full classification
    print("\n=== Testing Full Classification ===")
    try:
        result = classify_regime(df)
        print(f"Full classification result: {result}")
        if isinstance(result, tuple) and len(result) >= 2:
            regime, probs = result
            print(f"Regime: {regime}")
            print(f"Probabilities: {probs}")
        else:
            print(f"Unexpected result format: {type(result)}")
    except Exception as e:
        print(f"Error in full classification: {e}")
        import traceback
        traceback.print_exc()

    # Debug configuration
    print("\n=== Configuration Check ===")
    print(f"ML enabled: {CONFIG.get('use_ml_regime_classifier', False)}")
    print(f"ML min bars: {CONFIG.get('ml_min_bars', 20)}")
    print(f"Data has {len(df)} rows")
    print(f"Required columns present: {all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])}")

    # Test with higher timeframe data
    print("\n=== Testing with Higher Timeframe ===")
    # Create higher timeframe data (4h)
    df_high_tf = df.copy()
    df_high_tf['timestamp'] = pd.to_datetime(df_high_tf['timestamp'], unit='ms')
    higher_df = (
        df_high_tf
        .set_index('timestamp')
        .resample('4H')
        .agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        .dropna()
        .reset_index()
    )
    higher_df['timestamp'] = (higher_df['timestamp'].astype('int64') // 1_000_000).astype(int)

    print(f"Higher TF data shape: {higher_df.shape}")

    try:
        result = classify_regime(df, higher_df)
        print(f"Classification with higher TF: {result}")
    except Exception as e:
        print(f"Error with higher TF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_regime_classification()
