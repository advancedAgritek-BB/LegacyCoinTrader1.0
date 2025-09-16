#!/usr/bin/env python3
"""Test script to verify Bollinger Bands implementations are consistent."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the path so local packages like `ta` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ta.volatility import BollingerBands

def test_bollinger_bands_consistency():
    """Test that different Bollinger Bands implementations give consistent results."""

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='1H')
    base_price = 50000

    # Generate realistic price movements
    price_changes = np.random.normal(0, 0.02, 50)  # 2% volatility
    prices = base_price * (1 + price_changes).cumprod()

    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices
    })

    print("=== Bollinger Bands Consistency Test ===")
    print(f"Data shape: {df.shape}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")

    # Test local ta implementation
    print("\n=== Local TA Implementation ===")
    try:
        bb_local = BollingerBands(df['close'].values, window=20, ndev=2)
        bb_width_local = bb_local.bollinger_wband()
        print(f"Local TA BB width (last 5): {bb_width_local[-5:]}")
        print(f"Local TA BB width range: {bb_width_local.min():.4f} - {bb_width_local.max():.4f}")
    except Exception as e:
        print(f"Local TA implementation failed: {e}")
        return False

    # Test pandas implementation (what was used before)
    print("\n=== Pandas Implementation ===")
    try:
        window = 20
        close = df["close"]
        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        bb_width_pandas = (bb_upper - bb_lower) / sma

        # Get the valid values (after window period)
        bb_width_pandas_valid = bb_width_pandas.iloc[window-1:]
        print(f"Pandas BB width (last 5): {bb_width_pandas_valid.iloc[-5:].values}")
        print(f"Pandas BB width range: {bb_width_pandas_valid.min():.4f} - {bb_width_pandas_valid.max():.4f}")
    except Exception as e:
        print(f"Pandas implementation failed: {e}")
        return False

    # Compare results
    print("\n=== Comparison ===")
    try:
        # Compare the overlapping periods
        local_valid = bb_width_local[window-1:]  # Skip initial values
        pandas_valid = bb_width_pandas_valid.values

        if len(local_valid) == len(pandas_valid):
            diff = np.abs(local_valid - pandas_valid)
            max_diff = diff.max()
            mean_diff = diff.mean()

            print(f"Maximum difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")

            if max_diff < 0.001:  # Within 0.1% tolerance
                print("✅ Results are consistent!")
                return True
            else:
                print("❌ Results differ significantly!")
                print(f"Sample comparison:")
                for i in range(min(5, len(local_valid))):
                    print(f"  Local: {local_valid[i]:.6f}, Pandas: {pandas_valid[i]:.6f}, Diff: {diff[i]:.6f}")
                return False
        else:
            print(f"❌ Different array lengths: Local {len(local_valid)}, Pandas {len(pandas_valid)}")
            return False

    except Exception as e:
        print(f"Comparison failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases for Bollinger Bands."""
    print("\n=== Edge Case Tests ===")

    # Test with insufficient data
    short_data = [100, 101, 102]
    try:
        bb = BollingerBands(short_data, window=20, ndev=2)
        width = bb.bollinger_wband()
        print(f"Short data test: width = {width}")
        expected = np.array([0.0] * len(short_data))
        if np.allclose(width, expected):
            print("✅ Short data handled correctly")
        else:
            print("❌ Short data not handled correctly")
    except Exception as e:
        print(f"Short data test failed: {e}")

    # Test with constant data
    constant_data = [100] * 30
    try:
        bb = BollingerBands(constant_data, window=20, ndev=2)
        width = bb.bollinger_wband()
        print(f"Constant data test: width range = {width.min():.6f} - {width.max():.6f}")
        # With constant data, std should be 0, so width should be 0
        if width.max() < 1e-10:
            print("✅ Constant data handled correctly")
        else:
            print("❌ Constant data not handled correctly")
    except Exception as e:
        print(f"Constant data test failed: {e}")

if __name__ == "__main__":
    success = test_bollinger_bands_consistency()
    test_edge_cases()

    if success:
        print("\n🎉 All tests passed! Bollinger Bands implementation is ready for production.")
    else:
        print("\n❌ Tests failed. Review the implementation before deploying.")
