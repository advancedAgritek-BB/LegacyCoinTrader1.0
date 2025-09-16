#!/usr/bin/env python3
"""Test script to verify volume calculation fix."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure local packages can be imported when running the script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_bot.risk.risk_manager import RiskManager, RiskConfig

def test_volume_calculation():
    """Test that volume uses the most recent complete candle."""

    # Create sample data similar to ANKR/USD from the logs
    timestamps = [1757975400000 - i * 3600000 for i in range(212)]  # 212 hours ago to now
    timestamps.reverse()  # chronological order

    # Create volume data with last candle having 0 volume (incomplete)
    volumes = np.random.uniform(1000, 5000, 211)  # 211 complete candles
    volumes = np.append(volumes, 0.0)  # Last candle incomplete with 0 volume

    # Create OHLC data
    closes = np.random.uniform(0.014, 0.016, 212)
    opens = closes + np.random.uniform(-0.001, 0.001, 212)
    highs = np.maximum(opens, closes) + np.random.uniform(0, 0.001, 212)
    lows = np.minimum(opens, closes) - np.random.uniform(0, 0.001, 212)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

    print(f"DataFrame length: {len(df)}")
    print(f"Last volume (potentially incomplete): {df['volume'].iloc[-1]:.6f}")
    print(f"Second-to-last volume (complete): {df['volume'].iloc[-2]:.6f}")

    # Calculate what the old logic would give
    vol_mean_old = df["volume"].rolling(20).mean().iloc[-1]
    current_volume_old = df["volume"].iloc[-1]  # Old logic: uses last candle
    vol_threshold_old = vol_mean_old * 0.1

    print("\nOld logic (using last candle):")
    print(f"Volume: {current_volume_old:.4f}, Mean: {vol_mean_old:.4f}, Threshold: {vol_threshold_old:.4f}")
    print(f"Would pass check: {current_volume_old >= vol_threshold_old}")

    # Calculate what the new logic gives
    vol_mean_new = df["volume"].rolling(20).mean().iloc[-1]
    current_volume_idx = -2 if len(df) >= 2 else -1
    current_volume_new = df["volume"].iloc[current_volume_idx]  # New logic: uses second-to-last
    vol_threshold_new = vol_mean_new * 0.1

    print("\nNew logic (using second-to-last candle):")
    print(f"Volume: {current_volume_new:.4f}, Mean: {vol_mean_new:.4f}, Threshold: {vol_threshold_new:.4f}")
    print(f"Would pass check: {current_volume_new >= vol_threshold_new}")

    # Test with RiskManager
    config = RiskConfig(
        symbol="ANKR/USD",
        volume_threshold_ratio=0.1,
        min_volume=0.0,
        max_drawdown=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )

    risk_manager = RiskManager(config)
    allowed, reason = risk_manager.allow_trade(df)

    print(f"\nRiskManager result: {allowed} - {reason}")

    return allowed

if __name__ == "__main__":
    test_volume_calculation()
