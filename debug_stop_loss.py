#!/usr/bin/env python3
"""
Debug script to understand why stop loss is not triggering.
"""

import pandas as pd
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from crypto_bot.risk.exit_manager import should_exit, _assess_momentum_strength

def debug_stop_loss():
    """Debug stop loss trigger issue."""
    print("🔍 DEBUGGING STOP LOSS TRIGGER")
    print("=" * 50)
    
    # Create test data
    df = pd.DataFrame({
        "open": [100, 101, 102],
        "high": [100, 101, 102],
        "low": [100, 101, 102],
        "close": [100, 101, 102],
        "volume": [1000, 1000, 1000]
    })
    
    entry_price = 100.0
    current_price = 98.0  # 2% below entry
    trailing_stop = 99.0  # 1% below entry
    
    print(f"Entry price: {entry_price}")
    print(f"Current price: {current_price}")
    print(f"Trailing stop: {trailing_stop}")
    print(f"Price hit stop: {current_price < trailing_stop}")
    
    # Test momentum strength
    momentum_strength = _assess_momentum_strength(df)
    print(f"Momentum strength: {momentum_strength}")
    
    # Test with different configs
    configs = [
        {
            "name": "No momentum",
            "config": {
                "exit_strategy": {
                    "stop_loss_pct": 0.01,
                    "trailing_stop_pct": 0.008,
                    "take_profit_pct": 0.04,
                    "momentum_aware_exits": False
                }
            }
        },
        {
            "name": "High momentum threshold",
            "config": {
                "exit_strategy": {
                    "stop_loss_pct": 0.01,
                    "trailing_stop_pct": 0.008,
                    "take_profit_pct": 0.04,
                    "momentum_aware_exits": True,
                    "momentum_continuation": {
                        "min_momentum_strength": 0.95,
                        "min_continuation_probability": 0.95
                    }
                }
            }
        }
    ]
    
    for test_config in configs:
        print(f"\n--- Testing: {test_config['name']} ---")
        try:
            exit_signal, new_stop = should_exit(
                df, current_price, trailing_stop, test_config['config'],
                position_side="buy", entry_price=entry_price
            )
            print(f"Exit signal: {exit_signal}")
            print(f"New stop: {new_stop}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_stop_loss()
