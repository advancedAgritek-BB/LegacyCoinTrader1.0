#!/usr/bin/env python3
"""
Quick test to verify the config is being loaded correctly.
"""

import yaml
from pathlib import Path

# Load the config
config_path = Path("crypto_bot/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("🔍 Config verification:")
print(f"use_enhanced_ohlcv_fetcher: {config.get('use_enhanced_ohlcv_fetcher', 'NOT SET')}")
print(f"max_concurrent_ohlcv: {config.get('max_concurrent_ohlcv', 'NOT SET')}")
print(f"cycle_delay_seconds: {config.get('cycle_delay_seconds', 'NOT SET')}")
print(f"min_confidence_score: {config.get('min_confidence_score', 'NOT SET')}")

# Check if the enhanced fetcher is disabled
if config.get('use_enhanced_ohlcv_fetcher') == False:
    print("✅ Enhanced OHLCV fetcher is DISABLED")
else:
    print("❌ Enhanced OHLCV fetcher is ENABLED")
