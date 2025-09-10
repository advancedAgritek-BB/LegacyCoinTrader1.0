#!/usr/bin/env python3
"""
Simple startup script that bypasses symbol loading issues
"""

import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Patch symbol loading to use only supported symbols
def patched_get_filtered_symbols(exchange, config):
    supported = ['BTC/USD', 'ETH/USD', 'SOL/USD']
    return [(symbol, 1.0) for symbol in supported]

# Apply patch before importing main
import crypto_bot.utils.symbol_utils
crypto_bot.utils.symbol_utils.get_filtered_symbols = patched_get_filtered_symbols

print("✅ Symbol loading patched")
print("🚀 Starting bot with minimal configuration...")

# Import and run main
import crypto_bot.main
