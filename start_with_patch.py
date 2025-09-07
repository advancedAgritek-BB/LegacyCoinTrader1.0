#!/usr/bin/env python3
"""
Symbol loading patch to ensure only supported symbols are used
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Patch the symbol loading function
def patched_get_filtered_symbols(exchange, config):
    """Patched version that only returns supported symbols."""
    
    # Define supported symbols
    supported_symbols = [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
        "LINK/USD", "UNI/USD", "AAVE/USD", "AVAX/USD",
        "BTC/EUR", "ETH/EUR", "SOL/EUR", "ADA/EUR"
    ]
    
    # Return only supported symbols with default scores
    return [(symbol, 1.0) for symbol in supported_symbols]

# Apply the patch
import crypto_bot.utils.symbol_utils
crypto_bot.utils.symbol_utils.get_filtered_symbols = patched_get_filtered_symbols

print("✅ Symbol loading patch applied")

# Start the main application
if __name__ == "__main__":
    import crypto_bot.main
