#!/usr/bin/env python3
"""
Script to reset failed symbols cache to allow disabled symbols to be traded again.
"""

import sys
import os

# Add the crypto_bot directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crypto_bot'))

from crypto_bot.utils.market_loader import reset_failed_symbols, get_failed_symbols_info

def main():
    """Reset failed symbols and show status."""
    print("🔄 Resetting failed symbols cache...")

    # Show current failed symbols before reset
    failed_info = get_failed_symbols_info()
    if failed_info:
        print(f"📊 Found {len(failed_info)} failed/disabled symbols:")
        for symbol, info in failed_info.items():
            status = "DISABLED" if info.get("disabled") else "RETRYING"
            count = info.get("count", 0)
            print(f"  - {symbol}: {status} (failures: {count})")
    else:
        print("✅ No failed symbols found - cache is clean")

    # Reset all failed symbols
    reset_count = reset_failed_symbols()
    print(f"✅ Reset {reset_count} symbols")

    # Verify the reset worked
    remaining = get_failed_symbols_info()
    if remaining:
        print(f"⚠️  {len(remaining)} symbols still in cache after reset")
    else:
        print("🎉 All symbols successfully reset!")

    print("\n💡 Tip: You can also reset specific symbols by passing them as arguments:")
    print("   python reset_failed_symbols.py BTC/USD ETH/USD")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Reset specific symbols
        symbols_to_reset = sys.argv[1:]
        print(f"🔄 Resetting specific symbols: {symbols_to_reset}")
        reset_count = reset_failed_symbols(symbols_to_reset)
        print(f"✅ Reset {reset_count} symbols")
    else:
        main()
