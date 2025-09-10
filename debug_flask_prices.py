#!/usr/bin/env python3
"""
Debug script to test Flask get_open_positions function directly.

This script calls the get_open_positions function directly to see
the debugging output and understand why current_price is None.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_get_open_positions():
    """Debug the get_open_positions function directly."""
    print("🔍 Debugging get_open_positions function...")

    try:
        # Import the function directly
        from frontend.app import get_open_positions

        print("✅ Successfully imported get_open_positions")

        # Call the function
        print("📡 Calling get_open_positions()...")
        positions = get_open_positions()

        print(f"📊 Function returned {len(positions)} positions")

        if positions:
            # Show details of first position
            pos = positions[0]
            print(f"📋 First position details:")
            print(f"   Symbol: {pos.get('symbol')}")
            print(f"   Current Price: {pos.get('current_price')}")
            print(f"   Entry Price: {pos.get('entry_price')}")
            print(f"   PnL: {pos.get('pnl')}")

            # Check if current_price is None for all positions
            none_count = sum(1 for p in positions if p.get('current_price') is None)
            print(f"⚠️ Positions with None current_price: {none_count}/{len(positions)}")

        return positions

    except Exception as e:
        print(f"❌ Error in debug_get_open_positions: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_trade_manager_cache():
    """Debug the TradeManager price cache directly."""
    print("🔍 Debugging TradeManager price cache...")

    try:
        from crypto_bot.utils.trade_manager import get_trade_manager

        tm = get_trade_manager()
        print("✅ TradeManager loaded")

        cache = tm.price_cache
        print(f"📊 Price cache contains {len(cache)} entries:")

        for symbol, price in cache.items():
            print(f"   {symbol}: ${price}")

        if not cache:
            print("⚠️ Price cache is empty!")

        # Check positions
        positions = tm.get_all_positions()
        print(f"📊 TradeManager has {len(positions)} open positions:")

        for pos in positions:
            cached_price = cache.get(pos.symbol)
            print(f"   {pos.symbol}: cached_price={cached_price}, entry_price=${pos.average_price}")

        return cache, positions

    except Exception as e:
        print(f"❌ Error in debug_trade_manager_cache: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main debug function."""
    print("=" * 60)
    print("FLASK PRICE DEBUGGING")
    print("=" * 60)

    # Debug TradeManager cache first
    cache, tm_positions = debug_trade_manager_cache()

    print("\n" + "=" * 40)

    # Debug get_open_positions function
    positions = debug_get_open_positions()

    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)

    if cache is not None and tm_positions is not None:
        print(f"✅ TradeManager cache: {len(cache)} entries")
        print(f"✅ TradeManager positions: {len(tm_positions)} positions")

    if positions is not None:
        print(f"✅ get_open_positions returned: {len(positions)} positions")

        # Check for issues
        none_prices = [p for p in positions if p.get('current_price') is None]
        if none_prices:
            print(f"❌ ISSUE: {len(none_prices)} positions have current_price = None")
            print("💡 This means the Flask API is not fetching prices correctly")
        else:
            print("✅ All positions have current_price values")

    print("\n💡 If current_price is None, check:")
    print("   1. Is the exchange properly configured?")
    print("   2. Are API keys valid?")
    print("   3. Is the exchange rate limiting?")


if __name__ == "__main__":
    main()
