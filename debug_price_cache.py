#!/usr/bin/env python3
"""
Debug script to check price cache status and force updates.
"""

import sys
import os
sys.path.append('/Users/brandonburnette/Downloads/LegacyCoinTrader1.0')

from crypto_bot.utils.trade_manager import get_trade_manager
from crypto_bot.utils.price_monitor import get_price_monitor, start_price_monitoring
import json
from pathlib import Path

def main():
    print("=== Price Cache Debug ===")

    # Get trade manager
    tm = get_trade_manager()
    print(f"Trade manager price cache: {dict(tm.price_cache)}")

    # Get price monitor
    pm = get_price_monitor(tm)
    print(f"Price monitor status: {pm.get_price_status()}")

    # Load state file
    state_file = Path("crypto_bot/logs/trade_manager_state.json")
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
        print(f"State file price cache: {state.get('price_cache', {})}")

    # Get active positions
    positions = tm.get_all_positions()
    print(f"Active positions: {[pos.symbol for pos in positions]}")

    # Try to force update prices
    print("\n=== Force Updating Prices ===")
    symbols = {pos.symbol for pos in positions}
    if symbols:
        print(f"Updating prices for: {symbols}")

        # Import exchange if available
        try:
            from crypto_bot.main import get_exchange
            exchange = get_exchange()
        except:
            exchange = None
            print("No exchange available for price updates")

        # Force update (async call)
        import asyncio
        async def do_update():
            return await pm.update_prices_for_symbols(symbols, exchange)

        updated = asyncio.run(do_update())
        print(f"Updated prices: {updated}")

        # Check cache after update
        print(f"Trade manager price cache after update: {dict(tm.price_cache)}")

        # Save state
        tm.save_state()
        print("Trade manager state saved")

        # Reload state file
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            print(f"State file price cache after save: {state.get('price_cache', {})}")
    else:
        print("No positions to update")

if __name__ == "__main__":
    main()
