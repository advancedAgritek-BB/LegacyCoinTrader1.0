#!/usr/bin/env python3
"""
Remove all BTC/USD trades and positions from TradeManager state for testing.

This script removes all records of BTC/USD trades and positions from the TradeManager
state file to allow for clean testing.
"""

import json
import sys
from pathlib import Path
from decimal import Decimal

def remove_btc_from_trade_manager():
    """Remove all BTC/USD records from TradeManager state."""

    state_file = Path('crypto_bot/logs/trade_manager_state.json')

    if not state_file.exists():
        print("❌ TradeManager state file not found")
        return False

    # Load current state
    with open(state_file, 'r') as f:
        state = json.load(f)

    print("🔍 Current state before removal:")
    print(f"  Total trades: {len(state.get('trades', []))}")
    print(f"  Total positions: {len(state.get('positions', {}))}")
    print(f"  BTC/USD in price cache: {'BTC/USD' in state.get('price_cache', {})}")

    # Find and track BTC/USD trades for statistics adjustment
    btc_trades = []
    non_btc_trades = []

    for trade in state.get('trades', []):
        if trade.get('symbol') == 'BTC/USD':
            btc_trades.append(trade)
        else:
            non_btc_trades.append(trade)

    # Calculate statistics to remove
    btc_volume = sum(float(trade['amount']) * float(trade['price']) for trade in btc_trades)
    btc_fees = sum(float(trade['fees']) for trade in btc_trades)

    # Remove BTC/USD trades
    state['trades'] = non_btc_trades

    # Remove BTC/USD position
    if 'BTC/USD' in state.get('positions', {}):
        btc_position = state['positions']['BTC/USD']
        btc_realized_pnl = float(btc_position['realized_pnl'])
        print(f"  Removing BTC/USD position with realized P&L: ${btc_realized_pnl:.2f}")
        del state['positions']['BTC/USD']
    else:
        btc_realized_pnl = 0.0

    # Remove BTC/USD from price cache
    if 'BTC/USD' in state.get('price_cache', {}):
        del state['price_cache']['BTC/USD']

    # Update statistics
    stats = state.get('statistics', {})
    stats['total_trades'] = len(state['trades'])
    stats['total_volume'] = float(Decimal(str(stats.get('total_volume', 0))) - Decimal(str(btc_volume)))
    stats['total_fees'] = float(Decimal(str(stats.get('total_fees', 0))) - Decimal(str(btc_fees)))
    stats['total_realized_pnl'] = float(Decimal(str(stats.get('total_realized_pnl', 0))) - Decimal(str(btc_realized_pnl)))

    print("\n📊 Statistics adjustments:")
    print(f"  Removed volume: ${btc_volume:.2f}")
    print(f"  Removed fees: ${btc_fees:.2f}")
    print(f"  Removed realized P&L: ${btc_realized_pnl:.2f}")

    # Save updated state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print("\n✅ Successfully removed BTC/USD records:")
    print(f"  Removed {len(btc_trades)} BTC/USD trades")
    print(f"  Removed 1 BTC/USD position")
    print(f"  Removed BTC/USD from price cache")
    print(f"  Updated statistics")

    print("\n🔍 Final state:")
    print(f"  Total trades: {len(state.get('trades', []))}")
    print(f"  Total positions: {len(state.get('positions', {}))}")
    print(f"  Final total volume: ${stats['total_volume']:.2f}")
    print(f"  Final total fees: ${stats['total_fees']:.2f}")
    print(f"  Final realized P&L: ${stats['total_realized_pnl']:.2f}")

    return True

def main():
    print("🧹 Removing all BTC/USD records from TradeManager for testing")
    print("=" * 60)

    try:
        success = remove_btc_from_trade_manager()
        if success:
            print("\n✅ BTC/USD records successfully removed!")
            print("The TradeManager state is now clean for testing.")
        else:
            print("\n❌ Failed to remove BTC/USD records")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error removing BTC/USD records: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
