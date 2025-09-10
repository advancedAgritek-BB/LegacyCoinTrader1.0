#!/usr/bin/env python3
"""
Diagnostic script to check P&L calculation issues.
"""

import json
import sys
import os
from datetime import datetime
from decimal import Decimal

# Add the project root to Python path
sys.path.insert(0, '/Users/brandonburnette/Downloads/LegacyCoinTrader1.0')

from crypto_bot.utils.trade_manager import get_trade_manager
from crypto_bot.execution.kraken_ws import KrakenWSClient

def check_trade_manager_status():
    """Check trade manager state."""
    print("=== TRADE MANAGER STATUS ===")

    try:
        trade_manager = get_trade_manager()

        # Force reload state from disk
        print("Forcing reload from state file...")
        trade_manager._load_state()

        # Check price cache
        print(f"Price cache size: {len(trade_manager.price_cache)}")
        if trade_manager.price_cache:
            print("Price cache contents:")
            for symbol, price in trade_manager.price_cache.items():
                print(f"  {symbol}: {price}")
        else:
            print("Price cache is EMPTY!")

        # Check positions
        open_positions = trade_manager.get_all_positions()
        closed_positions = trade_manager.get_closed_positions()
        print(f"\nOpen positions: {len(open_positions)}")
        print(f"Closed positions: {len(closed_positions)}")

        if open_positions:
            print("Open Positions:")
            for pos in open_positions:
                print(f"  {pos.symbol}: {pos.side} {pos.total_amount} @ {pos.average_price}")
        else:
            print("No open positions found!")

        if closed_positions:
            print("\nClosed Positions:")
            total_realized_pnl = Decimal('0')
            for pos in closed_positions:
                print(".6f")
                total_realized_pnl += pos.realized_pnl
            print(".6f")

        # Check trades
        print(f"\nTrades count: {len(trade_manager.trades)}")
        if trade_manager.trades:
            print("Recent trades:")
            for trade in trade_manager.trades[-3:]:  # Show last 3
                print(f"  {trade.symbol}: {trade.side} {trade.amount} @ {trade.price}")

        # Check total stats
        print("\nTrading Statistics:")
        print(f"  Total trades: {trade_manager.total_trades}")
        print(f"  Total volume: ${trade_manager.total_volume:,.2f}")
        print(f"  Total fees: ${trade_manager.total_fees:,.2f}")
        print(f"  Total realized P&L: ${trade_manager.total_realized_pnl:,.2f}")

    except Exception as e:
        print(f"Error checking trade manager: {e}")
        import traceback
        traceback.print_exc()

def check_websocket_status():
    """Check WebSocket client status."""
    print("\n=== WEBSOCKET STATUS ===")

    try:
        ws_client = KrakenWSClient()

        # Check price cache
        print(f"WS Price cache size: {len(ws_client.price_cache)}")
        if ws_client.price_cache:
            print("WS Price cache contents:")
            for symbol, data in ws_client.price_cache.items():
                print(f"  {symbol}: {data}")
        else:
            print("WS Price cache is EMPTY!")

        # Check connection status
        print(f"Public WS connected: {ws_client.public_ws is not None}")
        print(f"Private WS connected: {ws_client.private_ws is not None}")
        print(f"WS Token: {ws_client.token is not None}")

    except Exception as e:
        print(f"Error checking WebSocket: {e}")

def check_frontend_data():
    """Check what the frontend would see."""
    print("\n=== FRONTEND DATA CHECK ===")

    try:
        from crypto_bot.utils.trade_manager import get_trade_manager
        trade_manager = get_trade_manager()

        positions = trade_manager.get_all_positions()
        positions_data = []

        for position in positions:
            # Get current price for unrealized PnL calculation
            current_price = trade_manager.price_cache.get(position.symbol)
            print(f"Position {position.symbol}:")
            print(f"  Current price from cache: {current_price}")
            print(f"  Position avg price: {position.average_price}")
            print(f"  Position amount: {position.total_amount}")

            if current_price:
                unrealized_pnl, unrealized_pct = position.calculate_unrealized_pnl(current_price)
                print(f"  Unrealized P&L: ${unrealized_pnl}, {unrealized_pct}%")
            else:
                print("  NO CURRENT PRICE - P&L will be $0.00!")
    except Exception as e:
        print(f"Error checking frontend data: {e}")

def main():
    """Run all diagnostics."""
    print("P&L Diagnostic Report")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")

    check_trade_manager_status()
    check_websocket_status()
    check_frontend_data()

    print("\n=== SUMMARY ===")
    print("If price caches are empty, that's why P&L shows $0.00")
    print("The issue is likely:")
    print("1. WebSocket not connected/receiving data")
    print("2. Position monitor not running")
    print("3. Price updates not being propagated to trade manager")

if __name__ == "__main__":
    main()
