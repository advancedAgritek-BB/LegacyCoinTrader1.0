#!/usr/bin/env python3
"""
Trade Synchronization Script

This script syncs trades from the CSV log to the TradeManager to fix the frontend display issue.
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import json

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.trade_manager import get_trade_manager, create_trade
from crypto_bot.utils.logger import LOG_DIR

def sync_csv_trades_to_trade_manager():
    """Sync all trades from trades.csv to TradeManager."""

    csv_path = LOG_DIR / "trades.csv"
    if not csv_path.exists():
        print(f"❌ trades.csv not found at {csv_path}")
        return False

    print("📊 Trade Synchronization Script")
    print(f"📁 CSV file: {csv_path}")
    print("=" * 60)

    # Get TradeManager instance
    trade_manager = get_trade_manager()

    # Read existing trades from TradeManager
    existing_trade_ids = {trade.id for trade in trade_manager.trades}
    print(f"📈 TradeManager currently has {len(existing_trade_ids)} trades")

    # Read CSV file
    csv_trades = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 5:
                symbol, side, amount_str, price_str, timestamp_str = row[:5]
                csv_trades.append({
                    'symbol': symbol,
                    'side': side,
                    'amount': amount_str,
                    'price': price_str,
                    'timestamp': timestamp_str
                })

    print(f"📄 CSV file contains {len(csv_trades)} trades")

    # Sync trades
    synced_count = 0
    skipped_count = 0
    error_count = 0

    for trade_data in csv_trades:
        try:
            # Create a unique ID for this trade
            trade_key = f"{trade_data['symbol']}_{trade_data['side']}_{trade_data['amount']}_{trade_data['price']}_{trade_data['timestamp']}"

            # Skip if already exists
            if trade_key in existing_trade_ids:
                skipped_count += 1
                continue

            # Parse timestamp
            try:
                # Handle Unix timestamp (milliseconds)
                timestamp_ms = int(trade_data['timestamp'])
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            except ValueError:
                # Handle ISO format
                timestamp = datetime.fromisoformat(trade_data['timestamp'])

            # Create trade object (without timestamp - let it use current time)
            trade = create_trade(
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                amount=Decimal(trade_data['amount']),
                price=Decimal(trade_data['price']),
                strategy="csv_sync",
                exchange="kraken"
            )

            # Override the timestamp after creation
            trade.timestamp = timestamp

            # Override the ID to use our trade_key for deduplication
            trade.id = trade_key

            # Record to TradeManager
            trade_manager.record_trade(trade)
            synced_count += 1

            if synced_count % 10 == 0:
                print(f"✅ Synced {synced_count} trades...")

        except Exception as e:
            print(f"❌ Error syncing trade {trade_data}: {e}")
            error_count += 1

    # Save state
    trade_manager.save_state()

    # Summary
    print("\n" + "=" * 60)
    print("🎯 SYNCHRONIZATION COMPLETE")
    print(f"✅ Successfully synced: {synced_count} trades")
    print(f"⏭️  Skipped (already exist): {skipped_count} trades")
    print(f"❌ Errors: {error_count} trades")
    print(f"📊 TradeManager now has {len(trade_manager.trades)} total trades")
    print(f"📊 TradeManager has {len(trade_manager.positions)} open positions")

    return synced_count > 0

def verify_sync():
    """Verify that the sync worked by checking TradeManager state."""
    print("\n🔍 VERIFICATION")
    print("-" * 30)

    trade_manager = get_trade_manager()
    positions = trade_manager.get_all_positions()

    if positions:
        print(f"✅ Found {len(positions)} open positions:")
        for pos in positions:
            print(f"   • {pos.symbol}: {pos.side} {pos.total_amount} @ {pos.average_price}")
    else:
        print("❌ No positions found in TradeManager")

    return len(positions) > 0

if __name__ == "__main__":
    try:
        print("🚀 Starting trade synchronization...")

        # Perform sync
        success = sync_csv_trades_to_trade_manager()

        if success:
            # Verify
            verify_sync()
            print("\n🎉 SUCCESS: Trades have been synchronized to TradeManager!")
            print("📱 Your frontend dashboard should now display the positions.")
        else:
            print("\n❌ FAILED: No trades were synchronized.")

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        sys.exit(1)
