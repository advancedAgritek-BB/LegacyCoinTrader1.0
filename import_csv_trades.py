#!/usr/bin/env python3
"""
Import trades from CSV file into TradeManager.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from crypto_bot.utils.trade_manager import get_trade_manager, Trade

# Import the module to access the global variable
import crypto_bot.utils.trade_manager as tm_module

def import_csv_trades(csv_file_path: str) -> None:
    """Import trades from CSV into TradeManager."""
    print(f"Importing trades from {csv_file_path}...")

    # Reset the singleton instance to ensure we get a fresh TradeManager
    print("Resetting TradeManager singleton...")
    tm_module._trade_manager_instance = None

    # Get fresh TradeManager instance
    trade_manager = get_trade_manager()

    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Found {len(df)} trades in CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Get TradeManager instance
    trade_manager = get_trade_manager()

    # Convert CSV rows to Trade objects and record them
    imported_count = 0
    skipped_count = 0

    for _, row in df.iterrows():
        try:
            # Parse the timestamp - handle different formats
            timestamp_str = str(row['timestamp'])
            if 'T' in timestamp_str:
                # ISO format with T
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # Regular format
                timestamp = datetime.fromisoformat(timestamp_str)

            # Create trade object directly
            trade = Trade(
                id=str(uuid.uuid4()),
                symbol=row['symbol'],
                side=row['side'],
                amount=Decimal(str(row['amount'])),
                price=Decimal(str(row['price'])),
                timestamp=timestamp,
                strategy="imported",  # Mark as imported
                exchange="kraken",  # Default exchange
                fees=Decimal('0'),  # No fee data in CSV
                metadata={'imported_from_csv': True}
            )

            # Record the trade
            trade_id = trade_manager.record_trade(trade)
            imported_count += 1

            if imported_count % 10 == 0:
                print(f"Imported {imported_count} trades...")

        except Exception as e:
            print(f"Error importing trade {row.get('symbol', 'unknown')}: {e}")
            skipped_count += 1
            continue

    # Save the updated state
    print(f"\nBefore save - TradeManager has {len(trade_manager.trades)} trades and {len(trade_manager.positions)} positions")
    try:
        trade_manager.save_state()
        print("✅ State saved successfully")
    except Exception as e:
        print(f"❌ Error saving state: {e}")
        import traceback
        traceback.print_exc()

    # Verify the save worked
    print("\nVerifying save...")
    try:
        state_file = PROJECT_ROOT / 'crypto_bot' / 'logs' / 'trade_manager_state.json'
        with open(state_file, 'r') as f:
            import json
            saved_state = json.load(f)
            saved_trades = len(saved_state.get('trades', []))
            saved_positions = len(saved_state.get('positions', {}))
            print(f"Saved file has {saved_trades} trades and {saved_positions} positions")
    except Exception as e:
        print(f"Error checking saved file: {e}")

    print("\nImport Summary:")
    print(f"  ✅ Imported: {imported_count} trades")
    print(f"  ❌ Skipped: {skipped_count} trades")
    print(f"  📊 Total positions: {len(trade_manager.get_all_positions())}")

    # Show some position info
    positions = trade_manager.get_all_positions()
    if positions:
        print("\nCurrent Positions:")
        for pos in positions[:5]:  # Show first 5
            print(".6f")
        if len(positions) > 5:
            print(f"  ... and {len(positions) - 5} more")

def main():
    """Main function."""
    csv_path = PROJECT_ROOT / "crypto_bot" / "logs" / "trades.csv"

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    import_csv_trades(str(csv_path))

if __name__ == "__main__":
    main()
