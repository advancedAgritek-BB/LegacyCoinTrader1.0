#!/usr/bin/env python3
"""
Clear All Positions and Trades Script

This script completely clears all trade data from the SingleSourceTradeManager
to provide a fresh start for the application.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from decimal import Decimal

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.single_source_trade_manager import get_single_source_trade_manager
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__)

def clear_all_positions():
    """Clear all positions, trades, and reset the TradeManager to a clean state."""

    print("🧹 Clearing All Positions and Trades")
    print("=" * 50)

    try:
        # Get the single source trade manager
        trade_manager = get_single_source_trade_manager()

        # Show current state before clearing
        current_positions = trade_manager.get_all_positions()
        current_trades = trade_manager.trades if hasattr(trade_manager, 'trades') else []
        closed_positions = trade_manager.get_closed_positions()

        print(f"📊 Current State:")
        print(f"   - Open Positions: {len(current_positions)}")
        print(f"   - Total Trades: {len(current_trades)}")
        print(f"   - Closed Positions: {len(closed_positions)}")

        if current_positions:
            print("   - Position Symbols:", [p.symbol for p in current_positions])

        # Clear all data
        print("\n🗑️  Clearing all data...")

        # Clear positions
        for position in current_positions:
            print(f"   - Clearing position: {position.symbol}")
            # Force clear by directly modifying the internal state
            if hasattr(trade_manager, 'positions'):
                del trade_manager.positions[position.symbol]

        # Clear trades
        if hasattr(trade_manager, 'trades'):
            trade_manager.trades.clear()

        # Clear closed positions
        if hasattr(trade_manager, 'closed_positions'):
            trade_manager.closed_positions.clear()

        # Clear price cache
        if hasattr(trade_manager, 'price_cache'):
            trade_manager.price_cache.clear()

        # Reset statistics
        if hasattr(trade_manager, 'total_trades'):
            trade_manager.total_trades = 0
        if hasattr(trade_manager, 'total_volume'):
            trade_manager.total_volume = Decimal('0')
        if hasattr(trade_manager, 'total_fees'):
            trade_manager.total_fees = Decimal('0')
        if hasattr(trade_manager, 'total_realized_pnl'):
            trade_manager.total_realized_pnl = Decimal('0')

        # Force save the cleared state
        if hasattr(trade_manager, 'save_state'):
            trade_manager.save_state()

        print("✅ All data cleared successfully!")

        # Verify clearing
        print("\n🔍 Verification:")
        final_positions = trade_manager.get_all_positions()
        final_trades = trade_manager.trades if hasattr(trade_manager, 'trades') else []
        final_closed = trade_manager.get_closed_positions()

        print(f"   - Open Positions: {len(final_positions)}")
        print(f"   - Total Trades: {len(final_trades)}")
        print(f"   - Closed Positions: {len(final_closed)}")

        if final_positions or final_trades:
            print("❌ WARNING: Some data may still remain!")
            return False
        else:
            print("✅ Verification successful - all data cleared!")
            return True

    except Exception as e:
        print(f"❌ Error clearing positions: {e}")
        import traceback
        traceback.print_exc()
        return False

def backup_current_state():
    """Create a backup of the current trade manager state before clearing."""

    try:
        trade_manager = get_single_source_trade_manager()

        # Create backup directory if it doesn't exist
        backup_dir = Path("crypto_bot/logs/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"trade_manager_backup_{timestamp}.json"

        # Get current state
        state = trade_manager._load_state()

        # Save backup
        with open(backup_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        print(f"💾 Backup created: {backup_file}")
        return backup_file

    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return None

def clear_csv_logs():
    """Clear the CSV trade log file."""

    try:
        csv_path = Path("crypto_bot/logs/trades.csv")

        if csv_path.exists():
            # Create backup first
            backup_path = csv_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            csv_path.rename(backup_path)
            print(f"💾 CSV backup created: {backup_path}")

            # Clear the CSV file (keep header)
            with open(csv_path, 'w') as f:
                f.write("timestamp,symbol,side,amount,price,strategy,exchange,fees,status,order_id,client_order_id\n")

            print("✅ CSV trade log cleared (header preserved)")
        else:
            print("ℹ️  No CSV trade log found")

    except Exception as e:
        print(f"❌ Error clearing CSV logs: {e}")

def main():
    """Main function to clear all positions with confirmation."""

    print("⚠️  WARNING: This will permanently delete all trade data!")
    print("   Make sure you have backups of any important data.")
    print()

    # Ask for confirmation
    response = input("Are you sure you want to clear ALL positions and trades? (type 'YES' to confirm): ")

    if response != "YES":
        print("❌ Operation cancelled.")
        return

    print()

    # Create backup
    backup_file = backup_current_state()
    if backup_file:
        print(f"📁 Backup saved to: {backup_file}")
    print()

    # Clear all positions
    success = clear_all_positions()

    if success:
        # Clear CSV logs too
        clear_csv_logs()

        print("\n🎉 SUCCESS: All positions and trades have been cleared!")
        print("   The application is ready for a fresh start.")
    else:
        print("\n❌ FAILED: Some data may not have been cleared properly.")
        print("   Please check the logs and try again.")

if __name__ == "__main__":
    main()
