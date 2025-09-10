#!/usr/bin/env python3
"""
Migrate and Verify TradeManager State

This script migrates CSV trades to TradeManager and immediately verifies the state.
"""

import sys
from pathlib import Path

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

from migrate_csv_to_trademanager import migrate_csv_trades_to_trademanager
from crypto_bot.utils.trade_manager import get_trade_manager

def main():
    print("🔄 Migrating and Verifying TradeManager State")
    print("=" * 50)
    
    try:
        # Run migration
        print("📥 Running migration...")
        migrate_csv_trades_to_trademanager()
        
        # Immediately check TradeManager state
        print("\n🔍 Checking TradeManager state...")
        tm = get_trade_manager()
        
        print(f"  Trades in memory: {len(tm.trades)}")
        print(f"  Positions in memory: {len(tm.positions)}")
        
        if tm.trades:
            print("\n📊 Trades in TradeManager:")
            for trade in tm.trades:
                print(f"  {trade.symbol} {trade.side} {trade.amount} @ {trade.price}")
        
        if tm.positions:
            print("\n📈 Positions in TradeManager:")
            for symbol, pos in tm.positions.items():
                print(f"  {symbol}: {pos.side} {pos.total_amount} @ {pos.average_price}")
        
        # Force save and check state file
        print("\n💾 Saving state...")
        tm.save_state()
        
        state_file = Path("crypto_bot/logs/trade_manager_state.json")
        if state_file.exists():
            with open(state_file, 'r') as f:
                import json
                state = json.load(f)
                print(f"  Trades in state file: {len(state.get('trades', []))}")
                print(f"  Positions in state file: {len(state.get('positions', {}))}")
        
        # Get portfolio summary
        print("\n📈 Portfolio Summary:")
        summary = tm.get_portfolio_summary()
        print(f"  Total P&L: ${summary['total_pnl']:.2f}")
        print(f"  Realized P&L: ${summary['total_realized_pnl']:.2f}")
        print(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}")
        print(f"  Open Positions: {summary['open_positions_count']}")
        print(f"  Total Trades: {summary['total_trades']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
