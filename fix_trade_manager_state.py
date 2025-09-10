#!/usr/bin/env python3
"""
Fix TradeManager State - Manual Save

This script manually saves the TradeManager state to ensure all migrated trades
are properly persisted to disk.
"""

import sys
from pathlib import Path

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import get_trade_manager

def main():
    print("🔧 Fixing TradeManager State")
    print("=" * 40)
    
    try:
        # Get TradeManager instance
        tm = get_trade_manager()
        
        print(f"Current trades in memory: {len(tm.trades)}")
        print(f"Current positions in memory: {len(tm.positions)}")
        
        if tm.trades:
            print("\n📊 Trades in memory:")
            for trade in tm.trades:
                print(f"  {trade.symbol} {trade.side} {trade.amount} @ {trade.price}")
        
        if tm.positions:
            print("\n📈 Positions in memory:")
            for symbol, pos in tm.positions.items():
                print(f"  {symbol}: {pos.side} {pos.total_amount} @ {pos.average_price}")
        
        # Force save state
        print("\n💾 Saving TradeManager state...")
        tm.save_state()
        
        # Verify save
        print("\n✅ Verification:")
        summary = tm.get_portfolio_summary()
        print(f"  Total P&L: ${summary['total_pnl']:.2f}")
        print(f"  Realized P&L: ${summary['total_realized_pnl']:.2f}")
        print(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}")
        print(f"  Open Positions: {summary['open_positions_count']}")
        print(f"  Closed Positions: {summary['closed_positions_count']}")
        print(f"  Total Trades: {summary['total_trades']}")
        
        print("\n🎉 TradeManager state fixed!")
        
    except Exception as e:
        print(f"❌ Error fixing TradeManager state: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
