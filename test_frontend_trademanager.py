#!/usr/bin/env python3
"""
Test TradeManager State in Frontend Context

This script tests the TradeManager state in the same context as the frontend.
"""

import sys
from pathlib import Path

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import get_trade_manager

def main():
    print("🧪 Testing TradeManager in Frontend Context")
    print("=" * 50)
    
    try:
        # Get TradeManager instance
        tm = get_trade_manager()
        
        print(f"TradeManager instance: {tm}")
        print(f"Storage path: {tm.storage_path}")
        print(f"Trades in memory: {len(tm.trades)}")
        print(f"Positions in memory: {len(tm.positions)}")
        
        if tm.trades:
            print("\n📊 Trades:")
            for trade in tm.trades:
                print(f"  {trade.symbol} {trade.side} {trade.amount} @ {trade.price}")
        
        if tm.positions:
            print("\n📈 Positions:")
            for symbol, pos in tm.positions.items():
                print(f"  {symbol}: {pos.side} {pos.total_amount} @ {pos.average_price}")
        
        # Get portfolio summary
        print("\n📈 Portfolio Summary:")
        summary = tm.get_portfolio_summary()
        print(f"  Total P&L: ${summary['total_pnl']:.2f}")
        print(f"  Realized P&L: ${summary['total_realized_pnl']:.2f}")
        print(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}")
        print(f"  Open Positions: {summary['open_positions_count']}")
        print(f"  Total Trades: {summary['total_trades']}")
        
        # Check if state file exists and has data
        print(f"\n💾 State file: {tm.storage_path}")
        if tm.storage_path.exists():
            with open(tm.storage_path, 'r') as f:
                import json
                state = json.load(f)
                print(f"  Trades in file: {len(state.get('trades', []))}")
                print(f"  Positions in file: {len(state.get('positions', {}))}")
        else:
            print("  State file does not exist!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
