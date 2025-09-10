#!/usr/bin/env python3
"""
Fix BTC Position Tracking Issue

This script fixes the incorrect BTC position tracking that's causing the inflated P&L.
"""

import sys
from pathlib import Path
import json
from decimal import Decimal
from datetime import datetime

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import get_trade_manager
from crypto_bot import log_reader

def analyze_btc_trades():
    """Analyze BTC trades to understand the position tracking issue."""
    print("🔍 Analyzing BTC Trade History")
    print("=" * 50)
    
    # Read trade history
    df = log_reader._read_trades(Path('crypto_bot/logs/trades.csv'))
    btc_trades = df[df['symbol'] == 'BTC/USD']
    
    if btc_trades.empty:
        print("No BTC trades found")
        return
    
    print(f"Total BTC trades: {len(btc_trades)}")
    
    # Simulate position tracking manually
    position_amount = 0.0
    position_value = 0.0
    realized_pnl = 0.0
    
    for _, trade in btc_trades.iterrows():
        side = trade['side']
        amount = float(trade['amount'])
        price = float(trade['price'])
        timestamp = trade['timestamp']
        
        print(f"\nTrade: {timestamp}")
        print(f"  {side} {amount} @ ${price:.2f}")
        print(f"  Before: {position_amount} BTC, ${position_value:.2f}")
        
        if side == 'buy':
            # Add to position
            position_amount += amount
            position_value += amount * price
            print(f"  After: {position_amount} BTC, ${position_value:.2f}")
        else:  # sell
            # Close position
            if position_amount > 0:
                # Calculate P&L
                avg_price = position_value / position_amount
                pnl = (price - avg_price) * min(amount, position_amount)
                realized_pnl += pnl
                
                # Reduce position
                close_amount = min(amount, position_amount)
                position_amount -= close_amount
                position_value -= close_amount * avg_price
                
                print(f"  P&L: ${pnl:.2f}")
                print(f"  After: {position_amount} BTC, ${position_value:.2f}")
        
        print(f"  Realized P&L: ${realized_pnl:.2f}")
    
    print(f"\n📊 Final Position:")
    print(f"  Remaining BTC: {position_amount}")
    print(f"  Position Value: ${position_value:.2f}")
    print(f"  Total Realized P&L: ${realized_pnl:.2f}")
    
    return position_amount, realized_pnl

def fix_btc_position():
    """Fix the BTC position in TradeManager."""
    print("\n🔧 Fixing BTC Position in TradeManager")
    print("=" * 50)
    
    # Analyze trades first
    correct_amount, correct_realized_pnl = analyze_btc_trades()
    
    # Get TradeManager
    tm = get_trade_manager()
    
    # Check current BTC position
    btc_position = tm.get_position("BTC/USD")
    if btc_position:
        print(f"\nCurrent BTC position:")
        print(f"  Amount: {btc_position.total_amount}")
        print(f"  Realized P&L: ${btc_position.realized_pnl}")
        
        # Fix the position
        if abs(float(btc_position.total_amount) - correct_amount) > 0.001:
            print(f"\n⚠️  Position amount mismatch detected!")
            print(f"  Current: {btc_position.total_amount}")
            print(f"  Correct: {correct_amount}")
            
            # Update the position
            btc_position.total_amount = Decimal(str(correct_amount))
            btc_position.realized_pnl = Decimal(str(correct_realized_pnl))
            btc_position.last_update = datetime.utcnow()  # Fix the datetime issue
            
            # If position is closed, remove it
            if correct_amount == 0:
                print("  Removing closed BTC position")
                del tm.positions["BTC/USD"]
            else:
                print("  Updating BTC position")
            
            # Save the updated state
            tm.save_state()
            print("  ✅ Position fixed and saved")
        else:
            print("  ✅ Position amount is correct")
    else:
        print("  No BTC position found in TradeManager")

def verify_fix():
    """Verify that the fix worked."""
    print("\n✅ Verifying Fix")
    print("=" * 50)
    
    tm = get_trade_manager()
    summary = tm.get_portfolio_summary()
    
    print(f"Total P&L: ${summary['total_pnl']:.2f}")
    print(f"Realized P&L: ${summary['total_realized_pnl']:.2f}")
    print(f"Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}")
    print(f"Open Positions: {summary['open_positions_count']}")
    
    # Check BTC position specifically
    btc_position = tm.get_position("BTC/USD")
    if btc_position:
        print(f"BTC Position: {btc_position.total_amount} @ ${btc_position.average_price:.2f}")
    else:
        print("BTC Position: Closed")

def main():
    print("🔧 BTC Position Fix Script")
    print("=" * 60)
    
    # Step 1: Analyze the issue
    analyze_btc_trades()
    
    # Step 2: Fix the position
    fix_btc_position()
    
    # Step 3: Verify the fix
    verify_fix()
    
    print("\n🎉 Fix completed!")

if __name__ == "__main__":
    main()
