#!/usr/bin/env python3
"""
Debug BTC Position Issue

This script investigates the specific BTC position that's causing the P&L calculation issue.
"""

import sys
from pathlib import Path

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import get_trade_manager
from crypto_bot import log_reader
import json
from decimal import Decimal

def main():
    print("🔍 Debugging BTC Position Issue")
    print("=" * 50)
    
    # Get TradeManager
    tm = get_trade_manager()
    
    # Check BTC position specifically
    btc_position = tm.get_position("BTC/USD")
    if btc_position:
        print("\n📊 BTC/USD Position Details:")
        print(f"  Symbol: {btc_position.symbol}")
        print(f"  Side: {btc_position.side}")
        print(f"  Total Amount: {btc_position.total_amount}")
        print(f"  Average Price: ${btc_position.average_price:.2f}")
        print(f"  Position Value: ${btc_position.position_value:.2f}")
        print(f"  Realized P&L: ${btc_position.realized_pnl:.2f}")
        print(f"  Entry Time: {btc_position.entry_time}")
        print(f"  Last Update: {btc_position.last_update}")
        
        # Calculate unrealized P&L manually
        current_price = tm.price_cache.get("BTC/USD")
        if current_price:
            current_price = float(current_price)
            avg_price = float(btc_position.average_price)
            total_amount = float(btc_position.total_amount)
            
            if btc_position.side == 'long':
                unrealized_pnl = (current_price - avg_price) * total_amount
            else:  # short
                unrealized_pnl = (avg_price - current_price) * total_amount
            
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  Unrealized P&L: ${unrealized_pnl:.2f}")
            print(f"  P&L Percentage: {(unrealized_pnl / float(btc_position.position_value) * 100):.2f}%")
            
            # Check if this is reasonable
            if unrealized_pnl > 50000:  # More than $50k unrealized P&L
                print("⚠️  WARNING: Unrealized P&L seems unusually high!")
                print("   This could indicate:")
                print("   1. Position size is too large")
                print("   2. Entry price is incorrect")
                print("   3. Current price is incorrect")
                print("   4. Position side is incorrect")
                
                # Let's verify the calculation
                print(f"\n🔢 P&L Calculation Breakdown:")
                print(f"   Entry Price: ${avg_price:.2f}")
                print(f"   Current Price: ${current_price:.2f}")
                print(f"   Price Difference: ${current_price - avg_price:.2f}")
                print(f"   Position Size: {total_amount}")
                print(f"   Unrealized P&L: ${(current_price - avg_price) * total_amount:.2f}")
                
                # Check if this is a reasonable position size
                if total_amount > 10:
                    print("⚠️  Position size seems unusually large!")
                elif avg_price < 10000:
                    print("⚠️  Entry price seems unusually low!")
    else:
        print("❌ No BTC/USD position found in TradeManager")
    
    # Check BTC position from TradeManager (primary source)
    print("\n📋 BTC/USD Position from TradeManager:")
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager
        trade_manager = get_trade_manager()

        btc_position = trade_manager.get_position('BTC/USD')
        if btc_position:
            current_price = float(trade_manager.price_cache.get('BTC/USD', btc_position.average_price))
            pnl, pnl_pct = btc_position.calculate_unrealized_pnl(Decimal(str(current_price)))

            print(f"  Symbol: BTC/USD")
            print(f"  Side: {btc_position.side}")
            print(f"  Amount: {btc_position.total_amount}")
            print(f"  Entry Price: ${btc_position.average_price}")
            print(f"  Current Price: ${current_price}")
            print(f"  Unrealized P&L: ${float(pnl):.2f} ({float(pnl_pct):.2f}%)")
            print(f"  Entry Time: {btc_position.entry_time}")
        else:
            print("  No BTC position found in TradeManager")
    except Exception as e:
        print(f"  Error getting BTC position from TradeManager: {e}")

    # Check trade history for BTC (secondary/fallback)
    print("\n📄 BTC/USD Trade History from CSV (Secondary):")
    df = log_reader._read_trades(Path('crypto_bot/logs/trades.csv'))

    btc_trades = df[df['symbol'] == 'BTC/USD']
    if not btc_trades.empty:
        print(f"  Total BTC trades: {len(btc_trades)}")
        for _, trade in btc_trades.iterrows():
            side = trade['side']
            amount = float(trade['amount'])
            price = float(trade['price'])
            timestamp = trade['timestamp']
            print(f"    {timestamp}: {side} {amount} @ ${price:.2f}")
    else:
        print("  No BTC trades found in CSV")
    
    # Check TradeManager state file for BTC position
    print("\n💾 TradeManager State File BTC Position:")
    tm_state_file = Path('crypto_bot/logs/trade_manager_state.json')
    
    if tm_state_file.exists():
        with open(tm_state_file, 'r') as f:
            state = json.load(f)
            positions = state.get('positions', {})
            btc_pos_data = positions.get('BTC/USD')
            if btc_pos_data:
                print(f"  Symbol: {btc_pos_data.get('symbol')}")
                print(f"  Side: {btc_pos_data.get('side')}")
                print(f"  Total Amount: {btc_pos_data.get('total_amount')}")
                print(f"  Average Price: ${btc_pos_data.get('average_price', 0):.2f}")
                print(f"  Realized P&L: ${btc_pos_data.get('realized_pnl', 0):.2f}")
            else:
                print("  No BTC/USD position in state file")
    
    # Check if there are multiple BTC positions or conflicting data
    print("\n🔍 Checking for Position Conflicts:")
    all_positions = tm.get_all_positions()
    btc_positions = [pos for pos in all_positions if pos.symbol == 'BTC/USD']
    print(f"  BTC positions in TradeManager: {len(btc_positions)}")
    
    if len(btc_positions) > 1:
        print("⚠️  WARNING: Multiple BTC positions found!")
        for i, pos in enumerate(btc_positions):
            print(f"    Position {i+1}: {pos.side} {pos.total_amount} @ ${pos.average_price:.2f}")

if __name__ == "__main__":
    main()
