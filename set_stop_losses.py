#!/usr/bin/env python3
"""
Set Stop Loss Prices for Positions
This script helps you set stop loss prices for your open positions.
"""

import json
import sys
from pathlib import Path

def get_positions():
    """Get current positions from TradeManager state."""
    state_file = Path("crypto_bot/logs/trade_manager_state.json")
    if not state_file.exists():
        print("❌ TradeManager state file not found")
        return {}
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    return state.get('positions', {})

def set_stop_loss(symbol, stop_loss_price):
    """Set stop loss price for a position."""
    state_file = Path("crypto_bot/logs/trade_manager_state.json")
    if not state_file.exists():
        print("❌ TradeManager state file not found")
        return False
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    positions = state.get('positions', {})
    if symbol not in positions:
        print(f"❌ Position {symbol} not found")
        return False
    
    # Update stop loss price
    positions[symbol]['stop_loss_price'] = stop_loss_price
    
    # Save updated state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"✅ Set stop loss for {symbol} to ${stop_loss_price}")
    return True

def main():
    print("🎯 Position Stop Loss Manager")
    print("=" * 40)
    
    # Get current positions
    positions = get_positions()
    if not positions:
        print("❌ No positions found")
        return
    
    print(f"Found {len(positions)} positions:")
    print()
    
    # Display positions
    for i, (symbol, pos) in enumerate(positions.items(), 1):
        current_stop = pos.get('stop_loss_price')
        entry_price = pos.get('average_price', 0)
        side = pos.get('side', 'unknown')
        
        print(f"{i}. {symbol}")
        print(f"   Side: {side}")
        print(f"   Entry: ${entry_price}")
        print(f"   Current Stop: {current_stop if current_stop else 'None'}")
        print()
    
    # Interactive stop loss setting
    while True:
        try:
            choice = input("Enter position number to set stop loss (or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                break
            
            position_num = int(choice)
            if position_num < 1 or position_num > len(positions):
                print("❌ Invalid position number")
                continue
            
            # Get the position
            symbol = list(positions.keys())[position_num - 1]
            position = positions[symbol]
            entry_price = position.get('average_price', 0)
            side = position.get('side', 'unknown')
            
            print(f"\nSetting stop loss for {symbol}")
            print(f"Entry price: ${entry_price}")
            print(f"Side: {side}")
            
            # Calculate suggested stop loss
            if side == 'long':
                suggested_stop = entry_price * 0.95  # 5% below entry
            else:  # short
                suggested_stop = entry_price * 1.05  # 5% above entry
            
            print(f"Suggested stop loss: ${suggested_stop:.6f}")
            
            stop_input = input("Enter stop loss price (or press Enter for suggested): ").strip()
            
            if stop_input:
                try:
                    stop_price = float(stop_input)
                except ValueError:
                    print("❌ Invalid price")
                    continue
            else:
                stop_price = suggested_stop
            
            # Confirm
            confirm = input(f"Set stop loss for {symbol} to ${stop_price:.6f}? (y/n): ").strip().lower()
            if confirm == 'y':
                if set_stop_loss(symbol, stop_price):
                    print("✅ Stop loss set successfully!")
                    print("🔄 Restart the frontend to see the changes in the charts")
                else:
                    print("❌ Failed to set stop loss")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
