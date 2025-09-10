#!/usr/bin/env python3
"""
Script to sell all current positions using the existing API endpoint.
This script uses the TradeManager as the source of truth and sells all positions.
"""

import requests
import json
import time
import sys

def get_positions():
    """Get current positions from the API."""
    try:
        response = requests.get('http://localhost:8000/api/open-positions')
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting positions: {e}")
        return []

def sell_position(symbol, amount):
    """Sell a single position using the API."""
    try:
        data = {
            'symbol': symbol,
            'amount': amount
        }
        response = requests.post(
            'http://localhost:8000/api/sell-position',
            headers={'Content-Type': 'application/json'},
            json=data
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error selling {symbol}: {e}")
        return {'success': False, 'error': str(e)}

def close_positions_in_trade_manager():
    """Close all positions in TradeManager by recording closing trades."""
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager, create_trade
        from decimal import Decimal
        
        trade_manager = get_trade_manager()
        
        # Get all open positions from TradeManager
        tm_positions = trade_manager.get_all_positions()
        closed_count = 0
        
        for pos in tm_positions:
            if pos.is_open:
                # Create a closing trade to close the position
                current_price = float(trade_manager.price_cache.get(pos.symbol, pos.average_price))
                
                # Create closing trade (opposite side of position)
                closing_side = 'sell' if pos.side == 'long' else 'buy'
                closing_trade = create_trade(
                    symbol=pos.symbol,
                    side=closing_side,
                    amount=pos.total_amount,
                    price=Decimal(str(current_price)),
                    strategy="manual_sell_all",
                    exchange="manual",
                    metadata={"reason": "sell_all_positions"}
                )
                
                # Record the closing trade in TradeManager
                trade_manager.record_trade(closing_trade)
                print(f"Closed position {pos.symbol} in TradeManager via closing trade")
                closed_count += 1
        
        return closed_count
    except Exception as e:
        print(f"Error closing positions in TradeManager: {e}")
        return 0

def main():
    """Main function to sell all positions."""
    print("=== Selling All Positions ===")
    
    # Get current positions
    positions = get_positions()
    
    if not positions:
        print("No positions to sell")
        return
    
    print(f"Found {len(positions)} positions to sell:")
    for pos in positions:
        print(f"  {pos['symbol']}: {pos['amount']} ({pos['side']}) @ ${pos['current_price']:.4f}")
    
    # Confirm with user
    confirm = input("\nAre you sure you want to sell ALL positions? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled")
        return
    
    # Sell each position
    sold_count = 0
    failed_sells = []
    
    print("\nSelling positions...")
    for position in positions:
        print(f"Selling {position['amount']} {position['symbol']}...")
        
        result = sell_position(position['symbol'], position['amount'])
        
        if result.get('success', False):
            sold_count += 1
            print(f"✓ Successfully sold {position['symbol']}")
        else:
            failed_sells.append({
                'symbol': position['symbol'],
                'error': result.get('error', 'Unknown error')
            })
            print(f"✗ Failed to sell {position['symbol']}: {result.get('error', 'Unknown error')}")
        
        # Small delay between sells
        time.sleep(0.5)
    
    # Close positions in TradeManager
    print("\nClosing positions in TradeManager...")
    tm_closed = close_positions_in_trade_manager()
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Positions sold via API: {sold_count}/{len(positions)}")
    print(f"Positions closed in TradeManager: {tm_closed}")
    
    if failed_sells:
        print(f"Failed sells: {len(failed_sells)}")
        for failed in failed_sells:
            print(f"  {failed['symbol']}: {failed['error']}")
    
    print("\nAll positions have been processed.")

if __name__ == "__main__":
    main()
