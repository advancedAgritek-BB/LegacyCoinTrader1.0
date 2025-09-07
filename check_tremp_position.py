#!/usr/bin/env python3

from crypto_bot.utils.trade_manager import TradeManager
import ccxt
from decimal import Decimal

def main():
    tm = TradeManager()
    positions = tm.get_all_positions()
    
    print("All positions:")
    for p in positions:
        print(f"{p.symbol}: {p.side} {p.total_amount} @ ${p.average_price:.6f}")
        
        # Check specifically for TREMP
        if "TREMP" in p.symbol:
            print(f"TREMP position details:")
            print(f"  Symbol: {p.symbol}")
            print(f"  Side: {p.side}")
            print(f"  Amount: {p.total_amount}")
            print(f"  Average Price: ${p.average_price:.6f}")
            print(f"  Is Open: {p.is_open}")
            
            # Get current price using ccxt
            try:
                exchange = ccxt.kraken()
                ticker = exchange.fetch_ticker(p.symbol)
                current_price = Decimal(str(ticker['last']))
                print(f"  Current Price: ${current_price:.6f}")
                
                # Calculate P&L using the position's method
                pnl, pnl_pct = p.calculate_unrealized_pnl(current_price)
                print(f"  P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                
                # Check if entry price line should be above or below current price
                print(f"  Entry Price: ${p.average_price:.6f}")
                print(f"  Current Price: ${current_price:.6f}")
                if p.average_price > current_price:
                    print(f"  Entry price is ABOVE current price (should show negative P&L)")
                else:
                    print(f"  Entry price is BELOW current price (should show positive P&L)")
                
            except Exception as e:
                print(f"  Error getting current price: {e}")

if __name__ == "__main__":
    main()
