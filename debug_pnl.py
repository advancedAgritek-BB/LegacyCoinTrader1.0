#!/usr/bin/env python3
"""
Debug P&L Calculation Issue

This script investigates the P&L calculation discrepancy in the dashboard.
"""

import sys
from pathlib import Path

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import get_trade_manager
from crypto_bot import log_reader
import yaml
import json

def main():
    print("🔍 Debugging P&L Calculation Issue")
    print("=" * 50)
    
    # Get TradeManager portfolio summary
    try:
        tm = get_trade_manager()
        summary = tm.get_portfolio_summary()
        
        print("\n📊 TradeManager Portfolio Summary:")
        print(f"  Total P&L: ${summary['total_pnl']:.2f}")
        print(f"  Realized P&L: ${summary['total_realized_pnl']:.2f}")
        print(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}")
        print(f"  Open Positions: {summary['open_positions_count']}")
        print(f"  Closed Positions: {summary['closed_positions_count']}")
        print(f"  Total Trades: {summary['total_trades']}")
        print(f"  Total Volume: ${summary['total_volume']:.2f}")
        print(f"  Total Fees: ${summary['total_fees']:.2f}")
        
        if summary['positions']:
            print("\n📈 Open Positions with Current Prices:")
            for pos in summary['positions']:
                symbol = pos['symbol']
                side = pos['side']
                amount = pos['total_amount']
                avg_price = pos['average_price']
                
                # Get current price from TradeManager
                current_price = tm.price_cache.get(symbol)
                if current_price:
                    current_price = float(current_price)
                    # Calculate unrealized P&L manually
                    if side == 'long':
                        unrealized_pnl = (current_price - avg_price) * amount
                    else:  # short
                        unrealized_pnl = (avg_price - current_price) * amount
                    
                    print(f"  {symbol}: {side} {amount} @ ${avg_price:.2f}")
                    print(f"    Current Price: ${current_price:.2f}")
                    print(f"    Unrealized P&L: ${unrealized_pnl:.2f}")
                    print(f"    Position Value: ${amount * avg_price:.2f}")
                    print()
                else:
                    print(f"  {symbol}: {side} {amount} @ ${avg_price:.2f} (No current price)")
        
    except Exception as e:
        print(f"❌ Error getting TradeManager summary: {e}")
    
    # Check trade history from TradeManager (primary source)
    try:
        print("\n📋 Trade History Analysis:")
        from crypto_bot.utils.trade_manager import get_trade_manager
        trade_manager = get_trade_manager()

        positions = trade_manager.get_all_positions()
        if positions:
            print(f"  Total positions from TradeManager: {len(positions)}")

            # Calculate P&L from TradeManager positions
            total_realized_pnl = float(trade_manager.total_realized_pnl)
            total_unrealized_pnl = 0.0

            for pos in positions:
                if pos.is_open:
                    # Get current price from cache or use entry price as fallback
                    current_price = float(trade_manager.price_cache.get(pos.symbol, pos.average_price))
                    if current_price > 0:
                        from decimal import Decimal
                        pnl, _ = pos.calculate_unrealized_pnl(Decimal(str(current_price)))
                        total_unrealized_pnl += float(pnl)

            print(f"  Realized P&L: ${total_realized_pnl:.2f}")
            print(f"  Unrealized P&L: ${total_unrealized_pnl:.2f}")
            print(f"  Total P&L: ${total_realized_pnl + total_unrealized_pnl:.2f}")

        else:
            print("  No positions found in TradeManager, checking CSV fallback...")

    except Exception as e:
        print(f"❌ Error getting trade data from TradeManager: {e}")

    # Check CSV trade history (secondary/fallback)
    try:
        print("\n📄 CSV Trade History Analysis (Secondary):")
        df = log_reader._read_trades(Path('crypto_bot/logs/trades.csv'))

        if not df.empty:
            print(f"  Total trades in CSV: {len(df)}")

            # Calculate P&L from trade history
            position_history = {}
            realized_pnl = 0.0

            for _, row in df.iterrows():
                symbol = str(row.get('symbol', ''))
                side = str(row.get('side', ''))
                amount = float(row.get('amount', 0))
                price = float(row.get('price', 0))
                
                if symbol and amount > 0 and price > 0:
                    total = amount * price
                    
                    if symbol in position_history:
                        existing = position_history[symbol]
                        
                        # Check if this closes the position
                        if ((side == 'sell' and existing['side'] == 'buy') or
                            (side == 'buy' and existing['side'] == 'sell')):
                            
                            # Calculate realized P&L
                            if side == 'sell':  # Closing long
                                pnl = (price - existing['price']) * min(amount, existing['amount'])
                            else:  # Closing short
                                pnl = (existing['price'] - price) * min(amount, existing['amount'])
                            
                            realized_pnl += pnl
                            print(f"    Closed {symbol}: P&L = ${pnl:.2f}")
                            
                            # Update or remove position
                            if amount >= existing['amount']:
                                del position_history[symbol]
                            else:
                                position_history[symbol]['amount'] -= amount
                        else:
                            # Same side - average the position
                            total_cost = (existing['price'] * existing['amount']) + total
                            total_amount = existing['amount'] + amount
                            position_history[symbol] = {
                                'side': side,
                                'price': total_cost / total_amount,
                                'amount': total_amount
                            }
                    else:
                        # New position
                        position_history[symbol] = {'side': side, 'price': price, 'amount': amount}
            
            print(f"  Calculated realized P&L from CSV: ${realized_pnl:.2f}")
            print(f"  Remaining open positions: {len(position_history)}")
            
            for symbol, pos in position_history.items():
                print(f"    {symbol}: {pos['side']} {pos['amount']} @ ${pos['price']:.2f}")
        
    except Exception as e:
        print(f"❌ Error analyzing trade history: {e}")
    
    # Check paper wallet state
    try:
        print("\n💰 Paper Wallet State:")
        paper_wallet_file = Path('crypto_bot/logs/paper_wallet_state.yaml')
        
        if paper_wallet_file.exists():
            with open(paper_wallet_file, 'r') as f:
                state = yaml.safe_load(f) or {}
                print(f"  Balance: ${state.get('balance', 0):.2f}")
                print(f"  Initial Balance: ${state.get('initial_balance', 0):.2f}")
                print(f"  Realized P&L: ${state.get('realized_pnl', 0):.2f}")
        else:
            print("  No paper wallet state file found")
            
    except Exception as e:
        print(f"❌ Error reading paper wallet state: {e}")
    
    # Check TradeManager state file
    try:
        print("\n💾 TradeManager State File:")
        tm_state_file = Path('crypto_bot/logs/trade_manager_state.json')
        
        if tm_state_file.exists():
            with open(tm_state_file, 'r') as f:
                state = json.load(f)
                stats = state.get('statistics', {})
                print(f"  Total Realized P&L: ${stats.get('total_realized_pnl', 0):.2f}")
                print(f"  Total Volume: ${stats.get('total_volume', 0):.2f}")
                print(f"  Total Fees: ${stats.get('total_fees', 0):.2f}")
                print(f"  Saved Trades: {len(state.get('trades', []))}")
                print(f"  Saved Positions: {len(state.get('positions', {}))}")
                
                # Check price cache
                price_cache = state.get('price_cache', {})
                print(f"  Price Cache Entries: {len(price_cache)}")
                if price_cache:
                    print("  Current Prices:")
                    for symbol, price in price_cache.items():
                        print(f"    {symbol}: ${price:.2f}")
        else:
            print("  No TradeManager state file found")
            
    except Exception as e:
        print(f"❌ Error reading TradeManager state: {e}")

if __name__ == "__main__":
    main()
