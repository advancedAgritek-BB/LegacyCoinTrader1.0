#!/usr/bin/env python3
"""
Investigate Phantom Trades

This script investigates the phantom trades that are appearing in the trading history.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import get_trade_manager
from crypto_bot import log_reader

def investigate_phantom_trades():
    """Investigate phantom trades in the trading history."""
    print("🔍 Investigating Phantom Trades")
    print("=" * 60)
    
    # Check trades.csv file
    print("\n📋 Raw Trades.csv Analysis:")
    print("-" * 40)
    
    trades_file = Path('crypto_bot/logs/trades.csv')
    if trades_file.exists():
        with open(trades_file, 'r') as f:
            lines = f.readlines()
        
        print(f"Total lines in trades.csv: {len(lines)}")
        
        # Look for suspicious entries
        suspicious_trades = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line and ',' in line:
                parts = line.split(',')
                if len(parts) >= 5:
                    try:
                        symbol = parts[0].strip()
                        side = parts[1].strip()
                        amount = float(parts[2])
                        price = float(parts[3])
                        timestamp = parts[4].strip()
                        
                        # Check for suspicious patterns
                        if amount >= 1.0:  # Large amounts
                            suspicious_trades.append({
                                'line': i + 1,
                                'symbol': symbol,
                                'side': side,
                                'amount': amount,
                                'price': price,
                                'timestamp': timestamp,
                                'reason': 'Large amount'
                            })
                        
                        if price > 100000:  # Unrealistic prices
                            suspicious_trades.append({
                                'line': i + 1,
                                'symbol': symbol,
                                'side': side,
                                'amount': amount,
                                'price': price,
                                'timestamp': timestamp,
                                'reason': 'Unrealistic price'
                            })
                        
                        # Check for recent trades (today)
                        try:
                            trade_time = datetime.fromisoformat(timestamp.replace('T', ' '))
                            if trade_time.date() == datetime.now().date():
                                suspicious_trades.append({
                                    'line': i + 1,
                                    'symbol': symbol,
                                    'side': side,
                                    'amount': amount,
                                    'price': price,
                                    'timestamp': timestamp,
                                    'reason': 'Recent trade'
                                })
                        except:
                            pass
                            
                    except (ValueError, IndexError) as e:
                        print(f"  Line {i+1}: Parse error - {e}")
                        print(f"    Raw line: {line}")
        
        if suspicious_trades:
            print(f"\n⚠️  Found {len(suspicious_trades)} suspicious trades:")
            for trade in suspicious_trades:
                print(f"  Line {trade['line']}: {trade['symbol']} {trade['side']} {trade['amount']} @ ${trade['price']:.2f}")
                print(f"    Time: {trade['timestamp']}")
                print(f"    Reason: {trade['reason']}")
                print()
        else:
            print("  No suspicious trades found in raw CSV")
    else:
        print("  trades.csv file not found")
    
    # Check TradeManager state
    print("\n💾 TradeManager State Analysis:")
    print("-" * 40)
    
    tm = get_trade_manager()
    trades = tm.get_trade_history()
    
    print(f"Total trades in TradeManager: {len(trades)}")
    
    # Check for recent trades
    today = datetime.now().date()
    recent_trades = []
    
    for trade in trades:
        try:
            trade_date = trade.timestamp.date()
            if trade_date == today:
                recent_trades.append(trade)
        except:
            pass
    
    if recent_trades:
        print(f"\n⚠️  Found {len(recent_trades)} trades from today:")
        for trade in recent_trades:
            print(f"  {trade.timestamp}: {trade.symbol} {trade.side} {trade.amount} @ ${trade.price:.2f}")
            print(f"    Trade ID: {trade.id}")
            print(f"    Fees: ${trade.fees:.2f}")
            print()
    else:
        print("  No trades from today found")
    
    # Check for large trades
    large_trades = [t for t in trades if float(t.amount) >= 1.0]
    if large_trades:
        print(f"\n⚠️  Found {len(large_trades)} large trades (>= 1.0):")
        for trade in large_trades:
            print(f"  {trade.timestamp}: {trade.symbol} {trade.side} {trade.amount} @ ${trade.price:.2f}")
            print(f"    Trade ID: {trade.id}")
            print(f"    Total Value: ${float(trade.amount) * float(trade.price):.2f}")
            print()
    
    # Check TradeManager state file
    print("\n📄 TradeManager State File Analysis:")
    print("-" * 40)
    
    tm_state_file = Path('crypto_bot/logs/trade_manager_state.json')
    if tm_state_file.exists():
        with open(tm_state_file, 'r') as f:
            state = json.load(f)
        
        trades_data = state.get('trades', [])
        print(f"Trades in state file: {len(trades_data)}")
        
        # Check for recent trades in state file
        recent_state_trades = []
        for trade_data in trades_data:
            try:
                timestamp = trade_data.get('timestamp', '')
                if 'T' in timestamp:
                    trade_time = datetime.fromisoformat(timestamp)
                    if trade_time.date() == today:
                        recent_state_trades.append(trade_data)
            except:
                pass
        
        if recent_state_trades:
            print(f"\n⚠️  Found {len(recent_state_trades)} recent trades in state file:")
            for trade_data in recent_state_trades:
                print(f"  {trade_data.get('timestamp')}: {trade_data.get('symbol')} {trade_data.get('side')} {trade_data.get('amount')} @ ${trade_data.get('price')}")
                print(f"    Trade ID: {trade_data.get('id')}")
                print()

def check_paper_wallet_consistency():
    """Check if paper wallet balance is consistent with trade history."""
    print("\n💰 Paper Wallet Consistency Check:")
    print("-" * 40)
    
    # Get paper wallet balance
    try:
        from frontend.app import get_paper_wallet_balance
        current_balance = get_paper_wallet_balance()
        print(f"Current paper wallet balance: ${current_balance:.2f}")
    except Exception as e:
        print(f"Error getting paper wallet balance: {e}")
        current_balance = 10000.0  # Default
    
    # Calculate expected balance from TradeManager (primary source)
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager
        trade_manager = get_trade_manager()

        expected_balance = 10000.0 + float(trade_manager.total_realized_pnl)
        print(f"Expected balance from TradeManager: ${expected_balance:.2f}")

        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        positions = trade_manager.get_all_positions()
        for pos in positions:
            if pos.is_open:
                current_price = float(trade_manager.price_cache.get(pos.symbol, pos.average_price))
                if current_price > 0:
                    from decimal import Decimal
                    pnl, _ = pos.calculate_unrealized_pnl(Decimal(str(current_price)))
                    unrealized_pnl += float(pnl)

        expected_balance += unrealized_pnl
        print(f"Expected balance with unrealized P&L: ${expected_balance:.2f}")

    except Exception as e:
        print(f"Error getting balance from TradeManager: {e}")

    # Calculate expected balance from CSV trades (secondary/fallback)
    df = log_reader._read_trades(Path('crypto_bot/logs/trades.csv'))

    if not df.empty:
        initial_balance = 10000.0
        expected_balance = initial_balance
        
        for _, trade in df.iterrows():
            side = trade['side']
            amount = float(trade['amount'])
            price = float(trade['price'])
            
            if side == 'buy':
                expected_balance -= amount * price
            else:  # sell
                expected_balance += amount * price
        
        print(f"Expected balance from trade history: ${expected_balance:.2f}")
        print(f"Difference: ${current_balance - expected_balance:.2f}")
        
        if abs(current_balance - expected_balance) > 100:
            print("⚠️  Significant balance discrepancy detected!")

def main():
    print("🔍 Phantom Trade Investigation")
    print("=" * 60)
    
    investigate_phantom_trades()
    check_paper_wallet_consistency()
    
    print("\n🎯 Next Steps:")
    print("1. Review suspicious trades above")
    print("2. Check if any automated trading was running")
    print("3. Verify paper wallet balance")
    print("4. Consider resetting TradeManager state if needed")

if __name__ == "__main__":
    main()
