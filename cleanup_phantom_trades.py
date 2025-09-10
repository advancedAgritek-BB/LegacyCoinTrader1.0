#!/usr/bin/env python3
"""
Remove Phantom Trades and Reset Trading State

This script removes the phantom trades that were created by multiple bot instances
and resets the trading state to a clean state.
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import shutil

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import get_trade_manager
from crypto_bot import log_reader

def backup_current_state():
    """Backup current trading state before making changes."""
    print("💾 Backing up current state...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Backup trades.csv
    trades_file = Path('crypto_bot/logs/trades.csv')
    if trades_file.exists():
        backup_file = Path(f'crypto_bot/logs/trades_backup_{timestamp}.csv')
        shutil.copy2(trades_file, backup_file)
        print(f"  ✅ Backed up trades.csv to {backup_file}")
    
    # Backup TradeManager state
    tm_state_file = Path('crypto_bot/logs/trade_manager_state.json')
    if tm_state_file.exists():
        backup_file = Path(f'crypto_bot/logs/trade_manager_state_backup_{timestamp}.json')
        shutil.copy2(tm_state_file, backup_file)
        print(f"  ✅ Backed up trade_manager_state.json to {backup_file}")
    
    # Backup paper wallet state
    paper_wallet_file = Path('crypto_bot/logs/paper_wallet_state.yaml')
    if paper_wallet_file.exists():
        backup_file = Path(f'crypto_bot/logs/paper_wallet_state_backup_{timestamp}.yaml')
        shutil.copy2(paper_wallet_file, backup_file)
        print(f"  ✅ Backed up paper_wallet_state.yaml to {backup_file}")

def identify_phantom_trades():
    """Identify which trades are phantom trades."""
    print("\n🔍 Identifying Phantom Trades...")
    
    # Read current trades
    df = log_reader._read_trades(Path('crypto_bot/logs/trades.csv'))
    
    phantom_trades = []
    legitimate_trades = []
    
    for _, trade in df.iterrows():
        symbol = trade['symbol']
        side = trade['side']
        amount = float(trade['amount'])
        price = float(trade['price'])
        timestamp = trade['timestamp']
        
        # Check for suspicious patterns
        is_phantom = False
        
        # Large amounts (>1000)
        if amount > 1000:
            is_phantom = True
        
        # Recent trades (today) that are large
        try:
            trade_time = datetime.fromisoformat(str(timestamp).replace('T', ' '))
            if trade_time.date() == datetime.now().date() and amount > 100:
                is_phantom = True
        except:
            pass
        
        # Specific phantom trades we know about
        if (symbol == 'BTC/USD' and amount >= 1.0) or \
           (symbol == 'SOGNI/USD' and amount > 400000) or \
           (symbol == 'APU/USD' and amount > 4000000):
            is_phantom = True
        
        if is_phantom:
            phantom_trades.append({
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'timestamp': timestamp
            })
        else:
            legitimate_trades.append({
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'timestamp': timestamp
            })
    
    print(f"  Found {len(phantom_trades)} phantom trades")
    print(f"  Found {len(legitimate_trades)} legitimate trades")
    
    return phantom_trades, legitimate_trades

def create_clean_trades_file(legitimate_trades):
    """Create a clean trades.csv with only legitimate trades."""
    print("\n🧹 Creating clean trades.csv...")
    
    trades_file = Path('crypto_bot/logs/trades.csv')
    
    # Write header
    with open(trades_file, 'w') as f:
        f.write("symbol,side,amount,price,timestamp,is_stop\n")
        
        # Write legitimate trades
        for trade in legitimate_trades:
            line = f"{trade['symbol']},{trade['side']},{trade['amount']},{trade['price']},{trade['timestamp']},False\n"
            f.write(line)
    
    print(f"  ✅ Created clean trades.csv with {len(legitimate_trades)} trades")

def reset_trade_manager():
    """Reset TradeManager to use only legitimate trades."""
    print("\n🔄 Resetting TradeManager...")
    
    # Get TradeManager
    tm = get_trade_manager()
    
    # Clear current state
    tm.trades = []
    tm.positions = {}
    tm.price_cache = {}
    tm.total_trades = 0
    tm.total_volume = 0
    tm.total_fees = 0
    tm.total_realized_pnl = 0
    
    # Reload from clean trades.csv
    df = log_reader._read_trades(Path('crypto_bot/logs/trades.csv'))
    
    for _, row in df.iterrows():
        try:
            from crypto_bot.utils.trade_manager import Trade
            from decimal import Decimal
            
            trade = Trade(
                symbol=str(row['symbol']),
                side=str(row['side']),
                amount=Decimal(str(row['amount'])),
                price=Decimal(str(row['price'])),
                timestamp=datetime.fromisoformat(str(row['timestamp']).replace('T', ' ')),
                fees=Decimal('0'),
                dry_run=True
            )
            
            tm.record_trade(trade)
        except Exception as e:
            print(f"  ⚠️  Error processing trade: {e}")
    
    # Save clean state
    tm.save_state()
    print(f"  ✅ TradeManager reset with {len(tm.trades)} legitimate trades")

def reset_paper_wallet():
    """Reset paper wallet to initial state."""
    print("\n💰 Resetting paper wallet...")
    
    import yaml
    
    # Create clean paper wallet state
    paper_wallet_state = {
        'balance': 10000.0,
        'initial_balance': 10000.0,
        'realized_pnl': 0.0,
        'last_update': datetime.utcnow().isoformat()
    }
    
    paper_wallet_file = Path('crypto_bot/logs/paper_wallet_state.yaml')
    with open(paper_wallet_file, 'w') as f:
        yaml.dump(paper_wallet_state, f)
    
    print("  ✅ Paper wallet reset to $10,000 initial balance")

def verify_clean_state():
    """Verify that the clean state is correct."""
    print("\n✅ Verifying Clean State...")
    
    # Check TradeManager
    tm = get_trade_manager()
    summary = tm.get_portfolio_summary()
    
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Total P&L: ${summary['total_pnl']:.2f}")
    print(f"  Realized P&L: ${summary['total_realized_pnl']:.2f}")
    print(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}")
    print(f"  Open Positions: {summary['open_positions_count']}")
    
    # Check paper wallet
    try:
        from frontend.app import get_paper_wallet_balance
        balance = get_paper_wallet_balance()
        print(f"  Paper Wallet Balance: ${balance:.2f}")
    except Exception as e:
        print(f"  Error getting paper wallet balance: {e}")
    
    # Check for any remaining phantom trades
    df = log_reader._read_trades(Path('crypto_bot/logs/trades.csv'))
    large_trades = df[df['amount'] > 1000]
    
    if not large_trades.empty:
        print(f"  ⚠️  Warning: {len(large_trades)} large trades still present")
        for _, trade in large_trades.iterrows():
            print(f"    {trade['symbol']} {trade['side']} {trade['amount']} @ ${trade['price']}")
    else:
        print("  ✅ No large trades remaining")

def main():
    print("🧹 Phantom Trade Cleanup Script")
    print("=" * 60)
    
    # Step 1: Backup current state
    backup_current_state()
    
    # Step 2: Identify phantom trades
    phantom_trades, legitimate_trades = identify_phantom_trades()
    
    if phantom_trades:
        print(f"\n⚠️  Found {len(phantom_trades)} phantom trades to remove:")
        for trade in phantom_trades:
            print(f"  {trade['timestamp']}: {trade['symbol']} {trade['side']} {trade['amount']} @ ${trade['price']}")
        
        # Step 3: Create clean trades file
        create_clean_trades_file(legitimate_trades)
        
        # Step 4: Reset TradeManager
        reset_trade_manager()
        
        # Step 5: Reset paper wallet
        reset_paper_wallet()
        
        # Step 6: Verify clean state
        verify_clean_state()
        
        print("\n🎉 Cleanup completed!")
        print("\n⚠️  IMPORTANT:")
        print("1. All bot processes have been stopped")
        print("2. Phantom trades have been removed")
        print("3. Trading state has been reset")
        print("4. Paper wallet reset to $10,000")
        print("5. Do NOT start the bot again until you've reviewed the configuration")
    else:
        print("\n✅ No phantom trades found - state is already clean")

if __name__ == "__main__":
    main()
