#!/usr/bin/env python3
"""
Diagnostic script to analyze wallet balance discrepancy between TradeManager and PaperWallet

This script will help identify why the frontend shows $803.83 available balance
while the paper wallet shows a negative balance of -$1,702.87.
"""

import yaml
import json
from pathlib import Path
from decimal import Decimal
from typing import Dict, Any, List

def analyze_balance_discrepancy():
    """Analyze the balance discrepancy between different systems."""
    
    print("=" * 80)
    print("💰 WALLET BALANCE DISCREPANCY ANALYSIS")
    print("=" * 80)
    
    # 1. Check PaperWallet state
    print("\n📊 PAPER WALLET STATE ANALYSIS:")
    paper_wallet_file = Path("crypto_bot/logs/paper_wallet_state.yaml")
    if paper_wallet_file.exists():
        with open(paper_wallet_file, 'r') as f:
            pw_state = yaml.safe_load(f)
        
        initial_balance = pw_state.get('initial_balance', 0)
        current_balance = pw_state.get('balance', 0)
        positions = pw_state.get('positions', {})
        
        print(f"   Initial Balance: ${initial_balance:.2f}")
        print(f"   Current Balance: ${current_balance:.2f}")
        print(f"   Balance Change: ${current_balance - initial_balance:.2f}")
        print(f"   Open Positions: {len(positions)}")
        
        # Calculate total position value
        total_position_value = 0.0
        for trade_id, pos in positions.items():
            symbol = pos.get('symbol', 'Unknown')
            side = pos.get('side', 'Unknown')
            amount = pos.get('amount', pos.get('size', 0))
            entry_price = pos.get('entry_price', 0)
            position_value = amount * entry_price
            total_position_value += position_value
            
            print(f"     {symbol}: {side.upper()} {amount:.4f} @ ${entry_price:.6f} = ${position_value:.2f}")
        
        print(f"   Total Position Value: ${total_position_value:.2f}")
        
        # Calculate expected balance
        expected_balance = initial_balance - total_position_value
        print(f"   Expected Balance: ${expected_balance:.2f}")
        print(f"   Balance Discrepancy: ${current_balance - expected_balance:.2f}")
        
    else:
        print("   ❌ Paper wallet state file not found")
    
    # 2. Check TradeManager state
    print("\n📈 TRADE MANAGER STATE ANALYSIS:")
    trade_manager_file = Path("crypto_bot/logs/trade_manager_state.json")
    if trade_manager_file.exists():
        with open(trade_manager_file, 'r') as f:
            tm_state = json.load(f)
        
        trades = tm_state.get('trades', [])
        positions = tm_state.get('positions', {})
        
        print(f"   Total Trades: {len(trades)}")
        print(f"   Open Positions: {len(positions)}")
        
        # Calculate total volume and fees
        total_volume = 0.0
        total_fees = 0.0
        for trade in trades:
            volume = trade.get('amount', 0) * trade.get('price', 0)
            total_volume += volume
            total_fees += trade.get('fees', 0)
        
        print(f"   Total Volume: ${total_volume:.2f}")
        print(f"   Total Fees: ${total_fees:.2f}")
        
        # Analyze positions
        tm_position_value = 0.0
        for symbol, pos_data in positions.items():
            if pos_data.get('total_amount', 0) > 0:  # Only open positions
                amount = pos_data.get('total_amount', 0)
                avg_price = pos_data.get('average_price', 0)
                position_value = amount * avg_price
                tm_position_value += position_value
                
                side = pos_data.get('side', 'Unknown')
                print(f"     {symbol}: {side.upper()} {amount:.4f} @ ${avg_price:.6f} = ${position_value:.2f}")
        
        print(f"   TradeManager Position Value: ${tm_position_value:.2f}")
        
    else:
        print("   ❌ TradeManager state file not found")
    
    # 3. Check frontend balance calculation
    print("\n🖥️ FRONTEND BALANCE CALCULATION:")
    
    # Simulate frontend calculation
    try:
        # Get positions from API
        from frontend.api import positions
        api_positions = positions()
        
        print(f"   API Positions Count: {len(api_positions)}")
        
        # Calculate position value as frontend does
        api_position_value = 0.0
        for pos in api_positions:
            current_price = pos.get('current_price', 0)
            amount = pos.get('amount', 0)
            position_value = current_price * amount
            api_position_value += position_value
            
            symbol = pos.get('symbol', 'Unknown')
            side = pos.get('side', 'Unknown')
            print(f"     {symbol}: {side.upper()} {amount:.4f} @ ${current_price:.6f} = ${position_value:.2f}")
        
        print(f"   API Position Value: ${api_position_value:.2f}")
        
        # Simulate frontend available balance calculation
        from frontend.app import get_paper_wallet_balance, get_available_balance
        frontend_total_balance = get_paper_wallet_balance()
        frontend_available_balance = get_available_balance(api_positions)
        
        print(f"   Frontend Total Balance: ${frontend_total_balance:.2f}")
        print(f"   Frontend Available Balance: ${frontend_available_balance:.2f}")
        print(f"   Frontend Position Value: ${frontend_total_balance - frontend_available_balance:.2f}")
        
    except Exception as e:
        print(f"   ❌ Error analyzing frontend calculation: {e}")
    
    # 4. Check configuration files
    print("\n⚙️ CONFIGURATION FILES:")
    
    config_files = [
        ("crypto_bot/logs/paper_wallet.yaml", "paper_wallet.yaml"),
        ("crypto_bot/user_config.yaml", "user_config.yaml"),
        ("crypto_bot/paper_wallet_config.yaml", "paper_wallet_config.yaml"),
    ]
    
    for file_path, name in config_files:
        config_file = Path(file_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
                
                balance = config.get('initial_balance', config.get('paper_wallet_balance', 'Not found'))
                print(f"   {name}: ${balance}")
            except Exception as e:
                print(f"   {name}: Error reading - {e}")
        else:
            print(f"   {name}: File not found")
    
    # 5. Summary and recommendations
    print("\n" + "=" * 80)
    print("🔍 SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n📋 ISSUE IDENTIFICATION:")
    print("   1. PaperWallet shows negative balance (-$1,702.87)")
    print("   2. Frontend calculates available balance as $803.83")
    print("   3. This suggests the frontend is using a different balance source")
    print("   4. TradeManager and PaperWallet may be out of sync")
    
    print("\n🔧 ROOT CAUSE ANALYSIS:")
    print("   1. PaperWallet balance calculation includes all position costs")
    print("   2. Frontend uses a fallback balance source (likely $10,000 default)")
    print("   3. Frontend calculates available balance as: total_balance - position_values")
    print("   4. Position values may be calculated differently between systems")
    
    print("\n✅ RECOMMENDED FIXES:")
    print("   1. Synchronize PaperWallet with TradeManager")
    print("   2. Update frontend to use TradeManager as single source of truth")
    print("   3. Fix balance calculation logic to be consistent")
    print("   4. Reset paper wallet balance to match actual available funds")
    print("   5. Ensure all systems use the same position valuation method")
    
    print("\n🚀 IMMEDIATE ACTIONS:")
    print("   1. Run: python fix_trade_manager_state.py")
    print("   2. Run: python migrate_to_trade_manager.py")
    print("   3. Restart the bot to sync all systems")
    print("   4. Monitor balance consistency after restart")

if __name__ == "__main__":
    analyze_balance_discrepancy()
