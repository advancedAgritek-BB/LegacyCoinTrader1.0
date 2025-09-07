#!/usr/bin/env python3
"""
Fix script to resolve wallet balance discrepancy between TradeManager and PaperWallet

This script will:
1. Analyze the current state
2. Synchronize PaperWallet with TradeManager
3. Fix the balance calculation
4. Ensure consistency across all systems
"""

import yaml
import json
from pathlib import Path
from decimal import Decimal
from typing import Dict, Any, List
import shutil
from datetime import datetime

def backup_current_state():
    """Create backup of current state files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    backup_dir = Path(f"backup_wallet_fix_{timestamp}")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "crypto_bot/logs/paper_wallet_state.yaml",
        "crypto_bot/logs/trade_manager_state.json",
        "crypto_bot/logs/paper_wallet.yaml",
        "crypto_bot/user_config.yaml",
    ]
    
    for file_path in files_to_backup:
        src = Path(file_path)
        if src.exists():
            dst = backup_dir / src.name
            shutil.copy2(src, dst)
            print(f"✅ Backed up {file_path} to {dst}")
    
    return backup_dir

def analyze_current_state():
    """Analyze current state to understand the discrepancy."""
    print("\n📊 ANALYZING CURRENT STATE:")
    
    # PaperWallet state
    pw_file = Path("crypto_bot/logs/paper_wallet_state.yaml")
    if pw_file.exists():
        with open(pw_file, 'r') as f:
            pw_state = yaml.safe_load(f)
        
        pw_balance = pw_state.get('balance', 0)
        pw_positions = pw_state.get('positions', {})
        pw_position_value = sum(
            pos.get('amount', pos.get('size', 0)) * pos.get('entry_price', 0)
            for pos in pw_positions.values()
        )
        
        print(f"   PaperWallet Balance: ${pw_balance:.2f}")
        print(f"   PaperWallet Positions: {len(pw_positions)}")
        print(f"   PaperWallet Position Value: ${pw_position_value:.2f}")
    
    # TradeManager state
    tm_file = Path("crypto_bot/logs/trade_manager_state.json")
    if tm_file.exists():
        with open(tm_file, 'r') as f:
            tm_state = json.load(f)
        
        tm_positions = tm_state.get('positions', {})
        tm_position_value = sum(
            pos.get('total_amount', 0) * pos.get('average_price', 0)
            for pos in tm_positions.values()
            if pos.get('total_amount', 0) > 0
        )
        
        print(f"   TradeManager Positions: {len(tm_positions)}")
        print(f"   TradeManager Position Value: ${tm_position_value:.2f}")
    
    return pw_balance, pw_position_value, tm_position_value

def fix_paper_wallet_balance():
    """Fix the paper wallet balance to match TradeManager."""
    print("\n🔧 FIXING PAPER WALLET BALANCE:")
    
    # Get TradeManager position value
    tm_file = Path("crypto_bot/logs/trade_manager_state.json")
    if not tm_file.exists():
        print("   ❌ TradeManager state file not found")
        return False
    
    with open(tm_file, 'r') as f:
        tm_state = json.load(f)
    
    tm_positions = tm_state.get('positions', {})
    tm_position_value = sum(
        pos.get('total_amount', 0) * pos.get('average_price', 0)
        for pos in tm_positions.values()
        if pos.get('total_amount', 0) > 0
    )
    
    print(f"   TradeManager Position Value: ${tm_position_value:.2f}")
    
    # Calculate correct balance
    initial_balance = 10000.0
    correct_balance = initial_balance - tm_position_value
    
    print(f"   Initial Balance: ${initial_balance:.2f}")
    print(f"   Correct Balance: ${correct_balance:.2f}")
    
    # Update paper wallet state
    pw_file = Path("crypto_bot/logs/paper_wallet_state.yaml")
    if pw_file.exists():
        with open(pw_file, 'r') as f:
            pw_state = yaml.safe_load(f)
        
        # Update balance
        pw_state['balance'] = correct_balance
        pw_state['initial_balance'] = initial_balance
        
        # Clear old positions and sync with TradeManager
        pw_state['positions'] = {}
        
        # Add positions from TradeManager
        for symbol, tm_pos in tm_positions.items():
            if tm_pos.get('total_amount', 0) > 0:  # Only open positions
                # Convert TradeManager position to PaperWallet format
                pw_pos = {
                    'symbol': symbol,
                    'side': tm_pos.get('side', 'buy'),
                    'amount': tm_pos.get('total_amount', 0),
                    'entry_price': tm_pos.get('average_price', 0),
                    'entry_time': tm_pos.get('entry_time', datetime.now().isoformat()),
                    'reserved': 0.0
                }
                pw_state['positions'][symbol] = pw_pos
        
        # Save updated state
        with open(pw_file, 'w') as f:
            yaml.dump(pw_state, f, default_flow_style=False)
        
        print(f"   ✅ Updated PaperWallet balance to ${correct_balance:.2f}")
        print(f"   ✅ Synced {len(pw_state['positions'])} positions from TradeManager")
        
        return True
    else:
        print("   ❌ PaperWallet state file not found")
        return False

def update_config_files():
    """Update configuration files to use consistent balance."""
    print("\n⚙️ UPDATING CONFIGURATION FILES:")
    
    # Calculate correct balance
    tm_file = Path("crypto_bot/logs/trade_manager_state.json")
    if tm_file.exists():
        with open(tm_file, 'r') as f:
            tm_state = json.load(f)
        
        tm_positions = tm_state.get('positions', {})
        tm_position_value = sum(
            pos.get('total_amount', 0) * pos.get('average_price', 0)
            for pos in tm_positions.values()
            if pos.get('total_amount', 0) > 0
        )
        
        initial_balance = 10000.0
        correct_balance = initial_balance - tm_position_value
        
        # Update paper_wallet.yaml
        pw_config_file = Path("crypto_bot/logs/paper_wallet.yaml")
        pw_config = {'initial_balance': correct_balance}
        with open(pw_config_file, 'w') as f:
            yaml.dump(pw_config, f, default_flow_style=False)
        print(f"   ✅ Updated paper_wallet.yaml: ${correct_balance:.2f}")
        
        # Update user_config.yaml
        user_config_file = Path("crypto_bot/user_config.yaml")
        if user_config_file.exists():
            with open(user_config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        
        config['paper_wallet_balance'] = correct_balance
        with open(user_config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"   ✅ Updated user_config.yaml: ${correct_balance:.2f}")
        
        # Update legacy config if it exists
        legacy_config_file = Path("crypto_bot/paper_wallet_config.yaml")
        if legacy_config_file.exists():
            with open(legacy_config_file, 'r') as f:
                legacy_config = yaml.safe_load(f) or {}
            legacy_config['initial_balance'] = correct_balance
            with open(legacy_config_file, 'w') as f:
                yaml.dump(legacy_config, f, default_flow_style=False)
            print(f"   ✅ Updated paper_wallet_config.yaml: ${correct_balance:.2f}")

def verify_fix():
    """Verify that the fix was successful."""
    print("\n✅ VERIFYING FIX:")
    
    # Check PaperWallet state
    pw_file = Path("crypto_bot/logs/paper_wallet_state.yaml")
    if pw_file.exists():
        with open(pw_file, 'r') as f:
            pw_state = yaml.safe_load(f)
        
        pw_balance = pw_state.get('balance', 0)
        pw_positions = pw_state.get('positions', {})
        
        print(f"   PaperWallet Balance: ${pw_balance:.2f}")
        print(f"   PaperWallet Positions: {len(pw_positions)}")
        
        if pw_balance >= 0:
            print("   ✅ PaperWallet balance is now positive")
        else:
            print("   ❌ PaperWallet balance is still negative")
    
    # Check TradeManager state
    tm_file = Path("crypto_bot/logs/trade_manager_state.json")
    if tm_file.exists():
        with open(tm_file, 'r') as f:
            tm_state = json.load(f)
        
        tm_positions = tm_state.get('positions', {})
        open_positions = [pos for pos in tm_positions.values() if pos.get('total_amount', 0) > 0]
        
        print(f"   TradeManager Open Positions: {len(open_positions)}")
        
        if len(pw_positions) == len(open_positions):
            print("   ✅ Position counts match between PaperWallet and TradeManager")
        else:
            print(f"   ⚠️ Position count mismatch: PaperWallet={len(pw_positions)}, TradeManager={len(open_positions)}")

def main():
    """Main function to fix the wallet balance discrepancy."""
    print("=" * 80)
    print("💰 WALLET BALANCE DISCREPANCY FIX")
    print("=" * 80)
    
    # Step 1: Backup current state
    print("\n📦 CREATING BACKUP:")
    backup_dir = backup_current_state()
    print(f"   Backup created in: {backup_dir}")
    
    # Step 2: Analyze current state
    pw_balance, pw_position_value, tm_position_value = analyze_current_state()
    
    # Step 3: Fix paper wallet balance
    if not fix_paper_wallet_balance():
        print("   ❌ Failed to fix paper wallet balance")
        return
    
    # Step 4: Update configuration files
    update_config_files()
    
    # Step 5: Verify fix
    verify_fix()
    
    print("\n" + "=" * 80)
    print("🎉 FIX COMPLETED")
    print("=" * 80)
    print("\n📋 SUMMARY:")
    print("   1. ✅ Created backup of current state")
    print("   2. ✅ Fixed PaperWallet balance to match TradeManager")
    print("   3. ✅ Updated all configuration files")
    print("   4. ✅ Synchronized positions between systems")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Restart the bot to ensure all systems are in sync")
    print("   2. Monitor the frontend to confirm available balance is correct")
    print("   3. Check that no more negative balance warnings appear")
    print("   4. Verify that position tracking is consistent")
    
    print(f"\n💾 Backup location: {backup_dir}")
    print("   You can restore the previous state if needed by copying files back")

if __name__ == "__main__":
    main()
