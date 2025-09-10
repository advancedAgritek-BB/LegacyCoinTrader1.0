#!/usr/bin/env python3
"""
Diagnostic script to analyze position count mismatch and negative wallet balance issues.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trade_manager_state() -> Dict[str, Any]:
    """Load TradeManager state from JSON file."""
    try:
        with open('crypto_bot/logs/trade_manager_state.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load TradeManager state: {e}")
        return {}

def load_paper_wallet_state() -> Dict[str, Any]:
    """Load PaperWallet state from YAML file."""
    try:
        with open('crypto_bot/logs/paper_wallet_state.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load PaperWallet state: {e}")
        return {}

def analyze_position_mismatch():
    """Analyze the mismatch between TradeManager and PaperWallet positions."""
    
    print("=" * 80)
    print("POSITION MISMATCH DIAGNOSTIC")
    print("=" * 80)
    
    # Load states
    tm_state = load_trade_manager_state()
    pw_state = load_paper_wallet_state()
    
    if not tm_state or not pw_state:
        print("❌ Failed to load state files")
        return
    
    # Analyze TradeManager positions
    print("\n📊 TRADE MANAGER ANALYSIS:")
    print("-" * 40)
    
    tm_positions = tm_state.get('positions', {})
    tm_open_positions = {symbol: pos for symbol, pos in tm_positions.items() 
                        if pos.get('total_amount', 0) > 0}
    
    print(f"Total positions in TradeManager: {len(tm_positions)}")
    print(f"Open positions in TradeManager: {len(tm_open_positions)}")
    
    if tm_open_positions:
        print("\nOpen positions:")
        for symbol, pos in tm_open_positions.items():
            print(f"  • {symbol}: {pos['side']} {pos['total_amount']} @ ${pos['average_price']}")
    
    # Analyze PaperWallet positions
    print("\n💰 PAPER WALLET ANALYSIS:")
    print("-" * 40)
    
    pw_positions = pw_state.get('positions', {})
    pw_balance = pw_state.get('balance', 0)
    pw_initial_balance = pw_state.get('initial_balance', 0)
    
    print(f"PaperWallet balance: ${pw_balance:.2f}")
    print(f"Initial balance: ${pw_initial_balance:.2f}")
    print(f"Balance change: ${pw_balance - pw_initial_balance:.2f}")
    print(f"Total positions in PaperWallet: {len(pw_positions)}")
    
    if pw_positions:
        print("\nPaperWallet positions:")
        for pos_id, pos in pw_positions.items():
            symbol = pos.get('symbol', pos_id)
            side = pos.get('side', 'unknown')
            size = pos.get('size', pos.get('amount', 0))
            entry_price = pos.get('entry_price', 0)
            print(f"  • {symbol}: {side} {size} @ ${entry_price}")
    
    # Calculate position values
    print("\n💡 POSITION VALUE ANALYSIS:")
    print("-" * 40)
    
    # TradeManager position values
    tm_total_value = 0
    for symbol, pos in tm_open_positions.items():
        value = pos['total_amount'] * pos['average_price']
        tm_total_value += value
        print(f"  TM {symbol}: ${value:.2f}")
    
    print(f"TradeManager total position value: ${tm_total_value:.2f}")
    
    # PaperWallet position values
    pw_total_value = 0
    for pos_id, pos in pw_positions.items():
        size = pos.get('size', pos.get('amount', 0))
        entry_price = pos.get('entry_price', 0)
        value = size * entry_price
        pw_total_value += value
        symbol = pos.get('symbol', pos_id)
        print(f"  PW {symbol}: ${value:.2f}")
    
    print(f"PaperWallet total position value: ${pw_total_value:.2f}")
    
    # Identify mismatches
    print("\n🔍 MISMATCH ANALYSIS:")
    print("-" * 40)
    
    tm_symbols = set(tm_open_positions.keys())
    pw_symbols = {pos.get('symbol', pos_id) for pos_id, pos in pw_positions.items()}
    
    print(f"TradeManager symbols: {sorted(tm_symbols)}")
    print(f"PaperWallet symbols: {sorted(pw_symbols)}")
    
    tm_only = tm_symbols - pw_symbols
    pw_only = pw_symbols - tm_symbols
    common = tm_symbols & pw_symbols
    
    if tm_only:
        print(f"❌ Symbols only in TradeManager: {sorted(tm_only)}")
    if pw_only:
        print(f"❌ Symbols only in PaperWallet: {sorted(pw_only)}")
    if common:
        print(f"✅ Common symbols: {sorted(common)}")
    
    # Root cause analysis
    print("\n🎯 ROOT CAUSE ANALYSIS:")
    print("-" * 40)
    
    if pw_balance < 0:
        print(f"❌ PaperWallet has negative balance: ${pw_balance:.2f}")
        print("   This suggests positions were opened without proper balance validation")
    
    if len(pw_positions) > len(tm_open_positions):
        print(f"❌ PaperWallet has more positions ({len(pw_positions)}) than TradeManager ({len(tm_open_positions)})")
        print("   This suggests PaperWallet is tracking positions that don't exist in TradeManager")
    
    if abs(pw_total_value - tm_total_value) > 100:  # $100 tolerance
        print(f"❌ Large difference in position values: PW=${pw_total_value:.2f} vs TM=${tm_total_value:.2f}")
        print("   This suggests position sizes or prices are not synchronized")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    print("1. Reset PaperWallet to match TradeManager state")
    print("2. Ensure proper synchronization between TradeManager and PaperWallet")
    print("3. Validate position opening logic to prevent negative balances")
    print("4. Implement position reconciliation checks")
    
    return {
        'tm_positions': tm_open_positions,
        'pw_positions': pw_positions,
        'tm_total_value': tm_total_value,
        'pw_total_value': pw_total_value,
        'pw_balance': pw_balance,
        'mismatches': {
            'tm_only': list(tm_only),
            'pw_only': list(pw_only),
            'common': list(common)
        }
    }

if __name__ == "__main__":
    analyze_position_mismatch()
