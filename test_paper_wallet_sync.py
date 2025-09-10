#!/usr/bin/env python3
"""
Test script to verify paper wallet synchronization fix.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_bot.paper_wallet import PaperWallet
import yaml
from pathlib import Path

def test_paper_wallet_synchronization():
    """Test that the paper wallet is properly synchronized."""
    print("=== Testing Paper Wallet Synchronization ===")
    
    # Test 1: Check paper wallet state
    pw = PaperWallet(10000.0)
    pw.load_state()
    
    print(f"Paper wallet balance: ${pw.balance:.2f}")
    print(f"Paper wallet positions: {len(pw.positions)}")
    print(f"Initial balance: ${pw.initial_balance:.2f}")
    
    # Test 2: Check configuration
    config_path = Path("config/trading_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"TradeManager as source: {config.get('use_trade_manager_as_source', False)}")
        print(f"Position sync enabled: {config.get('position_sync_enabled', False)}")
    else:
        print("Configuration file not found")
    
    # Test 3: Check positions log
    positions_log = Path("crypto_bot/logs/positions.log")
    if positions_log.exists():
        with open(positions_log, 'r') as f:
            lines = f.readlines()
        print(f"Positions log lines: {len(lines)}")
    else:
        print("Positions log not found")
    
    # Test 4: Verify synchronization
    if len(pw.positions) == 0 and pw.balance == pw.initial_balance:
        print("✅ Paper wallet is properly synchronized")
        return True
    else:
        print("❌ Paper wallet has synchronization issues")
        return False

if __name__ == "__main__":
    success = test_paper_wallet_synchronization()
    sys.exit(0 if success else 1)
