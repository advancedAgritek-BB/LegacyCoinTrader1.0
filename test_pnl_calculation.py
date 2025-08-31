#!/usr/bin/env python3
"""
Test script to verify PnL calculation is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.app import calculate_wallet_pnl, get_open_positions
from frontend.utils import compute_performance
from crypto_bot.log_reader import _read_trades
from pathlib import Path

def test_pnl_calculation():
    """Test the PnL calculation with current data."""
    print("=== PnL Calculation Test ===\n")
    
    # Test open positions
    print("1. Testing open positions:")
    open_positions = get_open_positions()
    print(f"   Found {len(open_positions)} open positions:")
    for pos in open_positions:
        print(f"   - {pos['symbol']}: {pos['side']} {pos['amount']} @ {pos['entry_price']} (current: {pos['current_price']})")
    
    # Test trades data
    print("\n2. Testing trades data:")
    trade_file = Path("crypto_bot/logs/trades.csv")
    if trade_file.exists():
        df = _read_trades(trade_file)
        print(f"   Found {len(df)} trades in CSV")
        if not df.empty:
            print("   Recent trades:")
            for _, row in df.tail(3).iterrows():
                print(f"   - {row.get('symbol', 'N/A')}: {row.get('side', 'N/A')} {row.get('amount', 0)} @ {row.get('price', 0)}")
    else:
        print("   No trades.csv file found")
    
    # Test performance calculation
    print("\n3. Testing performance calculation:")
    if trade_file.exists():
        df = _read_trades(trade_file)
        perf = compute_performance(df)
        print(f"   Total PnL from trades: ${perf.get('total_pnl', 0.0):.2f}")
        print(f"   Total trades: {perf.get('total_trades', 0)}")
        print(f"   Win rate: {perf.get('win_rate', 0.0)*100:.1f}%")
    
    # Test wallet PnL calculation
    print("\n4. Testing wallet PnL calculation:")
    pnl_data = calculate_wallet_pnl()
    print(f"   Initial balance: ${pnl_data.get('initial_balance', 0.0):.2f}")
    print(f"   Current balance: ${pnl_data.get('current_balance', 0.0):.2f}")
    print(f"   Realized PnL: ${pnl_data.get('realized_pnl', 0.0):.2f}")
    print(f"   Unrealized PnL: ${pnl_data.get('unrealized_pnl', 0.0):.2f}")
    print(f"   Total PnL: ${pnl_data.get('total_pnl', 0.0):.2f}")
    print(f"   PnL percentage: {pnl_data.get('pnl_percentage', 0.0):.2f}%")
    print(f"   Open positions: {pnl_data.get('position_count', 0)}")
    
    if 'error' in pnl_data:
        print(f"   ERROR: {pnl_data['error']}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_pnl_calculation()
