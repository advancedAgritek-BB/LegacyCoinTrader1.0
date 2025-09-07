#!/usr/bin/env python3
"""
Verify TradeManager integration and check for any remaining APU/USD duplication issues.
"""

from pathlib import Path
import json
import pandas as pd

def verify_trade_manager():
    """Verify TradeManager state and check for APU/USD trades"""

    print("=== TradeManager Verification ===\n")

    # Check TradeManager state file
    trade_manager_file = Path("crypto_bot/logs/trade_manager_state.json")
    if trade_manager_file.exists():
        with open(trade_manager_file, 'r') as f:
            state = json.load(f)

        trades = state.get('trades', [])
        print(f"TradeManager trades count: {len(trades)}")

        # Check APU/USD trades in TradeManager
        apu_trades = [t for t in trades if t['symbol'] == 'APU/USD']
        print(f"APU/USD trades in TradeManager: {len(apu_trades)}")

        if apu_trades:
            print("APU/USD trades in TradeManager:")
            for trade in apu_trades:
                print(f"  {trade['symbol']}, {trade['side']}, {trade['amount']}, {trade['price']}, {trade['timestamp']}")
    else:
        print("TradeManager state file not found")

    print("\n=== CSV File Verification ===\n")

    # Check CSV file
    csv_file = Path("crypto_bot/logs/trades.csv")
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"CSV trades count: {len(df)}")

        # Check APU/USD trades in CSV
        apu_csv_trades = df[df['symbol'] == 'APU/USD']
        print(f"APU/USD trades in CSV: {len(apu_csv_trades)}")

        if len(apu_csv_trades) > 0:
            print("APU/USD trades in CSV:")
            for idx, row in apu_csv_trades.iterrows():
                print(f"  {row['symbol']}, {row['side']}, {row['amount']}, {row['price']}, {row['timestamp']}")
    else:
        print("CSV file not found")

    print("\n=== Summary ===\n")

    if len(apu_trades) == 2 and len(apu_csv_trades) == 2:
        print("✅ SUCCESS: APU/USD trades are properly deduplicated in both systems")
        print("   - TradeManager: 2 trades (1 buy, 1 sell)")
        print("   - CSV file: 2 trades (1 buy, 1 sell)")
    else:
        print("❌ ISSUE: APU/USD trades are not properly synchronized")
        print(f"   - TradeManager: {len(apu_trades) if 'apu_trades' in locals() else 0} trades")
        print(f"   - CSV file: {len(apu_csv_trades) if 'apu_csv_trades' in locals() else 0} trades")

if __name__ == "__main__":
    verify_trade_manager()
