#!/usr/bin/env python3
"""
Clean duplicate APU/USD trades from the CSV file.

This script removes duplicate APU/USD sell entries while preserving the legitimate buy trade.
"""

import pandas as pd
import os
from pathlib import Path

def clean_duplicate_apu_trades():
    """Remove duplicate APU/USD sell trades from trades.csv"""

    # Path to the trades CSV file
    trades_file = Path("crypto_bot/logs/trades.csv")

    if not trades_file.exists():
        print(f"Trades file not found: {trades_file}")
        return

    # Read the CSV file
    df = pd.read_csv(trades_file)
    print(f"Original trades count: {len(df)}")

    # Filter APU/USD trades
    apu_trades = df[df['symbol'] == 'APU/USD']
    print(f"APU/USD trades found: {len(apu_trades)}")

    if len(apu_trades) > 0:
        print("APU/USD trades:")
        for idx, row in apu_trades.iterrows():
            print(f"  {row['symbol']}, {row['side']}, {row['amount']}, {row['price']}, {row['timestamp']}")

    # Keep only the first buy trade and the first sell trade for APU/USD
    # Remove all other APU/USD sell trades (duplicates)
    apu_buy_mask = (df['symbol'] == 'APU/USD') & (df['side'] == 'buy')
    apu_sell_mask = (df['symbol'] == 'APU/USD') & (df['side'] == 'sell')

    # Get the first sell trade timestamp (the legitimate one)
    if len(df[apu_sell_mask]) > 0:
        first_sell_timestamp = df[apu_sell_mask]['timestamp'].iloc[0]
        print(f"Keeping first sell trade: {first_sell_timestamp}")

        # Keep only the first sell trade, remove others
        keep_sell_mask = apu_sell_mask & (df['timestamp'] == first_sell_timestamp)
        remove_sell_mask = apu_sell_mask & (df['timestamp'] != first_sell_timestamp)

        print(f"Removing {len(df[remove_sell_mask])} duplicate APU/USD sell trades")

        # Remove duplicate sell trades
        df = df[~(remove_sell_mask)]

    # Show cleaned results
    cleaned_apu_trades = df[df['symbol'] == 'APU/USD']
    print(f"Cleaned APU/USD trades: {len(cleaned_apu_trades)}")

    if len(cleaned_apu_trades) > 0:
        print("Remaining APU/USD trades:")
        for idx, row in cleaned_apu_trades.iterrows():
            print(f"  {row['symbol']}, {row['side']}, {row['amount']}, {row['price']}, {row['timestamp']}")

    # Backup original file
    backup_file = trades_file.with_suffix('.csv.backup_before_apu_clean')
    if not backup_file.exists():
        df_original = pd.read_csv(trades_file)
        df_original.to_csv(backup_file, index=False)
        print(f"Created backup: {backup_file}")

    # Save cleaned data
    df.to_csv(trades_file, index=False)
    print(f"Cleaned trades saved to: {trades_file}")
    print(f"Final trades count: {len(df)}")

if __name__ == "__main__":
    clean_duplicate_apu_trades()
