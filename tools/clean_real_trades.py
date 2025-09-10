#!/usr/bin/env python3
"""
Clean Real Trades

This script removes all the fake XBT trades and keeps only the legitimate trades
that the user actually made (BTC, HBAR, USDUC).
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "crypto_bot" / "logs"
TRADES_FILE = LOGS_DIR / "trades.csv"

def clean_real_trades():
    """Remove fake XBT trades and keep only legitimate trades."""
    if not TRADES_FILE.exists():
        logger.error(f"Trades file not found: {TRADES_FILE}")
        return False
    
    try:
        # Read current trades
        df = pd.read_csv(TRADES_FILE)
        logger.info(f"Current trades file contains {len(df)} trades")
        
        # Show current breakdown
        symbol_counts = df['symbol'].value_counts()
        logger.info("Current trades by symbol:")
        for symbol, count in symbol_counts.items():
            logger.info(f"  {symbol}: {count} trades")
        
        # Define legitimate symbols (trades the user actually made)
        legitimate_symbols = ['BTC/USD', 'HBAR/USD', 'USDUC/USD']
        
        # Filter to keep only legitimate trades
        legitimate_trades = df[df['symbol'].isin(legitimate_symbols)].copy()
        
        logger.info(f"\nLegitimate trades found:")
        for symbol in legitimate_symbols:
            symbol_trades = legitimate_trades[legitimate_trades['symbol'] == symbol]
            if len(symbol_trades) > 0:
                logger.info(f"  {symbol}: {len(symbol_trades)} trades")
                for _, trade in symbol_trades.iterrows():
                    logger.info(f"    {trade['side']} {trade['amount']} @ {trade['price']}")
        
        # Remove fake XBT trades
        fake_trades_removed = len(df) - len(legitimate_trades)
        logger.info(f"\nRemoving {fake_trades_removed} fake XBT trades...")
        
        # Create backup before cleaning
        backup_file = LOGS_DIR / f"trades_backup_before_cleaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False)
        logger.info(f"Created backup: {backup_file}")
        
        # Save cleaned trades
        legitimate_trades.to_csv(TRADES_FILE, index=False)
        logger.info(f"Cleaned trades file now contains {len(legitimate_trades)} legitimate trades")
        
        # Verify the cleaned file
        verify_cleaned_trades(legitimate_trades)
        
        return True
        
    except Exception as e:
        logger.error(f"Error cleaning trades: {e}")
        return False

def verify_cleaned_trades(df):
    """Verify that the cleaned trades file contains only legitimate trades."""
    logger.info("\n=== Verification ===")
    
    # Check for any remaining XBT trades
    xbt_trades = df[df['symbol'] == 'XBT/USDT']
    if len(xbt_trades) > 0:
        logger.error(f"ERROR: Still found {len(xbt_trades)} XBT trades!")
        return False
    else:
        logger.info("✓ No XBT trades found (good!)")
    
    # Verify legitimate trades are present
    legitimate_symbols = ['BTC/USD', 'HBAR/USD', 'USDUC/USD']
    for symbol in legitimate_symbols:
        symbol_trades = df[df['symbol'] == symbol]
        if len(symbol_trades) > 0:
            logger.info(f"✓ {symbol}: {len(symbol_trades)} trades preserved")
        else:
            logger.warning(f"⚠ {symbol}: No trades found")
    
    # Calculate P&L for BTC
    btc_trades = df[df['symbol'] == 'BTC/USD']
    if len(btc_trades) > 0:
        buy_trades = btc_trades[btc_trades['side'] == 'buy']
        sell_trades = btc_trades[btc_trades['side'] == 'sell']
        
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            total_bought = (buy_trades['amount'] * buy_trades['price']).sum()
            total_sold = (sell_trades['amount'] * sell_trades['price']).sum()
            pnl = total_sold - total_bought
            
            logger.info(f"✓ BTC P&L: ${pnl:,.2f}")
            
            if abs(pnl - 2943) < 1:
                logger.info("✓ BTC P&L is correct ($2,943)")
            else:
                logger.warning(f"⚠ BTC P&L shows ${pnl:,.2f}, expected $2,943")
    
    logger.info(f"\n✓ Cleaned trades file contains {len(df)} legitimate trades")
    return True

def main():
    """Main function."""
    logger.info("=== Cleaning Real Trades ===")
    
    if clean_real_trades():
        logger.info("\n✓ Successfully cleaned trades file!")
        logger.info("✓ Removed all fake XBT trades")
        logger.info("✓ Preserved your legitimate BTC, HBAR, and USDUC trades")
        logger.info("✓ P&L calculations should now be accurate")
    else:
        logger.error("Failed to clean trades file")

if __name__ == "__main__":
    main()
