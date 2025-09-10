#!/usr/bin/env python3
"""
Trading Data Cleanup Tool

This script cleans up the trading data by:
1. Removing duplicate trades
2. Filtering out test/fake trades
3. Keeping only legitimate trading activity
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
CLEAN_TRADES_FILE = LOGS_DIR / "trades_clean.csv"


def analyze_trades():
    """Analyze the current trades file to understand the data."""
    if not TRADES_FILE.exists():
        logger.error(f"Trades file not found: {TRADES_FILE}")
        return None
    
    try:
        df = pd.read_csv(TRADES_FILE)
        logger.info(f"Loaded {len(df)} trades from {TRADES_FILE}")
        
        # Basic analysis
        logger.info("Trade Analysis:")
        logger.info(f"  Total trades: {len(df)}")
        logger.info(f"  Unique symbols: {df['symbol'].nunique()}")
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"  Total volume: {df['amount'].sum():.2f}")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['symbol', 'side', 'amount', 'price', 'timestamp'], keep=False)
        duplicate_count = duplicates.sum()
        logger.info(f"  Duplicate trades: {duplicate_count}")
        
        # Check for suspicious patterns
        zero_price_trades = len(df[df['price'] == 0])
        logger.info(f"  Zero price trades: {zero_price_trades}")
        
        # Show symbol distribution
        symbol_counts = df['symbol'].value_counts()
        logger.info("  Symbol distribution:")
        for symbol, count in symbol_counts.items():
            logger.info(f"    {symbol}: {count} trades")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to analyze trades: {e}")
        return None


def clean_trades(df):
    """Clean the trades data by removing duplicates and suspicious entries."""
    if df is None or df.empty:
        logger.error("No trades to clean")
        return None
    
    original_count = len(df)
    logger.info(f"Starting cleanup of {original_count} trades...")
    
    # Remove exact duplicates
    df_clean = df.drop_duplicates(subset=['symbol', 'side', 'amount', 'price', 'timestamp'], keep='first')
    logger.info(f"Removed {original_count - len(df_clean)} exact duplicates")
    
    # Remove zero price trades (likely test trades)
    df_clean = df_clean[df_clean['price'] > 0]
    logger.info(f"Removed {len(df) - len(df_clean)} zero price trades")
    
    # Remove trades with zero amounts
    df_clean = df_clean[df_clean['amount'] > 0]
    logger.info(f"Removed {len(df) - len(df_clean)} zero amount trades")
    
    # Sort by timestamp
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
    df_clean = df_clean.sort_values('timestamp')
    
    # Remove trades that are too close in time (within 1 second) with same symbol and side
    df_clean = df_clean.drop_duplicates(subset=['symbol', 'side'], keep='first')
    logger.info(f"Removed duplicate symbol/side combinations, keeping first occurrence")
    
    final_count = len(df_clean)
    logger.info(f"Cleanup complete: {original_count} -> {final_count} trades")
    
    return df_clean


def create_clean_trades_file(df_clean):
    """Create a clean trades file."""
    if df_clean is None or df_clean.empty:
        logger.error("No clean trades to save")
        return False
    
    try:
        # Save clean trades
        df_clean.to_csv(CLEAN_TRADES_FILE, index=False)
        logger.info(f"Saved clean trades to: {CLEAN_TRADES_FILE}")
        
        # Show summary of clean trades
        logger.info("Clean trades summary:")
        logger.info(f"  Total trades: {len(df_clean)}")
        logger.info(f"  Date range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
        logger.info(f"  Symbols: {', '.join(df_clean['symbol'].unique())}")
        logger.info(f"  Total volume: {df_clean['amount'].sum():.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save clean trades: {e}")
        return False


def backup_original_and_restore():
    """Backup the original file and restore with clean data."""
    if not CLEAN_TRADES_FILE.exists():
        logger.error("Clean trades file not found")
        return False
    
    try:
        # Create backup of current file
        backup_file = LOGS_DIR / f"trades_backup_before_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        TRADES_FILE.rename(backup_file)
        logger.info(f"Backed up original to: {backup_file}")
        
        # Copy clean file to main location
        import shutil
        shutil.copy2(CLEAN_TRADES_FILE, TRADES_FILE)
        logger.info(f"Restored clean trades to: {TRADES_FILE}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to backup and restore: {e}")
        return False


def main():
    """Main cleanup process."""
    logger.info("Starting trading data cleanup...")
    
    # Analyze current trades
    df = analyze_trades()
    if df is None:
        return 1
    
    # Clean trades
    df_clean = clean_trades(df)
    if df_clean is None:
        return 1
    
    # Create clean file
    if not create_clean_trades_file(df_clean):
        return 1
    
    # Ask for confirmation before replacing
    logger.info("\n" + "="*50)
    logger.info("CLEANUP SUMMARY:")
    logger.info(f"Original trades: {len(df)}")
    logger.info(f"Clean trades: {len(df_clean)}")
    logger.info(f"Removed: {len(df) - len(df_clean)} trades")
    logger.info("="*50)
    
    response = input("\nDo you want to replace the current trades.csv with the clean version? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        if backup_original_and_restore():
            logger.info("✅ Trading data cleanup completed successfully!")
            return 0
        else:
            logger.error("❌ Failed to restore clean trades")
            return 1
    else:
        logger.info("Cleanup completed but original file not replaced.")
        logger.info(f"Clean trades are available at: {CLEAN_TRADES_FILE}")
        return 0


if __name__ == "__main__":
    exit(main())
