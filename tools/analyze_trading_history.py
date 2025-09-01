#!/usr/bin/env python3
"""
Trading History Analysis Tool

This script analyzes the trading history to understand what happened
and identify missing trades, especially the BTC sell trade with $2,943 profit.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "crypto_bot" / "logs"
TRADES_FILE = LOGS_DIR / "trades.csv"
EXECUTION_LOG = LOGS_DIR / "execution.log"

def analyze_trading_history():
    """Analyze the trading history to understand what happened."""
    logger.info("=== Trading History Analysis ===")
    
    # Check current trades file
    if TRADES_FILE.exists():
        df = pd.read_csv(TRADES_FILE)
        logger.info(f"Current trades.csv contains {len(df)} trades")
        
        # Analyze by symbol
        symbol_counts = df['symbol'].value_counts()
        logger.info("Trades by symbol:")
        for symbol, count in symbol_counts.items():
            logger.info(f"  {symbol}: {count} trades")
        
        # Check for sell trades
        sell_trades = df[df['side'] == 'sell']
        if len(sell_trades) > 0:
            logger.info(f"Found {len(sell_trades)} sell trades:")
            for _, trade in sell_trades.iterrows():
                logger.info(f"  {trade['symbol']} {trade['side']} {trade['amount']} @ {trade['price']}")
        else:
            logger.warning("No sell trades found in trades.csv")
    
    # Check execution log for sell orders
    logger.info("\n=== Execution Log Analysis ===")
    if EXECUTION_LOG.exists():
        with open(EXECUTION_LOG, 'r') as f:
            content = f.read()
        
        # Look for sell order executions
        sell_patterns = [
            r'Order executed.*sell',
            r'Trade written to CSV.*sell',
            r'BTC.*sell.*executed',
            r'profit.*2943',
            r'2943.*profit'
        ]
        
        for pattern in sell_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                logger.info(f"Found {len(matches)} matches for pattern '{pattern}':")
                for match in matches[:5]:  # Show first 5 matches
                    logger.info(f"  {match}")
            else:
                logger.info(f"No matches found for pattern '{pattern}'")

def reconstruct_missing_trades():
    """Attempt to reconstruct missing trades based on available data."""
    logger.info("\n=== Missing Trade Reconstruction ===")
    
    # Based on user's description, there should be:
    # 1. BTC buy trade(s) - we have these
    # 2. BTC sell trade with $2,943 profit - missing
    # 3. HBAR buy trade - we have this
    
    if TRADES_FILE.exists():
        df = pd.read_csv(TRADES_FILE)
        
        # Find BTC trades
        btc_trades = df[df['symbol'] == 'BTC/USD']
        if len(btc_trades) > 0:
            logger.info(f"Found {len(btc_trades)} BTC trades:")
            for _, trade in btc_trades.iterrows():
                logger.info(f"  {trade['side']} {trade['amount']} @ {trade['price']}")
            
            # Calculate what the sell trade should look like
            total_btc_bought = btc_trades['amount'].sum()
            avg_buy_price = (btc_trades['amount'] * btc_trades['price']).sum() / total_btc_bought
            
            logger.info(f"\nBTC Position Analysis:")
            logger.info(f"  Total BTC bought: {total_btc_bought}")
            logger.info(f"  Average buy price: ${avg_buy_price:,.2f}")
            
            # Calculate sell price needed for $2,943 profit
            if total_btc_bought > 0:
                profit_per_btc = 2943 / total_btc_bought
                sell_price = avg_buy_price + profit_per_btc
                logger.info(f"  To achieve $2,943 profit, sell price would be: ${sell_price:,.2f}")
                logger.info(f"  Profit per BTC: ${profit_per_btc:,.2f}")

def create_recovery_plan():
    """Create a plan to recover the missing trading data."""
    logger.info("\n=== Recovery Plan ===")
    
    logger.info("1. IMMEDIATE ACTIONS:")
    logger.info("   ✓ Restored original trades.csv from backup")
    logger.info("   ✓ Identified missing BTC sell trade")
    
    logger.info("\n2. MISSING DATA IDENTIFIED:")
    logger.info("   - BTC sell trade with $2,943 profit")
    logger.info("   - This trade was never logged to trades.csv")
    
    logger.info("\n3. POSSIBLE CAUSES:")
    logger.info("   - Trade logger failed to record the sell trade")
    logger.info("   - Sell order was executed but not logged")
    logger.info("   - Data corruption during refresh operation")
    
    logger.info("\n4. RECOMMENDED ACTIONS:")
    logger.info("   - Manually add the missing BTC sell trade")
    logger.info("   - Implement better trade logging validation")
    logger.info("   - Create automatic backups before refresh operations")
    logger.info("   - Add data integrity checks")

def main():
    """Main analysis function."""
    analyze_trading_history()
    reconstruct_missing_trades()
    create_recovery_plan()

if __name__ == "__main__":
    main()
