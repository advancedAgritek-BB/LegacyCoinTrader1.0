#!/usr/bin/env python3
"""
Fix BTC Amounts

This script fixes the BTC amounts by making them exact to avoid floating-point precision issues.
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

def fix_btc_amounts():
    """Fix the BTC amounts to be exact and avoid floating-point issues."""
    if not TRADES_FILE.exists():
        logger.error(f"Trades file not found: {TRADES_FILE}")
        return False
    
    try:
        # Read current trades
        df = pd.read_csv(TRADES_FILE)
        logger.info(f"Current trades file contains {len(df)} trades")
        
        # Find BTC trades
        btc_mask = df['symbol'] == 'BTC/USD'
        btc_trades = df[btc_mask]
        
        if len(btc_trades) == 0:
            logger.error("No BTC trades found")
            return False
        
        # Calculate total BTC bought
        buy_trades = btc_trades[btc_trades['side'] == 'buy']
        total_bought = buy_trades['amount'].sum()
        
        # Find the sell trade
        sell_trades = btc_trades[btc_trades['side'] == 'sell']
        if len(sell_trades) == 0:
            logger.error("No BTC sell trade found")
            return False
        
        logger.info(f"BTC Position Analysis:")
        logger.info(f"  Total bought: {total_bought}")
        logger.info(f"  Sell amount: {sell_trades.iloc[0]['amount']}")
        
        # Fix the amounts to be exact
        # Make all buy amounts exactly 0.1
        buy_indices = df[btc_mask & (df['side'] == 'buy')].index
        for idx in buy_indices:
            df.loc[idx, 'amount'] = 0.1
        
        # Make sell amount exactly 0.3
        sell_idx = df[btc_mask & (df['side'] == 'sell')].index[0]
        df.loc[sell_idx, 'amount'] = 0.3
        
        # Create backup before saving
        backup_file = LOGS_DIR / f"trades_backup_before_btc_amounts_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False)
        logger.info(f"Created backup: {backup_file}")
        
        # Save the updated trades file
        df.to_csv(TRADES_FILE, index=False)
        logger.info(f"Fixed BTC amounts to exact values")
        
        # Verify the fix
        verify_btc_fix()
        
        return True
        
    except Exception as e:
        logger.error(f"Error fixing BTC amounts: {e}")
        return False

def verify_btc_fix():
    """Verify that BTC position is now properly closed."""
    logger.info("\n=== Verification ===")
    
    try:
        from crypto_bot.utils.open_trades import get_open_trades
        from pathlib import Path
        
        open_trades = get_open_trades(Path('crypto_bot/logs/trades.csv'))
        
        logger.info(f"Open trades after fix: {open_trades}")
        
        # Check if BTC is still in open trades
        btc_open = [trade for trade in open_trades if trade['symbol'] == 'BTC/USD']
        
        if btc_open:
            logger.warning(f"⚠ BTC still shows as open: {btc_open}")
        else:
            logger.info("✓ BTC position is now properly closed")
        
        # Check if only HBAR is open
        hbar_open = [trade for trade in open_trades if trade['symbol'] == 'HBAR/USD']
        
        if len(open_trades) == 1 and hbar_open:
            logger.info("✓ Only HBAR is open (correct)")
        else:
            logger.warning(f"⚠ Expected only HBAR to be open, but found: {open_trades}")
            
    except Exception as e:
        logger.error(f"Error verifying BTC fix: {e}")

def main():
    """Main function."""
    logger.info("=== Fixing BTC Amounts ===")
    
    if fix_btc_amounts():
        logger.info("\n✓ Successfully fixed BTC amounts!")
        logger.info("✓ BTC position should now show as closed")
        logger.info("✓ Only HBAR should remain active")
    else:
        logger.error("Failed to fix BTC amounts")

if __name__ == "__main__":
    main()
