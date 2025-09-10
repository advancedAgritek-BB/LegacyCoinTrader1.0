#!/usr/bin/env python3
"""
Fix BTC Precision Issue

This script fixes the floating-point precision issue in the BTC sell trade
that's causing it to show as still having an open position.
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

def fix_btc_precision():
    """Fix the BTC sell amount precision issue."""
    if not TRADES_FILE.exists():
        logger.error(f"Trades file not found: {TRADES_FILE}")
        return False
    
    try:
        # Read current trades
        df = pd.read_csv(TRADES_FILE)
        logger.info(f"Current trades file contains {len(df)} trades")
        
        # Find BTC trades
        btc_trades = df[df['symbol'] == 'BTC/USD']
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
        
        sell_trade_idx = sell_trades.index[0]
        current_sell_amount = df.loc[sell_trade_idx, 'amount']
        
        logger.info(f"BTC Position Analysis:")
        logger.info(f"  Total bought: {total_bought}")
        logger.info(f"  Current sell amount: {current_sell_amount}")
        logger.info(f"  Difference: {total_bought - current_sell_amount}")
        
        # Fix the sell amount to exactly match the total bought
        df.loc[sell_trade_idx, 'amount'] = total_bought
        
        # Create backup before saving
        backup_file = LOGS_DIR / f"trades_backup_before_btc_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False)
        logger.info(f"Created backup: {backup_file}")
        
        # Save the updated trades file
        df.to_csv(TRADES_FILE, index=False)
        logger.info(f"Fixed BTC sell amount from {current_sell_amount} to {total_bought}")
        
        # Verify the fix
        verify_btc_fix()
        
        return True
        
    except Exception as e:
        logger.error(f"Error fixing BTC precision: {e}")
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
    logger.info("=== Fixing BTC Precision Issue ===")
    
    if fix_btc_precision():
        logger.info("\n✓ Successfully fixed BTC precision issue!")
        logger.info("✓ BTC position should now show as closed")
        logger.info("✓ Only HBAR should remain active")
    else:
        logger.error("Failed to fix BTC precision issue")

if __name__ == "__main__":
    main()
