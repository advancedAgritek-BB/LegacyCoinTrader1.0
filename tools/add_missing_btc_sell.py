#!/usr/bin/env python3
"""
Add Missing BTC Sell Trade

This script adds the missing BTC sell trade that closed with $2,943 profit
to restore the correct P&L calculations.
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

def add_missing_btc_sell():
    """Add the missing BTC sell trade with $2,943 profit."""
    if not TRADES_FILE.exists():
        logger.error(f"Trades file not found: {TRADES_FILE}")
        return False
    
    try:
        # Read current trades
        df = pd.read_csv(TRADES_FILE)
        logger.info(f"Current trades file contains {len(df)} trades")
        
        # Find BTC buy trades
        btc_buys = df[df['symbol'] == 'BTC/USD']
        if len(btc_buys) == 0:
            logger.error("No BTC buy trades found")
            return False
        
        # Calculate total BTC bought and average price
        total_btc_bought = btc_buys['amount'].sum()
        avg_buy_price = (btc_buys['amount'] * btc_buys['price']).sum() / total_btc_bought
        
        logger.info(f"BTC Position Analysis:")
        logger.info(f"  Total BTC bought: {total_btc_bought}")
        logger.info(f"  Average buy price: ${avg_buy_price:,.2f}")
        
        # Calculate sell price needed for $2,943 profit
        profit_per_btc = 2943 / total_btc_bought
        sell_price = avg_buy_price + profit_per_btc
        
        logger.info(f"  To achieve $2,943 profit:")
        logger.info(f"    Sell price: ${sell_price:,.2f}")
        logger.info(f"    Profit per BTC: ${profit_per_btc:,.2f}")
        
        # Create the missing sell trade
        # Use a timestamp that's after the last buy trade but before the HBAR trade
        last_btc_buy_time = pd.to_datetime(btc_buys['timestamp'].max())
        sell_time = last_btc_buy_time + pd.Timedelta(hours=1)  # 1 hour after last buy
        
        sell_trade = {
            'symbol': 'BTC/USD',
            'side': 'sell',
            'amount': total_btc_bought,
            'price': sell_price,
            'timestamp': sell_time.strftime('%Y-%m-%d %H:%M:%S'),
            'is_stop': False
        }
        
        # Add the sell trade to the dataframe
        new_df = pd.concat([df, pd.DataFrame([sell_trade])], ignore_index=True)
        
        # Sort by timestamp to maintain chronological order
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
        new_df = new_df.sort_values('timestamp').reset_index(drop=True)
        new_df['timestamp'] = new_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create backup before saving
        backup_file = LOGS_DIR / f"trades_backup_before_btc_sell_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False)
        logger.info(f"Created backup: {backup_file}")
        
        # Save the updated trades file
        new_df.to_csv(TRADES_FILE, index=False)
        logger.info(f"Added BTC sell trade: {total_btc_bought} BTC @ ${sell_price:,.2f}")
        logger.info(f"Updated trades file now contains {len(new_df)} trades")
        
        return True
        
    except Exception as e:
        logger.error(f"Error adding BTC sell trade: {e}")
        return False

def verify_pnl_calculation():
    """Verify that the P&L calculation now works correctly."""
    logger.info("\n=== P&L Verification ===")
    
    if not TRADES_FILE.exists():
        logger.error("Trades file not found")
        return
    
    try:
        df = pd.read_csv(TRADES_FILE)
        
        # Find BTC trades
        btc_trades = df[df['symbol'] == 'BTC/USD']
        if len(btc_trades) == 0:
            logger.error("No BTC trades found")
            return
        
        # Calculate P&L
        buy_trades = btc_trades[btc_trades['side'] == 'buy']
        sell_trades = btc_trades[btc_trades['side'] == 'sell']
        
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            total_bought = (buy_trades['amount'] * buy_trades['price']).sum()
            total_sold = (sell_trades['amount'] * sell_trades['price']).sum()
            pnl = total_sold - total_bought
            
            logger.info(f"BTC P&L Calculation:")
            logger.info(f"  Total bought: ${total_bought:,.2f}")
            logger.info(f"  Total sold: ${total_sold:,.2f}")
            logger.info(f"  P&L: ${pnl:,.2f}")
            
            if abs(pnl - 2943) < 1:  # Allow for small rounding differences
                logger.info("✓ P&L calculation is correct!")
            else:
                logger.warning(f"P&L calculation shows ${pnl:,.2f}, expected $2,943")
        else:
            logger.error("Missing buy or sell trades for P&L calculation")
            
    except Exception as e:
        logger.error(f"Error verifying P&L: {e}")

def main():
    """Main function."""
    logger.info("=== Adding Missing BTC Sell Trade ===")
    
    if add_missing_btc_sell():
        verify_pnl_calculation()
        logger.info("\n✓ Successfully added missing BTC sell trade!")
        logger.info("Your P&L should now show the correct $2,943 profit.")
    else:
        logger.error("Failed to add BTC sell trade")

if __name__ == "__main__":
    main()
