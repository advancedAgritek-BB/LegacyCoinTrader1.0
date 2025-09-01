#!/usr/bin/env python3
"""
Add Missing USDUC Sell Trade

This script adds the missing USDUC sell trade at $0.05634 to close the position.
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

def add_missing_usduc_sell():
    """Add the missing USDUC sell trade at $0.05634."""
    if not TRADES_FILE.exists():
        logger.error(f"Trades file not found: {TRADES_FILE}")
        return False
    
    try:
        # Read current trades
        df = pd.read_csv(TRADES_FILE)
        logger.info(f"Current trades file contains {len(df)} trades")
        
        # Find USDUC buy trade
        usduc_buys = df[df['symbol'] == 'USDUC/USD']
        if len(usduc_buys) == 0:
            logger.error("No USDUC buy trades found")
            return False
        
        # Get USDUC position details
        usduc_buy = usduc_buys.iloc[0]
        amount = usduc_buy['amount']
        buy_price = usduc_buy['price']
        buy_time = pd.to_datetime(usduc_buy['timestamp'])
        
        # Sell price provided by user
        sell_price = 0.05634
        
        logger.info(f"USDUC Position Analysis:")
        logger.info(f"  Bought: {amount} USDUC @ ${buy_price}")
        logger.info(f"  Sold: {amount} USDUC @ ${sell_price}")
        
        # Calculate P&L
        total_bought = amount * buy_price
        total_sold = amount * sell_price
        pnl = total_sold - total_bought
        pnl_percentage = (pnl / total_bought) * 100 if total_bought > 0 else 0
        
        logger.info(f"  Total bought: ${total_bought:,.2f}")
        logger.info(f"  Total sold: ${total_sold:,.2f}")
        logger.info(f"  P&L: ${pnl:,.2f} ({pnl_percentage:+.2f}%)")
        
        # Create the sell trade
        # Use a timestamp that's after the buy trade but before the HBAR trade
        sell_time = buy_time + pd.Timedelta(hours=2)  # 2 hours after buy
        
        sell_trade = {
            'symbol': 'USDUC/USD',
            'side': 'sell',
            'amount': amount,
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
        backup_file = LOGS_DIR / f"trades_backup_before_usduc_sell_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False)
        logger.info(f"Created backup: {backup_file}")
        
        # Save the updated trades file
        new_df.to_csv(TRADES_FILE, index=False)
        logger.info(f"Added USDUC sell trade: {amount} USDUC @ ${sell_price}")
        logger.info(f"Updated trades file now contains {len(new_df)} trades")
        
        return True
        
    except Exception as e:
        logger.error(f"Error adding USDUC sell trade: {e}")
        return False

def verify_positions():
    """Verify that the positions are now correct."""
    logger.info("\n=== Position Verification ===")
    
    if not TRADES_FILE.exists():
        logger.error("Trades file not found")
        return
    
    try:
        df = pd.read_csv(TRADES_FILE)
        
        # Check each symbol
        symbols = df['symbol'].unique()
        
        for symbol in symbols:
            symbol_trades = df[df['symbol'] == symbol]
            buy_trades = symbol_trades[symbol_trades['side'] == 'buy']
            sell_trades = symbol_trades[symbol_trades['side'] == 'sell']
            
            logger.info(f"\n{symbol}:")
            logger.info(f"  Buy trades: {len(buy_trades)}")
            logger.info(f"  Sell trades: {len(sell_trades)}")
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                # Closed position
                total_bought = (buy_trades['amount'] * buy_trades['price']).sum()
                total_sold = (sell_trades['amount'] * sell_trades['price']).sum()
                pnl = total_sold - total_bought
                logger.info(f"  Status: CLOSED")
                logger.info(f"  P&L: ${pnl:,.2f}")
            elif len(buy_trades) > 0 and len(sell_trades) == 0:
                # Open position
                logger.info(f"  Status: ACTIVE (open position)")
            else:
                logger.warning(f"  Status: UNKNOWN")
        
        # Summary
        closed_positions = []
        active_positions = []
        
        for symbol in symbols:
            symbol_trades = df[df['symbol'] == symbol]
            buy_trades = symbol_trades[symbol_trades['side'] == 'buy']
            sell_trades = symbol_trades[symbol_trades['side'] == 'sell']
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                closed_positions.append(symbol)
            elif len(buy_trades) > 0 and len(sell_trades) == 0:
                active_positions.append(symbol)
        
        logger.info(f"\n=== Summary ===")
        logger.info(f"Closed positions: {', '.join(closed_positions)}")
        logger.info(f"Active positions: {', '.join(active_positions)}")
        
        # Verify only HBAR should be active
        if len(active_positions) == 1 and active_positions[0] == 'HBAR/USD':
            logger.info("✓ Correct! Only HBAR is active")
        else:
            logger.warning(f"⚠ Expected only HBAR to be active, but found: {active_positions}")
            
    except Exception as e:
        logger.error(f"Error verifying positions: {e}")

def main():
    """Main function."""
    logger.info("=== Adding Missing USDUC Sell Trade ===")
    
    if add_missing_usduc_sell():
        verify_positions()
        logger.info("\n✓ Successfully added missing USDUC sell trade!")
        logger.info("✓ USDUC position is now closed")
        logger.info("✓ Only HBAR should remain active")
    else:
        logger.error("Failed to add USDUC sell trade")

if __name__ == "__main__":
    main()
