#!/usr/bin/env python3
"""
Trading History Recovery Tool

This script attempts to recover missing trading history by:
1. Parsing execution logs to find executed trades
2. Reconstructing the trades.csv file
3. Creating backups and validating data integrity
"""

import re
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
EXECUTION_LOG = LOGS_DIR / "execution.log"
TRADES_FILE = LOGS_DIR / "trades.csv"
BACKUP_FILE = LOGS_DIR / "trades_backup.csv"


def parse_execution_log():
    """Parse execution log to extract trade information."""
    trades = []
    
    if not EXECUTION_LOG.exists():
        logger.error(f"Execution log not found: {EXECUTION_LOG}")
        return trades
    
    # Patterns to match trade execution entries
    patterns = [
        r'Order executed (\w+) (\w+/\w+) ([\d.]+) \(id/tx: (\w+)\)',
        r'Order executed - id=(\w+) side=(\w+) amount=([\d.]+) price=([\d.]+) dry_run=(\w+)',
        r'Trade written to CSV: (\w+/\w+) (\w+) ([\d.]+) @ ([\d.]+)',
        r'Logged trade: \{.*?"symbol": "([^"]+)",.*?"side": "([^"]+)",.*?"amount": ([\d.]+),.*?"price": ([\d.]+)'
    ]
    
    logger.info(f"Parsing execution log: {EXECUTION_LOG}")
    
    with open(EXECUTION_LOG, 'r') as f:
        for line_num, line in enumerate(f, 1):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        if 'Trade written to CSV' in line:
                            # Extract from "Trade written to CSV" format
                            symbol = match.group(1)
                            side = match.group(2)
                            amount = float(match.group(3))
                            price = float(match.group(4))
                            
                            # Extract timestamp from the line
                            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().isoformat()
                            
                            trades.append({
                                'symbol': symbol,
                                'side': side,
                                'amount': amount,
                                'price': price,
                                'timestamp': timestamp,
                                'is_stop': False,
                                'source': 'execution_log',
                                'line': line_num
                            })
                            logger.debug(f"Found trade: {symbol} {side} {amount} @ {price}")
                            break
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse line {line_num}: {line.strip()} - {e}")
                        continue
    
    logger.info(f"Found {len(trades)} trades in execution log")
    return trades


def create_backup():
    """Create backup of current trades file."""
    if TRADES_FILE.exists():
        try:
            import shutil
            shutil.copy2(TRADES_FILE, BACKUP_FILE)
            logger.info(f"Created backup: {BACKUP_FILE}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    return True


def reconstruct_trades_file(trades):
    """Reconstruct the trades.csv file from recovered data."""
    if not trades:
        logger.warning("No trades to reconstruct")
        return False
    
    # Create DataFrame
    df = pd.DataFrame(trades)
    
    # Ensure we have the required columns
    required_columns = ['symbol', 'side', 'amount', 'price', 'timestamp', 'is_stop']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    # Select only the required columns for CSV
    csv_df = df[required_columns].copy()
    
    # Sort by timestamp
    csv_df['timestamp'] = pd.to_datetime(csv_df['timestamp'], errors='coerce')
    csv_df = csv_df.sort_values('timestamp')
    
    # Write to CSV
    try:
        csv_df.to_csv(TRADES_FILE, index=False)
        logger.info(f"Reconstructed trades.csv with {len(csv_df)} trades")
        return True
    except Exception as e:
        logger.error(f"Failed to write trades.csv: {e}")
        return False


def validate_trades_file():
    """Validate the trades.csv file."""
    if not TRADES_FILE.exists():
        logger.error("trades.csv does not exist")
        return False
    
    try:
        df = pd.read_csv(TRADES_FILE)
        logger.info(f"trades.csv is valid, contains {len(df)} trades")
        
        # Show summary
        if not df.empty:
            logger.info("Trade summary:")
            logger.info(f"  Total trades: {len(df)}")
            logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"  Symbols: {', '.join(df['symbol'].unique())}")
            logger.info(f"  Total volume: {df['amount'].sum():.2f}")
        
        return True
    except Exception as e:
        logger.error(f"trades.csv is corrupted: {e}")
        return False


def main():
    """Main recovery process."""
    logger.info("Starting trading history recovery...")
    
    # Create backup
    create_backup()
    
    # Parse execution log
    trades = parse_execution_log()
    
    if not trades:
        logger.warning("No trades found in execution log")
        return
    
    # Reconstruct trades file
    if reconstruct_trades_file(trades):
        # Validate the result
        if validate_trades_file():
            logger.info("✅ Trading history recovery completed successfully!")
        else:
            logger.error("❌ Trading history recovery failed validation")
    else:
        logger.error("❌ Failed to reconstruct trades file")


if __name__ == "__main__":
    main()
