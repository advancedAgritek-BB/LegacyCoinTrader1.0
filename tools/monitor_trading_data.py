#!/usr/bin/env python3
"""
Trading Data Monitor

This script monitors the trading data files and performs periodic backups
to prevent data loss. It can be run as a cron job or as a background process.
"""

import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.trade_logger import get_trade_summary, periodic_backup_and_validate, validate_trades_file
from crypto_bot.utils.logger import LOG_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "trading_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_trading_data_health():
    """Check the health of trading data files."""
    trades_file = LOG_DIR / "trades.csv"
    
    if not trades_file.exists():
        logger.error("❌ trades.csv file is missing!")
        return False
    
    # Get file stats
    stat = trades_file.stat()
    file_size = stat.st_size
    last_modified = datetime.fromtimestamp(stat.st_mtime)
    age_hours = (datetime.now() - last_modified).total_seconds() / 3600
    
    logger.info(f"📊 Trades file stats:")
    logger.info(f"   Size: {file_size} bytes")
    logger.info(f"   Last modified: {last_modified}")
    logger.info(f"   Age: {age_hours:.1f} hours")
    
    # Check if file is too old (more than 24 hours)
    if age_hours > 24:
        logger.warning(f"⚠️  Trades file is {age_hours:.1f} hours old")
    
    # Validate file integrity
    if not validate_trades_file():
        logger.error("❌ Trades file validation failed!")
        return False
    
    # Get trade summary
    summary = get_trade_summary()
    if "error" in summary:
        logger.error(f"❌ Failed to get trade summary: {summary['error']}")
        return False
    
    logger.info(f"📈 Trade summary:")
    logger.info(f"   Total trades: {summary['total_trades']}")
    logger.info(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    logger.info(f"   Symbols: {', '.join(summary['symbols'])}")
    logger.info(f"   Total volume: {summary['total_volume']:.2f}")
    logger.info(f"   Buy trades: {summary['buy_trades']}, Sell trades: {summary['sell_trades']}")
    
    return True


def perform_backup():
    """Perform a backup of trading data."""
    logger.info("🔄 Starting periodic backup...")
    
    if periodic_backup_and_validate():
        logger.info("✅ Backup completed successfully")
        return True
    else:
        logger.error("❌ Backup failed")
        return False


def main():
    """Main monitoring function."""
    logger.info("🚀 Starting trading data monitor...")
    
    # Check data health
    if not check_trading_data_health():
        logger.error("❌ Trading data health check failed")
        return 1
    
    # Perform backup
    if not perform_backup():
        logger.error("❌ Backup failed")
        return 1
    
    logger.info("✅ Trading data monitor completed successfully")
    return 0


def run_continuous_monitor(interval_minutes=60):
    """Run continuous monitoring with specified interval."""
    logger.info(f"🔄 Starting continuous monitoring (interval: {interval_minutes} minutes)")
    
    while True:
        try:
            logger.info("=" * 50)
            logger.info(f"🕐 Monitoring cycle at {datetime.now()}")
            
            # Check health
            if not check_trading_data_health():
                logger.error("❌ Health check failed")
            
            # Perform backup
            if not perform_backup():
                logger.error("❌ Backup failed")
            
            logger.info(f"💤 Sleeping for {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
            time.sleep(60)  # Wait 1 minute before retrying


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        run_continuous_monitor(interval)
    else:
        sys.exit(main())
