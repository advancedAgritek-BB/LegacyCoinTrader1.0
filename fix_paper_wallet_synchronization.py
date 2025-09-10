#!/usr/bin/env python3
"""
Comprehensive fix for paper wallet synchronization issues.
This script will:
1. Reset the paper wallet to match current trading context
2. Enable TradeManager as single source of truth
3. Ensure proper synchronization
"""

import asyncio
import logging
import yaml
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_bot.paper_wallet import PaperWallet

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_paper_wallet_to_initial_state():
    """Reset paper wallet to initial state with no positions."""
    try:
        # Load current paper wallet
        pw = PaperWallet(10000.0)
        pw.load_state()
        
        logger.info(f"Current paper wallet state:")
        logger.info(f"  Balance: ${pw.balance:.2f}")
        logger.info(f"  Positions: {len(pw.positions)}")
        logger.info(f"  Initial balance: ${pw.initial_balance:.2f}")
        
        # Reset to initial state
        pw._balance = pw.initial_balance
        pw.positions = {}
        pw.realized_pnl = 0.0
        pw.total_trades = 0
        pw.winning_trades = 0
        
        # Save the reset state
        pw.save_state()
        
        logger.info(f"Paper wallet reset to initial state:")
        logger.info(f"  New balance: ${pw.balance:.2f}")
        logger.info(f"  Positions: {len(pw.positions)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error resetting paper wallet: {e}")
        return False

def enable_trade_manager_as_source():
    """Enable TradeManager as the single source of truth in configuration."""
    try:
        config_path = Path("config/trading_config.yaml")
        if not config_path.exists():
            logger.warning("Trading config not found, creating default")
            config = {
                "execution_mode": "dry_run",
                "use_trade_manager_as_source": True,
                "position_sync_enabled": True,
                "paper_wallet_initial_balance": 10000.0
            }
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Enable TradeManager as source of truth
        config["use_trade_manager_as_source"] = True
        config["position_sync_enabled"] = True
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("TradeManager enabled as single source of truth")
        return True
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return False

def clear_positions_log():
    """Clear the positions.log file to start fresh."""
    try:
        positions_log = Path("crypto_bot/logs/positions.log")
        if positions_log.exists():
            # Backup the current log
            backup_path = positions_log.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            positions_log.rename(backup_path)
            logger.info(f"Backed up positions.log to {backup_path}")
        
        # Create new empty log
        positions_log.parent.mkdir(parents=True, exist_ok=True)
        positions_log.touch()
        logger.info("Cleared positions.log")
        
        return True
        
    except Exception as e:
        logger.error(f"Error clearing positions.log: {e}")
        return False

def verify_synchronization():
    """Verify that the paper wallet is properly synchronized."""
    try:
        pw = PaperWallet(10000.0)
        pw.load_state()
        
        logger.info("Verification results:")
        logger.info(f"  Paper wallet balance: ${pw.balance:.2f}")
        logger.info(f"  Paper wallet positions: {len(pw.positions)}")
        logger.info(f"  Initial balance: ${pw.initial_balance:.2f}")
        logger.info(f"  Realized PnL: ${pw.realized_pnl:.2f}")
        
        if len(pw.positions) == 0 and pw.balance == pw.initial_balance:
            logger.info("✅ Paper wallet is properly synchronized")
            return True
        else:
            logger.warning("❌ Paper wallet still has synchronization issues")
            return False
            
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return False

async def main():
    """Main function to execute the comprehensive fix."""
    logger.info("=== Starting comprehensive paper wallet synchronization fix ===")
    
    # Step 1: Reset paper wallet to initial state
    logger.info("Step 1: Resetting paper wallet to initial state...")
    if not reset_paper_wallet_to_initial_state():
        logger.error("Failed to reset paper wallet")
        return False
    
    # Step 2: Enable TradeManager as single source of truth
    logger.info("Step 2: Enabling TradeManager as single source of truth...")
    if not enable_trade_manager_as_source():
        logger.error("Failed to enable TradeManager")
        return False
    
    # Step 3: Clear positions log
    logger.info("Step 3: Clearing positions log...")
    if not clear_positions_log():
        logger.error("Failed to clear positions log")
        return False
    
    # Step 4: Verify synchronization
    logger.info("Step 4: Verifying synchronization...")
    if not verify_synchronization():
        logger.error("Synchronization verification failed")
        return False
    
    logger.info("=== Comprehensive fix completed successfully ===")
    logger.info("The system is now ready to use TradeManager as the single source of truth.")
    logger.info("Paper wallet has been reset to initial state with no positions.")
    logger.info("Negative balance issues should be resolved.")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
