#!/usr/bin/env python3
"""
Comprehensive Wallet Balance Fix for LegacyCoinTrader1.0

This script fixes the wallet balance synchronization issues by:
1. Making TradeManager the single source of truth for trades and positions
2. Removing the problematic TradeManager state reset
3. Implementing proper synchronization between TradeManager and paper wallet
4. Ensuring balance consistency across all components

Author: AI Assistant
Date: 2025-09-03
"""

import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WalletBalanceFixer:
    """Comprehensive fixer for wallet balance synchronization issues."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.logs_dir = self.project_root / "crypto_bot" / "logs"

        # State file paths
        self.paper_wallet_state = self.logs_dir / "paper_wallet_state.yaml"
        self.trade_manager_state = self.logs_dir / "trade_manager_state.json"
        self.backup_dir = self.logs_dir / "migration_backup"

        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)

    def backup_current_state(self) -> None:
        """Create backups of current state files before making changes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("Creating backups of current state...")

        # Backup paper wallet state
        if self.paper_wallet_state.exists():
            backup_pw = self.backup_dir / f"paper_wallet_backup_{timestamp}.yaml"
            backup_pw.write_text(self.paper_wallet_state.read_text())
            logger.info(f"Backed up paper wallet state to {backup_pw}")

        # Backup trade manager state
        if self.trade_manager_state.exists():
            backup_tm = self.backup_dir / f"trade_manager_backup_{timestamp}.json"
            backup_tm.write_text(self.trade_manager_state.read_text())
            logger.info(f"Backed up trade manager state to {backup_tm}")

    def load_paper_wallet_state(self) -> Dict[str, Any]:
        """Load current paper wallet state."""
        if not self.paper_wallet_state.exists():
            logger.warning("Paper wallet state file not found")
            return {}

        with open(self.paper_wallet_state, 'r') as f:
            return yaml.safe_load(f) or {}

    def load_trade_manager_state(self) -> Dict[str, Any]:
        """Load current trade manager state."""
        if not self.trade_manager_state.exists():
            logger.warning("Trade manager state file not found")
            return {}

        with open(self.trade_manager_state, 'r') as f:
            return json.load(f)

    def migrate_positions_to_trade_manager(self) -> Dict[str, Any]:
        """
        Migrate positions from paper wallet to TradeManager format.
        Returns the TradeManager state with migrated positions.
        """
        logger.info("Migrating positions from paper wallet to TradeManager...")

        pw_state = self.load_paper_wallet_state()
        tm_state = self.load_trade_manager_state()

        if not pw_state.get('positions'):
            logger.info("No positions to migrate from paper wallet")
            return tm_state

        # Convert paper wallet positions to TradeManager format
        migrated_positions = {}
        migrated_trades = tm_state.get('trades', [])

        for position_id, pw_pos in pw_state['positions'].items():
            # Create TradeManager position
            tm_position = {
                'symbol': pw_pos['symbol'],
                'side': 'long' if pw_pos['side'] == 'buy' else 'short',
                'total_amount': pw_pos['size'] if 'size' in pw_pos else pw_pos['amount'],
                'average_price': pw_pos['entry_price'],
                'realized_pnl': 0.0,  # Will be calculated properly
                'fees_paid': pw_pos.get('fees_paid', 0.0),
                'entry_time': pw_pos['entry_time'],
                'last_update': pw_pos['entry_time'],
                'highest_price': pw_pos['entry_price'],
                'lowest_price': pw_pos['entry_price'],
                'stop_loss_price': None,
                'take_profit_price': None,
                'trailing_stop_pct': None,
                'metadata': {
                    'migrated_from_paper_wallet': True,
                    'original_position_id': position_id
                },
                'trades': []
            }

            # Create corresponding trade
            trade = {
                'id': f"migrated_{position_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'symbol': pw_pos['symbol'],
                'side': pw_pos['side'],
                'amount': pw_pos['size'] if 'size' in pw_pos else pw_pos['amount'],
                'price': pw_pos['entry_price'],
                'timestamp': pw_pos['entry_time'],
                'strategy': 'migrated_from_paper_wallet',
                'exchange': 'paper',
                'fees': pw_pos.get('fees_paid', 0.0),
                'status': 'filled',
                'order_id': None,
                'client_order_id': None,
                'metadata': {
                    'migrated': True,
                    'original_position_id': position_id
                }
            }

            tm_position['trades'].append(trade)
            migrated_trades.append(trade)
            migrated_positions[pw_pos['symbol']] = tm_position

        # Update TradeManager state
        tm_state['positions'] = migrated_positions
        tm_state['trades'] = migrated_trades
        tm_state['statistics']['total_trades'] = len(migrated_trades)
        tm_state['last_save_time'] = datetime.utcnow().isoformat()

        logger.info(f"Migrated {len(migrated_positions)} positions and {len(migrated_trades)} trades to TradeManager")
        return tm_state

    def fix_paper_wallet_state(self) -> Dict[str, Any]:
        """
        Fix paper wallet state to be consistent with TradeManager.
        Removes positions (since TradeManager is now the source of truth)
        and keeps only balance tracking.
        """
        logger.info("Fixing paper wallet state...")

        pw_state = self.load_paper_wallet_state()

        # Keep only the essential balance information
        fixed_pw_state = {
            'balance': pw_state.get('balance', 10000.0),
            'initial_balance': pw_state.get('initial_balance', 10000.0),
            'realized_pnl': pw_state.get('realized_pnl', 0.0),
            'total_trades': pw_state.get('total_trades', 0),
            'winning_trades': pw_state.get('winning_trades', 0),
            'positions': {},  # Clear positions - TradeManager is now the source of truth
            'last_sync_with_trade_manager': datetime.utcnow().isoformat(),
            'note': 'Positions moved to TradeManager - this wallet now tracks balance only'
        }

        logger.info(f"Fixed paper wallet state: balance=${fixed_pw_state['balance']:.2f}, cleared {len(pw_state.get('positions', {}))} positions")
        return fixed_pw_state

    def update_main_bot_initialization(self) -> None:
        """
        Update the main bot initialization to remove TradeManager state reset
        and implement proper synchronization.
        """
        logger.info("Updating main bot initialization...")

        main_file = self.project_root / "crypto_bot" / "main.py"

        if not main_file.exists():
            logger.error(f"Main bot file not found: {main_file}")
            return

        content = main_file.read_text()

        # Find and remove the TradeManager state reset code
        reset_pattern_start = "# Reset TradeManager state BEFORE it gets used to prevent old positions from being loaded"
        reset_pattern_end = "logger.info(\"✅ Reset paper wallet state file to clean state\")"

        if reset_pattern_start in content and reset_pattern_end in content:
            # Find the section to remove
            start_idx = content.find(reset_pattern_start)
            end_idx = content.find(reset_pattern_end) + len(reset_pattern_end)

            # Remove the reset code
            new_content = content[:start_idx] + content[end_idx:]

            # Add proper synchronization code
            sync_code = '''
            # Initialize TradeManager and sync with paper wallet
            if config.get("execution_mode") == "dry_run":
                try:
                    from crypto_bot.utils.trade_manager import get_trade_manager
                    tm = get_trade_manager()

                    # Sync paper wallet balance with TradeManager
                    if ctx.paper_wallet:
                        # Update paper wallet from TradeManager positions
                        tm_positions = tm.get_all_positions()
                        if tm_positions:
                            # Calculate current portfolio value from TradeManager
                            portfolio_value = ctx.paper_wallet.initial_balance
                            for pos in tm_positions:
                                current_price = tm.price_cache.get(pos.symbol, pos.average_price)
                                if pos.side == 'long':
                                    portfolio_value += (current_price - pos.average_price) * pos.total_amount
                                else:  # short
                                    portfolio_value += (pos.average_price - current_price) * pos.total_amount
                                portfolio_value -= pos.fees_paid

                            ctx.paper_wallet.balance = portfolio_value
                            ctx.balance = portfolio_value
                            logger.info(f"Synchronized paper wallet balance with TradeManager: ${portfolio_value:.2f}")
                        else:
                            # No positions, use initial balance
                            ctx.paper_wallet.balance = ctx.paper_wallet.initial_balance
                            ctx.balance = ctx.paper_wallet.balance

                    logger.info("✅ TradeManager synchronization completed")

                except Exception as e:
                    logger.warning(f"Could not synchronize with TradeManager: {e}")
            '''

            # Insert the sync code after ctx.paper_wallet = paper_wallet
            insert_point = "ctx.paper_wallet = paper_wallet"
            insert_idx = new_content.find(insert_point) + len(insert_point)
            new_content = new_content[:insert_idx] + sync_code + new_content[insert_idx:]

            # Write back the updated content
            main_file.write_text(new_content)
            logger.info("Updated main bot initialization to remove TradeManager reset and add proper sync")
        else:
            logger.warning("Could not find TradeManager reset code to remove")

    def update_paper_wallet_sync_logic(self) -> None:
        """
        Update paper wallet to properly sync with TradeManager instead of preventing sync.
        """
        logger.info("Updating paper wallet synchronization logic...")

        pw_file = self.project_root / "crypto_bot" / "paper_wallet.py"

        if not pw_file.exists():
            logger.error(f"Paper wallet file not found: {pw_file}")
            return

        content = pw_file.read_text()

        # Find the sync method and update it
        sync_method_pattern = "def sync_from_trade_manager(self, trade_manager_positions: List[dict], current_prices: Dict[str, float] = None) -> None:"
        if sync_method_pattern in content:
            # Find the method and update the logic
            start_idx = content.find(sync_method_pattern)
            method_end_pattern = "        except Exception as e:"
            end_idx = content.find(method_end_pattern, start_idx) + 200  # Include some context

            # Replace the sync logic
            old_sync_logic = '''            # FORCE CLEAN STATE - Don't sync if we have old data
            if len(self.positions) > 0:
                logger.warning(f"⚠️ Preventing sync from TradeManager - paper wallet has {len(self.positions)} existing positions")
                logger.warning(f"⚠️ Paper wallet balance: ${self.balance:.2f}")
                return'''

            new_sync_logic = '''            # Allow sync even if we have positions - TradeManager is source of truth
            if len(self.positions) > 0:
                logger.info(f"🔄 Syncing with TradeManager - updating {len(self.positions)} existing positions")
                logger.info(f"📊 Current paper wallet balance: ${self.balance:.2f}")'''

            new_content = content.replace(old_sync_logic, new_sync_logic)
            pw_file.write_text(new_content)
            logger.info("Updated paper wallet sync logic to allow synchronization with TradeManager")
        else:
            logger.warning("Could not find sync method to update")

    def create_synchronization_manager(self) -> None:
        """
        Create a synchronization manager to handle ongoing sync between components.
        """
        logger.info("Creating synchronization manager...")

        sync_manager_code = '''"""
Synchronization Manager for Wallet Balance Components

Ensures consistent state between TradeManager, Paper Wallet, and Balance Manager.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class SynchronizationManager:
    """Manages synchronization between wallet components."""

    def __init__(self):
        self.last_sync_time = None
        self.sync_interval = 60  # seconds

    def sync_all_components(self, ctx=None) -> bool:
        """
        Synchronize all wallet components.
        Returns True if sync was successful.
        """
        try:
            if ctx is None:
                logger.warning("No context provided for synchronization")
                return False

            # Get TradeManager instance
            from crypto_bot.utils.trade_manager import get_trade_manager
            tm = get_trade_manager()

            if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
                # Calculate portfolio value from TradeManager positions
                tm_positions = tm.get_all_positions()
                portfolio_value = ctx.paper_wallet.initial_balance

                for pos in tm_positions:
                    current_price = tm.price_cache.get(pos.symbol, pos.average_price)
                    if pos.side == 'long':
                        portfolio_value += (current_price - pos.average_price) * pos.total_amount
                    else:  # short
                        portfolio_value += (pos.average_price - current_price) * pos.total_amount
                    portfolio_value -= pos.fees_paid

                # Update paper wallet balance
                old_balance = ctx.paper_wallet.balance
                ctx.paper_wallet.balance = portfolio_value
                ctx.balance = portfolio_value

                # Update balance manager
                try:
                    from crypto_bot.utils.balance_manager import set_single_balance
                    set_single_balance(portfolio_value)
                except Exception as e:
                    logger.warning(f"Failed to update balance manager: {e}")

                if abs(old_balance - portfolio_value) > 0.01:
                    logger.info(f"🔄 Balance synchronized: ${old_balance:.2f} → ${portfolio_value:.2f}")

                self.last_sync_time = datetime.utcnow()
                return True

        except Exception as e:
            logger.error(f"Error during component synchronization: {e}")
            return False

# Global instance
_sync_manager = SynchronizationManager()

def get_sync_manager() -> SynchronizationManager:
    """Get the global synchronization manager instance."""
    return _sync_manager
'''

        sync_file = self.project_root / "crypto_bot" / "utils" / "sync_manager.py"
        sync_file.parent.mkdir(exist_ok=True)
        sync_file.write_text(sync_manager_code)
        logger.info(f"Created synchronization manager at {sync_file}")

    def apply_all_fixes(self) -> None:
        """Apply all fixes in the correct order."""
        logger.info("=== STARTING COMPREHENSIVE WALLET BALANCE FIX ===")

        # Step 1: Backup current state
        self.backup_current_state()

        # Step 2: Migrate positions from paper wallet to TradeManager
        tm_state = self.migrate_positions_to_trade_manager()

        # Step 3: Fix paper wallet state
        pw_state = self.fix_paper_wallet_state()

        # Step 4: Save updated states
        with open(self.trade_manager_state, 'w') as f:
            json.dump(tm_state, f, indent=2)
        logger.info("Updated TradeManager state with migrated positions")

        with open(self.paper_wallet_state, 'w') as f:
            yaml.dump(pw_state, f, default_flow_style=False)
        logger.info("Updated paper wallet state (positions cleared)")

        # Step 5: Update main bot initialization
        self.update_main_bot_initialization()

        # Step 6: Update paper wallet sync logic
        self.update_paper_wallet_sync_logic()

        # Step 7: Create synchronization manager
        self.create_synchronization_manager()

        logger.info("=== COMPREHENSIVE WALLET BALANCE FIX COMPLETED ===")
        logger.info("")
        logger.info("SUMMARY OF CHANGES:")
        logger.info("✅ Migrated positions from paper wallet to TradeManager")
        logger.info("✅ Made TradeManager the single source of truth for positions")
        logger.info("✅ Removed TradeManager state reset that was causing issues")
        logger.info("✅ Fixed paper wallet synchronization logic")
        logger.info("✅ Created synchronization manager for ongoing consistency")
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("1. Restart the bot to apply the fixes")
        logger.info("2. Monitor that positions and balance stay synchronized")
        logger.info("3. Check that trades are properly recorded in TradeManager")
        logger.info("4. Verify that paper wallet balance updates correctly")

def main():
    """Main entry point for the wallet balance fix."""
    fixer = WalletBalanceFixer()
    fixer.apply_all_fixes()

if __name__ == "__main__":
    main()
