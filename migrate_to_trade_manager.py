#!/usr/bin/env python3
"""
Migration Script: TradeManager as Single Source of Truth

This script helps migrate your bot from multiple position tracking systems
to using TradeManager as the single source of truth.

Run this script to:
1. Enable TradeManager as the primary position source
2. Sync existing positions from legacy systems to TradeManager
3. Update configuration to use the new system
4. Validate the migration was successful

Usage:
    python migrate_to_trade_manager.py [--dry-run] [--backup]

Options:
    --dry-run    : Show what would be changed without making changes
    --backup     : Create backup of current state before migration
"""

import sys
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import argparse

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import get_trade_manager, create_trade
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.phase_runner import BotContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeManagerMigration:
    """Handles migration to TradeManager as single source of truth."""

    def __init__(self, dry_run: bool = False, create_backup: bool = True):
        self.dry_run = dry_run
        self.create_backup = create_backup
        self.trade_manager = get_trade_manager()
        self.backup_dir = Path("crypto_bot/logs/migration_backup")

        if create_backup and not dry_run:
            self.backup_dir.mkdir(exist_ok=True)

    def do_create_backup(self) -> bool:
        """Create backup of current state."""
        try:
            if self.dry_run:
                logger.info("DRY RUN: Would create backup")
                return True

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"migration_backup_{timestamp}.json"

            # Backup current TradeManager state
            tm_state = {
                'trades': [t.to_dict() for t in self.trade_manager.trades],
                'positions': {k: v.to_dict() for k, v in self.trade_manager.positions.items()},
                'statistics': {
                    'total_trades': self.trade_manager.total_trades,
                    'total_volume': float(self.trade_manager.total_volume),
                    'total_fees': float(self.trade_manager.total_fees),
                    'total_realized_pnl': float(self.trade_manager.total_realized_pnl)
                }
            }

            with open(backup_file, 'w') as f:
                json.dump(tm_state, f, indent=2, default=str)

            # Backup paper wallet if exists
            paper_wallet_file = Path("crypto_bot/logs/paper_wallet_state.yaml")
            if paper_wallet_file.exists():
                backup_wallet = self.backup_dir / f"paper_wallet_backup_{timestamp}.yaml"
                with open(paper_wallet_file, 'r') as src, open(backup_wallet, 'w') as dst:
                    dst.write(src.read())

            logger.info(f"Backup created: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def migrate_legacy_positions(self, ctx: BotContext) -> bool:
        """Migrate positions from legacy systems to TradeManager."""
        try:
            migrated_count = 0

            # Migrate from ctx.positions
            for symbol, pos_data in ctx.positions.items():
                if self.dry_run:
                    logger.info(f"DRY RUN: Would migrate {symbol} from ctx.positions")
                    continue

                # Check if position already exists in TradeManager
                existing_pos = self.trade_manager.get_position(symbol)
                if existing_pos:
                    logger.info(f"Position {symbol} already exists in TradeManager, skipping")
                    continue

                # Create trade to represent the position
                side = 'buy' if pos_data.get('side') == 'buy' else 'sell'
                amount = abs(pos_data.get('size', pos_data.get('amount', 0)))
                price = pos_data.get('entry_price', 0)

                if amount > 0 and price > 0:
                    trade = create_trade(
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=price,
                        strategy='migrated_from_legacy'
                    )

                    self.trade_manager.record_trade(trade)
                    migrated_count += 1
                    logger.info(f"Migrated {symbol}: {side} {amount} @ {price}")

            # Migrate from paper wallet positions
            if ctx.paper_wallet:
                for symbol, pos_data in ctx.paper_wallet.positions.items():
                    if self.dry_run:
                        logger.info(f"DRY RUN: Would migrate {symbol} from paper wallet")
                        continue

                    # Check if position already exists in TradeManager
                    existing_pos = self.trade_manager.get_position(symbol)
                    if existing_pos:
                        continue

                    # Create trade to represent the position
                    side = pos_data.get('side', 'buy')
                    amount = pos_data.get('amount', 0)
                    price = pos_data.get('entry_price', 0)

                    if amount > 0 and price > 0:
                        trade = create_trade(
                            symbol=symbol,
                            side=side,
                            amount=amount,
                            price=price,
                            strategy='migrated_from_paper_wallet'
                        )

                        self.trade_manager.record_trade(trade)
                        migrated_count += 1
                        logger.info(f"Migrated {symbol}: {side} {amount} @ {price}")

            if not self.dry_run:
                self.trade_manager.save_state()

            logger.info(f"Migration complete: {migrated_count} positions migrated")
            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    def update_config_for_trade_manager(self, config_path: str) -> bool:
        """Update bot configuration to enable TradeManager as source of truth."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}")
                return True

            if self.dry_run:
                logger.info(f"DRY RUN: Would update config at {config_path}")
                return True

            # Read current config
            with open(config_file, 'r') as f:
                if config_file.suffix == '.yaml':
                    config = yaml.safe_load(f) or {}
                else:
                    config = json.load(f)

            # Add TradeManager migration settings
            if 'trade_manager' not in config:
                config['trade_manager'] = {}

            config['trade_manager']['enabled'] = True
            config['trade_manager']['single_source_of_truth'] = True
            config['trade_manager']['migration_complete'] = True
            config['trade_manager']['migration_date'] = datetime.now().isoformat()

            # Write updated config
            with open(config_file, 'w') as f:
                if config_file.suffix == '.yaml':
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config, f, indent=2)

            logger.info(f"Updated config: {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False

    def enable_trade_manager_in_context(self, ctx: BotContext) -> bool:
        """Enable TradeManager as source of truth in BotContext."""
        try:
            if self.dry_run:
                logger.info("DRY RUN: Would enable TradeManager as source of truth")
                return True

            # Enable TradeManager as source of truth
            ctx.use_trade_manager_as_source = True

            # Sync positions from TradeManager to legacy systems for backward compatibility
            if hasattr(ctx, 'sync_positions_from_trade_manager'):
                ctx.sync_positions_from_trade_manager()

            logger.info("TradeManager enabled as single source of truth")
            return True

        except Exception as e:
            logger.error(f"Failed to enable TradeManager: {e}")
            return False

    def validate_migration(self, ctx: BotContext) -> bool:
        """Validate that migration was successful."""
        try:
            # Check that TradeManager has positions
            tm_positions = self.trade_manager.get_all_positions()
            logger.info(f"TradeManager positions: {len(tm_positions)}")

            # Check position consistency
            if hasattr(ctx, 'validate_position_consistency'):
                is_consistent = ctx.validate_position_consistency()
                if not is_consistent:
                    logger.warning("Position systems are not consistent after migration")
                    return False

            # Check that we have PositionSyncManager
            if not hasattr(ctx, 'position_sync_manager') or not ctx.position_sync_manager:
                logger.warning("PositionSyncManager not initialized")
                return False

            logger.info("Migration validation passed")
            return True

        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False

    def run_migration(self, config_path: str = "crypto_bot/config.yaml") -> bool:
        """Run the complete migration process."""
        logger.info("Starting TradeManager migration...")

        if self.dry_run:
            logger.info("RUNNING IN DRY RUN MODE - No changes will be made")

        # Step 1: Create backup
        if self.create_backup and not self.do_create_backup():
            logger.error("Backup creation failed, aborting migration")
            return False

        # Step 2: Create BotContext for migration
        try:
            # Load config
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    if config_file.suffix == '.yaml':
                        config = yaml.safe_load(f) or {}
                    else:
                        config = json.load(f)
            else:
                config = {}

            # Create context
            ctx = BotContext(
                positions={},
                df_cache={},
                regime_cache={},
                config=config,
                trade_manager=self.trade_manager
            )

            # Initialize paper wallet if config indicates dry run
            if config.get('execution_mode') == 'dry_run':
                ctx.paper_wallet = PaperWallet(balance=10000.0)

        except Exception as e:
            logger.error(f"Failed to create context: {e}")
            return False

        # Step 3: Migrate legacy positions to TradeManager
        if not self.migrate_legacy_positions(ctx):
            logger.error("Position migration failed")
            return False

        # Step 4: Enable TradeManager as source of truth
        if not self.enable_trade_manager_in_context(ctx):
            logger.error("Failed to enable TradeManager as source")
            return False

        # Step 5: Update configuration
        if not self.update_config_for_trade_manager(config_path):
            logger.error("Config update failed")
            return False

        # Step 6: Validate migration
        if not self.validate_migration(ctx):
            logger.error("Migration validation failed")
            return False

        logger.info("✅ TradeManager migration completed successfully!")
        logger.info("Your bot now uses TradeManager as the single source of truth for positions.")
        logger.info("Legacy position tracking systems will remain synchronized for backward compatibility.")

        return True


def main():
    parser = argparse.ArgumentParser(description="Migrate to TradeManager as single source of truth")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--backup', action='store_true', default=True, help='Create backup before migration')
    parser.add_argument('--config', default='crypto_bot/config.yaml', help='Path to config file')

    args = parser.parse_args()

    migration = TradeManagerMigration(dry_run=args.dry_run, create_backup=args.backup)

    success = migration.run_migration(args.config)

    if success:
        print("\n🎉 Migration completed successfully!")
        print("Your bot now uses TradeManager as the single source of truth.")
        print("Run your bot to test that everything works correctly.")
    else:
        print("\n❌ Migration failed!")
        print("Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
