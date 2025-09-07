#!/usr/bin/env python3
"""
Enable TradeManager as Single Source of Truth

This script updates your bot configuration to enable TradeManager as the
single source of truth for position tracking.

Usage:
    python enable_trade_manager_sot.py [--config CONFIG_FILE]
"""

import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_config_for_trade_manager_sot(config_path: str) -> bool:
    """Update configuration to enable TradeManager as single source of truth."""
    try:
        config_file = Path(config_path)

        if not config_file.exists():
            logger.error(f"Config file not found: {config_path}")
            return False

        # Read current config
        with open(config_file, 'r') as f:
            if config_file.suffix == '.yaml':
                config = yaml.safe_load(f) or {}
            else:
                config = json.load(f)

        # Enable TradeManager as single source of truth
        if 'trade_manager' not in config:
            config['trade_manager'] = {}

        config['trade_manager'].update({
            'enabled': True,
            'single_source_of_truth': True,
            'migration_complete': True,
            'migration_date': datetime.now().isoformat(),
            'sync_legacy_systems': True,  # Keep legacy systems in sync
            'validate_consistency': True  # Enable consistency validation
        })

        # Add position monitoring configuration
        if 'position_monitor' not in config:
            config['position_monitor'] = {}

        config['position_monitor'].update({
            'use_trade_manager': True,
            'sync_with_legacy': True
        })

        # Add paper wallet synchronization settings
        if 'paper_wallet' not in config:
            config['paper_wallet'] = {}

        config['paper_wallet'].update({
            'sync_with_trade_manager': True,
            'auto_sync_on_trade': True
        })

        # Write updated config
        with open(config_file, 'w') as f:
            if config_file.suffix == '.yaml':
                yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
            else:
                json.dump(config, f, indent=2, sort_keys=False)

        logger.info(f"✅ Updated config: {config_path}")
        logger.info("TradeManager is now enabled as single source of truth")

        return True

    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return False


def validate_config(config_path: str) -> bool:
    """Validate that the configuration is properly set up."""
    try:
        config_file = Path(config_path)

        if not config_file.exists():
            logger.error(f"Config file not found: {config_path}")
            return False

        # Read config
        with open(config_file, 'r') as f:
            if config_file.suffix == '.yaml':
                config = yaml.safe_load(f) or {}
            else:
                config = json.load(f)

        # Check TradeManager settings
        tm_config = config.get('trade_manager', {})
        if not tm_config.get('enabled', False):
            logger.warning("TradeManager is not enabled")
            return False

        if not tm_config.get('single_source_of_truth', False):
            logger.warning("TradeManager is not set as single source of truth")
            return False

        logger.info("✅ Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Enable TradeManager as single source of truth")
    parser.add_argument('--config', default='crypto_bot/config.yaml', help='Path to config file')
    parser.add_argument('--validate-only', action='store_true', help='Only validate current config')

    args = parser.parse_args()

    if args.validate_only:
        success = validate_config(args.config)
    else:
        success = update_config_for_trade_manager_sot(args.config)
        if success:
            # Also validate the updated config
            success = validate_config(args.config)

    if success:
        print("\n🎉 Configuration updated successfully!")
        print("TradeManager is now configured as the single source of truth.")
        print("Run your bot to start using the new system.")
    else:
        print("\n❌ Configuration update failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
