#!/usr/bin/env python3
"""
Startup validation script for the crypto bot.
Runs configuration validation and fixes common issues before startup.
"""

import sys
import os
import yaml
import logging
import time
from pathlib import Path

# Add the crypto_bot directory to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from crypto_bot.utils.config_validator import validate_config, fix_timeframe_config, log_config_summary
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(current_dir))
    from utils.config_validator import validate_config, fix_timeframe_config, log_config_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return {}

def save_config(config: dict, config_path: str) -> bool:
    """Save configuration to YAML file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved updated configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        return False

def create_backup(config_path: str) -> str:
    """Create a backup of the configuration file."""
    backup_path = f"{config_path}.backup.{int(time.time())}"
    try:
        import shutil
        shutil.copy2(config_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return ""

def validate_and_fix_config(config_path: str, auto_fix: bool = True) -> bool:
    """
    Validate configuration and optionally fix issues.
    
    Args:
        config_path: Path to configuration file
        auto_fix: Whether to automatically fix configuration issues
        
    Returns:
        True if configuration is valid, False otherwise
    """
    logger.info("Starting configuration validation...")
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        return False
    
    # Log current configuration summary
    log_config_summary(config)
    
    # Validate configuration
    errors = validate_config(config)
    
    if errors:
        logger.error("Configuration validation failed with the following errors:")
        for error in errors:
            logger.error(f"  - {error}")
        
        if auto_fix:
            logger.info("Attempting to fix configuration issues automatically...")
            
            # Create backup before making changes
            backup_path = create_backup(config_path)
            
            # Fix timeframe configuration
            exchange_name = config.get('exchange', 'unknown')
            fixed_config = fix_timeframe_config(config, exchange_name)
            
            # Save fixed configuration
            if save_config(fixed_config, config_path):
                logger.info("Configuration has been automatically fixed and saved.")
                
                # Re-validate after fixes
                logger.info("Re-validating configuration after fixes...")
                new_errors = validate_config(fixed_config)
                
                if new_errors:
                    logger.error("Some configuration issues remain after automatic fixes:")
                    for error in new_errors:
                        logger.error(f"  - {error}")
                    return False
                else:
                    logger.info("All configuration issues have been resolved!")
                    return True
            else:
                logger.error("Failed to save fixed configuration.")
                return False
        else:
            logger.error("Configuration validation failed. Set auto_fix=True to attempt automatic fixes.")
            return False
    else:
        logger.info("Configuration validation passed successfully!")
        return True

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate and fix crypto bot configuration')
    parser.add_argument(
        '--config', 
        default='crypto_bot/config.yaml',
        help='Path to configuration file (default: crypto_bot/config.yaml)'
    )
    parser.add_argument(
        '--no-auto-fix',
        action='store_true',
        help='Disable automatic configuration fixes'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Validate and fix configuration
    auto_fix = not args.no_auto_fix
    success = validate_and_fix_config(args.config, auto_fix)
    
    if success:
        logger.info("Configuration validation completed successfully!")
        sys.exit(0)
    else:
        logger.error("Configuration validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
