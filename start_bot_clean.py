#!/usr/bin/env python3
"""
CLEAN BOT STARTUP - Forces complete clean state from beginning
"""

import asyncio
import sys
import os
import yaml
import json
import shutil
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def force_clean_state():
    """Force clean state by resetting all files and clearing all caches."""
    print("🧹 FORCING COMPLETE CLEAN STATE...")
    
    # Kill any running processes
    try:
        import subprocess
        subprocess.run(['pkill', '-f', 'python.*bot'], capture_output=True)
        print("✅ Killed any running bot processes")
    except Exception as e:
        print(f"⚠️ Could not kill processes: {e}")
    
    # Files to completely reset
    files_to_reset = [
        "crypto_bot/logs/paper_wallet_state.yaml",
        "crypto_bot/logs/trade_manager_state.json",
        "crypto_bot/logs/paper_wallet.yaml",
        "crypto_bot/paper_wallet_config.yaml",
        "crypto_bot/user_config.yaml",
        "crypto_bot/logs/positions.log"
    ]
    
    # Clean paper wallet state
    clean_paper_wallet_state = {
        'balance': 10000.0,
        'initial_balance': 10000.0,
        'realized_pnl': 0.0,
        'total_trades': 0,
        'winning_trades': 0,
        'positions': {}
    }
    
    # Clean trade manager state
    clean_trade_manager_state = {
        "trades": [],
        "positions": {},
        "price_cache": {},
        "statistics": {
            "total_trades": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
            "total_realized_pnl": 0.0
        },
        "last_save_time": "2025-09-03T08:35:00.000000"
    }
    
    # Clean paper wallet config
    clean_paper_wallet_config = {
        'initial_balance': 10000.0
    }
    
    # Clean user config
    clean_user_config = {
        'coinbase_api_key': '',
        'coinbase_api_secret': '',
        'coinbase_passphrase': '',
        'exchange': 'kraken',
        'mode': 'cex',
        'paper_wallet_balance': 10000.0,
        'telegram_chat_id': '827777274',
        'telegram_token': '8126215032:AAEhQZLiXpssauKf0ktQsq1XqXl94QriCdE',
        'wallet_address': 'EoiVpzLA6b6JBKXTB5WRFor3mPkseM6UisLHt8qK9g1c'
    }
    
    # Reset each file
    for file_path in files_to_reset:
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.endswith('.yaml'):
                if 'user_config' in file_path:
                    with open(path, 'w') as f:
                        yaml.dump(clean_user_config, f, default_flow_style=False)
                elif 'paper_wallet_config' in file_path or 'paper_wallet.yaml' in file_path:
                    with open(path, 'w') as f:
                        yaml.dump(clean_paper_wallet_config, f, default_flow_style=False)
                else:
                    with open(path, 'w') as f:
                        yaml.dump(clean_paper_wallet_state, f, default_flow_style=False)
            elif file_path.endswith('.json'):
                with open(path, 'w') as f:
                    json.dump(clean_trade_manager_state, f, indent=2)
            elif file_path.endswith('.log'):
                # Clear the log file
                with open(path, 'w') as f:
                    f.write("")
            
            print(f"✅ Reset {file_path}")
            
        except Exception as e:
            print(f"❌ Failed to reset {file_path}: {e}")
    
    # Clear ALL cache directories
    try:
        cache_dirs = [
            "crypto_bot/__pycache__",
            "__pycache__",
            "crypto_bot/logs/__pycache__",
            "tests/__pycache__",
            "crypto_bot/utils/__pycache__",
            "crypto_bot/solana/__pycache__"
        ]
        
        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                print(f"✅ Cleared cache: {cache_dir}")
    except Exception as e:
        print(f"⚠️ Could not clear cache: {e}")
    
    # Clear any backup files that might be auto-restored
    try:
        backup_patterns = [
            "*backup*",
            "*migration_backup*",
            "*negative_balance*",
            "*mismatch*"
        ]
        
        for pattern in backup_patterns:
            backup_files = list(Path("crypto_bot/logs").glob(pattern))
            for backup_file in backup_files:
                if backup_file.is_file():
                    backup_file.unlink()
                    print(f"✅ Removed backup file: {backup_file}")
    except Exception as e:
        print(f"⚠️ Could not remove backup files: {e}")
    
    # Clear Python module cache
    try:
        # Clear sys.modules for any cached modules
        modules_to_clear = [name for name in sys.modules.keys() if name.startswith('crypto_bot')]
        for module_name in modules_to_clear:
            del sys.modules[module_name]
        print(f"✅ Cleared {len(modules_to_clear)} cached Python modules")
    except Exception as e:
        print(f"⚠️ Could not clear Python modules: {e}")
    
    print("\n🎯 COMPLETE CLEAN STATE FORCED!")
    print("✅ All wallet state files reset to clean $10,000 balance")
    print("✅ All positions cleared")
    print("✅ All caches cleared")
    print("✅ All backup files removed")
    print("✅ All Python modules cleared")
    print("✅ All running processes killed")

async def main():
    """Start the trading bot with forced clean state"""
    print("🚀 Starting LegacyCoinTrader - CLEAN STATE")
    print("=" * 60)
    print("🤖 Trading Bot with FORCED CLEAN STATE")
    print("=" * 60)
    
    # Force clean state first
    force_clean_state()
    
    # Set environment variables
    os.environ['AUTO_START_TRADING'] = '1'
    os.environ['NON_INTERACTIVE'] = '1'
    
    try:
        # Import and run the main bot function
        from crypto_bot.main import _main_impl
        
        print("🎯 Starting trading bot with clean state...")
        print("-" * 60)
        
        # Run the main bot function
        notifier = await _main_impl()
        
        print("✅ Bot completed successfully")
        
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    except Exception as e:
        print(f"❌ Bot error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
