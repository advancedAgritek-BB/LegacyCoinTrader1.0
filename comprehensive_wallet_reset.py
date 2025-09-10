#!/usr/bin/env python3
"""
Comprehensive fix to completely reset all wallet state and ensure clean startup
"""

import sys
import os
import yaml
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def comprehensive_wallet_reset():
    """Completely reset all wallet state files to ensure clean startup."""
    print("🔄 Performing comprehensive wallet state reset...")
    
    # Files to reset
    files_to_reset = [
        "crypto_bot/logs/paper_wallet_state.yaml",
        "crypto_bot/logs/trade_manager_state.json",
        "crypto_bot/logs/paper_wallet.yaml",
        "crypto_bot/paper_wallet_config.yaml",
        "crypto_bot/user_config.yaml"
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
        "last_save_time": "2025-09-03T07:30:00.000000"
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
                elif 'paper_wallet_config' in file_path:
                    with open(path, 'w') as f:
                        yaml.dump(clean_paper_wallet_config, f, default_flow_style=False)
                elif 'paper_wallet.yaml' in file_path:
                    with open(path, 'w') as f:
                        yaml.dump(clean_paper_wallet_config, f, default_flow_style=False)
                else:
                    with open(path, 'w') as f:
                        yaml.dump(clean_paper_wallet_state, f, default_flow_style=False)
            elif file_path.endswith('.json'):
                with open(path, 'w') as f:
                    json.dump(clean_trade_manager_state, f, indent=2)
            
            print(f"✅ Reset {file_path}")
            
        except Exception as e:
            print(f"❌ Failed to reset {file_path}: {e}")
    
    # Clear any cached Python files
    try:
        import shutil
        cache_dirs = [
            "crypto_bot/__pycache__",
            "__pycache__"
        ]
        
        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                print(f"✅ Cleared cache: {cache_dir}")
    except Exception as e:
        print(f"⚠️ Could not clear cache: {e}")
    
    print("\n🎯 Comprehensive wallet reset completed!")
    print("✅ All wallet state files reset to clean $10,000 balance")
    print("✅ All positions cleared")
    print("✅ All caches cleared")
    print("\n🔄 Please restart your bot now.")

if __name__ == "__main__":
    comprehensive_wallet_reset()
