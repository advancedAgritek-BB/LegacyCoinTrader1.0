#!/usr/bin/env python3
"""
Final fix to resolve critical pipeline issues:
1. Filter out unsupported Solana tokens
2. Focus on supported Kraken pairs only
3. Ensure proper symbol validation
"""

import yaml
from pathlib import Path

def apply_final_fix():
    """Apply the final fix to resolve critical pipeline issues."""
    
    print("🔧 Applying final fix for critical pipeline issues...")
    
    # Load current config
    config_path = Path("crypto_bot/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Set a minimal list of supported Kraken symbols
    supported_symbols = [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
        "LINK/USD", "UNI/USD", "AAVE/USD", "AVAX/USD",
        "BTC/EUR", "ETH/EUR", "SOL/EUR", "ADA/EUR"
    ]
    
    config['symbols'] = supported_symbols
    print(f"✅ Set supported symbols to {len(supported_symbols)} known working pairs")
    
    # 2. Disable Solana scanning temporarily
    if 'solana_scanner' in config:
        config['solana_scanner']['enabled'] = False
        print("✅ Disabled Solana scanner to prevent unsupported token issues")
    
    # 3. Set conservative batch size
    config['symbol_batch_size'] = 10
    print("✅ Reduced batch size to 10 for stability")
    
    # 4. Add symbol validation
    symbol_validation = {
        'enabled': True,
        'require_kraken_support': True,
        'max_symbol_length': 20,
        'allowed_quotes': ['USD', 'EUR'],
        'skip_unknown_addresses': True
    }
    config['symbol_validation'] = symbol_validation
    print("✅ Added strict symbol validation")
    
    # 5. Ensure proper execution mode
    config['execution_mode'] = 'dry_run'
    config['mode'] = 'cex'
    config['testing_mode'] = True
    print("✅ Set to dry run testing mode")
    
    # 6. Add pipeline stability settings
    pipeline_config = {
        'max_concurrent_requests': 5,
        'request_timeout': 30,
        'retry_attempts': 2,
        'continue_on_error': True,
        'fallback_to_simple_mode': True
    }
    config['pipeline_stability'] = pipeline_config
    print("✅ Added pipeline stability configuration")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✅ Configuration updated successfully")
    
    # Create final restart script
    restart_script = """#!/bin/bash
echo "🔄 Final restart with stable configuration..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Start with stable config
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_final_stable.log 2>&1 &

echo "✅ Bot restarted with PID: $!"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_final_stable.log"
echo "🌐 Dashboard: http://localhost:8000"
"""
    
    with open("restart_stable.sh", 'w') as f:
        f.write(restart_script)
    
    import os
    os.chmod("restart_stable.sh", 0o755)
    print("✅ Created stable restart script: restart_stable.sh")
    
    print("\n🎉 Final fix applied!")
    print("📋 Changes made:")
    print("   - Limited to 12 supported Kraken pairs")
    print("   - Disabled Solana scanner")
    print("   - Reduced batch size to 10")
    print("   - Added strict symbol validation")
    print("   - Set to dry run mode")
    print("\n🚀 Run './restart_stable.sh' to restart with stable configuration")

if __name__ == "__main__":
    apply_final_fix()
