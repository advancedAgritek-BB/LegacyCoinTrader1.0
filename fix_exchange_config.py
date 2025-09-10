#!/usr/bin/env python3
"""
Fix to ensure the bot uses Kraken instead of Coinbase
"""

import yaml
from pathlib import Path

def fix_exchange_config():
    """Fix the exchange configuration to use Kraken instead of Coinbase."""
    
    print("🔧 Fixing exchange configuration to use Kraken...")
    
    # Load current config
    config_path = Path("crypto_bot/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add exchange setting to config
    config['exchange'] = 'kraken'
    print("✅ Added exchange: kraken to config")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✅ Configuration updated successfully")
    
    # Create restart script
    restart_script = """#!/bin/bash
echo "🔄 Restarting bot with Kraken exchange fix..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Start with Kraken exchange
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_kraken_fixed.log 2>&1 &

echo "✅ Bot restarted with Kraken exchange (PID: $!)"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_kraken_fixed.log"

# Wait and check
sleep 20
echo "🔍 Checking bot status..."
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ Bot is running successfully"
    
    # Check for Kraken usage and no Coinbase errors
    if grep -q "coinbase.*401 Unauthorized" crypto_bot/logs/bot_kraken_fixed.log; then
        echo "⚠️ Still seeing Coinbase errors - checking config..."
        echo "📋 Current exchange config:"
        grep -A 5 -B 5 "exchange" crypto_bot/config.yaml
    elif grep -q "kraken" crypto_bot/logs/bot_kraken_fixed.log; then
        echo "✅ SUCCESS: Bot is using Kraken!"
    elif grep -q "actionable signals" crypto_bot/logs/bot_kraken_fixed.log; then
        echo "🎉 SUCCESS: Found actionable signals!"
    elif grep -q "PHASE: analyse_batch completed" crypto_bot/logs/bot_kraken_fixed.log; then
        echo "✅ SUCCESS: Analysis pipeline working!"
    else
        echo "⏳ Bot is running, waiting for signal generation..."
    fi
else
    echo "❌ Bot failed to start"
    echo "📋 Check logs for errors"
fi
"""
    
    with open("restart_kraken_fixed.sh", 'w') as f:
        f.write(restart_script)
    
    import os
    os.chmod("restart_kraken_fixed.sh", 0o755)
    print("✅ Created restart script: restart_kraken_fixed.sh")
    
    print("\n🎉 Exchange configuration fix applied!")
    print("📋 Added exchange: kraken to config.yaml")
    print("\n🚀 Run './restart_kraken_fixed.sh' to restart with Kraken")

if __name__ == "__main__":
    fix_exchange_config()
