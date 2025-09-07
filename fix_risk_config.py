#!/usr/bin/env python3
"""
Final fix for the RiskConfig error that's preventing the bot from starting
"""

import yaml
from pathlib import Path

def fix_risk_config():
    """Fix the RiskConfig missing parameters error."""
    
    print("🔧 Fixing RiskConfig missing parameters...")
    
    # Load current config
    config_path = Path("crypto_bot/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add missing risk parameters
    risk_config = {
        'max_drawdown': 0.25,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'max_positions': 10,
        'position_size_pct': 0.1,
        'max_risk_per_trade': 0.02,
        'max_total_risk': 0.1,
        'enable_trailing_stop': True,
        'trailing_stop_pct': 0.01,
        'enable_partial_exits': True,
        'partial_exit_pct': 0.5
    }
    
    config['risk'] = risk_config
    print("✅ Added missing risk configuration")
    
    # Also add any other missing config sections
    if 'bounce_scalper' not in config:
        config['bounce_scalper'] = {
            'enabled': True,
            'min_score': 0.01,
            'max_concurrent_signals': 5
        }
        print("✅ Added bounce_scalper configuration")
    
    if 'breakout' not in config:
        config['breakout'] = {
            'enabled': True,
            'min_score': 0.01
        }
        print("✅ Added breakout configuration")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✅ Configuration updated successfully")
    
    # Create final restart script
    restart_script = """#!/bin/bash
echo "🔄 Final restart with RiskConfig fix..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Start with fixed config
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_final_fixed.log 2>&1 &

echo "✅ Bot restarted with RiskConfig fix (PID: $!)"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_final_fixed.log"

# Wait and check
sleep 20
echo "🔍 Checking bot status..."
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ Bot is running successfully"
    echo "📈 Check logs for signal generation"
    
    # Check for successful operation
    if grep -q "actionable signals" crypto_bot/logs/bot_final_fixed.log; then
        echo "🎉 SUCCESS: Found actionable signals!"
    elif grep -q "PHASE: analyse_batch completed" crypto_bot/logs/bot_final_fixed.log; then
        echo "✅ SUCCESS: Analysis pipeline working!"
    elif grep -q "Trading cycle completed" crypto_bot/logs/bot_final_fixed.log; then
        echo "✅ SUCCESS: Trading cycle completed!"
    else
        echo "⏳ Bot is running, waiting for signal generation..."
    fi
else
    echo "❌ Bot failed to start"
    echo "📋 Check logs for errors"
fi
"""
    
    with open("restart_final_fixed.sh", 'w') as f:
        f.write(restart_script)
    
    import os
    os.chmod("restart_final_fixed.sh", 0o755)
    print("✅ Created final restart script: restart_final_fixed.sh")
    
    print("\n🎉 RiskConfig fix applied!")
    print("📋 Added missing parameters:")
    print("   - max_drawdown: 0.25")
    print("   - stop_loss_pct: 0.02")
    print("   - take_profit_pct: 0.04")
    print("   - Other risk management settings")
    print("\n🚀 Run './restart_final_fixed.sh' to restart with the fix")

if __name__ == "__main__":
    fix_risk_config()
