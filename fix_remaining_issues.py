#!/usr/bin/env python3
"""
Quick fix for remaining evaluation pipeline issues:
1. Disable Coinbase API completely
2. Fix remaining strategy execution issues
3. Adjust signal filtering thresholds
"""

import yaml
import os
from pathlib import Path

def fix_remaining_issues():
    """Fix the remaining issues causing critical pipeline status."""
    
    print("🔧 Fixing remaining evaluation pipeline issues...")
    
    # Load current config
    config_path = Path("crypto_bot/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Completely disable Coinbase API
    if 'coinbase' in config:
        config['coinbase']['enabled'] = False
        print("✅ Disabled Coinbase API")
    
    # 2. Adjust signal filtering to be less restrictive
    if 'min_confidence_score' in config:
        config['min_confidence_score'] = 0.01  # Lower threshold
        print("✅ Lowered confidence threshold to 0.01")
    
    # 3. Add signal execution configuration
    signal_config = {
        'min_signal_strength': 0.01,
        'max_signals_per_cycle': 10,
        'signal_timeout_seconds': 30,
        'execute_all_signals': True,  # For testing
        'dry_run_mode': True  # Ensure we're in dry run
    }
    config['signal_execution'] = signal_config
    print("✅ Added signal execution configuration")
    
    # 4. Fix strategy timeout issues
    if 'strategy_timeout' not in config:
        config['strategy_timeout'] = 60  # Increase timeout
        print("✅ Set strategy timeout to 60 seconds")
    
    # 5. Add better error handling for strategy execution
    error_handling = {
        'continue_on_strategy_error': True,
        'max_strategy_errors_per_cycle': 10,
        'strategy_error_cooldown': 30,
        'fallback_to_simple_strategies': True
    }
    config['strategy_error_handling'] = error_handling
    print("✅ Enhanced strategy error handling")
    
    # 6. Ensure we're in dry run mode
    config['execution_mode'] = 'dry_run'
    config['mode'] = 'cex'
    print("✅ Set execution mode to dry_run")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✅ Configuration updated successfully")
    
    # 7. Create a simple restart script
    restart_script = """#!/bin/bash
echo "🔄 Restarting bot with final fixes..."

# Stop any running processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 2

# Start the bot
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_restart_final.log 2>&1 &

echo "✅ Bot restarted with PID: $!"
echo "📊 Monitor logs: tail -f crypto_bot/logs/bot_restart_final.log"
"""
    
    with open("restart_final.sh", 'w') as f:
        f.write(restart_script)
    
    os.chmod("restart_final.sh", 0o755)
    print("✅ Created restart script: restart_final.sh")
    
    print("\n🎉 Final fixes applied!")
    print("🚀 Run './restart_final.sh' to restart with all fixes")

if __name__ == "__main__":
    fix_remaining_issues()
