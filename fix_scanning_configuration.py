#!/usr/bin/env python3
"""
Fix Scanning Configuration

This script fixes the scanning configuration to match the Pydantic schema
requirements and ensure the bot starts properly.
"""

import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

def fix_scanning_configuration():
    """Fix the scanning configuration to match schema requirements."""
    
    print("🔧 Fixing scanning configuration...")
    
    # Load current config
    config_path = Path("crypto_bot/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Fix Solana scanner config to match schema
    solana_scanner_config = {
        'enabled': True,
        'interval_minutes': 30,  # Use the correct field name
        'min_volume_usd': 5000,
        'max_tokens_per_scan': 20,
        'gecko_search': True,
        'api_keys': {
            'moralis': os.getenv("MORALIS_KEY", "YOUR_KEY"),
            'bitquery': os.getenv("BITQUERY_KEY", "YOUR_KEY")
        }
    }
    
    config['solana_scanner'] = solana_scanner_config
    print("✅ Fixed Solana scanner configuration to match schema")
    
    # Add enhanced scanning configuration (separate from solana_scanner)
    enhanced_scanning_config = {
        'enabled': True,
        'scan_interval': 30,
        'max_tokens_per_scan': 20,
        'min_score_threshold': 0.4,
        'enable_sentiment': False,
        'enable_pyth_prices': True,
        'min_volume_usd': 5000,
        'max_spread_pct': 1.5,
        'min_liquidity_score': 0.6,
        'min_strategy_fit': 0.6,
        'min_confidence': 0.5,
        'discovery_sources': ['basic_scanner', 'dex_aggregators'],
        'data_sources': {
            'price': ['pyth', 'jupiter'],
            'volume': ['birdeye'],
            'orderbook': ['jupiter']
        }
    }
    
    config['enhanced_scanning'] = enhanced_scanning_config
    print("✅ Added enhanced scanning configuration")
    
    # Add scan cache configuration
    scan_cache_config = {
        'max_cache_size': 500,
        'review_interval_minutes': 20,
        'max_age_hours': 12,
        'min_score_threshold': 0.4,
        'persist_to_disk': True,
        'auto_cleanup': True
    }
    
    config['scan_cache'] = scan_cache_config
    print("✅ Added scan cache configuration")
    
    # Add symbol validation with safeguards
    symbol_validation_config = {
        'enabled': True,
        'require_kraken_support': False,
        'max_symbol_length': 25,
        'allowed_quotes': ['USD', 'EUR', 'USDC', 'USDT'],
        'skip_unknown_addresses': True,
        'strict_mode': False,
        'validate_volume': True,
        'min_volume_usd': 5000,
        'validate_liquidity': True,
        'min_liquidity_score': 0.6,
        'validate_price': True,
        'max_price_deviation': 0.1,
        'timeout_seconds': 10
    }
    
    config['symbol_validation'] = symbol_validation_config
    print("✅ Added symbol validation configuration")
    
    # Add pipeline stability for scanning
    pipeline_stability_config = {
        'max_concurrent_requests': 5,
        'request_timeout': 20,
        'retry_attempts': 2,
        'continue_on_error': True,
        'fallback_to_simple_mode': True,
        'disable_complex_filters': False,
        'enable_rate_limiting': True,
        'rate_limit_delay': 1.0,
        'max_failures_per_cycle': 5,
        'circuit_breaker_enabled': True,
        'circuit_breaker_threshold': 3
    }
    
    config['pipeline_stability'] = pipeline_stability_config
    print("✅ Added pipeline stability configuration")
    
    # Add error handling for scanning
    error_handling_config = {
        'continue_on_error': True,
        'exponential_backoff': True,
        'fallback_data_sources': True,
        'log_errors': True,
        'max_backoff': 30.0,
        'max_retries': 2,
        'retry_delay': 2.0,
        'scan_error_threshold': 5,
        'scan_error_cooldown': 300,
        'graceful_degradation': True
    }
    
    config['error_handling'] = error_handling_config
    print("✅ Enhanced error handling for scanning")
    
    # Add monitoring for scanning
    monitoring_config = {
        'enable_scan_metrics': True,
        'log_scan_results': True,
        'track_scan_performance': True,
        'alert_on_scan_failures': True,
        'scan_metrics_interval': 60,
        'max_scan_logs': 1000
    }
    
    config['scan_monitoring'] = monitoring_config
    print("✅ Added scanning monitoring configuration")
    
    # Update batch size for scanning
    config['symbol_batch_size'] = 8
    print("✅ Set moderate batch size for scanning")
    
    # Add scanning-specific symbols
    scanning_symbols = [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
        "LINK/USD", "UNI/USD", "AAVE/USD", "AVAX/USD",
        "MATIC/USD", "ATOM/USD", "NEAR/USD"
    ]
    
    config['symbols'] = scanning_symbols
    print(f"✅ Set scanning symbols to {len(scanning_symbols)} supported pairs")
    
    # Ensure proper execution mode for scanning
    config['execution_mode'] = 'dry_run'
    config['mode'] = 'auto'
    config['testing_mode'] = True
    print("✅ Set to dry run testing mode for safe scanning")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✅ Configuration updated successfully")
    
    # Create fixed restart script
    restart_script = """#!/bin/bash
echo "🔧 Restarting with fixed scanning configuration..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Clear scan cache if needed
rm -f crypto_bot/logs/scan_cache.json || true

# Start with fixed config
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_fixed_scanning.log 2>&1 &

echo "✅ Bot restarted with PID: $!"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_fixed_scanning.log"
echo "🌐 Dashboard: http://localhost:8000"

# Wait and check status
sleep 15
echo "🔍 Checking bot status..."
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ Bot is running successfully"
    echo "🔍 Scanning and token analysis should be active"
    echo "📈 Check logs for scan results and token discoveries"
else
    echo "❌ Bot failed to start"
    echo "📋 Check logs for errors"
fi
"""
    
    with open("restart_fixed_scanning.sh", 'w') as f:
        f.write(restart_script)
    
    os.chmod("restart_fixed_scanning.sh", 0o755)
    print("✅ Created fixed scanning restart script: restart_fixed_scanning.sh")
    
    print("\n🎉 Scanning configuration fixed!")
    print("📋 Changes made:")
    print("   - Fixed Solana scanner config to match Pydantic schema")
    print("   - Separated enhanced scanning config from solana_scanner")
    print("   - Maintained all scanning functionality with proper validation")
    print("   - Added comprehensive error handling and monitoring")
    print("\n🚀 Run './restart_fixed_scanning.sh' to restart with fixed configuration")

if __name__ == "__main__":
    fix_scanning_configuration()
