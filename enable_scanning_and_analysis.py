#!/usr/bin/env python3
"""
Enable Scanning and Token Analysis

This script re-enables the scanning and token analysis system with proper
safeguards to prevent the issues that caused it to be disabled previously.
"""

import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

def enable_scanning_and_analysis():
    """Re-enable scanning and token analysis with safeguards."""
    
    print("🔍 Re-enabling scanning and token analysis...")
    
    # Load current config
    config_path = Path("crypto_bot/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Re-enable Solana scanner with safeguards
    solana_scanner_config = {
        'enabled': True,
        'max_tokens_per_scan': 20,  # Conservative limit
        'min_volume_usd': 5000,     # Minimum volume threshold
        'scan_interval_minutes': 30, # Reasonable scan interval
        'enable_sentiment': False,   # Disable sentiment initially
        'enable_pyth_prices': True,  # Enable Pyth for reliable prices
        'max_spread_pct': 1.5,       # Tighter spread requirement
        'min_liquidity_score': 0.6,  # Higher liquidity requirement
        'timeout_seconds': 15,       # Shorter timeout
        'max_retries': 2,            # Fewer retries
        'rate_limit_delay': 1.0,     # Rate limiting
        'validate_symbols': True,    # Enable symbol validation
        'skip_unknown_tokens': True, # Skip tokens without proper data
        'fallback_to_cex': True      # Fallback to CEX if DEX fails
    }
    
    config['solana_scanner'] = solana_scanner_config
    print("✅ Re-enabled Solana scanner with conservative settings")
    
    # 2. Add enhanced scanning configuration
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
    
    # 3. Add scan cache configuration
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
    
    # 4. Add symbol validation with safeguards
    symbol_validation_config = {
        'enabled': True,
        'require_kraken_support': False,  # Allow non-Kraken tokens
        'max_symbol_length': 25,
        'allowed_quotes': ['USD', 'EUR', 'USDC', 'USDT'],
        'skip_unknown_addresses': True,
        'strict_mode': False,  # Less strict for scanning
        'validate_volume': True,
        'min_volume_usd': 5000,
        'validate_liquidity': True,
        'min_liquidity_score': 0.6,
        'validate_price': True,
        'max_price_deviation': 0.1,  # 10% max deviation
        'timeout_seconds': 10
    }
    
    config['symbol_validation'] = symbol_validation_config
    print("✅ Added symbol validation with scanning-friendly settings")
    
    # 5. Add pipeline stability for scanning
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
    
    # 6. Add error handling for scanning
    error_handling_config = {
        'continue_on_error': True,
        'exponential_backoff': True,
        'fallback_data_sources': True,
        'log_errors': True,
        'max_backoff': 30.0,
        'max_retries': 2,
        'retry_delay': 2.0,
        'scan_error_threshold': 5,
        'scan_error_cooldown': 300,  # 5 minutes
        'graceful_degradation': True
    }
    
    config['error_handling'] = error_handling_config
    print("✅ Enhanced error handling for scanning")
    
    # 7. Add monitoring for scanning
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
    
    # 8. Update batch size for scanning
    config['symbol_batch_size'] = 8  # Moderate batch size
    print("✅ Set moderate batch size for scanning")
    
    # 9. Add scanning-specific symbols
    scanning_symbols = [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
        "LINK/USD", "UNI/USD", "AAVE/USD", "AVAX/USD",
        "MATIC/USD", "ATOM/USD", "NEAR/USD"
    ]
    
    config['symbols'] = scanning_symbols
    print(f"✅ Set scanning symbols to {len(scanning_symbols)} supported pairs")
    
    # 10. Ensure proper execution mode for scanning
    config['execution_mode'] = 'dry_run'  # Safe mode
    config['mode'] = 'auto'  # Allow both CEX and onchain
    config['testing_mode'] = True
    print("✅ Set to dry run testing mode for safe scanning")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✅ Configuration updated successfully")
    
    # Create enhanced scanning restart script
    restart_script = """#!/bin/bash
echo "🔍 Restarting with enhanced scanning and token analysis..."

# Stop all processes
pkill -f "python.*main.py" || true
pkill -f "python.*crypto_bot" || true

sleep 3

# Clear scan cache if needed
rm -f crypto_bot/logs/scan_cache.json || true

# Start with enhanced scanning
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0
python3 crypto_bot/main.py > crypto_bot/logs/bot_enhanced_scanning.log 2>&1 &

echo "✅ Bot restarted with PID: $!"
echo "📊 Monitor: tail -f crypto_bot/logs/bot_enhanced_scanning.log"
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
    
    with open("restart_enhanced_scanning.sh", 'w') as f:
        f.write(restart_script)
    
    os.chmod("restart_enhanced_scanning.sh", 0o755)
    print("✅ Created enhanced scanning restart script: restart_enhanced_scanning.sh")
    
    # Create test script for scanning
    test_script = """#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_scanning():
    \"\"\"Test the scanning and token analysis system.\"\"\"
    try:
        from crypto_bot.solana.scanner import get_solana_new_tokens
        from crypto_bot.utils.logger import setup_logger, LOG_DIR
        
        logger = setup_logger("test_scanning", LOG_DIR / "test_scanning.log")
        
        # Load config
        import yaml
        with open("crypto_bot/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        scanner_config = config.get("solana_scanner", {})
        
        print("🔍 Testing Solana scanner...")
        logger.info("Starting scanner test")
        
        # Test scanner
        tokens = await get_solana_new_tokens(scanner_config)
        
        print(f"✅ Scanner test successful! Found {len(tokens)} tokens")
        logger.info(f"Scanner test completed: {len(tokens)} tokens found")
        
        if tokens:
            print("📋 Sample tokens:")
            for i, token in enumerate(tokens[:5]):
                print(f"   {i+1}. {token}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scanner test failed: {e}")
        logger.error(f"Scanner test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_scanning())
    sys.exit(0 if success else 1)
"""
    
    with open("test_scanning.py", 'w') as f:
        f.write(test_script)
    
    os.chmod("test_scanning.py", 0o755)
    print("✅ Created scanning test script: test_scanning.py")
    
    print("\n🎉 Scanning and token analysis re-enabled!")
    print("📋 Changes made:")
    print("   - Re-enabled Solana scanner with safeguards")
    print("   - Added enhanced scanning configuration")
    print("   - Added scan cache management")
    print("   - Added symbol validation for scanning")
    print("   - Enhanced error handling and monitoring")
    print("   - Set conservative limits and timeouts")
    print("   - Added fallback mechanisms")
    print("\n🚀 Next steps:")
    print("   1. Run './restart_enhanced_scanning.sh' to restart with scanning")
    print("   2. Run './test_scanning.py' to test the scanner")
    print("   3. Monitor logs for scan results and token discoveries")
    print("   4. Check the dashboard for scanning metrics")

if __name__ == "__main__":
    enable_scanning_and_analysis()
