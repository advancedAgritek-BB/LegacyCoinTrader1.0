#!/usr/bin/env python3
"""
Configuration validation script for the crypto bot.
Checks for common configuration issues and provides fixes.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def validate_telegram_config(config: Dict[str, Any]) -> List[str]:
    """Validate Telegram configuration."""
    issues = []
    telegram = config.get('telegram', {})
    
    if not telegram.get('enabled', False):
        return issues
    
    if not telegram.get('token'):
        issues.append("Telegram token is missing")
    
    if not telegram.get('chat_id'):
        issues.append("Telegram chat ID is missing")
    
    # Check for reasonable timeout values
    timeout = telegram.get('timeout_seconds', 30)
    if timeout < 10 or timeout > 60:
        issues.append(f"Telegram timeout_seconds ({timeout}) should be between 10-60")
    
    return issues

def validate_websocket_config(config: Dict[str, Any]) -> List[str]:
    """Validate WebSocket configuration."""
    issues = []
    
    if not config.get('use_websocket', False):
        return issues
    
    ws_timeout = config.get('ws_ohlcv_timeout', 20)
    if ws_timeout < 5 or ws_timeout > 30:
        issues.append(f"ws_ohlcv_timeout ({ws_timeout}) should be between 5-30")
    
    ws_ping = config.get('ws_ping_interval', 8)
    if ws_ping < 3 or ws_ping > 15:
        issues.append(f"ws_ping_interval ({ws_ping}) should be between 3-15")
    
    ws_failures = config.get('ws_failures_before_disable', 5)
    if ws_failures < 2 or ws_failures > 10:
        issues.append(f"ws_failures_before_disable ({ws_failures}) should be between 2-10")
    
    return issues

def validate_ohlcv_config(config: Dict[str, Any]) -> List[str]:
    """Validate OHLCV configuration."""
    issues = []
    
    ohlcv = config.get('ohlcv_quality', {})
    min_ratio = ohlcv.get('min_data_ratio', 0.6)
    if min_ratio < 0.3 or min_ratio > 0.9:
        issues.append(f"min_data_ratio ({min_ratio}) should be between 0.3-0.9")
    
    min_candles = ohlcv.get('min_required_candles', 30)
    if min_candles < 10 or min_candles > 100:
        issues.append(f"min_required_candles ({min_candles}) should be between 10-100")
    
    return issues

def validate_risk_config(config: Dict[str, Any]) -> List[str]:
    """Validate risk management configuration."""
    issues = []
    
    risk = config.get('risk', {})
    risk_pct = risk.get('risk_pct', 0.01)
    if risk_pct < 0.001 or risk_pct > 0.1:
        issues.append(f"risk_pct ({risk_pct}) should be between 0.001-0.1")
    
    max_dd = config.get('max_drawdown', 0.35)
    if max_dd < 0.1 or max_dd > 0.5:
        issues.append(f"max_drawdown ({max_dd}) should be between 0.1-0.5")
    
    return issues

def main():
    """Main validation function."""
    config_path = "crypto_bot/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return
    
    print("🔍 Validating crypto bot configuration...")
    config = load_config(config_path)
    
    if not config:
        print("❌ Failed to load configuration")
        return
    
    all_issues = []
    
    # Run all validations
    all_issues.extend(validate_telegram_config(config))
    all_issues.extend(validate_websocket_config(config))
    all_issues.extend(validate_ohlcv_config(config))
    all_issues.extend(validate_risk_config(config))
    
    if not all_issues:
        print("✅ Configuration validation passed!")
        return
    
    print(f"\n❌ Found {len(all_issues)} configuration issues:")
    for i, issue in enumerate(all_issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 Recommendations:")
    print("  - Check the configuration values above")
    print("  - Ensure Telegram bot is properly configured")
    print("  - Verify WebSocket settings are reasonable")
    print("  - Review risk management parameters")

if __name__ == "__main__":
    main()
