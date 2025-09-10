#!/usr/bin/env python3
"""
Comprehensive bot health monitoring system.
Checks bot status, position monitoring, and risk management.
"""

import subprocess
import time
import sys
import os
from pathlib import Path
import json

def check_bot_process():
    """Check if bot process is running."""
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    bot_running = (
        'crypto_bot' in result.stdout or 
        'python.*main' in result.stdout or
        'start_bot' in result.stdout or
        'start_bot_final' in result.stdout or
        'start_bot_auto' in result.stdout or
        'start_bot_clean' in result.stdout
    )
    
    if bot_running:
        # Extract process info
        lines = result.stdout.splitlines()
        for line in lines:
            if any(keyword in line for keyword in ['crypto_bot', 'python.*main', 'start_bot']):
                parts = line.split()
                if len(parts) >= 2:
                    return True, parts[1]  # Return PID
    return False, None

def check_position_monitoring():
    """Check if position monitoring is active."""
    positions_file = Path("crypto_bot/logs/positions.log")
    if not positions_file.exists():
        return False, "No positions log found"
    
    # Check if positions log has recent entries
    try:
        mtime = positions_file.stat().st_mtime
        age_seconds = time.time() - mtime
        
        if age_seconds > 300:  # 5 minutes
            return False, f"Positions log is {age_seconds/60:.1f} minutes old"
        else:
            return True, f"Positions log updated {age_seconds:.0f} seconds ago"
    except Exception as e:
        return False, f"Error checking positions log: {e}"

def check_bot_logs():
    """Check bot logs for errors or issues."""
    log_file = Path("crypto_bot/logs/bot.log")
    if not log_file.exists():
        return False, "No bot log found"
    
    try:
        # Get last 50 lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            last_lines = lines[-50:] if len(lines) > 50 else lines
        
        # Check for errors
        errors = [line for line in last_lines if 'ERROR' in line or 'CRITICAL' in line]
        warnings = [line for line in last_lines if 'WARNING' in line]
        
        if errors:
            return False, f"Found {len(errors)} errors in recent logs"
        elif warnings:
            return True, f"Found {len(warnings)} warnings, no errors"
        else:
            return True, "No errors or warnings in recent logs"
    except Exception as e:
        return False, f"Error reading bot logs: {e}"

def check_configuration():
    """Check if configuration is valid."""
    config_file = Path("crypto_bot/config.yaml")
    if not config_file.exists():
        return False, "No config file found"
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check critical settings
        required_settings = [
            'exits.default_sl_pct',
            'exit_strategy.real_time_monitoring.enabled',
            'exit_strategy.real_time_monitoring.check_interval_seconds'
        ]
        
        for setting in required_settings:
            keys = setting.split('.')
            value = config
            for key in keys:
                if key not in value:
                    return False, f"Missing config: {setting}"
                value = value[key]
        
        return True, "Configuration valid"
    except Exception as e:
        return False, f"Error reading config: {e}"

def get_system_status():
    """Get comprehensive system status."""
    print("🔍 BOT HEALTH CHECK")
    print("=" * 50)
    
    # Check bot process
    bot_running, pid = check_bot_process()
    print(f"Bot Process: {'🟢 Running' if bot_running else '🔴 NOT RUNNING'}")
    if pid:
        print(f"  PID: {pid}")
    
    # Check position monitoring
    monitoring_ok, monitoring_msg = check_position_monitoring()
    print(f"Position Monitoring: {'🟢 Active' if monitoring_ok else '🔴 Inactive'}")
    print(f"  Status: {monitoring_msg}")
    
    # Check bot logs
    logs_ok, logs_msg = check_bot_logs()
    print(f"Bot Logs: {'🟢 OK' if logs_ok else '🔴 Issues'}")
    print(f"  Status: {logs_msg}")
    
    # Check configuration
    config_ok, config_msg = check_configuration()
    print(f"Configuration: {'🟢 Valid' if config_ok else '🔴 Invalid'}")
    print(f"  Status: {config_msg}")
    
    # Overall status
    all_ok = bot_running and monitoring_ok and logs_ok and config_ok
    print(f"\n📊 OVERALL STATUS: {'🟢 HEALTHY' if all_ok else '🔴 UNHEALTHY'}")
    
    return all_ok, {
        'bot_running': bot_running,
        'monitoring_active': monitoring_ok,
        'logs_healthy': logs_ok,
        'config_valid': config_ok
    }

def provide_recommendations(status):
    """Provide recommendations based on system status."""
    print("\n📋 RECOMMENDATIONS:")
    print("-" * 30)
    
    if not status['bot_running']:
        print("🚨 CRITICAL: Bot is not running!")
        print("  Action: Start bot immediately")
        print("  Command: python3 start_bot_auto.py")
        print("  Impact: No stop loss protection")
    
    if not status['monitoring_active']:
        print("⚠️  WARNING: Position monitoring inactive")
        print("  Action: Check bot logs for errors")
        print("  Command: tail -f crypto_bot/logs/bot.log")
        print("  Impact: Outdated position data")
    
    if not status['logs_healthy']:
        print("⚠️  WARNING: Bot logs show issues")
        print("  Action: Review recent errors")
        print("  Command: tail -50 crypto_bot/logs/bot.log")
        print("  Impact: Potential system instability")
    
    if not status['config_valid']:
        print("🚨 CRITICAL: Configuration invalid")
        print("  Action: Fix configuration file")
        print("  File: crypto_bot/config.yaml")
        print("  Impact: Bot may not function correctly")
    
    if all(status.values()):
        print("✅ System is healthy")
        print("  Recommendation: Continue monitoring")
        print("  Command: ./monitor_bot.sh")

def main():
    """Main health check function."""
    healthy, status = get_system_status()
    
    provide_recommendations(status)
    
    print("\n📈 MONITORING COMMANDS:")
    print("-" * 30)
    print("tail -f crypto_bot/logs/bot.log")
    print("tail -f crypto_bot/logs/positions.log")
    print("./monitor_bot.sh")
    print("python3 check_bot_health.py")
    
    print("\n⚠️  IMPORTANT:")
    print("-" * 30)
    print("• Stop losses only work when bot is running")
    print("• Position monitoring requires active bot process")
    print("• All positions are affected by bot status")
    print("• Regular health checks recommended")
    
    return 0 if healthy else 1

if __name__ == "__main__":
    sys.exit(main())
