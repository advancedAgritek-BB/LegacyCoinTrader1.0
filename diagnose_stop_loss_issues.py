#!/usr/bin/env python3
"""
Diagnostic script for stop loss and trailing stop loss issues.

This script performs a comprehensive analysis of the stop loss system
without requiring complex imports that might fail.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

def check_config_file():
    """Check the configuration file for stop loss settings."""
    config_path = Path("crypto_bot/config.yaml")
    
    if not config_path.exists():
        return {"error": "Configuration file not found"}
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check exit strategy configuration
        exit_cfg = config.get("exit_strategy", {})
        
        return {
            "stop_loss_pct": exit_cfg.get("stop_loss_pct"),
            "take_profit_pct": exit_cfg.get("take_profit_pct"),
            "trailing_stop_pct": exit_cfg.get("trailing_stop_pct"),
            "min_gain_to_trail": exit_cfg.get("min_gain_to_trail"),
            "real_time_monitoring_enabled": exit_cfg.get("real_time_monitoring", {}).get("enabled", False),
            "momentum_aware_exits": exit_cfg.get("momentum_aware_exits", False),
            "momentum_tp_scaling": exit_cfg.get("momentum_tp_scaling", False),
            "momentum_trail_adjustment": exit_cfg.get("momentum_trail_adjustment", False)
        }
    except Exception as e:
        return {"error": f"Failed to parse config: {e}"}

def check_main_file():
    """Check the main.py file for stop loss implementation."""
    main_path = Path("crypto_bot/main.py")
    
    if not main_path.exists():
        return {"error": "Main file not found"}
    
    try:
        with open(main_path, 'r') as f:
            content = f.read()
        
        issues = []
        
        # Check for critical components
        if 'async def handle_exits(ctx: BotContext) -> None:' not in content:
            issues.append("handle_exits function missing")
        
        if 'should_exit(' not in content:
            issues.append("should_exit function not called")
        
        if 'ctx.position_monitor = PositionMonitor(' not in content:
            issues.append("Position monitor not initialized")
        
        if 'handle_exits,' not in content:
            issues.append("handle_exits not included in phase runner")
        
        if 'ctx.position_monitor.start_monitoring(' not in content:
            issues.append("Position monitoring not started")
        
        if 'ctx.position_monitor.stop_monitoring(' not in content:
            issues.append("Position monitoring not stopped")
        
        if 'trailing_stop' not in content:
            issues.append("Trailing stop logic missing")
        
        return {
            "issues_found": issues,
            "handle_exits_present": 'async def handle_exits(ctx: BotContext) -> None:' in content,
            "position_monitor_present": 'ctx.position_monitor = PositionMonitor(' in content,
            "phase_runner_integration": 'handle_exits,' in content,
            "trailing_stop_logic": 'trailing_stop' in content
        }
    except Exception as e:
        return {"error": f"Failed to analyze main file: {e}"}

def check_position_monitor():
    """Check the position monitor implementation."""
    monitor_path = Path("crypto_bot/position_monitor.py")
    
    if not monitor_path.exists():
        return {"error": "Position monitor file not found"}
    
    try:
        with open(monitor_path, 'r') as f:
            content = f.read()
        
        return {
            "class_present": 'class PositionMonitor:' in content,
            "start_monitoring_method": 'async def start_monitoring(' in content,
            "stop_monitoring_method": 'async def stop_monitoring(' in content,
            "check_exit_conditions": 'async def _check_exit_conditions(' in content,
            "update_trailing_stop": 'async def _update_trailing_stop(' in content,
            "real_time_monitoring": 'real_time_monitoring' in content
        }
    except Exception as e:
        return {"error": f"Failed to analyze position monitor: {e}"}

def check_exit_manager():
    """Check the exit manager implementation."""
    exit_path = Path("crypto_bot/risk/exit_manager.py")
    
    if not exit_path.exists():
        return {"error": "Exit manager file not found"}
    
    try:
        with open(exit_path, 'r') as f:
            content = f.read()
        
        return {
            "should_exit_function": 'def should_exit(' in content,
            "calculate_trailing_stop": 'def calculate_trailing_stop(' in content,
            "calculate_atr_trailing_stop": 'def calculate_atr_trailing_stop(' in content,
            "momentum_aware_exits": 'momentum_aware' in content,
            "position_side_handling": 'position_side' in content
        }
    except Exception as e:
        return {"error": f"Failed to analyze exit manager: {e}"}

def check_active_positions():
    """Check for active positions in the logs."""
    positions_log = Path("crypto_bot/logs/positions.log")
    
    if not positions_log.exists():
        return {"error": "Positions log not found"}
    
    try:
        with open(positions_log, 'r') as f:
            lines = f.readlines()
        
        active_positions = []
        for line in lines[-100:]:  # Check last 100 lines
            if "Active" in line and "entry" in line:
                parts = line.split()
                if len(parts) >= 8:
                    active_positions.append({
                        "symbol": parts[1],
                        "side": parts[2],
                        "size": parts[3],
                        "entry_price": parts[5],
                        "current_price": parts[7] if len(parts) > 7 else parts[5],
                        "timestamp": line.split(" - ")[0] if " - " in line else "unknown"
                    })
        
        return {
            "active_positions_count": len(active_positions),
            "active_positions": active_positions[-5:] if active_positions else []  # Last 5 positions
        }
    except Exception as e:
        return {"error": f"Failed to analyze positions log: {e}"}

def check_bot_status():
    """Check if the bot is currently running."""
    pid_file = Path("bot_pid.txt")
    
    if not pid_file.exists():
        return {"bot_running": False, "reason": "No PID file found"}
    
    try:
        with open(pid_file, 'r') as f:
            pid = f.read().strip()
        
        if not pid:
            return {"bot_running": False, "reason": "Empty PID file"}
        
        # Check if process is running
        import psutil
        try:
            process = psutil.Process(int(pid))
            if process.is_running():
                return {
                    "bot_running": True,
                    "pid": pid,
                    "process_name": process.name(),
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024
                }
            else:
                return {"bot_running": False, "reason": "Process not running"}
        except psutil.NoSuchProcess:
            return {"bot_running": False, "reason": "Process not found"}
        except Exception as e:
            return {"bot_running": False, "reason": f"Error checking process: {e}"}
            
    except Exception as e:
        return {"bot_running": False, "reason": f"Error reading PID file: {e}"}

def check_recent_logs():
    """Check recent logs for stop loss related messages."""
    bot_log = Path("crypto_bot/logs/bot.log")
    
    if not bot_log.exists():
        return {"error": "Bot log not found"}
    
    try:
        with open(bot_log, 'r') as f:
            lines = f.readlines()
        
        # Get last 100 lines
        recent_lines = lines[-100:] if len(lines) > 100 else lines
        
        stop_loss_messages = []
        error_messages = []
        exit_messages = []
        
        for line in recent_lines:
            if "stop" in line.lower() or "trail" in line.lower():
                stop_loss_messages.append(line.strip())
            if "error" in line.lower() or "exception" in line.lower():
                error_messages.append(line.strip())
            if "exit" in line.lower():
                exit_messages.append(line.strip())
        
        return {
            "stop_loss_messages": stop_loss_messages[-5:] if stop_loss_messages else [],
            "error_messages": error_messages[-5:] if error_messages else [],
            "exit_messages": exit_messages[-5:] if exit_messages else [],
            "total_lines_checked": len(recent_lines)
        }
    except Exception as e:
        return {"error": f"Failed to analyze bot log: {e}"}

def generate_diagnostic_report():
    """Generate a comprehensive diagnostic report."""
    print("🔍 STOP LOSS SYSTEM DIAGNOSTIC REPORT")
    print("=" * 60)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_analysis": check_config_file(),
        "main_file_analysis": check_main_file(),
        "position_monitor_analysis": check_position_monitor(),
        "exit_manager_analysis": check_exit_manager(),
        "active_positions_analysis": check_active_positions(),
        "bot_status": check_bot_status(),
        "recent_logs_analysis": check_recent_logs()
    }
    
    # Print summary
    print(f"📅 Timestamp: {report['timestamp']}")
    print()
    
    # Bot Status
    bot_status = report['bot_status']
    if bot_status.get('bot_running'):
        print(f"🤖 Bot Status: RUNNING (PID: {bot_status['pid']})")
        print(f"   CPU: {bot_status.get('cpu_percent', 'N/A')}%")
        print(f"   Memory: {bot_status.get('memory_mb', 'N/A'):.1f} MB")
    else:
        print(f"🤖 Bot Status: NOT RUNNING")
        print(f"   Reason: {bot_status.get('reason', 'Unknown')}")
    print()
    
    # Configuration
    config = report['config_analysis']
    if 'error' not in config:
        print("⚙️  Configuration Analysis:")
        print(f"   Stop Loss: {config.get('stop_loss_pct', 'Not set')}")
        print(f"   Take Profit: {config.get('take_profit_pct', 'Not set')}")
        print(f"   Trailing Stop: {config.get('trailing_stop_pct', 'Not set')}")
        print(f"   Min Gain to Trail: {config.get('min_gain_to_trail', 'Not set')}")
        print(f"   Real-time Monitoring: {'✓' if config.get('real_time_monitoring_enabled') else '✗'}")
        print(f"   Momentum Aware Exits: {'✓' if config.get('momentum_aware_exits') else '✗'}")
    else:
        print(f"⚙️  Configuration Error: {config['error']}")
    print()
    
    # Main File Analysis
    main_analysis = report['main_file_analysis']
    if 'error' not in main_analysis:
        print("📄 Main File Analysis:")
        print(f"   Handle Exits Function: {'✓' if main_analysis.get('handle_exits_present') else '✗'}")
        print(f"   Position Monitor: {'✓' if main_analysis.get('position_monitor_present') else '✗'}")
        print(f"   Phase Runner Integration: {'✓' if main_analysis.get('phase_runner_integration') else '✗'}")
        print(f"   Trailing Stop Logic: {'✓' if main_analysis.get('trailing_stop_logic') else '✗'}")
        
        if main_analysis.get('issues_found'):
            print("   Issues Found:")
            for issue in main_analysis['issues_found']:
                print(f"     • {issue}")
    else:
        print(f"📄 Main File Error: {main_analysis['error']}")
    print()
    
    # Active Positions
    positions = report['active_positions_analysis']
    if 'error' not in positions:
        print(f"📊 Active Positions: {positions.get('active_positions_count', 0)}")
        if positions.get('active_positions'):
            print("   Recent Positions:")
            for pos in positions['active_positions']:
                print(f"     • {pos['symbol']}: {pos['side']} {pos['size']} @ ${pos['entry_price']}")
    else:
        print(f"📊 Positions Error: {positions['error']}")
    print()
    
    # Recent Logs
    logs = report['recent_logs_analysis']
    if 'error' not in logs:
        print("📝 Recent Log Analysis:")
        print(f"   Stop Loss Messages: {len(logs.get('stop_loss_messages', []))}")
        print(f"   Error Messages: {len(logs.get('error_messages', []))}")
        print(f"   Exit Messages: {len(logs.get('exit_messages', []))}")
        
        if logs.get('error_messages'):
            print("   Recent Errors:")
            for error in logs['error_messages'][-3:]:
                print(f"     • {error}")
    else:
        print(f"📝 Logs Error: {logs['error']}")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    
    if not bot_status.get('bot_running'):
        print("   • Start the bot to test stop loss functionality")
    
    if 'error' not in config and not config.get('real_time_monitoring_enabled'):
        print("   • Enable real-time monitoring in configuration")
    
    if 'error' not in main_analysis and main_analysis.get('issues_found'):
        print("   • Fix identified issues in main.py")
    
    if 'error' not in positions and positions.get('active_positions_count', 0) > 0:
        print("   • Monitor active positions for stop loss execution")
    
    if 'error' not in logs and logs.get('error_messages'):
        print("   • Address recent errors in the logs")
    
    print("   • Run the comprehensive fix scripts if issues are found")
    print()
    
    # Save report
    with open("stop_loss_diagnostic_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("📋 Full report saved to: stop_loss_diagnostic_report.json")
    print("=" * 60)
    
    return report

if __name__ == "__main__":
    generate_diagnostic_report()
