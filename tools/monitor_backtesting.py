#!/usr/bin/env python3
"""
Backtesting Monitor - Verify that backtesting is active and producing results

This script monitors the backtesting system to ensure it's working correctly.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yaml

def load_config():
    """Load the main bot configuration."""
    config_path = Path("crypto_bot/config.yaml")
    if not config_path.exists():
        print("❌ Config file not found: crypto_bot/config.yaml")
        return None
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def check_optimization_config(config):
    """Check if optimization is properly configured."""
    print("🔍 Checking optimization configuration...")
    
    opt_config = config.get("optimization", {})
    if not opt_config.get("enabled"):
        print("❌ Optimization is disabled in config")
        return False
    
    print("✅ Optimization is enabled")
    
    param_ranges = opt_config.get("parameter_ranges", {})
    if not param_ranges:
        print("❌ No parameter ranges defined")
        return False
    
    print(f"✅ Found parameter ranges for {len(param_ranges)} strategies:")
    for strategy, ranges in param_ranges.items():
        sl_count = len(ranges.get("stop_loss", []))
        tp_count = len(ranges.get("take_profit", []))
        total_combinations = sl_count * tp_count
        print(f"   {strategy}: {total_combinations} parameter combinations")
    
    interval_days = opt_config.get("interval_days", 0)
    print(f"✅ Optimization interval: {interval_days} days")
    
    return True

def check_optimization_results():
    """Check if optimization has produced results."""
    print("\n🔍 Checking optimization results...")
    
    results_file = Path("crypto_bot/logs/optimized_params.json")
    if not results_file.exists():
        print("❌ No optimization results file found")
        return False
    
    try:
        with open(results_file) as f:
            results = json.load(f)
        
        if not results:
            print("❌ Optimization results file is empty")
            return False
        
        print(f"✅ Found optimization results for {len(results)} strategies:")
        for strategy, params in results.items():
            print(f"   {strategy}: SL={params.get('stop_loss_pct', 'N/A'):.3f}, "
                  f"TP={params.get('take_profit_pct', 'N/A'):.3f}, "
                  f"Sharpe={params.get('sharpe', 'N/A'):.2f}")
        
        # Check file modification time
        mod_time = datetime.fromtimestamp(results_file.stat().st_mtime)
        age = datetime.now() - mod_time
        print(f"   Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} ({age.days} days ago)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading optimization results: {e}")
        return False

def check_backtest_logs():
    """Check backtesting log files."""
    print("\n🔍 Checking backtesting logs...")
    
    log_files = [
        "crypto_bot/logs/optimizer.log",
        "crypto_bot/logs/bot.log"
    ]
    
    found_logs = False
    for log_file in log_files:
        log_path = Path(log_file)
        if log_path.exists():
            print(f"✅ Found log file: {log_file}")
            found_logs = True
            
            # Check for recent backtest activity
            try:
                with open(log_path) as f:
                    lines = f.readlines()
                    recent_lines = lines[-100:]  # Last 100 lines
                    
                backtest_activity = [line for line in recent_lines if 
                                   any(keyword in line.lower() for keyword in 
                                       ['backtest', 'optimization', 'sharpe', 'grid'])]
                
                if backtest_activity:
                    print(f"   📊 Recent backtest activity found ({len(backtest_activity)} entries)")
                    print(f"   Latest: {backtest_activity[-1].strip()}")
                else:
                    print(f"   ⚠️  No recent backtest activity in {log_file}")
                    
            except Exception as e:
                print(f"   ❌ Error reading {log_file}: {e}")
        else:
            print(f"❌ Log file not found: {log_file}")
    
    return found_logs

def check_backtest_cache():
    """Check for cached backtest results."""
    print("\n🔍 Checking backtest cache...")
    
    cache_dirs = [
        "crypto_bot/logs/backtest_results",
        "crypto_bot/logs"
    ]
    
    found_results = False
    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            csv_files = list(cache_path.glob("*.csv"))
            json_files = list(cache_path.glob("*backtest*.json"))
            
            if csv_files or json_files:
                print(f"✅ Found {len(csv_files)} CSV and {len(json_files)} JSON result files in {cache_dir}")
                found_results = True
                
                # Show most recent files
                all_files = csv_files + json_files
                if all_files:
                    latest_file = max(all_files, key=lambda x: x.stat().st_mtime)
                    mod_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
                    print(f"   Latest: {latest_file.name} ({mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    if not found_results:
        print("❌ No backtest result files found")
    
    return found_results

def check_strategy_weights():
    """Check if strategy weights are being updated."""
    print("\n🔍 Checking strategy weight updates...")
    
    config = load_config()
    if not config:
        return False
    
    strategy_allocation = config.get("strategy_allocation", {})
    if not strategy_allocation:
        print("❌ No strategy allocation found in config")
        return False
    
    print(f"✅ Found strategy allocation for {len(strategy_allocation)} strategies:")
    for strategy, weight in strategy_allocation.items():
        print(f"   {strategy}: {weight:.1%}")
    
    return True

def run_quick_backtest():
    """Run a quick backtest to verify functionality."""
    print("\n🧪 Running quick backtest verification...")
    
    try:
        import subprocess
        result = subprocess.run([
            "python3", "-c", 
            """
from crypto_bot.auto_optimizer import optimize_strategies
import sys
try:
    results = optimize_strategies()
    if results:
        print('✅ Quick optimization test successful')
        print(f'Tested strategies: {list(results.keys())}')
    else:
        print('⚠️ Optimization returned empty results')
        print('This might be normal if no data is available')
except Exception as e:
    print(f'❌ Quick optimization test failed: {e}')
    sys.exit(1)
"""
        ], capture_output=True, text=True, timeout=60)
        
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⚠️ Quick backtest test timed out (this may be normal)")
        return False
    except Exception as e:
        print(f"❌ Could not run quick backtest test: {e}")
        return False

def main():
    """Main monitoring function."""
    print("🤖 Backtesting Monitor")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    if not config:
        return
    
    # Run all checks
    checks = [
        ("Optimization Configuration", lambda: check_optimization_config(config)),
        ("Optimization Results", check_optimization_results),
        ("Backtesting Logs", check_backtest_logs),
        ("Backtest Cache", check_backtest_cache),
        ("Strategy Weights", check_strategy_weights),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results[check_name] = False
    
    # Run quick test if requested
    if input("\n🧪 Run quick backtest verification? (y/N): ").lower().startswith('y'):
        results["Quick Backtest Test"] = run_quick_backtest()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 BACKTESTING STATUS SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, passed_check in results.items():
        status = "✅ PASS" if passed_check else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Backtesting appears to be working correctly.")
    elif passed >= total * 0.7:
        print("⚠️ Most checks passed. Some minor issues may need attention.")
    else:
        print("❌ Several issues detected. Backtesting may not be working properly.")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not results.get("Optimization Configuration", False):
        print("- Fix optimization configuration in crypto_bot/config.yaml")
    if not results.get("Optimization Results", False):
        print("- Wait for optimization to run (check interval_days setting)")
        print("- Or run manual optimization: python3 -m crypto_bot.auto_optimizer")
    if not results.get("Backtesting Logs", False):
        print("- Check if the bot is running and logging enabled")
    if not results.get("Backtest Cache", False):
        print("- Run continuous backtesting: python3 -m crypto_bot.backtest.cli continuous")

if __name__ == "__main__":
    main()
