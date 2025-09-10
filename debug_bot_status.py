#!/usr/bin/env python3
"""
Debug script to test bot status detection
"""

import subprocess
import json
from pathlib import Path

def test_bot_status_detection():
    """Test the bot status detection logic"""
    print("🔍 Testing bot status detection...")
    
    status = {
        'bot_running': False,
        'execution_running': False,
        'trading_active': False,
        'processes': [],
        'last_trade': None,
        'last_execution': None
    }

    # Check if bot processes are running
    try:
        # Check for main bot process - look for multiple possible process names
        bot_process_patterns = [
            'main.py',
            'start_bot_final.py',
            'start_bot_',
            'crypto_bot'
        ]
        
        print(f"🔍 Checking process patterns: {bot_process_patterns}")
        
        for pattern in bot_process_patterns:
            print(f"  Testing pattern: '{pattern}'")
            result = subprocess.run(
                ['pgrep', '-f', pattern],
                capture_output=True,
                text=True
            )
            print(f"    Return code: {result.returncode}")
            print(f"    Output: '{result.stdout.strip()}'")
            print(f"    Error: '{result.stderr.strip()}'")
            
            if result.returncode == 0:
                status['bot_running'] = True
                processes = result.stdout.strip().split('\n')
                status['processes'].extend([p for p in processes if p])
                print(f"    ✅ Found processes: {status['processes']}")
                break
            else:
                print(f"    ❌ No processes found for pattern '{pattern}'")

        # Check for execution processes
        print(f"🔍 Checking for execution processes...")
        result = subprocess.run(
            ['pgrep', '-f', 'execution'],
            capture_output=True,
            text=True
        )
        print(f"  Return code: {result.returncode}")
        print(f"  Output: '{result.stdout.strip()}'")
        
        if result.returncode == 0:
            status['execution_running'] = True
            status['processes'].extend(result.stdout.strip().split('\n'))
    except Exception as e:
        print(f"❌ Error in process detection: {e}")

    # Check if there are recent trades (indicating active trading)
    trades_file = Path(__file__).parent / 'crypto_bot' / 'logs' / 'trades.csv'
    print(f"🔍 Checking trades file: {trades_file}")
    print(f"  File exists: {trades_file.exists()}")
    
    if trades_file.exists():
        try:
            # Get file modification time
            mod_time = trades_file.stat().st_mtime
            print(f"  Modification time: {mod_time}")
            print(f"  Current time: {time.time()}")
            print(f"  Time difference: {time.time() - mod_time}")
            
            # If modified within last 5 minutes, consider trading active
            if time.time() - mod_time < 300:  # 5 minutes
                status['trading_active'] = True
                status['last_trade'] = mod_time
                print(f"  ✅ Trading active (modified within 5 minutes)")
            else:
                print(f"  ❌ Trading not active (modified more than 5 minutes ago)")
        except Exception as e:
            print(f"  ❌ Error checking trades file: {e}")

    # Check execution log for recent activity
    execution_log = Path(__file__).parent / 'crypto_bot' / 'logs' / 'execution.log'
    print(f"🔍 Checking execution log: {execution_log}")
    print(f"  File exists: {execution_log.exists()}")
    
    if execution_log.exists():
        try:
            mod_time = execution_log.stat().st_mtime
            status['last_execution'] = mod_time
            print(f"  Last execution: {mod_time}")
        except Exception as e:
            print(f"  ❌ Error checking execution log: {e}")

    print("\n📊 Final Status:")
    print(json.dumps(status, indent=2))
    
    return status

if __name__ == "__main__":
    import time
    test_bot_status_detection()
