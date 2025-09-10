#!/usr/bin/env python3
import time
import subprocess
import sys
from pathlib import Path

def test_pipeline():
    print("🧪 Testing evaluation pipeline...")
    
    # Wait for bot to start
    time.sleep(15)
    
    # Check if bot is running
    result = subprocess.run(['pgrep', '-f', 'python.*main.py'], capture_output=True)
    if result.returncode != 0:
        print("❌ Bot is not running")
        return False
    
    print("✅ Bot is running")
    
    # Check logs for successful operation
    log_file = Path("crypto_bot/logs/bot_comprehensive.log")
    if log_file.exists():
        with open(log_file, 'r') as f:
            content = f.read()
            
        if "actionable signals" in content:
            print("✅ Found actionable signals in logs")
            return True
        elif "PHASE: analyse_batch completed" in content:
            print("✅ Found analysis completion in logs")
            return True
        else:
            print("⚠️ No signal generation found yet")
            return False
    else:
        print("❌ Log file not found")
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
