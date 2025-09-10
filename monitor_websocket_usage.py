#!/usr/bin/env python3
"""
Real-time WebSocket monitoring script for the trading bot.
This script monitors the bot's logs to verify WebSocket usage during trading.
"""

import time
import re
import subprocess
from pathlib import Path
from datetime import datetime

def monitor_websocket_usage():
    """Monitor bot logs for WebSocket usage indicators."""
    
    log_file = Path("crypto_bot/logs/bot.log")
    
    if not log_file.exists():
        print("❌ Bot log file not found. Is the bot running?")
        return
    
    print("🔍 Monitoring WebSocket Usage in Trading Bot")
    print("=" * 50)
    print(f"📁 Monitoring log file: {log_file}")
    print("⏰ Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("\n🔍 Looking for WebSocket indicators...")
    print("   - WebSocket connections")
    print("   - Real-time price updates")
    print("   - Position monitoring")
    print("   - Order executions")
    print("\n" + "="*50)
    
    # WebSocket indicators to look for
    ws_indicators = {
        "websocket": 0,
        "ws_": 0,
        "watch_": 0,
        "real-time": 0,
        "position monitoring": 0,
        "trailing stop": 0,
        "exit triggered": 0,
        "price update": 0
    }
    
    # HTTP/REST indicators
    rest_indicators = {
        "fetch_": 0,
        "http": 0,
        "rest": 0,
        "api call": 0
    }
    
    try:
        # Use tail to follow the log file
        process = subprocess.Popen(
            ["tail", "-f", str(log_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        start_time = time.time()
        
        while True:
            line = process.stdout.readline()
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
            
            # Check for WebSocket indicators
            for indicator in ws_indicators:
                if indicator.lower() in line.lower():
                    ws_indicators[indicator] += 1
                    print(f"✅ WebSocket: {line}")
            
            # Check for REST indicators
            for indicator in rest_indicators:
                if indicator.lower() in line.lower():
                    rest_indicators[indicator] += 1
                    print(f"📡 REST: {line}")
            
            # Print summary every 30 seconds
            elapsed = time.time() - start_time
            if elapsed > 30:
                print("\n📊 WebSocket Usage Summary:")
                print("=" * 30)
                
                ws_total = sum(ws_indicators.values())
                rest_total = sum(rest_indicators.values())
                
                print(f"WebSocket indicators: {ws_total}")
                print(f"REST indicators: {rest_total}")
                
                if ws_total > rest_total:
                    print("✅ WebSocket appears to be the primary method")
                elif rest_total > ws_total:
                    print("⚠️  REST API appears to be the primary method")
                else:
                    print("ℹ️  Mixed usage detected")
                
                print(f"Elapsed time: {elapsed:.1f}s")
                print("=" * 30)
                start_time = time.time()
                
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        print("\n📊 Final Summary:")
        print("=" * 30)
        
        ws_total = sum(ws_indicators.values())
        rest_total = sum(rest_indicators.values())
        
        print(f"Total WebSocket indicators: {ws_total}")
        print(f"Total REST indicators: {rest_total}")
        
        if ws_total > rest_total:
            print("✅ WebSocket is being used for trading operations")
        elif rest_total > ws_total:
            print("❌ REST API is being used more than WebSocket")
        else:
            print("⚠️  Mixed usage - check configuration")
        
        print("\nDetailed WebSocket indicators:")
        for indicator, count in ws_indicators.items():
            if count > 0:
                print(f"  {indicator}: {count}")
        
        print("\nDetailed REST indicators:")
        for indicator, count in rest_indicators.items():
            if count > 0:
                print(f"  {indicator}: {count}")
    
    except Exception as e:
        print(f"❌ Error monitoring logs: {e}")

def check_bot_status():
    """Check if the bot is currently running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "crypto_bot.main"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"✅ Bot is running (PIDs: {', '.join(pids)})")
            return True
        else:
            print("❌ Bot is not running")
            return False
            
    except Exception as e:
        print(f"❌ Error checking bot status: {e}")
        return False

def main():
    """Main function."""
    print("🚀 WebSocket Usage Monitor")
    print("=" * 50)
    
    # Check if bot is running
    if not check_bot_status():
        print("\n💡 To start monitoring WebSocket usage:")
        print("   1. Start your trading bot")
        print("   2. Run this script again")
        return
    
    print("\n🔍 Starting WebSocket usage monitoring...")
    print("Press Ctrl+C to stop monitoring")
    
    monitor_websocket_usage()

if __name__ == "__main__":
    main()
