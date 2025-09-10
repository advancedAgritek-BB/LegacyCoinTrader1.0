#!/usr/bin/env python3
"""
Enhanced WebSocket Connection Monitor for Trading Bot
Monitors WebSocket connections, position monitoring, and real-time data flow
"""

import asyncio
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
import threading
from typing import Dict, List, Optional

class WebSocketMonitor:
    def __init__(self):
        self.log_file = Path("crypto_bot/logs/bot.log")
        self.execution_log = Path("crypto_bot/logs/execution.log")
        self.websocket_stats = {
            "connections_attempted": 0,
            "connections_successful": 0,
            "price_updates_received": 0,
            "position_monitoring_active": 0,
            "exit_triggers": 0,
            "errors": 0
        }
        self.last_stats_time = time.time()
        
    def check_bot_status(self) -> bool:
        """Check if the trading bot is running."""
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
    
    def monitor_websocket_indicators(self):
        """Monitor logs for WebSocket usage indicators."""
        print("🔍 Monitoring WebSocket Usage in Trading Bot")
        print("=" * 60)
        print(f"📁 Monitoring log files:")
        print(f"   - Bot log: {self.log_file}")
        print(f"   - Execution log: {self.execution_log}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🔍 Looking for WebSocket indicators...")
        print("   - WebSocket connections")
        print("   - Real-time price updates")
        print("   - Position monitoring")
        print("   - Exit triggers")
        print("   - Ticker subscriptions")
        print("\n" + "="*60)
        
        # WebSocket indicators to look for
        ws_indicators = {
            "websocket": 0,
            "ws_": 0,
            "watch_": 0,
            "real-time": 0,
            "position monitoring": 0,
            "trailing stop": 0,
            "exit triggered": 0,
            "price update": 0,
            "ticker": 0,
            "subscribe": 0,
            "krakenwsclient": 0,
            "websocket client": 0
        }
        
        # HTTP/REST indicators
        rest_indicators = {
            "fetch_": 0,
            "http": 0,
            "rest": 0,
            "api call": 0,
            "mode=rest": 0
        }
        
        # Error indicators
        error_indicators = {
            "websocket.*error": 0,
            "ws.*error": 0,
            "connection.*failed": 0,
            "socket.*closed": 0
        }
        
        try:
            # Use tail to follow the log files
            processes = []
            
            # Monitor bot.log
            if self.log_file.exists():
                bot_process = subprocess.Popen(
                    ["tail", "-f", str(self.log_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                processes.append(("bot.log", bot_process))
            
            # Monitor execution.log
            if self.execution_log.exists():
                exec_process = subprocess.Popen(
                    ["tail", "-f", str(self.execution_log)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                processes.append(("execution.log", exec_process))
            
            if not processes:
                print("❌ No log files found. Is the bot running?")
                return
            
            start_time = time.time()
            last_summary_time = start_time
            
            while True:
                for log_name, process in processes:
                    line = process.stdout.readline()
                    if line:
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
                        
                        # Check for error indicators
                        for indicator in error_indicators:
                            if indicator.lower() in line.lower():
                                error_indicators[indicator] += 1
                                print(f"🚨 Error: {line}")
                
                # Print summary every 30 seconds
                current_time = time.time()
                if current_time - last_summary_time > 30:
                    self._print_summary(ws_indicators, rest_indicators, error_indicators, current_time - start_time)
                    last_summary_time = current_time
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            self._print_final_summary(ws_indicators, rest_indicators, error_indicators)
        
        except Exception as e:
            print(f"❌ Error monitoring logs: {e}")
    
    def _print_summary(self, ws_indicators: Dict, rest_indicators: Dict, error_indicators: Dict, elapsed: float):
        """Print monitoring summary."""
        print("\n📊 WebSocket Usage Summary:")
        print("=" * 40)
        
        ws_total = sum(ws_indicators.values())
        rest_total = sum(rest_indicators.values())
        error_total = sum(error_indicators.values())
        
        print(f"WebSocket indicators: {ws_total}")
        print(f"REST indicators: {rest_total}")
        print(f"Error indicators: {error_total}")
        
        if ws_total > rest_total:
            print("✅ WebSocket appears to be the primary method")
        elif rest_total > ws_total:
            print("⚠️  REST API appears to be the primary method")
        else:
            print("ℹ️  Mixed usage detected")
        
        if error_total > 0:
            print(f"🚨 {error_total} errors detected")
        
        print(f"Elapsed time: {elapsed:.1f}s")
        print("=" * 40)
    
    def _print_final_summary(self, ws_indicators: Dict, rest_indicators: Dict, error_indicators: Dict):
        """Print final monitoring summary."""
        print("\n📊 Final Summary:")
        print("=" * 40)
        
        ws_total = sum(ws_indicators.values())
        rest_total = sum(rest_indicators.values())
        error_total = sum(error_indicators.values())
        
        print(f"Total WebSocket indicators: {ws_total}")
        print(f"Total REST indicators: {rest_total}")
        print(f"Total error indicators: {error_total}")
        
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
        
        print("\nDetailed Error indicators:")
        for indicator, count in error_indicators.items():
            if count > 0:
                print(f"  {indicator}: {count}")
    
    def test_websocket_connection(self):
        """Test direct WebSocket connection to Kraken."""
        print("\n🔗 Testing Direct Kraken WebSocket Connection...")
        try:
            import websocket
            import json
            
            # Test public WebSocket
            ws = websocket.create_connection(
                "wss://ws.kraken.com/v2",
                timeout=10
            )
            
            # Subscribe to ticker
            subscribe_msg = {
                "method": "subscribe",
                "params": {
                    "channel": "ticker",
                    "symbol": ["XBT/USD"]
                }
            }
            
            ws.send(json.dumps(subscribe_msg))
            print("✅ WebSocket connection established")
            print("✅ Subscription message sent")
            
            # Wait for response
            response = ws.recv()
            data = json.loads(response)
            print(f"✅ Received response: {data}")
            
            if "channel" in data and data.get("channel") == "ticker":
                print("✅ WebSocket connection successful!")
                print("✅ Real-time data streaming enabled")
            else:
                print("⚠️  WebSocket connected but unexpected response format")
                print(f"   Response: {data}")
            
            ws.close()
            return True
            
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            print("   This may indicate network issues or Kraken API problems")
            return False

def main():
    """Main monitoring function."""
    print("🚀 Enhanced WebSocket Connection Monitor")
    print("=" * 60)
    
    monitor = WebSocketMonitor()
    
    # Check if bot is running
    if not monitor.check_bot_status():
        print("\n💡 To start monitoring WebSocket connections:")
        print("   1. Start your trading bot")
        print("   2. Run this script again")
        return
    
    # Test direct WebSocket connection
    monitor.test_websocket_connection()
    
    print("\n🔍 Starting WebSocket usage monitoring...")
    print("Press Ctrl+C to stop monitoring")
    
    monitor.monitor_websocket_indicators()

if __name__ == "__main__":
    main()
