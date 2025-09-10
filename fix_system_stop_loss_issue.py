#!/usr/bin/env python3
"""
System-wide stop loss monitoring fix.
This addresses the core issue: stop losses don't work when the bot isn't running.
"""

import subprocess
import time
import sys
import os
from pathlib import Path
import requests
import json

def check_bot_status():
    """Check if the trading bot is running."""
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    bot_running = 'crypto_bot' in result.stdout or 'python.*main' in result.stdout
    return bot_running

def get_all_positions():
    """Get all current positions from the position log."""
    positions_file = Path("crypto_bot/logs/positions.log")
    if not positions_file.exists():
        return []
    
    positions = []
    lines = positions_file.read_text().splitlines()
    
    for line in lines:
        if "Active" in line and "entry" in line:
            # Parse position line
            parts = line.split()
            if len(parts) >= 10:
                active_index = parts.index("Active")
                if active_index + 6 < len(parts):
                    symbol = parts[active_index + 1]
                    side = parts[active_index + 2]
                    amount = float(parts[active_index + 3])
                    entry_price = float(parts[active_index + 5])
                    current_price = float(parts[active_index + 7])
                    
                    positions.append({
                        'symbol': symbol,
                        'side': side,
                        'amount': amount,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'line': line
                    })
    
    return positions

def get_current_price(symbol):
    """Get current market price for a symbol."""
    try:
        # Convert symbol format (e.g., HBAR/USD -> HBARUSD)
        kraken_symbol = symbol.replace('/', '')
        
        response = requests.get(f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}")
        data = response.json()
        
        if data.get("error"):
            return None
            
        current_price = float(data["result"][kraken_symbol]["c"][0])
        return current_price
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

def analyze_position_risk(position):
    """Analyze risk for a single position."""
    current_price = get_current_price(position['symbol'])
    if not current_price:
        return None
    
    entry_price = position['entry_price']
    side = position['side']
    
    # Calculate PnL
    if side == "buy":
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    else:  # sell/short
        pnl_pct = ((entry_price - current_price) / entry_price) * 100
    
    # Check stop loss levels
    stop_loss_levels = [
        ("Micro Scalp", 0.005),
        ("Sniper Bot", 0.0063797342686504575),
        ("Default", 0.008),
        ("Risk Manager", 0.01),
        ("Bounce Scalper", 0.01)
    ]
    
    triggered_stops = []
    for name, sl_pct in stop_loss_levels:
        if side == "buy":
            stop_price = entry_price * (1 - sl_pct)
            if current_price <= stop_price:
                triggered_stops.append((name, sl_pct, stop_price))
        else:  # sell/short
            stop_price = entry_price * (1 + sl_pct)
            if current_price >= stop_price:
                triggered_stops.append((name, sl_pct, stop_price))
    
    return {
        'symbol': position['symbol'],
        'side': side,
        'entry_price': entry_price,
        'current_price': current_price,
        'pnl_pct': pnl_pct,
        'triggered_stops': triggered_stops,
        'at_risk': len(triggered_stops) > 0
    }

def start_bot():
    """Start the trading bot."""
    print("🚀 Starting Trading Bot...")
    
    try:
        process = subprocess.Popen([
            sys.executable, 'start_bot_auto.py'
        ], 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
        )
        
        print(f"✅ Bot started (PID: {process.pid})")
        
        # Wait for bot to initialize
        time.sleep(5)
        
        # Verify bot is running
        if check_bot_status():
            print("✅ Bot is running and monitoring positions")
            return True
        else:
            print("❌ Bot failed to start properly")
            return False
            
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        return False

def create_monitoring_script():
    """Create a monitoring script to ensure bot stays running."""
    script_content = '''#!/bin/bash
# Bot monitoring script
BOT_PID_FILE="bot_pid.txt"
LOG_FILE="crypto_bot/logs/bot_monitor.log"

while true; do
    if ! pgrep -f "crypto_bot" > /dev/null; then
        echo "$(date): Bot not running, restarting..." >> $LOG_FILE
        python3 start_bot_auto.py &
        echo $! > $BOT_PID_FILE
        sleep 10
    else
        echo "$(date): Bot running normally" >> $LOG_FILE
    fi
    sleep 30
done
'''
    
    with open("monitor_bot.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("monitor_bot.sh", 0o755)
    print("✅ Created bot monitoring script: monitor_bot.sh")

def main():
    """Main system-wide stop loss fix."""
    print("🔧 SYSTEM-WIDE STOP LOSS FIX")
    print("=" * 50)
    
    # Check bot status
    bot_running = check_bot_status()
    print(f"Bot Status: {'🟢 Running' if bot_running else '🔴 NOT RUNNING'}")
    
    # Get all positions
    positions = get_all_positions()
    print(f"Active Positions: {len(positions)}")
    
    if not positions:
        print("No active positions found.")
        return
    
    # Analyze each position
    print("\n📊 Position Risk Analysis:")
    print("-" * 50)
    
    at_risk_positions = []
    for position in positions:
        risk_analysis = analyze_position_risk(position)
        if risk_analysis:
            print(f"\n{risk_analysis['symbol']} ({risk_analysis['side']}):")
            print(f"  Entry: ${risk_analysis['entry_price']:.6f}")
            print(f"  Current: ${risk_analysis['current_price']:.6f}")
            print(f"  PnL: {risk_analysis['pnl_pct']:.2f}%")
            
            if risk_analysis['at_risk']:
                print("  🚨 STOP LOSSES SHOULD HAVE TRIGGERED:")
                for name, pct, price in risk_analysis['triggered_stops']:
                    print(f"    - {name} ({pct*100:.1f}%): ${price:.6f}")
                at_risk_positions.append(risk_analysis)
            else:
                print("  ✅ Position within risk parameters")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print("-" * 50)
    print(f"Total Positions: {len(positions)}")
    print(f"At Risk: {len(at_risk_positions)}")
    
    if at_risk_positions:
        print("\n🚨 CRITICAL ISSUE: Stop losses should have triggered!")
        print("Root Cause: Bot is not running - no position monitoring active")
        
        if not bot_running:
            print("\n🛠️ SOLUTION: Starting bot now...")
            if start_bot():
                print("✅ Bot started successfully")
                print("📊 Position monitoring now active")
                print("🛑 Stop losses will trigger automatically")
            else:
                print("❌ Failed to start bot automatically")
                print("Manual intervention required")
        
        # Create monitoring script
        create_monitoring_script()
        
        print("\n📈 Monitoring Commands:")
        print("tail -f crypto_bot/logs/bot.log")
        print("tail -f crypto_bot/logs/positions.log")
        print("./monitor_bot.sh  # Keep bot running")
    
    else:
        print("✅ All positions within risk parameters")
        if not bot_running:
            print("⚠️  Warning: Bot not running - no position monitoring")
            print("Recommendation: Start bot for proper risk management")
    
    print("\n⚠️  IMPORTANT:")
    print("• Stop losses only work when bot is actively running")
    print("• Position monitoring requires real-time price feeds")
    print("• Consider using monitor_bot.sh to keep bot running")
    print("• All positions are affected, not just individual symbols")

if __name__ == "__main__":
    main()
