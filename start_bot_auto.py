#!/usr/bin/env python3
"""
Automated bot startup script that starts trading immediately
"""

import subprocess
import time
import sys
import os

def start_bot_automatically():
    """Start the bot with automatic trading enabled"""
    
    # Set environment variable to auto-start trading
    env = os.environ.copy()
    env['AUTO_START_TRADING'] = '1'
    
    # Start the bot process
    process = subprocess.Popen([
        sys.executable, 'start_bot_noninteractive.py'
    ], 
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env,
    text=True,
    bufsize=1
    )
    
    # Send the paper trading balance and start command
    commands = [
        "10000\n",  # Paper trading balance
        "start\n",   # Start trading
    ]
    
    for cmd in commands:
        process.stdin.write(cmd)
        process.stdin.flush()
        time.sleep(1)
    
    print("Bot started automatically with trading enabled")
    return process

if __name__ == "__main__":
    start_bot_automatically()
