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
    env=env,
    text=True,
    bufsize=1,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
    )
    
    print("Bot started automatically with trading enabled")
    print("Note: Bot no longer prompts for paper trading balance - it uses config files automatically")
    return process

if __name__ == "__main__":
    start_bot_automatically()
