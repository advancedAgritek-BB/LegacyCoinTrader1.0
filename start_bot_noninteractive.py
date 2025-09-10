#!/usr/bin/env python3
"""Simple script to start the trading bot in non-interactive mode."""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_bot.main import main

if __name__ == "__main__":
    # Set environment variable to indicate non-interactive mode
    os.environ['NON_INTERACTIVE'] = '1'
    
    try:
        # Run the bot
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Bot error: {e}")
        sys.exit(1)
