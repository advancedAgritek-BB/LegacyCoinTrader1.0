#!/usr/bin/env python3
"""
Simple direct bot startup - no web server integration
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def main():
    """Start the trading bot directly"""
    print("🚀 Starting LegacyCoinTrader - Trading Bot Only")
    print("=" * 60)
    print("🤖 Trading Bot with TradeManager as Single Source of Truth")
    print("=" * 60)
    
    # Set environment variables
    os.environ['AUTO_START_TRADING'] = '1'
    os.environ['NON_INTERACTIVE'] = '1'
    
    try:
        # Import and run the main bot function
        from crypto_bot.main import _main_impl
        
        print("🎯 Starting trading bot...")
        print("-" * 60)
        
        # Run the main bot function
        notifier = await _main_impl()
        
        print("✅ Bot completed successfully")
        
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
    except Exception as e:
        print(f"❌ Bot error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
