#!/usr/bin/env python3
"""
Simple test to verify the main bot function works
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_main_bot():
    """Test if the main bot function can be imported and run"""
    try:
        print("Testing main bot function...")
        
        # Set environment variables
        os.environ['AUTO_START_TRADING'] = '1'
        os.environ['NON_INTERACTIVE'] = '1'
        
        # Import the main function
        from crypto_bot.main import _main_impl
        print("✅ Main function imported successfully")
        
        # Try to run it for a short time
        print("🎯 Starting main bot function...")
        
        # Create a task that will be cancelled after 10 seconds
        task = asyncio.create_task(_main_impl())
        
        try:
            # Wait for 10 seconds
            await asyncio.wait_for(task, timeout=10.0)
        except asyncio.TimeoutError:
            print("⏰ Timeout reached - cancelling bot")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                print("✅ Bot cancelled successfully")
        
        print("✅ Main bot function test completed")
        
    except Exception as e:
        print(f"❌ Error testing main bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_main_bot())
