#!/usr/bin/env python3
"""
Test script for interactive shutdown functionality.
This simulates a long-running bot process to test the shutdown mechanisms.
"""

import sys
import time
import signal
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
crypto_bot_path = project_root / "crypto_bot"
if str(crypto_bot_path) not in sys.path:
    sys.path.insert(0, str(crypto_bot_path))

from crypto_bot.interactive_shutdown import setup_interactive_shutdown
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBot:
    """A simple test bot that simulates the main trading bot."""
    
    def __init__(self):
        self.running = True
        self.state = {
            "running": False,
            "reload": False,
            "shutdown_requested": False
        }
        
        # Setup interactive shutdown
        self.shutdown_system, self.console_control = setup_interactive_shutdown(
            self.state, self.cleanup
        )
    
    async def cleanup(self):
        """Test cleanup function."""
        logger.info("🧹 Test bot cleanup started")
        await asyncio.sleep(1)  # Simulate cleanup work
        logger.info("✅ Test bot cleanup completed")
    
    async def main_loop(self):
        """Simulate the main bot loop."""
        logger.info("🤖 Test bot started")
        
        # Start console control
        control_task = asyncio.create_task(self.console_control.control_loop())
        
        try:
            iteration = 0
            while not self.shutdown_system.is_shutdown_requested():
                iteration += 1
                
                if self.state["running"]:
                    print(f"🔄 Bot working... (iteration {iteration})")
                else:
                    print(f"⏸️ Bot paused... (iteration {iteration})")
                
                # Check for reload
                if self.state["reload"]:
                    print("🔄 Reloading configuration...")
                    self.state["reload"] = False
                
                await asyncio.sleep(2)  # Simulate work
            
            logger.info("🛑 Main loop exiting due to shutdown request")
            
        except asyncio.CancelledError:
            logger.info("🛑 Main loop cancelled")
        finally:
            # Cancel control task
            control_task.cancel()
            try:
                await control_task
            except asyncio.CancelledError:
                pass
        
        logger.info("👋 Test bot finished")


async def main():
    """Main test function."""
    print("🧪 Testing Interactive Shutdown System")
    print("=" * 50)
    print("💡 Try the following:")
    print("   • Press Ctrl+C to test signal handling")
    print("   • Press Enter to test quick shutdown")
    print("   • Type 'start' to start the test bot")
    print("   • Type 'stop' to stop the test bot")
    print("   • Type 'quit' to shutdown completely")
    print("=" * 50)
    
    bot = TestBot()
    
    try:
        await bot.main_loop()
    except KeyboardInterrupt:
        print("\n✅ Keyboard interrupt handled by interactive shutdown system")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    
    print("✅ Interactive shutdown test completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"❌ Test failed: {e}")
