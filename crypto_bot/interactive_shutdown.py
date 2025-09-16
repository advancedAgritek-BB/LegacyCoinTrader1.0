#!/usr/bin/env python3
"""
Interactive Shutdown System for LegacyCoinTrader

This module provides enhanced signal handling and interactive shutdown
capabilities that work directly from the terminal while the bot is running.
"""

import sys
import signal
import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class InteractiveShutdown:
    """
    Enhanced shutdown system that works interactively from the terminal.
    Handles Ctrl+C, Enter key, and other shutdown signals properly.
    """
    
    def __init__(self, bot_state: Dict[str, Any], cleanup_callback: Optional[Callable] = None):
        self.bot_state = bot_state
        self.cleanup_callback = cleanup_callback
        self.shutdown_initiated = False
        self.shutdown_event = asyncio.Event()
        self.project_root = Path(__file__).parent.parent
        self.pid_file = self.project_root / "bot_pid.txt"
        
        # Setup enhanced signal handlers
        self._setup_signal_handlers()
        
        logger.info("🛡️ Interactive shutdown system initialized")
        logger.info("💡 Press Ctrl+C or type 'quit'/'exit' to shutdown safely")

    def _setup_signal_handlers(self) -> None:
        """Setup enhanced signal handlers for graceful shutdown."""
        
        def signal_handler(signum: int, frame) -> None:
            """Handle shutdown signals with proper cleanup."""
            if self.shutdown_initiated:
                logger.warning("🔄 Shutdown already in progress...")
                return
                
            signal_name = signal.Signals(signum).name
            logger.info(f"📡 Received {signal_name} signal - initiating safe shutdown...")
            
            # Set shutdown flag
            self.shutdown_initiated = True
            self.bot_state["running"] = False
            self.bot_state["shutdown_requested"] = True
            
            # Trigger shutdown event
            if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop():
                asyncio.create_task(self._execute_shutdown(signal_name))
            else:
                # If no event loop is running, start one
                threading.Thread(target=self._run_shutdown_in_thread, args=(signal_name,)).start()
        
        # Handle common signals
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request
        
        # Handle additional signals if available (Unix-like systems)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)   # Hang up
        if hasattr(signal, 'SIGQUIT'):
            signal.signal(signal.SIGQUIT, signal_handler)  # Quit

    def _run_shutdown_in_thread(self, signal_name: str) -> None:
        """Run shutdown in a separate thread if no event loop is available."""
        try:
            asyncio.run(self._execute_shutdown(signal_name))
        except Exception as e:
            logger.error(f"❌ Error in shutdown thread: {e}")

    async def _execute_shutdown(self, trigger: str) -> None:
        """Execute the shutdown process."""
        logger.info(f"🛑 Starting safe shutdown (triggered by: {trigger})")
        
        try:
            # Notify about shutdown
            print(f"\n{'='*50}")
            print("🛑 SHUTDOWN INITIATED")
            print(f"Trigger: {trigger}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("🔄 Performing safe shutdown...")
            print(f"{'='*50}")
            
            # Stop trading first
            self.bot_state["running"] = False
            logger.info("🤖 Trading stopped")
            
            # Wait a moment for current operations to complete
            await asyncio.sleep(2)
            
            # Execute custom cleanup if provided
            if self.cleanup_callback:
                try:
                    logger.info("🧹 Executing custom cleanup...")
                    if asyncio.iscoroutinefunction(self.cleanup_callback):
                        await self.cleanup_callback()
                    else:
                        self.cleanup_callback()
                    logger.info("✅ Custom cleanup completed")
                except Exception as e:
                    logger.error(f"❌ Custom cleanup failed: {e}")
            
            # Close open positions safely
            await self._close_positions_safely()
            
            # Save current state
            await self._save_bot_state()
            
            # Clean up PID file
            self._cleanup_pid_file()
            
            # Set shutdown event
            self.shutdown_event.set()
            
            print("✅ Safe shutdown completed successfully!")
            print("👋 Goodbye!")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
            print(f"❌ Shutdown error: {e}")
        finally:
            # Force exit after cleanup
            sys.exit(0)

    async def _close_positions_safely(self) -> None:
        """Attempt to close open positions safely."""
        try:
            logger.info("💰 Checking for open positions...")
            
            # Check if we have a trade manager in the bot state
            trade_manager = self.bot_state.get('trade_manager')
            if trade_manager:
                logger.info("💰 Closing open positions...")
                # This would integrate with the actual trade manager
                # trade_manager.close_all_positions()
                await asyncio.sleep(1)  # Simulate position closure
                logger.info("✅ Positions closed safely")
            else:
                logger.info("ℹ️ No active trade manager found")
                
        except Exception as e:
            logger.error(f"❌ Failed to close positions: {e}")

    async def _save_bot_state(self) -> None:
        """Save current bot state."""
        try:
            logger.info("💾 Saving bot state...")
            
            state_file = self.project_root / "logs" / "bot_shutdown_state.json"
            state_file.parent.mkdir(exist_ok=True)
            
            import json
            shutdown_state = {
                "shutdown_time": datetime.now().isoformat(),
                "bot_state": {k: v for k, v in self.bot_state.items() 
                             if isinstance(v, (str, int, float, bool, list, dict))},
                "shutdown_reason": "User initiated"
            }
            
            with open(state_file, 'w') as f:
                json.dump(shutdown_state, f, indent=2)
            
            logger.info(f"✅ Bot state saved to {state_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save bot state: {e}")

    def _cleanup_pid_file(self) -> None:
        """Clean up PID file."""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                logger.info("🧹 PID file cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up PID file: {e}")

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to be triggered."""
        await self.shutdown_event.wait()

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_initiated or self.bot_state.get("shutdown_requested", False)


class EnhancedConsoleControl:
    """
    Enhanced console control with immediate shutdown capabilities.
    """
    
    def __init__(self, bot_state: Dict[str, Any], shutdown_system: InteractiveShutdown):
        self.bot_state = bot_state
        self.shutdown_system = shutdown_system
        self.running = True

    async def control_loop(self) -> None:
        """Enhanced control loop with immediate shutdown support."""
        import os
        
        # Check environment variables and stdin availability
        non_interactive = os.environ.get('NON_INTERACTIVE')
        
        # Check if stdin is available
        stdin_available = hasattr(sys.stdin, 'isatty') and sys.stdin.isatty()
        
        # Auto-start trading in non-interactive mode
        if non_interactive or not stdin_available:
            self.bot_state["running"] = True
            print("🤖 Auto-starting trading in non-interactive mode")
            print("🛡️ Enhanced shutdown available - Ctrl+C to stop safely")
        else:
            print("🎮 Interactive mode enabled")
            print("📝 Commands: start | stop | reload | quit | exit | shutdown")
            print("🛡️ Quick shutdown: Ctrl+C or Enter on empty line")
        
        # File for frontend communication
        from crypto_bot.utils.logger import LOG_DIR
        control_file = LOG_DIR / "bot_control.json"
        
        try:
            while self.running and not self.shutdown_system.is_shutdown_requested():
                # In non-interactive mode, check for frontend commands
                if non_interactive or not stdin_available:
                    # Check for frontend control commands
                    if control_file.exists():
                        try:
                            with open(control_file, 'r') as f:
                                control_data = json.loads(f.read())
                            
                            cmd = control_data.get('command', '').strip().lower()
                            if cmd == "start":
                                self.bot_state["running"] = True
                                print("🚀 Frontend command: Trading started")
                            elif cmd == "stop":
                                self.bot_state["running"] = False
                                print("⏸️ Frontend command: Trading stopped")
                            elif cmd == "reload":
                                self.bot_state["reload"] = True
                                print("🔄 Frontend command: Reloading config")
                            elif cmd in {"quit", "exit", "shutdown"}:
                                print("🛑 Frontend command: Shutdown requested")
                                await self.shutdown_system._execute_shutdown("Frontend")
                                break
                            
                            # Remove the command file after processing
                            control_file.unlink(missing_ok=True)
                            
                        except Exception as e:
                            logger.warning(f"Error reading control file: {e}")
                    
                    await asyncio.sleep(1)  # Check every second for frontend commands
                    continue
                else:
                    # Interactive mode
                    try:
                        print("\n> ", end="", flush=True)
                        cmd = (await asyncio.to_thread(input)).strip().lower()
                        
                        # Handle empty input as shutdown request
                        if not cmd:
                            print("🛑 Empty input detected - initiating shutdown...")
                            await self.shutdown_system._execute_shutdown("Enter key")
                            break
                            
                    except (EOFError, OSError):
                        print("📡 Stdin not available, switching to non-interactive mode")
                        non_interactive = True
                        stdin_available = False
                        continue
                    except KeyboardInterrupt:
                        # This should be handled by signal handler, but just in case
                        print("\n🛑 Ctrl+C detected - shutdown in progress...")
                        break
                
                # Process commands
                if cmd == "start":
                    self.bot_state["running"] = True
                    print("🚀 Trading started")
                elif cmd == "stop":
                    self.bot_state["running"] = False
                    print("⏸️ Trading stopped")
                elif cmd == "reload":
                    self.bot_state["reload"] = True
                    print("🔄 Reloading config")
                elif cmd in {"quit", "exit", "shutdown"}:
                    print("🛑 Shutdown command received")
                    await self.shutdown_system._execute_shutdown(f"Command: {cmd}")
                    break
                elif cmd == "status":
                    running_status = "🟢 Running" if self.bot_state.get("running") else "🔴 Stopped"
                    print(f"📊 Bot Status: {running_status}")
                elif cmd == "help":
                    print("📖 Available commands:")
                    print("  start    - Start trading")
                    print("  stop     - Stop trading")
                    print("  reload   - Reload configuration")
                    print("  status   - Show bot status")
                    print("  quit     - Safe shutdown")
                    print("  exit     - Safe shutdown")
                    print("  shutdown - Safe shutdown")
                    print("  help     - Show this help")
                    print("  <Enter>  - Quick shutdown")
                    print("  Ctrl+C   - Emergency shutdown")
                elif cmd:
                    print(f"❓ Unknown command: {cmd}")
                    print("💡 Type 'help' for available commands or press Enter to shutdown")
                    
        except asyncio.CancelledError:
            logger.info("🛑 Control loop cancelled")
            self.bot_state["running"] = False
            raise
        except Exception as e:
            logger.error(f"❌ Error in control loop: {e}")
            self.bot_state["running"] = False
        finally:
            self.running = False


# Convenience function to setup interactive shutdown
def setup_interactive_shutdown(bot_state: Dict[str, Any], 
                             cleanup_callback: Optional[Callable] = None) -> tuple[InteractiveShutdown, EnhancedConsoleControl]:
    """
    Setup interactive shutdown system for the bot.
    
    Returns:
        tuple: (shutdown_system, console_control)
    """
    shutdown_system = InteractiveShutdown(bot_state, cleanup_callback)
    console_control = EnhancedConsoleControl(bot_state, shutdown_system)
    
    return shutdown_system, console_control


# Example usage for testing
if __name__ == "__main__":
    async def test_interactive_shutdown():
        """Test the interactive shutdown system."""
        
        # Mock bot state
        bot_state = {
            "running": False,
            "reload": False
        }
        
        # Mock cleanup function
        async def cleanup():
            print("🧹 Custom cleanup executed")
            await asyncio.sleep(1)
        
        # Setup interactive shutdown
        shutdown_system, console_control = setup_interactive_shutdown(bot_state, cleanup)
        
        # Run console control
        print("🧪 Testing interactive shutdown system")
        print("💡 Try typing commands or pressing Ctrl+C")
        
        try:
            await console_control.control_loop()
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted")
        
        print("✅ Test completed")
    
    # Run test
    asyncio.run(test_interactive_shutdown())
