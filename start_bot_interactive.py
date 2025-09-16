#!/usr/bin/env python3
"""
Interactive Bot Launcher for LegacyCoinTrader

This script starts the trading bot with enhanced interactive shutdown capabilities.
It allows you to stop the bot using Ctrl+C or Enter key directly from the terminal.
"""

import os
import sys
import signal
import asyncio
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "crypto_bot"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InteractiveBotLauncher:
    """
    Launches the bot with interactive shutdown capabilities.
    """
    
    def __init__(self):
        self.project_root = project_root
        self.bot_process: Optional[subprocess.Popen] = None
        self.shutdown_requested = False
        self.input_thread: Optional[threading.Thread] = None
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum: int, frame):
            signal_name = signal.Signals(signum).name
            print(f"\n📡 Received {signal_name} signal")
            self.request_shutdown(f"Signal: {signal_name}")
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
    
    def start_input_monitor(self):
        """Start monitoring for Enter key presses."""
        def input_monitor():
            try:
                while not self.shutdown_requested:
                    try:
                        user_input = input().strip().lower()
                        
                        if not user_input:  # Empty input (just Enter)
                            print("🛑 Enter key detected - requesting shutdown...")
                            self.request_shutdown("Enter key")
                            break
                        elif user_input in ['quit', 'exit', 'stop', 'shutdown']:
                            print(f"🛑 Command '{user_input}' detected - requesting shutdown...")
                            self.request_shutdown(f"Command: {user_input}")
                            break
                        elif user_input in ['help', 'h', '?']:
                            self.show_help()
                        elif user_input == 'status':
                            self.show_status()
                        else:
                            print(f"❓ Unknown command: {user_input}")
                            print("💡 Press Enter to shutdown, or type 'help' for commands")
                            
                    except EOFError:
                        # Input stream closed
                        break
                        
            except Exception as e:
                logger.error(f"Input monitor error: {e}")
        
        self.input_thread = threading.Thread(target=input_monitor, daemon=True)
        self.input_thread.start()
    
    def show_help(self):
        """Show available commands."""
        print("\n📖 Available commands:")
        print("  <Enter>           - Safe shutdown")
        print("  quit, exit, stop  - Safe shutdown")
        print("  status            - Show bot status")
        print("  help              - Show this help")
        print("  Ctrl+C            - Emergency shutdown")
        print()
    
    def show_status(self):
        """Show bot status."""
        if self.bot_process:
            if self.bot_process.poll() is None:
                print(f"🟢 Bot Status: Running (PID: {self.bot_process.pid})")
            else:
                print(f"🔴 Bot Status: Stopped (Exit code: {self.bot_process.returncode})")
        else:
            print("🔴 Bot Status: Not started")
    
    def request_shutdown(self, reason: str):
        """Request shutdown of the bot."""
        if self.shutdown_requested:
            return
            
        self.shutdown_requested = True
        print(f"🛑 Shutdown requested: {reason}")
        print("🔄 Initiating safe shutdown...")
        
        if self.bot_process and self.bot_process.poll() is None:
            try:
                # Try graceful shutdown first
                print("📤 Sending SIGTERM to bot process...")
                self.bot_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.bot_process.wait(timeout=10)
                    print("✅ Bot stopped gracefully")
                except subprocess.TimeoutExpired:
                    print("⏰ Graceful shutdown timeout, force killing...")
                    self.bot_process.kill()
                    self.bot_process.wait()
                    print("💀 Bot force killed")
                    
            except Exception as e:
                logger.error(f"Error shutting down bot: {e}")
        
        # Clean up PID file
        pid_file = self.project_root / "bot_pid.txt"
        if pid_file.exists():
            try:
                pid_file.unlink()
                print("🧹 PID file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean PID file: {e}")
    
    def start_bot(self):
        """Start the bot process."""
        print("🚀 Starting LegacyCoinTrader bot...")
        
        # Find Python executable
        python_cmd = sys.executable
        
        # Bot script path
        bot_script = self.project_root / "crypto_bot" / "main.py"
        if not bot_script.exists():
            print(f"❌ Bot script not found: {bot_script}")
            return False
        
        try:
            # Start bot process
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "crypto_bot")
            
            self.bot_process = subprocess.Popen(
                [python_cmd, str(bot_script)],
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print(f"✅ Bot started (PID: {self.bot_process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start bot: {e}")
            return False
    
    def monitor_bot_output(self):
        """Monitor and display bot output."""
        if not self.bot_process:
            return
            
        try:
            for line in iter(self.bot_process.stdout.readline, ''):
                if self.shutdown_requested:
                    break
                print(line.rstrip())
                
        except Exception as e:
            logger.error(f"Error reading bot output: {e}")
    
    def run(self):
        """Main run method."""
        print("🎮 LegacyCoinTrader Interactive Launcher")
        print("=" * 50)
        print("💡 Interactive controls:")
        print("   • Press Ctrl+C for emergency shutdown")
        print("   • Press Enter for safe shutdown")
        print("   • Type 'help' for more commands")
        print("=" * 50)
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Start the bot
        if not self.start_bot():
            return 1
        
        # Start input monitoring
        self.start_input_monitor()
        
        try:
            # Monitor bot output
            self.monitor_bot_output()
            
            # Wait for bot to finish
            if self.bot_process:
                exit_code = self.bot_process.wait()
                print(f"🏁 Bot exited with code: {exit_code}")
                return exit_code
                
        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C detected")
            self.request_shutdown("Ctrl+C")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.request_shutdown(f"Error: {e}")
        finally:
            # Ensure cleanup
            if self.bot_process and self.bot_process.poll() is None:
                self.request_shutdown("Cleanup")
        
        print("👋 Interactive launcher finished")
        return 0


def main():
    """Main entry point."""
    launcher = InteractiveBotLauncher()
    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
