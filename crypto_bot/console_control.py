from __future__ import annotations

"""Console utilities for starting and stopping the trading bot."""

import asyncio
import json
from typing import Dict, Any


import os

async def control_loop(state: Dict[str, Any]) -> None:
    """Listen for commands and update ``state`` accordingly."""
    
    # Check environment variables and stdin availability
    non_interactive = os.environ.get('NON_INTERACTIVE')
    auto_start = os.environ.get('AUTO_START_TRADING')
    
    # Check if stdin is available (for non-interactive detection)
    import sys
    stdin_available = hasattr(sys.stdin, 'isatty') and sys.stdin.isatty()
    
    # Auto-start trading in non-interactive mode, but keep control loop running
    if non_interactive or not stdin_available:
        state["running"] = True
        print("Auto-starting trading in non-interactive mode")
        print("Bot will respond to start/stop commands from frontend")
    
    print("Commands: start | stop | reload | quit")
    
    # File for frontend communication
    from crypto_bot.utils.logger import LOG_DIR
    control_file = LOG_DIR / "bot_control.json"
    
    try:
        while True:
            # In non-interactive mode, check for frontend commands
            if non_interactive or not stdin_available:
                # Check for frontend control commands
                if control_file.exists():
                    try:
                        with open(control_file, 'r') as f:
                            control_data = json.loads(f.read())
                        
                        cmd = control_data.get('command', '').strip().lower()
                        if cmd == "start":
                            state["running"] = True
                            print("Frontend command: Trading started")
                        elif cmd == "stop":
                            state["running"] = False
                            print("Frontend command: Trading stopped")
                        elif cmd == "reload":
                            state["reload"] = True
                            print("Frontend command: Reloading config")
                        
                        # Remove the command file after processing
                        control_file.unlink(missing_ok=True)
                        
                    except Exception as e:
                        print(f"Error reading control file: {e}")
                
                await asyncio.sleep(1)  # Check every second for frontend commands
                continue
            else:
                try:
                    cmd = (await asyncio.to_thread(input, "> ")).strip().lower()
                except (EOFError, OSError):
                    print("Stdin not available, switching to non-interactive mode")
                    non_interactive = True
                    stdin_available = False
                    continue
            if cmd == "start":
                state["running"] = True
                print("Trading started")
            elif cmd == "stop":
                state["running"] = False
                print("Trading stopped")
            elif cmd == "reload":
                state["reload"] = True
                print("Reloading config")
            elif cmd in {"quit", "exit"}:
                state["running"] = False
                break
    except asyncio.CancelledError:
        state["running"] = False
        raise

