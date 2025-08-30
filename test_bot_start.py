#!/usr/bin/env python3
"""Test script to check if the bot can start properly."""

import subprocess
import sys
import os
from pathlib import Path

def test_bot_start():
    """Test if the bot can start properly."""
    print("Testing bot startup...")
    
    # Get the project root
    project_root = Path(__file__).parent
    venv_python = project_root / 'venv' / 'bin' / 'python3'
    bot_script = project_root / 'start_bot_noninteractive.py'
    
    print(f"Project root: {project_root}")
    print(f"Python executable: {venv_python}")
    print(f"Bot script: {bot_script}")
    
    # Check if files exist
    if not venv_python.exists():
        print(f"ERROR: Python executable not found: {venv_python}")
        return False
    
    if not bot_script.exists():
        print(f"ERROR: Bot script not found: {bot_script}")
        return False
    
    print("Files found, attempting to start bot...")
    
    try:
        # Start the bot process
        process = subprocess.Popen(
            [str(venv_python), str(bot_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a few seconds to see if it starts
        import time
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("SUCCESS: Bot process started and is running")
            process.terminate()
            process.wait()
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"ERROR: Bot process failed to start")
            print(f"Return code: {process.returncode}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"ERROR: Exception starting bot: {e}")
        return False

if __name__ == "__main__":
    success = test_bot_start()
    sys.exit(0 if success else 1)
