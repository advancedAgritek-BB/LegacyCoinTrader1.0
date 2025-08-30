#!/usr/bin/env python3
"""
LegacyCoinTrader Launcher with Automatic Browser Opening
This script starts the application and automatically opens a browser to the frontend.
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser
from pathlib import Path

def check_venv():
    """Check if virtual environment exists and activate it."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run './startup.sh setup' first.")
        sys.exit(1)
    
    # Activate virtual environment
    if sys.platform == "win32":
        activate_script = venv_path / "Scripts" / "activate.bat"
        os.environ["VIRTUAL_ENV"] = str(venv_path.absolute())
        os.environ["PATH"] = f"{venv_path / 'Scripts'};{os.environ['PATH']}"
    else:
        activate_script = venv_path / "bin" / "activate"
        os.environ["VIRTUAL_ENV"] = str(venv_path.absolute())
        os.environ["PATH"] = f"{venv_path / 'bin'}:{os.environ['PATH']}"
    
    print("✅ Virtual environment activated")

def check_env():
    """Check if .env file exists with real API keys."""
    env_paths = [".env", "crypto_bot/.env"]
    env_found = False
    
    for env_path in env_paths:
        if Path(env_path).exists():
            print(f"✅ Found .env file at: {env_path}")
            # Check if it contains real API keys (not template values)
            with open(env_path, 'r') as f:
                content = f.read()
                if "your_kraken_api_key_here" in content or "your_telegram_token_here" in content or "your_helius_key_here" in content:
                    print("❌ .env file contains template values. Please edit with real API keys.")
                    sys.exit(1)
                else:
                    print("✅ .env file contains real API keys")
                    env_found = True
                    break
    
    if not env_found:
        print("❌ No valid .env file found. Please run './startup.sh setup' first.")
        sys.exit(1)

def open_browser_delayed(url, delay=3):
    """Open browser after a delay to ensure server is running."""
    def _open_browser():
        print(f"⏳ Waiting {delay} seconds for server to start...")
        time.sleep(delay)
        print("🌐 Opening browser...")
        try:
            webbrowser.open(url)
            print("✅ Browser opened successfully")
        except Exception as e:
            print(f"⚠️  Could not automatically open browser: {e}")
            print(f"Please manually navigate to: {url}")
    
    thread = threading.Thread(target=_open_browser, daemon=True)
    thread.start()

def start_services():
    """Start all the required services."""
    print("🚀 Starting LegacyCoinTrader...")
    
    # Start main application
    print("📊 Starting main trading bot...")
    main_proc = subprocess.Popen([sys.executable, "-m", "crypto_bot.main"])
    
    # Start web frontend
    print("🌐 Starting web dashboard...")
    frontend_proc = subprocess.Popen([sys.executable, "-m", "frontend.app"])
    
    # Start Telegram bot
    print("📱 Starting Telegram bot...")
    telegram_proc = subprocess.Popen([sys.executable, "telegram_ctl.py"])
    
    print("")
    print("🎉 LegacyCoinTrader is now running!")
    print("📊 Main bot PID:", main_proc.pid)
    print("🌐 Web dashboard: http://localhost:8000")
    print("📱 Telegram bot PID:", telegram_proc.pid)
    print("")
    
    # Open browser after a delay
    open_browser_delayed("http://localhost:8000", 3)
    
    print("Press Ctrl+C to stop all services")
    
    try:
        # Wait for all processes
        main_proc.wait()
        frontend_proc.wait()
        telegram_proc.wait()
    except KeyboardInterrupt:
        print("")
        print("🛑 Stopping LegacyCoinTrader...")
        main_proc.terminate()
        frontend_proc.terminate()
        telegram_proc.terminate()
        
        # Wait for processes to terminate gracefully
        try:
            main_proc.wait(timeout=5)
            frontend_proc.wait(timeout=5)
            telegram_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Some processes didn't terminate gracefully, forcing...")
            main_proc.kill()
            frontend_proc.kill()
            telegram_proc.kill()
        
        print("✅ All services stopped")

def main():
    """Main entry point."""
    print("🚀 LegacyCoinTrader Launcher with Browser")
    print("==========================================")
    
    # Check prerequisites
    check_venv()
    check_env()
    
    # Start services
    start_services()

if __name__ == "__main__":
    main()
