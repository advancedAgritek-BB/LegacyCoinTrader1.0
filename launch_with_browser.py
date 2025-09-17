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
            with open(env_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "MANAGED:" in content:
                    print("❌ .env file contains managed placeholders. Populate secrets before launch.")
                    sys.exit(1)
                else:
                    print("✅ .env file contains resolved secrets")
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

    # Start web frontend and capture its port
    print("🌐 Starting web dashboard...")
    import tempfile
    import time

    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    temp_file.close()

    frontend_proc = subprocess.Popen([sys.executable, "-m", "frontend.app"],
                                   stdout=open(temp_file.name, 'w'),
                                   stderr=subprocess.STDOUT)

    # Wait for Flask to start and write port info
    time.sleep(3)

    # Extract the port from the Flask output
    flask_port = 8000  # default fallback
    try:
        with open(temp_file.name, 'r') as f:
            content = f.read()
            # Look for FLASK_PORT= line
            for line in content.split('\n'):
                if line.startswith('FLASK_PORT='):
                    port_str = line.strip().split('=')[1]
                    flask_port = int(port_str)
                    print(f"✅ Detected Flask port: {flask_port}")
                    break
            else:
                # Try alternative pattern matching
                import re
                port_match = re.search(r'FLASK_PORT=(\d+)', content)
                if port_match:
                    flask_port = int(port_match.group(1))
                    print(f"✅ Detected Flask port via regex: {flask_port}")
                else:
                    print("⚠️ Could not detect Flask port, using default 8000")
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"⚠️ Error reading port from Flask output: {e}, using default 8000")
        flask_port = 8000  # fallback

    # Clean up temp file
    try:
        os.unlink(temp_file.name)
    except OSError:
        pass

    # Start Telegram bot
    print("📱 Starting Telegram bot...")
    telegram_proc = subprocess.Popen([sys.executable, "telegram_ctl.py"])

    print("")
    print("🎉 LegacyCoinTrader is now running!")
    print("📊 Main bot PID:", main_proc.pid)
    print(f"🌐 Web dashboard: http://localhost:{flask_port}")
    print("📱 Telegram bot PID:", telegram_proc.pid)
    print("")

    # Open browser after a delay
    open_browser_delayed(f"http://localhost:{flask_port}", 2)
    
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
