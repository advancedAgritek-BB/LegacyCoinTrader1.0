#!/usr/bin/env python3
"""
Fixed Frontend Launcher with proper Python path setup
"""

import sys
import os
import time
import threading
import webbrowser
from pathlib import Path

# Ensure we're in the project root
project_root = Path(__file__).parent
os.chdir(project_root)

# Add the project root to the Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def start_frontend():
    """Start the frontend with proper environment"""
    print("🚀 Starting LegacyCoinTrader Frontend (Fixed)")
    print("=" * 50)

    try:
        # Import Flask app
        print("📦 Importing Flask app...")
        from frontend.app import app

        # Find an available port
        import socket
        def find_free_port(start_port=8000, max_attempts=10):
            for port in range(start_port, start_port + max_attempts):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('', port))
                        return port
                except OSError:
                    continue
            return start_port

        port = find_free_port()
        print(f"🔌 Using port: {port}")

        # Function to open browser after delay
        def open_browser():
            time.sleep(3)  # Wait for Flask to start
            try:
                url = f"http://localhost:{port}"
                print(f"🌐 Opening browser to: {url}")
                webbrowser.open(url)
                print("✅ Browser opened successfully")
            except Exception as e:
                print(f"⚠️ Could not open browser automatically: {e}")
                print(f"🌐 Please manually navigate to: http://localhost:{port}")

        # Start browser opening in background
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()

        # Start Flask
        print(f"🌐 Starting Flask on port {port}...")
        print(f"📊 Dashboard: http://localhost:{port}")
        print(f"📋 System logs: http://localhost:{port}/system_logs")
        print(f"🔧 Test endpoint: http://localhost:{port}/test")
        print("-" * 50)
        print("✅ Balance values should now display correctly!")
        print("-" * 50)
        print("Press Ctrl+C to stop the frontend")
        print("-" * 50)

        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_frontend()
