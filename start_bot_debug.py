#!/usr/bin/env python3
"""
Debug version of the integrated bot startup script
"""

import asyncio
import threading
import time
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def start_web_server():
    """Start the Flask web server in a separate thread"""
    try:
        print("🌐 Starting integrated web server...")
        print("   Step 1: Importing Flask app...")

        # Import Flask app from frontend
        try:
            from frontend.app import app
            print("   ✅ Flask app imported successfully")
        except Exception as e:
            print(f"   ❌ Failed to import Flask app: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Import the main function to initialize monitoring
        try:
            from crypto_bot.main import _main_impl
            print("   ✅ Main bot function imported successfully")
        except Exception as e:
            print(f"   ❌ Failed to import main bot function: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Find an available port first
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
        print(f"   ✅ Found free port: {port}")

        # Start Flask in a separate thread with the port
        def run_flask(port_num):
            try:
                print(f"🌐 Web server running on http://localhost:{port_num}")
                print(f"📊 Monitoring dashboard: http://localhost:{port_num}/monitoring")
                print(f"📋 System logs: http://localhost:{port_num}/system_logs")
                print(f"🏠 Main dashboard: http://localhost:{port_num}")
                print("-" * 60)

                app.run(host='0.0.0.0', port=port_num, debug=False, use_reloader=False)

            except Exception as e:
                print(f"❌ Web server error: {e}")
                import traceback
                traceback.print_exc()

        # Start Flask in background thread
        print("   Step 2: Starting Flask thread...")
        flask_thread = threading.Thread(target=run_flask, args=(port,), daemon=True)
        flask_thread.start()
        print("   ✅ Flask thread started")

        print("✅ Web server started successfully")

        # Wait a moment for Flask to start
        print("   Step 3: Waiting for Flask to initialize...")
        time.sleep(3)

        # Try to open browser
        try:
            import webbrowser
            url = f"http://localhost:{port}"
            print(f"🌐 Opening browser to: {url}")
            webbrowser.open(url)
            print("✅ Browser opened successfully")
        except Exception as e:
            print(f"⚠️ Could not open browser automatically: {e}")
            print(f"🌐 Please manually navigate to: http://localhost:{port}")

        return flask_thread

    except Exception as e:
        print(f"⚠️ Failed to start web server: {e}")
        import traceback
        traceback.print_exc()
        return None

async def start_integrated_bot():
    """Start the complete integrated system"""
    print("🚀 Starting LegacyCoinTrader - Integrated Edition (DEBUG)")
    print("=" * 60)
    print("🤖 Trading Bot + 📊 Monitoring Dashboard + 🌐 Web Server")
    print("=" * 60)

    # Start web server first
    print("Step 1: Starting web server...")
    web_thread = start_web_server()

    if web_thread is None:
        print("❌ Web server failed to start, but continuing with bot...")
    else:
        print("✅ Web server started successfully")

    # Give web server time to start
    print("Step 2: Waiting for web server to initialize...")
    await asyncio.sleep(2)

    try:
        # Import and run the main bot function
        print("Step 3: Starting trading bot...")
        from crypto_bot.main import _main_impl

        print("🎯 Starting trading bot with integrated monitoring...")
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

    print("🛑 Shutting down integrated system...")
    print("✅ Shutdown complete")

if __name__ == "__main__":
    try:
        # Set environment variable to auto-start trading
        os.environ['AUTO_START_TRADING'] = '1'
        os.environ['NON_INTERACTIVE'] = '1'

        # Run the integrated system
        asyncio.run(start_integrated_bot())

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
