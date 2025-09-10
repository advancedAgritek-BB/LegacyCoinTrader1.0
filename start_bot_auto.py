#!/usr/bin/env python3
"""
Integrated bot startup script that combines trading bot and monitoring dashboard
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

        # Import Flask app from frontend
        try:
            from frontend.app import app
            print("✅ Flask app imported successfully")
        except Exception as e:
            print(f"❌ Failed to import Flask app: {e}")
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
        print(f"✅ Found free port: {port}")

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
        flask_thread = threading.Thread(target=run_flask, args=(port,), daemon=True)
        flask_thread.start()
        print("✅ Flask thread started")

        # Wait for Flask to start (but don't block indefinitely)
        print("⏳ Waiting for web server to initialize...")
        time.sleep(3)
        
        # Check if Flask is running by trying to connect
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:
                print(f"✅ Web server confirmed running on port {port}")
            else:
                print(f"⚠️ Web server may not be running on port {port}")
        except Exception as e:
            print(f"⚠️ Could not verify web server: {e}")

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
    print("🚀 Starting LegacyCoinTrader - Integrated Edition")
    print("=" * 60)
    print("🤖 Trading Bot + 📊 Monitoring Dashboard + 🌐 Web Server + 📈 OHLCV Fetching")
    print("=" * 60)

    # Step 1: Initialize OHLCV cache first
    print("Step 1: Initializing OHLCV data cache...")
    try:
        from crypto_bot.utils.market_loader import update_multi_tf_ohlcv_cache, load_kraken_symbols
        from dotenv import dotenv_values
        import ccxt
        import os

        print("Loading environment variables...")
        secrets = dotenv_values('.env')
        if not secrets:
            # Try loading from crypto_bot directory
            secrets = dotenv_values('crypto_bot/.env')
        os.environ.update(secrets)

        # Fix environment variable mapping for compatibility
        if 'KRAKEN_API_KEY' in secrets and 'API_KEY' not in os.environ:
            os.environ['API_KEY'] = secrets['KRAKEN_API_KEY']
        if 'KRAKEN_API_SECRET' in secrets and 'API_SECRET' not in os.environ:
            os.environ['API_SECRET'] = secrets['KRAKEN_API_SECRET']

        print("Setting up exchange connection...")
        exchange = ccxt.kraken({
            'apiKey': os.environ.get('API_KEY') or secrets.get('KRAKEN_API_KEY'),
            'secret': os.environ.get('API_SECRET') or secrets.get('KRAKEN_API_SECRET'),
        })

        print("Loading trading symbols...")
        symbols = await load_kraken_symbols(exchange, [], {})
        if symbols:
            print(f"✅ Found {len(symbols)} trading symbols")
            print("Initializing OHLCV cache for top symbols...")
            # Create a proper config dict for the cache function with production settings
            cache_config = {
                "timeframes": ['5m', '1h'],
                "ohlcv_timeout": 120,
                "max_ohlcv_failures": 3,
                "production_mode": True,  # Enable enhanced symbol validation
                "symbol_validation": {
                    "filter_invalid_symbols": True,
                    "min_liquidity_score": 0.6,
                    "min_volume_usd": 10000,
                    "strict_mode": True
                }
            }
            await update_multi_tf_ohlcv_cache(exchange, symbols[:20], cache_config)
            print("✅ OHLCV cache initialized successfully")
        else:
            print("⚠️ No symbols found, cache initialization skipped")

    except Exception as e:
        print(f"⚠️ OHLCV cache initialization failed (continuing): {e}")
        import traceback
        traceback.print_exc()

    # Step 2: Start web server
    print("Step 2: Starting web dashboard server...")
    web_thread = start_web_server()

    if web_thread is None:
        print("❌ Web server failed to start, but continuing with bot...")
    else:
        print("✅ Web server started successfully")

    # Give web server more time to start
    print("Step 3: Waiting for web server to fully initialize...")
    await asyncio.sleep(5)

    # Step 4: Start trading bot
    try:
        print("Step 4: Starting trading bot with integrated OHLCV fetching...")
        from crypto_bot.main import _main_impl

        print("🎯 Starting trading bot with integrated monitoring...")
        print("📊 OHLCV fetching will run continuously as part of trading cycles")
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
        # Set production environment variables
        os.environ['AUTO_START_TRADING'] = '1'
        os.environ['NON_INTERACTIVE'] = '1'
        os.environ['PRODUCTION'] = 'true'
        os.environ['FLASK_ENV'] = 'production'
        os.environ['LOG_LEVEL'] = 'INFO'

        # Production monitoring settings
        os.environ['ENABLE_METRICS'] = 'true'
        os.environ['ENABLE_POSITION_SYNC'] = 'true'
        os.environ['ENABLE_MEMORY_MANAGEMENT'] = 'true'

        print("🚀 Starting LegacyCoinTrader in PRODUCTION MODE")
        print("=" * 60)
        print("📊 Production Features Enabled:")
        print("  • Enhanced Symbol Validation")
        print("  • Production Memory Management")
        print("  • Position Synchronization")
        print("  • Circuit Breaker Protection")
        print("  • Production Monitoring")
        print("=" * 60)

        # Run the integrated system
        asyncio.run(start_integrated_bot())

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
