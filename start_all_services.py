#!/usr/bin/env python3
"""
Comprehensive Service Startup Script for LegacyCoinTrader

This script starts all required services in the correct order:
1. Enhanced Scanner Service
2. Strategy Router Service
3. WebSocket Monitor Service
4. Main Trading Bot
5. Flask Web Server
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path
from typing import List, Dict, Optional

import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "crypto_bot"))

class ServiceManager:
    """Manages startup and monitoring of all trading services."""

    def __init__(self):
        self.services: Dict[str, subprocess.Popen] = {}
        self.project_root = Path(__file__).parent
        self.venv_python = self.project_root / "modern_trader_env" / "bin" / "python"

        if not self.venv_python.exists():
            self.venv_python = self.project_root / "venv" / "bin" / "python"

    def start_service(self, name: str, command: List[str], cwd: Optional[Path] = None) -> bool:
        """Start a service and track it."""
        try:
            print(f"🚀 Starting {name}...")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "crypto_bot")

            process = subprocess.Popen(
                command,
                cwd=cwd or self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self.services[name] = process
            print(f"✅ {name} started (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False

    def start_enhanced_scanner(self) -> bool:
        """Start the enhanced scanner service."""
        scanner_script = self.project_root / "crypto_bot" / "enhanced_scanner_service.py"

        # Create the scanner service script if it doesn't exist
        if not scanner_script.exists():
            self._create_enhanced_scanner_service(scanner_script)

        command = [
            str(self.venv_python),
            str(scanner_script)
        ]
        return self.start_service("Enhanced Scanner", command)

    def start_strategy_router(self) -> bool:
        """Start the strategy router service."""
        router_script = self.project_root / "crypto_bot" / "strategy_router_service.py"

        # Create the strategy router service script if it doesn't exist
        if not router_script.exists():
            self._create_strategy_router_service(router_script)

        command = [
            str(self.venv_python),
            str(router_script)
        ]
        return self.start_service("Strategy Router", command)

    def start_websocket_monitor(self) -> bool:
        """Start the WebSocket monitor service."""
        monitor_script = self.project_root / "crypto_bot" / "websocket_monitor_service.py"

        # Create the WebSocket monitor service script if it doesn't exist
        if not monitor_script.exists():
            self._create_websocket_monitor_service(monitor_script)

        command = [
            str(self.venv_python),
            str(monitor_script)
        ]
        return self.start_service("WebSocket Monitor", command)

    def _pump_sniper_enabled(self) -> bool:
        """Return True if the pump sniper orchestrator is enabled in configuration."""

        config_path = self.project_root / "config" / "pump_sniper_config.yaml"
        if not config_path.exists():
            return False

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception as exc:
            print(f"⚠️ Unable to read pump sniper config: {exc}")
            return False

        orchestrator_cfg = data.get("pump_sniper_orchestrator", {})
        return bool(orchestrator_cfg.get("enabled", False))

    def _pump_live_requested(self) -> bool:
        value = os.environ.get("PUMP_SNIPER_LIVE", "0").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def start_pump_sniper(self) -> bool:
        """Start the pump sniper runtime if enabled."""

        if not self._pump_sniper_enabled():
            print("ℹ️ Pump Sniper runtime disabled or not configured; skipping startup")
            return False

        script_path = self.project_root / "start_pump_sniper.py"
        if not script_path.exists():
            print(f"⚠️ Pump Sniper script missing: {script_path} (skipping)")
            return False

        command = [
            str(self.venv_python),
            str(script_path),
            "--config",
            str(self.project_root / "config.yaml"),
        ]

        if self._pump_live_requested():
            command.append("--live")

        return self.start_service("Pump Sniper Runtime", command)

    def start_main_bot(self) -> bool:
        """Start the main trading bot."""
        command = [
            str(self.venv_python),
            "-m",
            "crypto_bot.main"
        ]
        return self.start_service("Main Trading Bot", command)

    def start_flask_server(self) -> bool:
        """Start the Flask web server."""
        command = [
            str(self.venv_python),
            "-m",
            "flask",
            "run",
            "--host=0.0.0.0",
            "--port=8001"
        ]
        env = os.environ.copy()
        env["FLASK_APP"] = str(self.project_root / "frontend" / "app.py")
        env["PYTHONPATH"] = str(self.project_root / "crypto_bot")

        try:
            print("🚀 Starting Flask Web Server...")
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.services["Flask Web Server"] = process
            print(f"✅ Flask Web Server started (PID: {process.pid})")
            return True
        except Exception as e:
            print(f"❌ Failed to start Flask Web Server: {e}")
            return False

    def _create_enhanced_scanner_service(self, script_path: Path):
        """Create the enhanced scanner service script."""
        script_content = '''#!/usr/bin/env python3
"""
Enhanced Scanner Service
Continuously scans for new trading opportunities and updates the pipeline.
"""

import time
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.enhanced_scanner import EnhancedScanner
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Main scanner service loop."""
    logger.info("🚀 Starting Enhanced Scanner Service")

    try:
        scanner = EnhancedScanner()

        while True:
            try:
                # Perform scan
                logger.info("🔍 Performing enhanced scan...")
                results = scanner.scan_tokens()

                if results:
                    logger.info(f"✅ Scan completed: {len(results)} tokens found")
                else:
                    logger.warning("⚠️ Scan returned no tokens")

                # Sleep before next scan
                time.sleep(30)  # Scan every 30 seconds

            except Exception as e:
                logger.error(f"❌ Scanner error: {e}")
                time.sleep(60)  # Wait longer on error

    except KeyboardInterrupt:
        logger.info("🛑 Scanner service stopped by user")
    except Exception as e:
        logger.error(f"❌ Scanner service failed: {e}")

if __name__ == "__main__":
    main()
'''
        script_path.write_text(script_content)
        script_path.chmod(0o755)

    def _create_strategy_router_service(self, script_path: Path):
        """Create the strategy router service script."""
        script_content = '''#!/usr/bin/env python3
"""
Strategy Router Service
Routes trading signals to appropriate strategies based on market conditions.
"""

import time
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_bot.strategies.strategy_router import StrategyRouter
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Main strategy router service loop."""
    logger.info("🚀 Starting Strategy Router Service")

    try:
        router = StrategyRouter()

        while True:
            try:
                # Process routing decisions
                logger.info("🎯 Processing strategy routing...")
                results = router.process_routing()

                if results:
                    logger.info(f"✅ Routing completed: {len(results)} decisions made")
                else:
                    logger.info("ℹ️ No routing decisions needed")

                # Sleep before next routing cycle
                time.sleep(15)  # Route every 15 seconds

            except Exception as e:
                logger.error(f"❌ Router error: {e}")
                time.sleep(30)  # Wait longer on error

    except KeyboardInterrupt:
        logger.info("🛑 Strategy router service stopped by user")
    except Exception as e:
        logger.error(f"❌ Strategy router service failed: {e}")

if __name__ == "__main__":
    main()
'''
        script_path.write_text(script_content)
        script_path.chmod(0o755)

    def _create_websocket_monitor_service(self, script_path: Path):
        """Create the WebSocket monitor service script."""
        script_content = '''#!/usr/bin/env python3
"""
WebSocket Monitor Service
Monitors WebSocket connections and handles real-time data streaming.
"""

import time
import sys
import asyncio
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_bot.execution.kraken_ws import KrakenWSClient
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__)

class WebSocketMonitorService:
    """Monitor and manage WebSocket connections."""

    def __init__(self):
        self.ws_client = None
        self.is_running = False

    async def monitor_connections(self):
        """Monitor WebSocket connections."""
        logger.info("🔌 Starting WebSocket monitoring...")

        while self.is_running:
            try:
                if not self.ws_client:
                    logger.info("🔗 Initializing WebSocket client...")
                    self.ws_client = KrakenWSClient()

                # Test connection health
                if hasattr(self.ws_client, 'test_connection'):
                    is_healthy = await self.ws_client.test_connection()
                    if is_healthy:
                        logger.debug("✅ WebSocket connection healthy")
                    else:
                        logger.warning("⚠️ WebSocket connection unhealthy")
                        # Reinitialize connection
                        self.ws_client = None

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"❌ WebSocket monitoring error: {e}")
                self.ws_client = None
                await asyncio.sleep(60)

    def start(self):
        """Start the WebSocket monitor service."""
        self.is_running = True
        logger.info("🚀 WebSocket Monitor Service starting...")

        try:
            asyncio.run(self.monitor_connections())
        except KeyboardInterrupt:
            logger.info("🛑 WebSocket monitor stopped by user")
        except Exception as e:
            logger.error(f"❌ WebSocket monitor failed: {e}")
        finally:
            self.is_running = False

def main():
    """Main WebSocket monitor service."""
    service = WebSocketMonitorService()
    service.start()

if __name__ == "__main__":
    main()
'''
        script_path.write_text(script_content)
        script_path.chmod(0o755)

    def wait_for_service(self, name: str, timeout: int = 10) -> bool:
        """Wait for a service to be ready."""
        print(f"⏳ Waiting for {name} to be ready...")
        time.sleep(timeout)
        return True  # Simple implementation

    def start_all_services(self):
        """Start all services in the correct order."""
        print("🚀 Starting LegacyCoinTrader Services")
        print("=" * 50)

        # Start services in order
        services_started = []

        # 1. Enhanced Scanner
        if self.start_enhanced_scanner():
            services_started.append("Enhanced Scanner")
            self.wait_for_service("Enhanced Scanner")

        # 2. Strategy Router
        if self.start_strategy_router():
            services_started.append("Strategy Router")
            self.wait_for_service("Strategy Router")

        # 3. WebSocket Monitor
        if self.start_websocket_monitor():
            services_started.append("WebSocket Monitor")
            self.wait_for_service("WebSocket Monitor")

        # 4. Pump Sniper Runtime
        pump_started = self.start_pump_sniper()
        if pump_started:
            services_started.append("Pump Sniper Runtime")
            self.wait_for_service("Pump Sniper Runtime")

        # 5. Main Trading Bot
        if self.start_main_bot():
            services_started.append("Main Trading Bot")
            self.wait_for_service("Main Trading Bot")

        # 6. Flask Web Server
        if self.start_flask_server():
            services_started.append("Flask Web Server")

        print("\n" + "=" * 50)
        print("🎉 Service Startup Complete!")
        print(f"✅ Services started: {len(services_started)}")
        for service in services_started:
            print(f"  • {service}")

        if services_started:
            print("
🌐 Dashboard: http://localhost:8001"            print("📊 Health Check: http://localhost:8001/api/monitoring/health"
        return len(services_started) > 0

    def stop_all_services(self, use_comprehensive_shutdown: bool = True):
        """Stop all running services."""
        if use_comprehensive_shutdown:
            print("🛑 Using comprehensive shutdown system...")
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, 
                    str(self.project_root / "safe_shutdown.py")
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print("✅ Comprehensive shutdown completed successfully")
                    self.services.clear()
                    return
                else:
                    print(f"⚠️ Comprehensive shutdown failed: {result.stderr}")
                    print("🔄 Falling back to basic shutdown...")
            except Exception as e:
                print(f"⚠️ Comprehensive shutdown error: {e}")
                print("🔄 Falling back to basic shutdown...")
        
        # Fallback to basic shutdown
        print("🛑 Stopping all services...")

        for name, process in self.services.items():
            try:
                print(f"Stopping {name} (PID: {process.pid})...")
                process.terminate()

                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    print(f"✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"⚠️ {name} didn't stop gracefully, force killing...")
                    process.kill()
                    process.wait()
                    print(f"💀 {name} force killed")

            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")

        self.services.clear()
        print("✅ All services stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n🛑 Shutdown signal received")
    manager.stop_all_services()
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create service manager
    manager = ServiceManager()

    try:
        # Start all services
        success = manager.start_all_services()

        if success:
            print("\n🔄 Services are running. Press Ctrl+C to stop all services.")
            # Keep main thread alive
            while True:
                time.sleep(1)
        else:
            print("❌ Failed to start services")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        manager.stop_all_services()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        manager.stop_all_services()
        sys.exit(1)
