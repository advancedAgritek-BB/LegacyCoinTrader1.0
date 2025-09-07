#!/usr/bin/env python3
"""
Simple WebSocket Monitor Fix for LegacyCoinTrader

This script monitors WebSocket connections and provides real-time status
to the health monitoring system.
"""

import time
import json
import websocket
import threading
from pathlib import Path
from typing import Dict, Any

class WebSocketMonitor:
    """Monitor WebSocket connections for the trading system."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.log_dir = self.project_root / "crypto_bot" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.monitoring_file = self.log_dir / "websocket_monitoring.json"
        self.status = {
            "websocket_active": False,
            "last_connection_attempt": None,
            "last_successful_connection": None,
            "connection_attempts": 0,
            "successful_connections": 0,
            "current_status": "disconnected",
            "error_message": None
        }

        self.running = False
        self.monitor_thread = None

    def test_kraken_connection(self) -> bool:
        """Test connection to Kraken WebSocket."""
        try:
            self.status["connection_attempts"] += 1
            self.status["last_connection_attempt"] = time.time()

            # Create WebSocket connection
            ws = websocket.create_connection(
                "wss://ws.kraken.com/v2",
                timeout=10
            )

            # Send ping
            ping_msg = {"method": "ping"}
            ws.send(json.dumps(ping_msg))

            # Wait for response
            response = ws.recv()
            ws.close()

            # Check response
            if "pong" in response.lower() or "status" in response.lower():
                self.status["successful_connections"] += 1
                self.status["last_successful_connection"] = time.time()
                self.status["websocket_active"] = True
                self.status["current_status"] = "connected"
                self.status["error_message"] = None
                return True
            else:
                self.status["websocket_active"] = False
                self.status["current_status"] = "protocol_error"
                self.status["error_message"] = f"Unexpected response: {response}"
                return False

        except Exception as e:
            self.status["websocket_active"] = False
            self.status["current_status"] = "connection_failed"
            self.status["error_message"] = str(e)
            return False

    def save_status(self):
        """Save current status to file for health monitoring."""
        status_copy = self.status.copy()
        status_copy["timestamp"] = time.time()

        try:
            with open(self.monitoring_file, 'w') as f:
                json.dump(status_copy, f, indent=2)
        except Exception as e:
            print(f"Error saving WebSocket status: {e}")

    def monitor_loop(self):
        """Main monitoring loop."""
        print("🔌 WebSocket Monitor started")

        while self.running:
            try:
                # Test connection
                self.test_kraken_connection()

                # Save status for health monitoring
                self.save_status()

                # Log status
                if self.status["websocket_active"]:
                    print("✅ WebSocket: Connected")
                else:
                    print(f"⚠️ WebSocket: {self.status['current_status']} - {self.status['error_message']}")

                # Wait before next check
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"❌ WebSocket monitor error: {e}")
                time.sleep(60)  # Wait longer on error

        print("🛑 WebSocket Monitor stopped")

    def start(self):
        """Start the WebSocket monitor."""
        if self.running:
            print("WebSocket monitor already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

        print("🚀 WebSocket Monitor service started")

    def stop(self):
        """Stop the WebSocket monitor."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        # Final status save
        self.save_status()
        print("🛑 WebSocket Monitor service stopped")

def main():
    """Main function."""
    monitor = WebSocketMonitor()

    try:
        monitor.start()

        # Keep running until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal")
        monitor.stop()
    except Exception as e:
        print(f"❌ WebSocket monitor crashed: {e}")
        monitor.stop()

if __name__ == "__main__":
    main()
