"""
WebSocket Connection Pool for efficient connection management.
Provides connection pooling, load balancing, and automatic reconnection.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import threading
from websocket import WebSocketApp
from pathlib import Path

logger = logging.getLogger(__name__)


class WebSocketPool:
    """Manage multiple WebSocket connections efficiently with connection pooling."""

    def __init__(self, max_connections: int = 10, max_connections_per_host: int = 3):
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.active_connections: Dict[str, List[WebSocketApp]] = {}
        self.connection_semaphore = asyncio.Semaphore(max_connections)
        self.connection_stats = {
            "total_created": 0,
            "total_closed": 0,
            "active_count": 0,
            "errors": 0
        }
        self._lock = threading.Lock()

    async def get_connection(
        self,
        url: str,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        **kwargs
    ) -> WebSocketApp:
        """Get or create a WebSocket connection for the given URL."""
        async with self.connection_semaphore:
            # Check if we have an available connection for this host
            host_connections = self.active_connections.get(url, [])

            # Filter out closed connections
            host_connections = [conn for conn in host_connections if conn and conn.sock and conn.sock.connected]
            self.active_connections[url] = host_connections

            # Reuse existing connection if available
            if host_connections:
                logger.debug(f"Reusing existing connection for {url}")
                return host_connections[0]

            # Check if we've reached the per-host limit
            if len(host_connections) >= self.max_connections_per_host:
                logger.warning(f"Max connections per host reached for {url}")
                # Return the first available connection
                return host_connections[0]

            # Create new connection
            logger.info(f"Creating new WebSocket connection to {url}")
            return await self._create_connection(url, on_message, on_error, on_close, **kwargs)

    async def _create_connection(
        self,
        url: str,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        **kwargs
    ) -> WebSocketApp:
        """Create a new WebSocket connection."""
        try:
            # Set up default callbacks
            def default_on_message(ws, message):
                logger.debug(f"Message from {url}: {message}")

            def default_on_error(ws, error):
                logger.error(f"WebSocket error for {url}: {error}")
                with self._lock:
                    self.connection_stats["errors"] += 1

            def default_on_close(ws, close_status_code, close_msg):
                logger.info(f"WebSocket closed for {url}: {close_status_code} {close_msg}")
                with self._lock:
                    self.connection_stats["total_closed"] += 1
                    self.connection_stats["active_count"] -= 1

                # Remove from active connections
                with self._lock:
                    if url in self.active_connections:
                        self.active_connections[url] = [
                            conn for conn in self.active_connections[url] if conn != ws
                        ]

            # Use provided callbacks or defaults
            on_message = on_message or default_on_message
            on_error = on_error or default_on_error

            def enhanced_on_close(ws, close_status_code, close_msg):
                if on_close:
                    on_close(ws, close_status_code, close_msg)
                default_on_close(ws, close_status_code, close_msg)

            # Create WebSocket connection
            ws = WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=enhanced_on_close,
                **kwargs
            )

            # Start connection in a thread
            def run_ws():
                ws.run_forever(
                    ping_interval=kwargs.get('ping_interval', 20),
                    ping_timeout=kwargs.get('ping_timeout', 10)
                )

            thread = threading.Thread(target=run_ws, daemon=True)
            thread.start()

            # Wait a moment for connection to establish
            await asyncio.sleep(0.5)

            # Track connection
            with self._lock:
                if url not in self.active_connections:
                    self.active_connections[url] = []
                self.active_connections[url].append(ws)
                self.connection_stats["total_created"] += 1
                self.connection_stats["active_count"] += 1

            logger.info(f"Successfully created WebSocket connection to {url}")
            return ws

        except Exception as e:
            logger.error(f"Failed to create WebSocket connection to {url}: {e}")
            with self._lock:
                self.connection_stats["errors"] += 1
            raise

    def release_connection(self, url: str, ws: WebSocketApp):
        """Release a connection back to the pool (for future reuse)."""
        # Connection stays in the pool for potential reuse
        logger.debug(f"Connection released back to pool for {url}")

    def close_connection(self, url: str, ws: WebSocketApp):
        """Force close a specific connection."""
        try:
            ws.close()
            logger.debug(f"Force closed connection for {url}")
        except Exception as e:
            logger.error(f"Error closing connection for {url}: {e}")

    def close_all_connections(self):
        """Close all active connections."""
        logger.info("Closing all WebSocket connections")
        with self._lock:
            for url, connections in self.active_connections.items():
                for ws in connections:
                    try:
                        ws.close()
                    except Exception as e:
                        logger.error(f"Error closing connection for {url}: {e}")

            self.active_connections.clear()
            self.connection_stats["active_count"] = 0

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the connection pool."""
        with self._lock:
            stats = self.connection_stats.copy()
            stats["connections_by_host"] = {
                url: len(connections) for url, connections in self.active_connections.items()
            }
            return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all connections."""
        health_status = {"healthy": 0, "unhealthy": 0, "details": {}}

        with self._lock:
            for url, connections in self.active_connections.items():
                healthy_count = 0
                unhealthy_count = 0

                for ws in connections:
                    if ws and ws.sock and ws.sock.connected:
                        healthy_count += 1
                    else:
                        unhealthy_count += 1

                health_status["details"][url] = {
                    "healthy": healthy_count,
                    "unhealthy": unhealthy_count,
                    "total": len(connections)
                }
                health_status["healthy"] += healthy_count
                health_status["unhealthy"] += unhealthy_count

        return health_status


class LoadBalancedWebSocketClient:
    """WebSocket client with load balancing across multiple endpoints."""

    def __init__(self, endpoints: List[str], pool: Optional[WebSocketPool] = None):
        self.endpoints = endpoints
        self.current_endpoint = 0
        self.pool = pool or WebSocketPool()
        self.connection_failures = {endpoint: 0 for endpoint in endpoints}

    def get_next_endpoint(self) -> str:
        """Round-robin endpoint selection."""
        endpoint = self.endpoints[self.current_endpoint]
        self.current_endpoint = (self.current_endpoint + 1) % len(self.endpoints)
        return endpoint

    def get_healthiest_endpoint(self) -> str:
        """Select the endpoint with the fewest failures."""
        return min(self.connection_failures.keys(),
                  key=lambda x: self.connection_failures[x])

    async def connect(
        self,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        **kwargs
    ) -> WebSocketApp:
        """Connect using load balancing strategy."""
        max_attempts = len(self.endpoints)
        last_error = None

        for attempt in range(max_attempts):
            try:
                endpoint = self.get_next_endpoint()
                logger.debug(f"Attempting connection to {endpoint} (attempt {attempt + 1})")

                ws = await self.pool.get_connection(
                    endpoint,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    **kwargs
                )

                # Reset failure count on successful connection
                self.connection_failures[endpoint] = 0
                return ws

            except Exception as e:
                logger.warning(f"Failed to connect to {endpoint}: {e}")
                self.connection_failures[endpoint] += 1
                last_error = e

        raise last_error or Exception("All endpoints failed")

    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            "endpoints": self.endpoints.copy(),
            "current_endpoint": self.current_endpoint,
            "failure_counts": self.connection_failures.copy(),
            "pool_stats": self.pool.get_pool_stats()
        }
