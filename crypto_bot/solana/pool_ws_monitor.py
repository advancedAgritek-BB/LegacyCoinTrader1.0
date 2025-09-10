from __future__ import annotations

import aiohttp
import asyncio
import json
import logging
from typing import AsyncGenerator, Any, Dict, Optional
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class EnhancedPoolMonitor:
    """Enhanced Solana pool WebSocket monitor with robust error handling and reconnection."""

    def __init__(self, api_key: str, pool_program: str, config: Optional[Dict[str, Any]] = None):
        self.api_key = api_key
        self.pool_program = pool_program
        self.config = config or self._get_default_config()

        # Connection parameters
        self.url = f"wss://atlas-mainnet.helius-rpc.com/?api-key={api_key}"
        self.reconnect_delay = self.config["reconnect"]["base_delay"]
        self.max_reconnect_delay = self.config["reconnect"]["max_delay"]
        self.backoff_factor = self.config["reconnect"]["backoff_factor"]
        self.max_attempts = self.config["reconnect"]["max_attempts"]

        # Connection state
        self.connection_attempts = 0
        self.last_connection_time = None
        self.is_connected = False
        self.total_messages_received = 0
        self.total_errors = 0
        self.start_time = datetime.now()

        # Subscription parameters
        self.subscription_id = 420
        self.subscription_params = {
            "failed": False,
            "accountInclude": [pool_program],
        }
        self.encoding_options = {
            "commitment": "processed",
            "encoding": "base64",
            "transactionDetails": "full",
            "showRewards": True,
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the monitor."""
        return {
            "reconnect": {
                "base_delay": 1.0,
                "max_delay": 60.0,
                "backoff_factor": 2.0,
                "max_attempts": 10,
            },
            "health_check": {
                "enabled": True,
                "interval": 30,
            },
            "message_validation": {
                "max_message_size": 10 * 1024 * 1024,  # 10MB
                "validate_json": True,
            },
            "logging": {
                "level": "INFO",
                "log_connection_events": True,
                "log_message_stats": True,
            }
        }

    def _calculate_reconnect_delay(self) -> float:
        """Calculate exponential backoff delay for reconnection."""
        delay = min(
            self.reconnect_delay * (self.backoff_factor ** self.connection_attempts),
            self.max_reconnect_delay
        )
        return delay

    def _should_attempt_reconnect(self) -> bool:
        """Determine if reconnection should be attempted."""
        if self.max_attempts == 0:  # Unlimited attempts
            return True
        return self.connection_attempts < self.max_attempts

    def _validate_message(self, message: str) -> bool:
        """Validate incoming WebSocket message."""
        try:
            # Check message size
            if len(message) > self.config["message_validation"]["max_message_size"]:
                logger.warning(f"Message too large: {len(message)} bytes")
                return False

            if self.config["message_validation"]["validate_json"]:
                json.loads(message)  # Validate JSON structure

            return True
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received in message")
            return False
        except Exception as e:
            logger.warning(f"Message validation error: {e}")
            return False

    def _log_connection_stats(self):
        """Log connection statistics."""
        if not self.config["logging"]["log_message_stats"]:
            return

        uptime = datetime.now() - self.start_time
        logger.info(
            f"Pool monitor stats - Uptime: {uptime}, "
            f"Messages: {self.total_messages_received}, "
            f"Errors: {self.total_errors}, "
            f"Reconnects: {self.connection_attempts}"
        )

    async def _create_subscription_message(self) -> Dict[str, Any]:
        """Create the subscription message for the WebSocket."""
        return {
            "jsonrpc": "2.0",
            "id": self.subscription_id,
            "method": "transactionSubscribe",
            "params": [
                self.subscription_params,
                self.encoding_options,
            ],
        }

    async def _handle_connection_success(self):
        """Handle successful connection establishment."""
        self.connection_attempts = 0  # Reset attempt counter
        self.last_connection_time = datetime.now()
        self.is_connected = True

        if self.config["logging"]["log_connection_events"]:
            logger.info(f"Successfully connected to Helius WebSocket for pool {self.pool_program}")

    async def _handle_connection_failure(self, error: Exception):
        """Handle connection failure with exponential backoff."""
        self.is_connected = False
        self.total_errors += 1

        if not self._should_attempt_reconnect():
            logger.error(f"Max reconnection attempts ({self.max_attempts}) reached. Giving up.")
            raise error

        delay = self._calculate_reconnect_delay()
        self.connection_attempts += 1

        logger.warning(
            f"WebSocket connection failed (attempt {self.connection_attempts}/{self.max_attempts}): {error}. "
            f"Retrying in {delay:.1f} seconds..."
        )

        await asyncio.sleep(delay)

    async def _health_check_loop(self):
        """Periodic health check for the connection."""
        if not self.config["health_check"]["enabled"]:
            return

        while True:
            try:
                await asyncio.sleep(self.config["health_check"]["interval"])
                if self.is_connected:
                    self._log_connection_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def watch_pool(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Connect to Helius enhanced websocket and yield transaction data with robust error handling."""

        health_check_task = None

        try:
            # Start health check task
            if self.config["health_check"]["enabled"]:
                health_check_task = asyncio.create_task(self._health_check_loop())

            while True:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.ws_connect(
                            self.url,
                            heartbeat=self.config.get("heartbeat", 30),
                            timeout=self.config.get("timeout", 30)
                        ) as ws:
                            await self._handle_connection_success()

                            # Send subscription message
                            sub_msg = await self._create_subscription_message()
                            await ws.send_json(sub_msg)

                            logger.info(f"Subscribed to pool {self.pool_program} transactions")

                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    try:
                                        # Validate message
                                        if not self._validate_message(msg.data):
                                            continue

                                        try:
                                            data = msg.json()
                                        except Exception:
                                            data = json.loads(msg.data)

                                        self.total_messages_received += 1

                                        result = None
                                        if isinstance(data, dict):
                                            result = data.get("params", {}).get("result")

                                        if result is not None:
                                            yield result

                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse JSON message: {e}")
                                        continue
                                    except Exception as e:
                                        logger.error(f"Error processing message: {e}")
                                        continue

                                elif msg.type == aiohttp.WSMsgType.ERROR:
                                    logger.error(f"WebSocket error: {msg}")
                                    break

                                elif msg.type == aiohttp.WSMsgType.CLOSED:
                                    logger.warning("WebSocket connection closed")
                                    break

                except aiohttp.ClientConnectorError as e:
                    await self._handle_connection_failure(e)
                    continue
                except asyncio.TimeoutError as e:
                    await self._handle_connection_failure(e)
                    continue
                except Exception as e:
                    await self._handle_connection_failure(e)
                    continue

        except Exception as e:
            logger.error(f"Fatal error in pool monitor: {e}")
            raise
        finally:
            # Clean up health check task
            if health_check_task and not health_check_task.done():
                health_check_task.cancel()
                try:
                    await health_check_task
                except asyncio.CancelledError:
                    pass


# Backwards compatibility function
async def watch_pool(api_key: str, pool_program: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Backwards compatible function that uses the enhanced monitor."""
    monitor = EnhancedPoolMonitor(api_key, pool_program)
    async for result in monitor.watch_pool():
        yield result
