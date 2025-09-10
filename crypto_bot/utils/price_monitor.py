"""
Price Monitoring Service for Real-time Price Updates

This module provides a dedicated service for monitoring and updating prices
for open positions to ensure accurate PnL calculations and trading decisions.
"""

import asyncio
import logging
import threading
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
from decimal import Decimal
import time

from .trade_manager import get_trade_manager, TradeManager
from .logger import LOG_DIR, setup_logger

logger = setup_logger("price_monitor", LOG_DIR / "price_monitor.log")


class PriceMonitor:
    """
    Dedicated price monitoring service that ensures consistent price updates
    for open positions and trading decisions.
    """

    def __init__(self, trade_manager: Optional[TradeManager] = None):
        self.trade_manager = trade_manager or get_trade_manager()
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.price_callbacks: List[Callable] = []
        self.symbol_subscriptions: Set[str] = set()
        self.last_price_update: Dict[str, datetime] = {}
        self.price_update_interval = 30  # seconds
        self.max_price_age = 300  # 5 minutes - consider price stale after this
        self.retry_attempts = 3
        self.retry_delay = 5  # seconds

    def add_price_callback(self, callback: Callable) -> None:
        """Add callback for price update notifications."""
        self.price_callbacks.append(callback)

    def subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to price updates for a specific symbol."""
        self.symbol_subscriptions.add(symbol)
        logger.info(f"Subscribed to price updates for {symbol}")

    def unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from price updates for a specific symbol."""
        self.symbol_subscriptions.discard(symbol)
        logger.info(f"Unsubscribed from price updates for {symbol}")

    def get_active_symbols(self) -> Set[str]:
        """Get all symbols that need price monitoring (open positions + subscriptions)."""
        # Get symbols from open positions
        open_positions = self.trade_manager.get_all_positions()
        position_symbols = {pos.symbol for pos in open_positions}

        # Combine with manual subscriptions
        return position_symbols.union(self.symbol_subscriptions)

    async def fetch_current_price(self, symbol: str, exchange=None) -> Optional[Decimal]:
        """
        Fetch current price for a symbol using multiple fallback methods.

        Returns:
            Current price as Decimal, or None if all methods fail
        """
        methods = []

        # Method 1: Try from TradeManager's price cache first
        cached_price = self.trade_manager.price_cache.get(symbol)
        if cached_price and self._is_price_fresh(symbol):
            return cached_price

        # Method 2: Try exchange ticker if available
        if exchange:
            try:
                if hasattr(exchange, 'fetch_ticker'):
                    if asyncio.iscoroutinefunction(exchange.fetch_ticker):
                        ticker = await exchange.fetch_ticker(symbol)
                    else:
                        ticker = exchange.fetch_ticker(symbol)

                    price = ticker.get('last') or ticker.get('close')
                    if price and float(price) > 0:
                        return Decimal(str(price))
            except Exception as e:
                logger.warning(f"Exchange ticker failed for {symbol}: {e}")

        # Method 3: Try df_cache if available
        try:
            # This would need access to the context's df_cache
            # For now, we'll rely on the above methods
            pass
        except Exception as e:
            logger.warning(f"df_cache lookup failed for {symbol}: {e}")

        logger.warning(f"All price fetching methods failed for {symbol}")
        return None

    def _is_price_fresh(self, symbol: str) -> bool:
        """Check if cached price is still fresh."""
        last_update = self.last_price_update.get(symbol)
        if not last_update:
            return False

        age = datetime.utcnow() - last_update
        return age.total_seconds() < self.max_price_age

    async def update_prices_for_symbols(self, symbols: Set[str], exchange=None) -> Dict[str, Decimal]:
        """
        Update prices for a set of symbols with retry logic.

        Returns:
            Dictionary of symbol -> price for successfully updated symbols
        """
        updated_prices = {}

        for symbol in symbols:
            for attempt in range(self.retry_attempts):
                try:
                    price = await self.fetch_current_price(symbol, exchange)
                    if price:
                        # Update TradeManager price cache
                        self.trade_manager.update_price(symbol, price)
                        self.last_price_update[symbol] = datetime.utcnow()
                        updated_prices[symbol] = price

                        logger.debug(f"Updated price for {symbol}: ${price}")
                        break
                    else:
                        if attempt < self.retry_attempts - 1:
                            await asyncio.sleep(self.retry_delay)
                        else:
                            logger.warning(f"Failed to update price for {symbol} after {self.retry_attempts} attempts")

                except Exception as e:
                    logger.error(f"Error updating price for {symbol} (attempt {attempt + 1}): {e}")
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(self.retry_delay)

        return updated_prices

    async def _monitor_loop(self, exchange=None) -> None:
        """Main monitoring loop that runs in background."""
        logger.info("Price monitoring service started")

        while self.running:
            try:
                # Get active symbols to monitor
                symbols = self.get_active_symbols()

                if symbols:
                    logger.debug(f"Monitoring prices for {len(symbols)} symbols: {list(symbols)}")

                    # Update prices for all symbols
                    updated_prices = await self.update_prices_for_symbols(symbols, exchange)

                    # Notify callbacks of price updates
                    for symbol, price in updated_prices.items():
                        for callback in self.price_callbacks:
                            try:
                                callback(symbol, price)
                            except Exception as e:
                                logger.error(f"Price callback failed: {e}")

                    # Log summary
                    if updated_prices:
                        logger.info(f"Successfully updated prices for {len(updated_prices)}/{len(symbols)} symbols")
                    else:
                        logger.warning("No prices were successfully updated")

                else:
                    logger.debug("No symbols to monitor")

            except Exception as e:
                logger.error(f"Error in price monitoring loop: {e}")

            # Wait for next update cycle
            await asyncio.sleep(self.price_update_interval)

        logger.info("Price monitoring service stopped")

    def start_monitoring(self, exchange=None) -> None:
        """Start the price monitoring service in a background thread."""
        if self.running:
            logger.warning("Price monitoring service is already running")
            return

        self.running = True

        def run_monitor():
            """Run the monitoring loop in a new event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._monitor_loop(exchange))
            except Exception as e:
                logger.error(f"Price monitoring thread error: {e}")
            finally:
                loop.close()

        self.monitor_thread = threading.Thread(target=run_monitor, daemon=True)
        self.monitor_thread.start()
        logger.info("Price monitoring service started in background thread")

    def stop_monitoring(self) -> None:
        """Stop the price monitoring service."""
        if not self.running:
            logger.info("Price monitoring service is not running")
            return

        self.running = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)
            if self.monitor_thread.is_alive():
                logger.warning("Price monitoring thread did not stop gracefully")

        logger.info("Price monitoring service stopped")

    def get_price_status(self) -> Dict[str, Dict]:
        """Get status of price monitoring for all symbols."""
        symbols = self.get_active_symbols()
        status = {}

        for symbol in symbols:
            cached_price = self.trade_manager.price_cache.get(symbol)
            last_update = self.last_price_update.get(symbol)
            is_fresh = self._is_price_fresh(symbol) if last_update else False

            status[symbol] = {
                'cached_price': float(cached_price) if cached_price else None,
                'last_update': last_update.isoformat() if last_update else None,
                'is_fresh': is_fresh,
                'age_seconds': (datetime.utcnow() - last_update).total_seconds() if last_update else None
            }

        return status

    def force_price_update(self, symbol: str, exchange=None) -> bool:
        """
        Force immediate price update for a specific symbol.

        Returns:
            True if update was successful, False otherwise
        """
        async def update_single():
            symbols = {symbol}
            updated = await self.update_prices_for_symbols(symbols, exchange)
            return symbol in updated

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(update_single())
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Force price update failed for {symbol}: {e}")
            return False


# Global price monitor instance
_price_monitor: Optional[PriceMonitor] = None
_price_monitor_lock = threading.Lock()


def get_price_monitor(trade_manager: Optional[TradeManager] = None) -> PriceMonitor:
    """Get the global PriceMonitor instance."""
    global _price_monitor
    if _price_monitor is None:
        with _price_monitor_lock:
            if _price_monitor is None:
                _price_monitor = PriceMonitor(trade_manager)
    return _price_monitor


def start_price_monitoring(exchange=None, trade_manager: Optional[TradeManager] = None) -> PriceMonitor:
    """Start the global price monitoring service."""
    monitor = get_price_monitor(trade_manager)
    monitor.start_monitoring(exchange)
    return monitor


def stop_price_monitoring() -> None:
    """Stop the global price monitoring service."""
    monitor = get_price_monitor()
    monitor.stop_monitoring()
