"""
Real-time position monitoring with WebSocket price feeds.

This module provides high-frequency monitoring of active positions to ensure
trailing stops are triggered promptly, even in fast-moving markets.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

import ccxt.async_support as ccxt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PositionMonitor:
    """Real-time position monitoring with WebSocket price feeds."""

    exchange: ccxt.Exchange
    config: dict
    positions: Dict[str, dict]
    notifier: Optional[object] = None
    trade_manager: Optional[object] = None  # Centralized trade manager

    # Monitoring state
    active_monitors: Dict[str, asyncio.Task] = field(default_factory=dict)
    price_cache: Dict[str, float] = field(default_factory=dict)
    last_update: Dict[str, float] = field(default_factory=dict)

    # Performance tracking
    monitoring_stats: Dict[str, int] = field(default_factory=dict)
    
    # Configuration
    check_interval_seconds: float = 5.0  # Check every 5 seconds
    max_monitor_age_seconds: float = 300.0  # 5 minutes
    price_update_threshold: float = 0.001  # 0.1% price change threshold
    
    def __post_init__(self):
        """Initialize monitoring statistics and configuration."""
        # Load configuration
        exit_cfg = self.config.get("exit_strategy", {})
        monitoring_cfg = exit_cfg.get("real_time_monitoring", {})
        
        if monitoring_cfg.get("enabled", True):
            self.check_interval_seconds = monitoring_cfg.get("check_interval_seconds", 5.0)
            self.max_monitor_age_seconds = monitoring_cfg.get("max_monitor_age_seconds", 300.0)
            self.price_update_threshold = monitoring_cfg.get("price_update_threshold", 0.001)
            self.use_websocket = monitoring_cfg.get("use_websocket_when_available", True)
            self.fallback_to_rest = monitoring_cfg.get("fallback_to_rest", True)
            self.max_execution_latency_ms = monitoring_cfg.get("max_execution_latency_ms", 1000)
        else:
            # Disable monitoring by setting very long intervals
            self.check_interval_seconds = 3600.0  # 1 hour
            self.max_monitor_age_seconds = 7200.0  # 2 hours
            # Set default values for WebSocket settings when disabled
            self.use_websocket = False
            self.fallback_to_rest = False
            self.max_execution_latency_ms = 1000

        # Initialize monitoring statistics
        self.monitoring_stats = {
            "positions_monitored": 0,
            "price_updates": 0,
            "trailing_stop_triggers": 0,
            "execution_latency_ms": 0,
            "missed_exits": 0
        }

        # Initialize WebSocket client for real-time price monitoring
        self.ws_client = None
        if self.use_websocket:
            try:
                from crypto_bot.execution.kraken_ws import KrakenWSClient
                self.ws_client = KrakenWSClient()
                logger.info("WebSocket client initialized for real-time position monitoring")
            except Exception as e:
                logger.warning(f"Failed to initialize WebSocket client: {e}")
                self.ws_client = None
    
    async def start_monitoring(self, symbol: str, position: dict = None) -> None:
        """Start real-time monitoring for a position."""
        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid symbol provided for monitoring: {symbol}")
            return

        if symbol in self.active_monitors:
            logger.warning(f"Already monitoring {symbol}")
            return

        logger.info(f"Starting real-time monitoring for {symbol}")

        # Get position data from TradeManager if available, otherwise use provided position
        if self.trade_manager:
            tm_position = self.trade_manager.get_position(symbol)
            if tm_position:
                # Use TradeManager position data
                position = {
                    'entry_price': float(tm_position.average_price),
                    'size': float(tm_position.total_amount),
                    'side': tm_position.side,
                    'symbol': tm_position.symbol,
                    'highest_price': float(tm_position.highest_price) if tm_position.highest_price else float(tm_position.average_price),
                    'lowest_price': float(tm_position.lowest_price) if tm_position.lowest_price else float(tm_position.average_price),
                    'trailing_stop': float(tm_position.stop_loss_price) if tm_position.stop_loss_price else 0.0,
                    'timestamp': tm_position.entry_time.isoformat(),
                }
            elif position and isinstance(position, dict):
                # Use provided position dict as fallback
                logger.info(f"Using provided position dict for {symbol} (no TradeManager position found)")
            else:
                logger.warning(f"No position found in TradeManager for {symbol} and no position dict provided")
                return
        elif not position or not isinstance(position, dict):
            logger.error(f"Invalid position provided for monitoring and no TradeManager available: {position}")
            return

        # Subscribe to WebSocket ticker if available
        if self.ws_client and self.use_websocket:
            try:
                # Subscribe to ticker data
                self.ws_client.subscribe_ticker(symbol)

                # Add price callback for real-time updates
                def price_callback(sym: str, price: float):
                    try:
                        asyncio.create_task(self._handle_price_update(sym, price))
                    except Exception as e:
                        logger.error(f"Error creating price update task for {sym}: {e}")

                self.ws_client.add_price_callback(symbol, price_callback)
                logger.info(f"Subscribed to WebSocket ticker for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to subscribe to WebSocket ticker for {symbol}: {e}")

        # Create monitoring task
        monitor_task = asyncio.create_task(
            self._monitor_position(symbol, position)
        )
        self.active_monitors[symbol] = monitor_task

        # Initialize tracking
        self.price_cache[symbol] = position.get("entry_price", 0.0)
        self.last_update[symbol] = time.time()
        self.monitoring_stats["positions_monitored"] += 1
    
    async def stop_monitoring(self, symbol: str) -> None:
        """Stop monitoring a position."""
        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid symbol provided for stopping monitoring: {symbol}")
            return

        if symbol in self.active_monitors:
            task = self.active_monitors[symbol]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.active_monitors[symbol]
            
        # Clean up WebSocket subscriptions
        if self.ws_client and symbol in self.ws_client.price_callbacks:
            del self.ws_client.price_callbacks[symbol]
            logger.info(f"Cleaned up WebSocket subscription for {symbol}")
            
        # Clean up tracking
        self.price_cache.pop(symbol, None)
        self.last_update.pop(symbol, None)
        logger.info(f"Stopped monitoring {symbol}")
    
    async def _monitor_position(self, symbol: str, position: dict) -> None:
        """Monitor a single position with high-frequency price updates."""
        try:
            while self._position_still_exists(symbol):
                start_time = time.time()
                
                # Get current price
                current_price = await self._get_current_price(symbol)
                if current_price is None or current_price <= 0:
                    logger.debug(f"Invalid price received for {symbol}: {current_price}")
                    await asyncio.sleep(self.check_interval_seconds)
                    continue
                
                # Update price cache
                old_price = self.price_cache.get(symbol, current_price)
                self.price_cache[symbol] = current_price
                self.last_update[symbol] = time.time()
                
                # Check for significant price movement
                price_change = abs(current_price - old_price) / old_price if old_price > 0 else 1.0
                if price_change > self.price_update_threshold:
                    self.monitoring_stats["price_updates"] += 1
                    logger.debug(f"Price update for {symbol}: {old_price:.6f} -> {current_price:.6f}")
                
                # Update position tracking
                await self._update_position_tracking(symbol, position, current_price)
                
                # Check exit conditions
                should_exit, exit_reason = await self._check_exit_conditions(
                    symbol, position, current_price
                )
                
                if should_exit:
                    await self._handle_exit(symbol, position, current_price, exit_reason)
                    break
                
                # Measure execution latency
                latency_ms = (time.time() - start_time) * 1000
                self.monitoring_stats["execution_latency_ms"] = max(
                    self.monitoring_stats["execution_latency_ms"], 
                    int(latency_ms)
                )
                
                # Alert if execution latency is too high
                if latency_ms > self.max_execution_latency_ms:
                    logger.warning(f"High execution latency for {symbol}: {latency_ms:.1f}ms (threshold: {self.max_execution_latency_ms}ms)")
                    if self.notifier:
                        self.notifier.notify(f"⚠️ High execution latency: {symbol} - {latency_ms:.1f}ms")
                
                await asyncio.sleep(self.check_interval_seconds)
                
        except asyncio.CancelledError:
            logger.info(f"Position monitoring cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Error monitoring position {symbol}: {e}")
            self.monitoring_stats["missed_exits"] += 1
    
    async def _handle_price_update(self, symbol: str, price: float) -> None:
        """Handle real-time price updates from WebSocket."""
        try:
            if not symbol or not isinstance(symbol, str):
                logger.error(f"Invalid symbol in price update: {symbol}")
                return

            if price <= 0:
                logger.warning(f"Invalid price in update for {symbol}: {price}")
                return

            if symbol not in self.positions:
                return
            
            # Update price cache
            old_price = self.price_cache.get(symbol, price)
            self.price_cache[symbol] = price
            self.last_update[symbol] = time.time()
            
            # Check for significant price movement
            price_change = abs(price - old_price) / old_price if old_price > 0 else 1.0
            if price_change > self.price_update_threshold:
                self.monitoring_stats["price_updates"] += 1
                logger.debug(f"Real-time price update for {symbol}: {old_price:.6f} -> {price:.6f}")
            
            # Update position tracking
            position = self.positions[symbol]
            await self._update_position_tracking(symbol, position, price)
            
            # Check exit conditions immediately
            should_exit, exit_reason = await self._check_exit_conditions(symbol, position, price)
            
            if should_exit:
                await self._handle_exit(symbol, position, price, exit_reason)
                
        except Exception as e:
            logger.error(f"Error handling real-time price update for {symbol}: {e}")

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol using the most efficient method."""
        try:
            # Try WebSocket cache first if available
            if self.ws_client and self.use_websocket:
                ws_price = self.ws_client.get_current_price(symbol)
                if ws_price is not None:
                    return ws_price
            
            # Try WebSocket first if available and enabled
            if self.use_websocket and hasattr(self.exchange, 'watch_ticker'):
                try:
                    if asyncio.iscoroutinefunction(self.exchange.watch_ticker):
                        ticker = await self.exchange.watch_ticker(symbol)
                    else:
                        ticker = self.exchange.watch_ticker(symbol)
                    return float(ticker['last'])
                except Exception as e:
                    logger.debug(f"WebSocket price fetch failed for {symbol}: {e}")
                    if not self.fallback_to_rest:
                        return None

            # Fallback to REST API
            if self.fallback_to_rest:
                if asyncio.iscoroutinefunction(self.exchange.fetch_ticker):
                    ticker = await self.exchange.fetch_ticker(symbol)
                else:
                    ticker = self.exchange.fetch_ticker(symbol)
                return float(ticker['last'])
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
            return None
    
    async def _update_position_tracking(self, symbol: str, position: dict, current_price: float) -> None:
        """Update position tracking (highest/lowest prices, PnL)."""
        try:
            # If we have a TradeManager, delegate position updates to it
            if self.trade_manager:
                from decimal import Decimal
                self.trade_manager.update_price(symbol, Decimal(str(current_price)))

                # Get updated position from TradeManager
                tm_position = self.trade_manager.get_position(symbol)
                if tm_position:
                    # Calculate unrealized PnL
                    unrealized_pnl, unrealized_pct = tm_position.calculate_unrealized_pnl(Decimal(str(current_price)))

                    # Update position data for backward compatibility
                    # Convert percentage to decimal for consistency with legacy format
                    position["pnl"] = float(unrealized_pct) / 100.0
                    position["highest_price"] = float(tm_position.highest_price) if tm_position.highest_price else current_price
                    position["lowest_price"] = float(tm_position.lowest_price) if tm_position.lowest_price else current_price
                    position["trailing_stop"] = float(tm_position.stop_loss_price) if tm_position.stop_loss_price else 0.0

                    logger.debug(f"Updated position tracking for {symbol}: PnL {unrealized_pct:.2f}% at price ${current_price:.6f}")
            else:
                # Fallback to legacy method
                await self._update_position_tracking_legacy(symbol, position, current_price)

        except Exception as e:
            logger.error(f"Error updating position tracking for {symbol}: {e}")

    async def _update_position_tracking_legacy(self, symbol: str, position: dict, current_price: float) -> None:
        """Legacy position tracking method for backward compatibility."""
        try:
            # Update highest/lowest price tracking
            if position["side"] == "buy":
                position["highest_price"] = max(
                    position.get("highest_price", current_price),
                    current_price
                )
            else:  # short position
                position["lowest_price"] = min(
                    position.get("lowest_price", current_price),
                    current_price
                )

            # Update trailing stop
            await self._update_trailing_stop(symbol, position, current_price)

            # Update PnL
            entry_price = position["entry_price"]
            pnl_pct = ((current_price - entry_price) / entry_price) * (
                1 if position["side"] == "buy" else -1
            )
            position["pnl"] = pnl_pct

        except Exception as e:
            logger.error(f"Error updating legacy position tracking for {symbol}: {e}")
    
    async def _update_trailing_stop(self, symbol: str, position: dict, current_price: float) -> None:
        """Update trailing stop based on current price and configuration."""
        try:
            # If we have a TradeManager, delegate trailing stop updates to it
            if self.trade_manager:
                tm_position = self.trade_manager.get_position(symbol)
                if tm_position:
                    # TradeManager handles trailing stop updates automatically
                    tm_position.update_trailing_stop(Decimal(str(current_price)))
                    # Update position dict for backward compatibility
                    position["trailing_stop"] = float(tm_position.stop_loss_price) if tm_position.stop_loss_price else 0.0
                    logger.debug(f"Updated trailing stop for {symbol} via TradeManager: {position['trailing_stop']:.6f}")
                return

            # Fallback to legacy method
            await self._update_trailing_stop_legacy(symbol, position, current_price)

        except Exception as e:
            logger.error(f"Error updating trailing stop for {symbol}: {e}")

    async def _update_trailing_stop_legacy(self, symbol: str, position: dict, current_price: float) -> None:
        """Legacy trailing stop method for backward compatibility."""
        try:
            exit_cfg = self.config.get("exit_strategy", {})
            min_gain_to_trail = exit_cfg.get("min_gain_to_trail", 0)

            # Calculate current PnL
            entry_price = position["entry_price"]
            pnl_pct = ((current_price - entry_price) / entry_price) * (
                1 if position["side"] == "buy" else -1
            )

            # Update highest/lowest prices continuously for tracking
            if position["side"] == "buy":
                position["highest_price"] = max(
                    position.get("highest_price", current_price),
                    current_price
                )
            else:  # short position
                position["lowest_price"] = min(
                    position.get("lowest_price", current_price),
                    current_price
                )

            # Only update trailing stop if we're in profit beyond minimum threshold
            if pnl_pct >= min_gain_to_trail:
                trailing_stop_pct = exit_cfg.get("trailing_stop_pct", 0.02)

                if position["side"] == "buy":
                    # Long position - trail below highest price
                    highest_price = position.get("highest_price", entry_price)
                    new_trailing_stop = highest_price * (1 - trailing_stop_pct)

                    # Only move trailing stop up (never down)
                    current_trailing_stop = position.get("trailing_stop", entry_price * (1 - 0.01))  # Default 1% below entry
                    if new_trailing_stop > current_trailing_stop:
                        position["trailing_stop"] = new_trailing_stop
                        logger.debug(f"Updated trailing stop for {symbol}: {current_trailing_stop:.6f} -> {new_trailing_stop:.6f}")

                else:
                    # Short position - trail above lowest price
                    lowest_price = position.get("lowest_price", entry_price)
                    new_trailing_stop = lowest_price * (1 + trailing_stop_pct)

                    # Only move trailing stop down (never up)
                    current_trailing_stop = position.get("trailing_stop", entry_price * (1 + 0.01))  # Default 1% above entry
                    if new_trailing_stop < current_trailing_stop:
                        position["trailing_stop"] = new_trailing_stop
                        logger.debug(f"Updated trailing stop for {symbol}: {current_trailing_stop:.6f} -> {new_trailing_stop:.6f}")

        except Exception as e:
            logger.error(f"Error updating legacy trailing stop for {symbol}: {e}")
    
    async def _check_exit_conditions(self, symbol: str, position: dict, current_price: float) -> tuple[bool, str]:
        """Check if position should be exited."""
        try:
            # If we have a TradeManager, delegate exit condition checking to it
            if self.trade_manager:
                tm_position = self.trade_manager.get_position(symbol)
                if tm_position:
                    should_exit, exit_reason = tm_position.should_exit(Decimal(str(current_price)))
                    if should_exit:
                        return True, exit_reason

            # Fallback to legacy method
            return await self._check_exit_conditions_legacy(symbol, position, current_price)

        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}")
            return False, ""

    async def _check_exit_conditions_legacy(self, symbol: str, position: dict, current_price: float) -> tuple[bool, str]:
        """Legacy exit condition checking for backward compatibility."""
        try:
            # Check trailing stop
            trailing_stop = position.get("trailing_stop", 0.0)

            if position["side"] == "buy":
                # Long position - exit if price falls below trailing stop
                if trailing_stop > 0 and current_price <= trailing_stop:
                    return True, "trailing_stop"
            else:
                # Short position - exit if price rises above trailing stop
                if trailing_stop < float('inf') and current_price >= trailing_stop:
                    return True, "trailing_stop"

            # Check take profit
            entry_price = position["entry_price"]
            exit_cfg = self.config.get("exit_strategy", {})
            take_profit_pct = exit_cfg.get("take_profit_pct", 0.0)

            if take_profit_pct > 0:
                if position["side"] == "buy":
                    take_profit_price = entry_price * (1 + take_profit_pct)
                    if current_price >= take_profit_price:
                        return True, "take_profit"
                else:
                    take_profit_price = entry_price * (1 - take_profit_pct)
                    if current_price <= take_profit_price:
                        return True, "take_profit"

            return False, ""

        except Exception as e:
            logger.error(f"Error checking legacy exit conditions for {symbol}: {e}")
            return False, ""
    
    async def _handle_exit(self, symbol: str, position: dict, current_price: float, exit_reason: str) -> None:
        """Handle position exit."""
        try:
            logger.info(f"Real-time exit triggered for {symbol}: {exit_reason} at {current_price:.6f}")
            
            # Record exit trigger
            self.monitoring_stats["trailing_stop_triggers"] += 1
            
            # Notify main system about exit
            if hasattr(self, 'on_exit_triggered'):
                await self.on_exit_triggered(symbol, position, current_price, exit_reason)
            
            # Send notification if available
            if self.notifier:
                pnl_pct = position.get("pnl", 0.0)
                pnl_emoji = "💰" if pnl_pct >= 0 else "📉"
                exit_msg = f"🚨 Real-time Exit {pnl_emoji}\n{position['side'].upper()} {position['size']:.4f} {symbol}\nEntry: ${position['entry_price']:.6f}\nExit: ${current_price:.6f}\nPnL: {pnl_pct:.2%}\nReason: {exit_reason}"
                self.notifier.notify(exit_msg)
                
        except Exception as e:
            logger.error(f"Error handling exit for {symbol}: {e}")
    
    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        return {
            **self.monitoring_stats,
            "active_monitors": len(self.active_monitors),
            "monitored_symbols": list(self.active_monitors.keys()),
            "price_cache_size": len(self.price_cache)
        }
    
    async def cleanup_old_monitors(self) -> None:
        """Clean up monitors for positions that no longer exist."""
        current_time = time.time()
        symbols_to_remove = []
        
        for symbol, last_update in self.last_update.items():
            if current_time - last_update > self.max_monitor_age_seconds:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            await self.stop_monitoring(symbol)
            logger.info(f"Cleaned up old monitor for {symbol}")
    
    async def stop_all_monitoring(self) -> None:
        """Stop all position monitoring."""
        logger.info("Stopping all position monitoring")
        
        for symbol in list(self.active_monitors.keys()):
            await self.stop_monitoring(symbol)
        
        # Close WebSocket client
        if self.ws_client:
            try:
                self.ws_client.close()
                logger.info("WebSocket client closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket client: {e}")
            finally:
                self.ws_client = None

        logger.info("All position monitoring stopped")

    def _position_still_exists(self, symbol: str) -> bool:
        """Check if position still exists using TradeManager or legacy positions."""
        if self.trade_manager:
            # Use TradeManager as source of truth
            position = self.trade_manager.get_position(symbol)
            return position is not None and position.is_open
        else:
            # Fallback to legacy positions dict
            return symbol in self.positions
