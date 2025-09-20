"""
Single Source of Truth Trade Manager

This module implements a unified trade management system where TradeManager is the
single source of truth for all trade and position data. All other components must
read from and write to this central system.

Key Principles:
- TradeManager owns all trade and position data
- Event-driven notifications for real-time updates
- CSV logging as audit trail (not primary storage)
- Consistent data flow across all components
- No manual synchronization needed
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import json
import threading
from queue import Queue
import time

from crypto_bot.utils.trade_manager import TradeManager, Trade, Position, create_trade
from crypto_bot.utils.logger import LOG_DIR
from libs.notifications import TelegramNotifier

logger = logging.getLogger(__name__)


@dataclass
class TradeEvent:
    """Event representing a trade-related change."""
    event_type: str  # 'trade_executed', 'position_updated', 'trade_cancelled'
    trade: Optional[Trade] = None
    position: Optional[Position] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class TradeEventBus:
    """
    Event-driven system for trade notifications.

    Components can subscribe to trade events and receive real-time updates
    when trades are executed or positions change.
    """

    def __init__(self):
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._event_queue = Queue()
        self._running = False
        self._worker_thread = None

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)
        logger.info(f"Subscribed to {event_type} events")

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from a specific event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    def publish(self, event: TradeEvent) -> None:
        """Publish an event to all subscribers."""
        self._event_queue.put(event)

    def _process_events(self) -> None:
        """Process events in background thread."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
                self._notify_subscribers(event)
                self._event_queue.task_done()
            except Exception:
                # Queue timeout or processing error
                continue

    def _notify_subscribers(self, event: TradeEvent) -> None:
        """Notify all subscribers of an event."""
        subscribers = self._subscribers.get(event.event_type, set()).copy()

        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback for {event.event_type}: {e}")

    def start(self) -> None:
        """Start the event processing thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_events,
            daemon=True,
            name="TradeEventBus"
        )
        self._worker_thread.start()
        logger.info("TradeEventBus started")

    def stop(self) -> None:
        """Stop the event processing thread."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        logger.info("TradeEventBus stopped")


class SingleSourceTradeManager(TradeManager):
    """
    Enhanced TradeManager that serves as the single source of truth.

    This extends the base TradeManager with:
    - Event-driven notifications
    - CSV audit logging (not primary storage)
    - Consistency guarantees
    - Real-time synchronization
    """

    def __init__(self, storage_path: Optional[str] = None, enable_csv_audit: bool = True):
        super().__init__(storage_path)

        # Event system
        self.event_bus = TradeEventBus()
        self.event_bus.start()

        # CSV audit logging
        self.enable_csv_audit = enable_csv_audit
        self.csv_lock = threading.Lock()

        # Component subscribers
        self._frontend_subscribers: Set[Callable] = set()
        self._portfolio_service_subscribers: Set[Callable] = set()
        self._notification_subscribers: Set[Callable] = set()

        # Register event handlers
        self._setup_event_handlers()

        logger.info("SingleSourceTradeManager initialized with event-driven architecture")

    def _setup_event_handlers(self) -> None:
        """Set up event handlers for different component types."""

        # Frontend updates
        def handle_frontend_update(event: TradeEvent) -> None:
            """Notify frontend of trade/position changes."""
            for callback in self._frontend_subscribers.copy():
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Frontend callback failed: {e}")

        # Portfolio service updates
        def handle_portfolio_update(event: TradeEvent) -> None:
            """Notify portfolio service of changes."""
            for callback in self._portfolio_service_subscribers.copy():
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Portfolio service callback failed: {e}")

        # Notification updates
        def handle_notification_update(event: TradeEvent) -> None:
            """Send notifications for important events."""
            for callback in self._notification_subscribers.copy():
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Notification callback failed: {e}")

        # Subscribe to relevant events
        self.event_bus.subscribe('trade_executed', handle_frontend_update)
        self.event_bus.subscribe('position_updated', handle_frontend_update)
        self.event_bus.subscribe('trade_executed', handle_portfolio_update)
        self.event_bus.subscribe('trade_executed', handle_notification_update)

    def record_trade(self, trade: Trade) -> str:
        """Record a trade and notify all subscribers."""

        # Call parent implementation
        trade_id = super().record_trade(trade)

        # Write to CSV audit log (as backup, not primary storage)
        if self.enable_csv_audit:
            self._write_csv_audit_entry(trade)

        # Publish events
        trade_event = TradeEvent(
            event_type='trade_executed',
            trade=trade,
            metadata={'trade_id': trade_id}
        )
        self.event_bus.publish(trade_event)

        # Publish position update if position was affected
        position = self.positions.get(trade.symbol)
        if position:
            position_event = TradeEvent(
                event_type='position_updated',
                position=position,
                trade=trade,
                metadata={'trade_id': trade_id}
            )
            self.event_bus.publish(position_event)

        logger.info(f"Trade {trade_id} recorded and events published")
        return trade_id

    def _write_csv_audit_entry(self, trade: Trade) -> None:
        """Write trade to CSV as audit trail (not primary storage)."""
        try:
            with self.csv_lock:
                csv_path = LOG_DIR / "trades_audit.csv"

                # Write header if file doesn't exist
                if not csv_path.exists():
                    with open(csv_path, 'w', newline='') as f:
                        import csv
                        writer = csv.writer(f)
                        writer.writerow([
                            'symbol', 'side', 'amount', 'price', 'timestamp',
                            'strategy', 'exchange', 'status', 'trade_id'
                        ])

                # Append trade entry
                with open(csv_path, 'a', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow([
                        trade.symbol,
                        trade.side,
                        float(trade.amount),
                        float(trade.price),
                        trade.timestamp.isoformat(),
                        trade.strategy,
                        trade.exchange,
                        trade.status,
                        trade.id
                    ])

        except Exception as e:
            logger.warning(f"Failed to write CSV audit entry: {e}")

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update price and notify subscribers of position changes."""
        old_price = self.price_cache.get(symbol)

        # Call parent implementation
        super().update_price(symbol, price)

        # Publish price update event
        if old_price != price:
            price_event = TradeEvent(
                event_type='price_updated',
                metadata={
                    'symbol': symbol,
                    'old_price': float(old_price) if old_price else None,
                    'new_price': float(price)
                }
            )
            self.event_bus.publish(price_event)

    def add_frontend_subscriber(self, callback: Callable) -> None:
        """Add a frontend component subscriber."""
        self._frontend_subscribers.add(callback)
        logger.info("Frontend subscriber added")

    def add_portfolio_service_subscriber(self, callback: Callable) -> None:
        """Add a portfolio service subscriber."""
        self._portfolio_service_subscribers.add(callback)
        logger.info("Portfolio service subscriber added")

    def add_notification_subscriber(self, callback: Callable) -> None:
        """Add a notification subscriber."""
        self._notification_subscribers.add(callback)
        logger.info("Notification subscriber added")

    def remove_frontend_subscriber(self, callback: Callable) -> None:
        """Remove a frontend subscriber."""
        self._frontend_subscribers.discard(callback)

    def remove_portfolio_service_subscriber(self, callback: Callable) -> None:
        """Remove a portfolio service subscriber."""
        self._portfolio_service_subscribers.discard(callback)

    def remove_notification_subscriber(self, callback: Callable) -> None:
        """Remove a notification subscriber."""
        self._notification_subscribers.discard(callback)

    def shutdown(self) -> None:
        """Shutdown the trade manager and event system."""
        self.event_bus.stop()
        super().shutdown()
        logger.info("SingleSourceTradeManager shutdown complete")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'trades_count': len(self.trades),
            'positions_count': len(self.positions),
            'closed_positions_count': len(self.closed_positions),
            'total_volume': float(self.total_volume),
            'total_pnl': float(self.total_realized_pnl),
            'frontend_subscribers': len(self._frontend_subscribers),
            'portfolio_subscribers': len(self._portfolio_service_subscribers),
            'notification_subscribers': len(self._notification_subscribers),
            'event_queue_size': self.event_bus._event_queue.qsize() if hasattr(self.event_bus, '_event_queue') else 0,
            'csv_audit_enabled': self.enable_csv_audit,
            'last_update': datetime.utcnow().isoformat()
        }


# Global instance management
_single_source_manager: Optional[SingleSourceTradeManager] = None
_manager_lock = threading.Lock()


def get_single_source_trade_manager() -> SingleSourceTradeManager:
    """
    Get the global SingleSourceTradeManager instance.

    This ensures all components use the same instance and single source of truth.
    """
    global _single_source_manager

    if _single_source_manager is None:
        with _manager_lock:
            if _single_source_manager is None:
                # Determine the correct path to the state file
                import os
                current_dir = os.getcwd()

                # Try different possible locations
                possible_paths = [
                    "crypto_bot/logs/trade_manager_state.json",
                    "/app/crypto_bot/logs/trade_manager_state.json",
                    "../crypto_bot/logs/trade_manager_state.json",
                    os.path.join(current_dir, "crypto_bot/logs/trade_manager_state.json"),
                    os.path.join(current_dir, "../crypto_bot/logs/trade_manager_state.json"),
                ]

                state_file_path = None
                for path in possible_paths:
                    if Path(path).exists():
                        state_file_path = path
                        break

                if state_file_path:
                    _single_source_manager = SingleSourceTradeManager(storage_path=state_file_path)
                else:
                    # Use default path
                    default_path = "crypto_bot/logs/trade_manager_state.json"
                    _single_source_manager = SingleSourceTradeManager(storage_path=default_path)

    return _single_source_manager


def reset_single_source_trade_manager() -> None:
    """Reset the global SingleSourceTradeManager instance."""
    global _single_source_manager
    with _manager_lock:
        if _single_source_manager is not None:
            _single_source_manager.shutdown()
        _single_source_manager = None
    logger.info("SingleSourceTradeManager global instance reset")


# Backward compatibility
def get_trade_manager() -> SingleSourceTradeManager:
    """Backward compatibility function for existing code."""
    return get_single_source_trade_manager()


# Component Integration Helpers

def create_frontend_subscriber(frontend_callback: Callable) -> Callable:
    """
    Create a frontend event subscriber.

    Usage:
        trade_manager = get_single_source_trade_manager()
        subscriber = create_frontend_subscriber(my_frontend_update_function)
        trade_manager.add_frontend_subscriber(subscriber)
    """
    def event_handler(event: TradeEvent) -> None:
        try:
            if event.event_type in ['trade_executed', 'position_updated', 'price_updated']:
                frontend_callback(event)
        except Exception as e:
            logger.error(f"Frontend event handler failed: {e}")

    return event_handler


def create_portfolio_subscriber(portfolio_callback: Callable) -> Callable:
    """
    Create a portfolio service event subscriber.

    Usage:
        trade_manager = get_single_source_trade_manager()
        subscriber = create_portfolio_subscriber(my_portfolio_update_function)
        trade_manager.add_portfolio_service_subscriber(subscriber)
    """
    def event_handler(event: TradeEvent) -> None:
        try:
            if event.event_type in ['trade_executed', 'position_updated']:
                portfolio_callback(event)
        except Exception as e:
            logger.error(f"Portfolio event handler failed: {e}")

    return event_handler


def create_notification_subscriber(notifier: TelegramNotifier, config: Dict[str, Any]) -> Callable:
    """
    Create a notification event subscriber.

    Usage:
        trade_manager = get_single_source_trade_manager()
        subscriber = create_notification_subscriber(my_notifier, config)
        trade_manager.add_notification_subscriber(subscriber)
    """
    def event_handler(event: TradeEvent) -> None:
        try:
            if event.event_type == 'trade_executed' and event.trade:
                trade = event.trade
                message = (
                    f"🎯 Trade Executed: {trade.symbol} {trade.side.upper()} "
                    f"{trade.amount} @ ${trade.price}"
                )
                if notifier:
                    notifier.notify(message)
        except Exception as e:
            logger.error(f"Notification event handler failed: {e}")

    return event_handler
