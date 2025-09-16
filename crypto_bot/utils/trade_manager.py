"""
Centralized Trade Manager - Single Source of Truth for Trades and Positions

This module provides a unified interface for all trade and position management,
ensuring consistent calculations and state management across the entire application.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import threading
from pathlib import Path
import json
import uuid
import time
from crypto_bot.utils.logger import LOG_DIR

logger = logging.getLogger(__name__)


def is_test_position(symbol: str) -> bool:
    """
    Check if a symbol represents a test position that should be filtered out.

    Test positions are identified by:
    - Symbols containing 'TEST' (case insensitive)
    - Symbols that are obviously fake/test symbols
    - Symbols from test exchanges or environments

    Args:
        symbol: Trading symbol to check

    Returns:
        True if this appears to be a test position
    """
    if not symbol or not isinstance(symbol, str):
        return False

    symbol_upper = symbol.upper()

    # Common test symbol patterns
    test_patterns = [
        "TEST",
        "FAKE",
        "DUMMY",
        "MOCK",
        "SAMPLE",
        "EXAMPLE",
        "DEMO",
        "SANDBOX",
    ]

    # Check for test patterns in symbol
    for pattern in test_patterns:
        if pattern in symbol_upper:
            logger.warning(f"Detected test position symbol: {symbol}")
            return True

    # Check for obviously fake trading pairs (single asset without quote currency)
    if "/" not in symbol and len(symbol) > 10:
        logger.warning(
            f"Detected potentially fake symbol (no quote currency): {symbol}"
        )
        return True

    # Check for test exchange prefixes
    test_exchanges = ["TESTEX", "FAKEEX", "MOCKEX"]
    if any(symbol_upper.startswith(exchange) for exchange in test_exchanges):
        logger.warning(f"Detected test exchange symbol: {symbol}")
        return True

    return False


def validate_position_symbol(symbol: str, context: str = "position") -> None:
    """
    Validate that a symbol is not a test position. Raises ValueError if invalid.

    Args:
        symbol: Symbol to validate
        context: Context for logging (e.g., "position", "trade")

    Raises:
        ValueError: If symbol is identified as a test position
    """
    if is_test_position(symbol):
        error_msg = f"Rejected {context} with test symbol: {symbol}"
        logger.error(error_msg)
        raise ValueError(error_msg)


@dataclass
class Trade:
    """Unified trade data model."""

    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: Decimal
    price: Decimal
    timestamp: datetime
    strategy: str = ""
    exchange: str = ""
    fees: Decimal = Decimal("0")
    status: str = "filled"  # 'pending', 'filled', 'cancelled', 'failed'
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure decimal types and validate data."""
        # Validate symbol is not a test position
        validate_position_symbol(self.symbol, "trade")

        # Validate inputs
        if self.amount < 0:
            raise ValueError("Trade amount cannot be negative")
        if self.price <= 0:
            raise ValueError("Trade price must be positive")

        # Handle extremely large numbers by using a more conservative quantization
        try:
            self.amount = Decimal(str(self.amount)).quantize(
                Decimal("0.00000001"), rounding=ROUND_DOWN
            )
            self.price = Decimal(str(self.price)).quantize(
                Decimal("0.00000001"), rounding=ROUND_DOWN
            )
            self.fees = Decimal(str(self.fees)).quantize(
                Decimal("0.00000001"), rounding=ROUND_DOWN
            )
        except InvalidOperation:
            # For extremely large numbers, use a more conservative approach
            max_digits = 28  # Maximum precision for Decimal
            if len(str(self.amount)) > max_digits:
                self.amount = Decimal("1e28")  # Cap at reasonable maximum
            else:
                self.amount = Decimal(str(self.amount))
            if len(str(self.price)) > max_digits:
                self.price = Decimal("1e28")
            else:
                self.price = Decimal(str(self.price))
            if len(str(self.fees)) > max_digits:
                self.fees = Decimal("1e28")
            else:
                self.fees = Decimal(str(self.fees))

    @property
    def total_value(self) -> Decimal:
        """Calculate total trade value (excluding fees)."""
        return self.amount * self.price

    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost including fees."""
        return self.total_value + self.fees

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "amount": float(self.amount),
            "price": float(self.price),
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy,
            "exchange": self.exchange,
            "fees": float(self.fees),
            "status": self.status,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Trade:
        """Create Trade from dictionary."""
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data_copy)


@dataclass
class Position:
    """Unified position data model."""

    symbol: str
    side: str  # 'long' or 'short'
    total_amount: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    trades: List[Trade] = field(default_factory=list)
    entry_time: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)
    highest_price: Optional[Decimal] = None
    lowest_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    trailing_stop_pct: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure decimal types and validate symbol."""
        # Validate symbol is not a test position
        validate_position_symbol(self.symbol, "position")

        self.total_amount = Decimal(str(self.total_amount)).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )
        self.average_price = Decimal(str(self.average_price)).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )
        self.realized_pnl = Decimal(str(self.realized_pnl)).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )
        self.fees_paid = Decimal(str(self.fees_paid)).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.total_amount > 0

    @property
    def position_value(self) -> Decimal:
        """Current position value at average price."""
        return self.total_amount * self.average_price

    def calculate_unrealized_pnl(
        self, current_price: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate unrealized PnL at current price.

        Returns:
            Tuple of (pnl_amount, pnl_percentage)
        """
        if not self.is_open or current_price <= 0:
            return Decimal("0"), Decimal("0")

        current_price = Decimal(str(current_price)).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )

        if self.side == "long":
            pnl = (current_price - self.average_price) * self.total_amount
        else:  # short
            pnl = (self.average_price - current_price) * self.total_amount

        pnl_pct = (
            (pnl / self.position_value) * 100
            if self.position_value > 0
            else Decimal("0")
        )
        return pnl, pnl_pct

    def update_price_levels(self, current_price: Decimal) -> None:
        """Update highest/lowest price tracking."""
        current_price = Decimal(str(current_price)).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )

        if self.highest_price is None or current_price > self.highest_price:
            self.highest_price = current_price

        if self.lowest_price is None or current_price < self.lowest_price:
            self.lowest_price = current_price

        self.last_update = datetime.utcnow()

    def update_trailing_stop(self, current_price: Decimal) -> None:
        """Update trailing stop based on current price."""
        if self.trailing_stop_pct is None or not self.is_open:
            return

        if self.side == "long":
            # Long position - trail below highest price
            if self.highest_price is not None:
                new_stop = self.highest_price * (
                    1 - self.trailing_stop_pct
                )
                if (
                    self.stop_loss_price is None
                    or new_stop > self.stop_loss_price
                ):
                    self.stop_loss_price = new_stop
        else:
            # Short position - trail above lowest price
            if self.lowest_price is not None:
                new_stop = self.lowest_price * (
                    1 + self.trailing_stop_pct
                )
                if (
                    self.stop_loss_price is None
                    or new_stop < self.stop_loss_price
                ):
                    self.stop_loss_price = new_stop

    def should_exit(self, current_price: Decimal) -> Tuple[bool, str]:
        """
        Check if position should be exited based on stop loss/take profit.

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        if not self.is_open:
            return False, ""

        current_price = Decimal(str(current_price)).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )

        # Check stop loss
        if self.stop_loss_price is not None:
            if self.side == "long" and current_price <= self.stop_loss_price:
                return True, "stop_loss"
            elif (
                self.side == "short" and current_price >= self.stop_loss_price
            ):
                return True, "stop_loss"

        # Check take profit
        if self.take_profit_price is not None:
            if self.side == "long" and current_price >= self.take_profit_price:
                return True, "take_profit"
            elif (
                self.side == "short"
                and current_price <= self.take_profit_price
            ):
                return True, "take_profit"

        return False, ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "total_amount": float(self.total_amount),
            "average_price": float(self.average_price),
            "realized_pnl": float(self.realized_pnl),
            "fees_paid": float(self.fees_paid),
            "entry_time": self.entry_time.isoformat(),
            "last_update": self.last_update.isoformat(),
            "highest_price": (
                float(self.highest_price) if self.highest_price else None
            ),
            "lowest_price": (
                float(self.lowest_price) if self.lowest_price else None
            ),
            "stop_loss_price": (
                float(self.stop_loss_price) if self.stop_loss_price else None
            ),
            "take_profit_price": (
                float(self.take_profit_price)
                if self.take_profit_price
                else None
            ),
            "trailing_stop_pct": (
                float(self.trailing_stop_pct)
                if self.trailing_stop_pct
                else None
            ),
            "metadata": self.metadata,
            "trades": [trade.to_dict() for trade in self.trades],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Position:
        """Create Position from dictionary."""
        data_copy = data.copy()
        data_copy["entry_time"] = datetime.fromisoformat(data["entry_time"])
        data_copy["last_update"] = datetime.fromisoformat(data["last_update"])
        data_copy["trades"] = [
            Trade.from_dict(t) for t in data.get("trades", [])
        ]

        # Handle optional decimal fields
        for fld in [
            "highest_price",
            "lowest_price",
            "stop_loss_price",
            "take_profit_price",
            "trailing_stop_pct",
        ]:
            if data_copy.get(fld) is not None:
                data_copy[fld] = Decimal(str(data_copy[fld]))

        return cls(**data_copy)


class TradeManager:
    """
    Centralized Trade Manager - Single Source of Truth

    This class manages all trades and positions with consistent calculations
    and provides a unified interface for the entire application.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(
            storage_path or "crypto_bot/logs/trade_manager_state.json"
        )
        self.lock = threading.RLock()

        # Core data structures
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.price_cache: Dict[str, Decimal] = {}
        self.closed_positions: List[Position] = []

        # Statistics
        self.total_trades = 0
        self.total_volume = Decimal("0")
        self.total_fees = Decimal("0")
        self.total_realized_pnl = Decimal("0")

        # Callbacks for real-time updates
        self.price_update_callbacks: List[callable] = []
        self.position_update_callbacks: List[callable] = []
        self.trade_callbacks: List[callable] = []

        # Sell operation locks to prevent duplicate concurrent operations
        self._sell_locks: set = set()
        self._lock_timeout = 300  # 5 minutes timeout for sell locks

        # Auto-save settings
        self.auto_save_enabled = (
            True  # Enable auto-save to persist price updates
        )
        self.auto_save_interval = 30  # seconds - match price monitor interval
        self.last_save_time = time.time()

        # Load existing state
        self._load_state()

        # Start auto-save thread to persist price updates
        self._start_auto_save()

    def _start_auto_save(self) -> None:
        """Start background thread for automatic state saving."""

        def auto_save_worker():
            while self.auto_save_enabled:
                time.sleep(self.auto_save_interval)
                try:
                    # Acquire lock before saving to prevent race conditions
                    with self.lock:
                        self.save_state()
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")

        self.auto_save_thread = threading.Thread(
            target=auto_save_worker, daemon=True
        )
        self.auto_save_thread.start()

    def add_price_callback(self, callback: callable) -> None:
        """Add callback for price updates."""
        self.price_update_callbacks.append(callback)

    def add_position_callback(self, callback: callable) -> None:
        """Add callback for position updates."""
        self.position_update_callbacks.append(callback)

    def add_trade_callback(self, callback: callable) -> None:
        """Add callback for trade events."""
        self.trade_callbacks.append(callback)

    def record_trade(self, trade: Trade) -> str:
        """
        Record a new trade and update position accordingly.

        Returns:
            The trade ID
        """
        with self.lock:
            # Add trade to history
            self.trades.append(trade)
            self.total_trades += 1
            self.total_volume += trade.total_value
            self.total_fees += trade.fees

            # Update position
            self._update_position_from_trade(trade)

            # Notify callbacks (with improved error isolation)
            failed_callbacks = []
            for i, callback in enumerate(self.trade_callbacks):
                try:
                    callback(trade)
                except Exception as e:
                    logger.error(f"Trade callback {i} failed: {e}")
                    failed_callbacks.append(callback)

            # Remove failed callbacks to prevent repeated failures
            for failed_callback in failed_callbacks:
                if failed_callback in self.trade_callbacks:
                    self.trade_callbacks.remove(failed_callback)

            # Trigger position update callbacks
            position = self.positions.get(trade.symbol)
            if position:
                failed_callbacks = []
                for i, callback in enumerate(self.position_update_callbacks):
                    try:
                        callback(position)
                    except Exception as e:
                        logger.error(f"Position callback {i} failed: {e}")
                        failed_callbacks.append(callback)

                # Remove failed callbacks
                for failed_callback in failed_callbacks:
                    if failed_callback in self.position_update_callbacks:
                        self.position_update_callbacks.remove(failed_callback)

            logger.info(
                f"Recorded trade: {trade.symbol} {trade.side} {trade.amount} @ {trade.price}"
            )
            try:
                self.save_state()
            except Exception as e:
                logger.error(f"Immediate save after record_trade failed: {e}")
            return trade.id

    def _update_position_from_trade(self, trade: Trade) -> None:
        """Update position based on new trade."""
        symbol = trade.symbol

        # Additional safety check - should already be caught by Trade.__post_init__
        if is_test_position(symbol):
            logger.error(
                f"Attempted to process trade for test position: {symbol}"
            )
            return

        position = self.positions.get(symbol)

        # Skip zero amount trades - they don't affect positions
        if trade.amount == 0:
            return

        if position is None:
            # Create new position
            position = Position(
                symbol=symbol,
                side="long" if trade.side == "buy" else "short",
                total_amount=trade.amount,
                average_price=trade.price,
                entry_time=trade.timestamp,
                fees_paid=trade.fees,
                highest_price=trade.price,  # Initialize with trade price
                lowest_price=trade.price,  # Initialize with trade price
            )
            position.trades.append(trade)
            self.positions[symbol] = position

            # Set up stop losses and take profits for new position
            self._setup_position_risk_management(position)

        else:
            # Update existing position
            # Determine trade direction: "buy" = long, "sell"/"short" = short
            trade_direction = "long" if trade.side == "buy" else "short"
            if position.side == trade_direction:
                # Same direction - add to position
                total_value = position.position_value + trade.total_value
                total_amount = position.total_amount + trade.amount
                position.average_price = total_value / total_amount
                position.total_amount = total_amount
                position.fees_paid += trade.fees
            else:
                # Opposite direction - reduce or reverse position
                pnl, remaining_position_amount = self._calculate_position_closure(
                    position, trade
                )

                position.realized_pnl += pnl
                self.total_realized_pnl += pnl

                # Calculate trade remaining (excess trade amount after closing position)
                trade_remaining = trade.amount - position.total_amount

                if trade_remaining > 0:
                    # Position reversal: close existing position and open new one in opposite direction
                    logger.info(f"Position reversal for {symbol}: closing {position.total_amount} and opening {trade_remaining} in opposite direction")

                    # Move current position to closed positions (fully closed portion)
                    closed_position = Position(
                        symbol=symbol,
                        side=position.side,
                        total_amount=position.total_amount,  # The amount being closed
                        average_price=position.average_price,
                        realized_pnl=position.realized_pnl,
                        fees_paid=position.fees_paid,
                        entry_time=position.entry_time,
                        last_update=trade.timestamp,
                        trades=position.trades.copy()  # Include all trades up to this point
                    )
                    closed_position.trades.append(trade)  # Add the closing trade
                    self.closed_positions.append(closed_position)

                    # Determine new position side
                    new_side = "short" if position.side == "long" else "long"

                    # Create new position with excess trade amount at the closing trade price
                    new_position = Position(
                        symbol=symbol,
                        side=new_side,
                        total_amount=trade_remaining,
                        average_price=trade.price,  # New position at closing trade price
                        entry_time=trade.timestamp,
                        fees_paid=trade.fees,
                    )

                    # Replace the old position with the new reversed position
                    self.positions[symbol] = new_position
                    new_position.trades.append(trade)  # Only the opening trade for the new position
                    new_position.last_update = trade.timestamp

                    # Set up stop losses and take profits for new reversed position
                    self._setup_position_risk_management(new_position)

                    logger.info(f"Position reversal complete: closed {position.side} position and opened {new_side} position for {symbol}")

                else:
                    # Full or partial closure - position is reduced or eliminated
                    position.total_amount = remaining_position_amount
                    position.fees_paid += trade.fees
                    position.trades.append(trade)
                    position.last_update = trade.timestamp

                    # Check if position is fully closed
                    if position.total_amount <= Decimal("0.00000001"):  # Use small epsilon for floating point comparison
                        logger.info(f"Position fully closed for {symbol}: moving to closed positions")
                        position.total_amount = Decimal("0")
                        self.closed_positions.append(position)
                        del self.positions[symbol]
                    else:
                        logger.debug(f"Partial position closure for {symbol}: remaining {position.total_amount}")

    def _calculate_position_closure(
        self, position: Position, closing_trade: Trade
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate PnL from closing part or all of a position.

        Returns:
            Tuple of (realized_pnl, remaining_amount)
        """
        if position.side == "long":
            # Closing long position with sell trade
            pnl_per_unit = closing_trade.price - position.average_price
        else:
            # Closing short position with buy trade
            pnl_per_unit = position.average_price - closing_trade.price

        # Calculate how much of the position we're closing
        close_amount = min(position.total_amount, closing_trade.amount)
        realized_pnl = pnl_per_unit * close_amount
        remaining_amount = position.total_amount - close_amount

        # CRITICAL: When partially closing a position, the average price of remaining shares
        # should NOT change. The current position's average price remains the same.
        # This is a fundamental accounting principle in position management.

        return realized_pnl, remaining_amount

    def _setup_position_risk_management(self, position: Position) -> None:
        """Set up stop losses and take profits for a position based on configuration."""
        try:
            # Load configuration
            import yaml
            from pathlib import Path

            config_path = Path(__file__).parent.parent / "config.yaml"
            if not config_path.exists():
                # Try alternative config locations
                alt_paths = [
                    Path(__file__).parent.parent.parent / "config.yaml",
                    Path(__file__).parent.parent.parent
                    / "crypto_bot"
                    / "config.yaml",
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        config_path = alt_path
                        break

            if not config_path.exists():
                logger.warning(
                    f"Could not find config.yaml for position risk management setup"
                )
                return

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            # Get risk management parameters
            risk_cfg = config.get("risk", {})
            exit_cfg = config.get("exit_strategy", {})

            # Use exit strategy config if available, otherwise fall back to risk config
            take_profit_pct = exit_cfg.get(
                "take_profit_pct", risk_cfg.get("take_profit_pct", 0.04)
            )
            stop_loss_pct = exit_cfg.get(
                "stop_loss_pct", risk_cfg.get("stop_loss_pct", 0.01)
            )  # Initial stop loss
            trailing_stop_pct = exit_cfg.get(
                "trailing_stop_pct", risk_cfg.get("trailing_stop_pct", 0.008)
            )  # Trailing stop

            # Calculate stop loss and take profit prices
            # Convert percentages to Decimal to avoid float/decimal multiplication issues
            stop_loss_decimal = Decimal(str(stop_loss_pct))
            take_profit_decimal = Decimal(str(take_profit_pct))

            if position.side == "long":
                # Long position: stop loss below entry price
                stop_loss_price = position.average_price * (
                    Decimal("1") - stop_loss_decimal
                )
                take_profit_price = position.average_price * (
                    Decimal("1") + take_profit_decimal
                )
            else:
                # Short position: stop loss above entry price
                stop_loss_price = position.average_price * (
                    Decimal("1") + stop_loss_decimal
                )
                take_profit_price = position.average_price * (
                    Decimal("1") - take_profit_decimal
                )

            # Set up the position stops
            position.stop_loss_price = stop_loss_price
            position.take_profit_price = take_profit_price
            position.trailing_stop_pct = Decimal(str(trailing_stop_pct))

            logger.info(
                f"Set up risk management for {position.symbol}: SL=${stop_loss_price:.6f}, TP=${take_profit_price:.6f}, TS={trailing_stop_pct*100}%"
            )

        except Exception as e:
            logger.error(
                f"Failed to set up risk management for {position.symbol}: {e}"
            )
            # Don't fail the trade if risk management setup fails

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update price for a symbol and trigger position updates."""
        with self.lock:
            old_price = self.price_cache.get(symbol)
            self.price_cache[symbol] = price

            # Update position price levels
            position = self.positions.get(symbol)
            if position:
                position.update_price_levels(price)
                position.update_trailing_stop(price)

                # Check for exit conditions
                should_exit, exit_reason = position.should_exit(price)
                if should_exit:
                    logger.info(
                        f"Exit condition met for {symbol}: {exit_reason}"
                    )

            # Notify price update callbacks
            if old_price != price:
                failed_callbacks = []
                for i, callback in enumerate(self.price_update_callbacks):
                    try:
                        callback(symbol, price)
                    except Exception as e:
                        logger.error(f"Price callback {i} failed: {e}")
                        failed_callbacks.append(callback)

                # Remove failed callbacks
                for failed_callback in failed_callbacks:
                    if failed_callback in self.price_update_callbacks:
                        self.price_update_callbacks.remove(failed_callback)

            # Save state immediately after price update for real-time persistence
            self.save_state()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return [pos for pos in self.positions.values() if pos.is_open]

    def get_closed_positions(self) -> List[Position]:
        """Get all closed positions."""
        return self.closed_positions

    def get_total_unrealized_pnl(self) -> Tuple[Decimal, Decimal]:
        """
        Calculate total unrealized PnL across all positions.

        Returns:
            Tuple of (total_pnl, total_percentage)
        """
        total_pnl = Decimal("0")
        total_value = Decimal("0")

        for position in self.get_all_positions():
            current_price = self.price_cache.get(position.symbol)
            if current_price:
                pnl, _ = position.calculate_unrealized_pnl(current_price)
                total_pnl += pnl
                total_value += position.position_value

        total_pct = (
            (total_pnl / total_value) * 100
            if total_value > 0
            else Decimal("0")
        )
        return total_pnl, total_pct

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        open_positions = self.get_all_positions()
        total_unrealized_pnl, total_unrealized_pct = (
            self.get_total_unrealized_pnl()
        )

        return {
            "total_trades": self.total_trades,
            "total_volume": float(self.total_volume),
            "total_fees": float(self.total_fees),
            "total_realized_pnl": float(self.total_realized_pnl),
            "total_unrealized_pnl": float(total_unrealized_pnl),
            "total_unrealized_pnl_pct": float(total_unrealized_pct),
            "total_pnl": float(self.total_realized_pnl + total_unrealized_pnl),
            "open_positions_count": len(open_positions),
            "closed_positions_count": len(self.get_closed_positions()),
            "positions": [pos.to_dict() for pos in open_positions],
            "last_update": datetime.utcnow().isoformat(),
        }

    def validate_position_consistency(self) -> Dict[str, Any]:
        """
        Validate position consistency and detect issues.

        Returns:
            Dict with validation results and any issues found
        """
        issues = []
        warnings = []

        try:
            # Check for positions with zero or negative amounts that are still open
            for symbol, position in self.positions.items():
                if position.total_amount <= 0:
                    issues.append(f"Position {symbol} has non-positive amount {position.total_amount} but is still open")
                    # Auto-fix: move to closed positions
                    logger.warning(f"Auto-fixing zero position for {symbol}")
                    position.total_amount = Decimal("0")
                    self.closed_positions.append(position)
                    del self.positions[symbol]

            # Check for duplicate positions (same symbol)
            symbol_counts = {}
            for symbol in self.positions.keys():
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

            for symbol, count in symbol_counts.items():
                if count > 1:
                    issues.append(f"Duplicate positions found for {symbol}: {count} instances")

            # Check for trades with inconsistent sides for the same symbol
            for symbol in self.positions.keys():
                position = self.positions[symbol]
                trades_for_symbol = [t for t in self.trades if t.symbol == symbol]

                if not trades_for_symbol:
                    warnings.append(f"No trades found for open position {symbol}")
                    continue

                # Check if position side matches the net effect of trades
                net_amount = sum(
                    (t.amount if t.side == "buy" else -t.amount)
                    for t in trades_for_symbol
                )

                expected_side = "long" if net_amount > 0 else "short"
                expected_amount = abs(net_amount)

                if position.side != expected_side:
                    issues.append(f"Position {symbol} side mismatch: position shows {position.side}, trades indicate {expected_side}")

                if abs(position.total_amount - expected_amount) > Decimal("0.0001"):
                    issues.append(f"Position {symbol} amount mismatch: position shows {position.total_amount}, trades indicate {expected_amount}")

            # Check for closed positions that still have amounts
            for position in self.closed_positions:
                if position.total_amount > 0:
                    warnings.append(f"Closed position {position.symbol} still has amount {position.total_amount}")

            # Validate trade amounts are reasonable
            for trade in self.trades:
                if trade.amount <= 0:
                    issues.append(f"Trade {trade.id} for {trade.symbol} has invalid amount {trade.amount}")

            logger.info(f"Position validation complete: {len(issues)} issues, {len(warnings)} warnings found")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "open_positions": len(self.positions),
                "closed_positions": len(self.closed_positions),
                "total_trades": len(self.trades)
            }

        except Exception as e:
            logger.error(f"Error during position validation: {e}")
            return {
                "valid": False,
                "issues": [f"Validation failed: {e}"],
                "warnings": [],
                "open_positions": len(self.positions),
                "closed_positions": len(self.closed_positions),
                "total_trades": len(self.trades)
            }

    def fix_data_inconsistencies(self) -> Dict[str, Any]:
        """
        Automatically fix common data inconsistencies.

        Returns:
            Dict with fixes applied and results
        """
        fixes_applied = []
        errors = []

        try:
            # Fix 1: Handle positions with zero/negative amounts
            positions_to_remove = []
            for symbol, position in self.positions.items():
                if position.total_amount <= Decimal("0.00000001"):
                    logger.info(f"Fixing zero position for {symbol}")
                    position.total_amount = Decimal("0")
                    self.closed_positions.append(position)
                    positions_to_remove.append(symbol)
                    fixes_applied.append(f"Moved zero position {symbol} to closed positions")

            for symbol in positions_to_remove:
                del self.positions[symbol]

            # Fix 2: Rebuild positions from trades to ensure consistency
            logger.info("Rebuilding positions from trade history to ensure consistency")

            # Clear existing positions and rebuild from trades
            original_positions = self.positions.copy()
            self.positions.clear()

            # Group trades by symbol and process chronologically
            trades_by_symbol = {}
            for trade in sorted(self.trades, key=lambda t: t.timestamp):
                if trade.symbol not in trades_by_symbol:
                    trades_by_symbol[trade.symbol] = []
                trades_by_symbol[trade.symbol].append(trade)

            # Rebuild positions for each symbol
            for symbol, symbol_trades in trades_by_symbol.items():
                position = None
                for trade in symbol_trades:
                    if position is None:
                        # Create new position
                        position = Position(
                            symbol=symbol,
                            side="long" if trade.side == "buy" else "short",
                            total_amount=trade.amount,
                            average_price=trade.price,
                            entry_time=trade.timestamp,
                            fees_paid=trade.fees,
                        )
                        position.trades.append(trade)
                        self.positions[symbol] = position
                    else:
                        # Update existing position
                        if position.side == ("long" if trade.side == "buy" else "short"):
                            # Same direction - add to position
                            total_value = position.position_value + trade.total_value
                            total_amount = position.total_amount + trade.amount
                            if total_amount > 0:
                                position.average_price = total_value / total_amount
                            position.total_amount = total_amount
                            position.fees_paid += trade.fees
                        else:
                            # Opposite direction - reduce position
                            pnl, remaining = self._calculate_position_closure(position, trade)
                            position.realized_pnl += pnl
                            self.total_realized_pnl += pnl
                            position.total_amount = remaining
                            position.fees_paid += trade.fees

                            # Check if position is closed
                            if position.total_amount <= Decimal("0.00000001"):
                                position.total_amount = Decimal("0")
                                position.trades.append(trade)
                                self.closed_positions.append(position)
                                del self.positions[symbol]
                                position = None
                                break
                            else:
                                position.trades.append(trade)

                        position.last_update = trade.timestamp

            # Compare with original positions and log differences
            for symbol in original_positions:
                if symbol not in self.positions and symbol not in [p.symbol for p in self.closed_positions]:
                    fixes_applied.append(f"Position {symbol} was lost during rebuild")
                elif symbol in self.positions:
                    orig_pos = original_positions[symbol]
                    new_pos = self.positions[symbol]
                    if (abs(orig_pos.total_amount - new_pos.total_amount) > Decimal("0.0001") or
                        orig_pos.side != new_pos.side):
                        fixes_applied.append(f"Position {symbol} corrected: {orig_pos.side} {orig_pos.total_amount} -> {new_pos.side} {new_pos.total_amount}")

            # Fix 3: Set up risk management for rebuilt positions
            for position in self.positions.values():
                if not position.stop_loss_price and not position.take_profit_price:
                    self._setup_position_risk_management(position)

            # Save the corrected state
            self.save_state()

            logger.info(f"Data inconsistency fixes applied: {len(fixes_applied)} corrections made")

            return {
                "success": True,
                "fixes_applied": fixes_applied,
                "errors": errors,
                "positions_rebuilt": len(self.positions),
                "positions_closed": len([p for p in self.closed_positions if p.total_amount == 0])
            }

        except Exception as e:
            error_msg = f"Error fixing data inconsistencies: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return {
                "success": False,
                "fixes_applied": fixes_applied,
                "errors": errors,
                "positions_rebuilt": len(self.positions),
                "positions_closed": len(self.closed_positions)
            }

    def get_trade_history(
        self, symbol: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Trade]:
        """Get trade history, optionally filtered by symbol."""
        trades = self.trades
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        if limit:
            trades = trades[-limit:]

        return trades

    def save_state(self) -> None:
        """Save current state to disk."""
        try:
            state = {
                "trades": [trade.to_dict() for trade in self.trades],
                "positions": {s: p.to_dict() for s, p in self.positions.items()},
                "closed_positions": [p.to_dict() for p in self.closed_positions],
                "price_cache": {s: float(p) for s, p in self.price_cache.items()},
                "statistics": {
                    "total_trades": self.total_trades,
                    "total_volume": float(self.total_volume),
                    "total_fees": float(self.total_fees),
                    "total_realized_pnl": float(self.total_realized_pnl),
                },
                "last_save_time": datetime.utcnow().isoformat(),
            }

            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(state, f, indent=2)

            logger.info(f"TradeManager state saved to {self.storage_path} with {len(self.positions)} positions")
            logger.info(f"Saved positions: {list(self.positions.keys())}")

        except Exception as e:
            logger.error(f"Failed to save TradeManager state: {e}")

    def cleanup_sell_locks(self) -> None:
        """Clean up sell locks (called periodically to prevent lock accumulation)."""
        # For now, just clear all locks periodically
        # In a production system, you'd want more sophisticated lock management
        if self._sell_locks:
            logger.info(f"Cleaning up {len(self._sell_locks)} sell locks")
            self._sell_locks.clear()

    def _load_state(self) -> None:
        """Load state from disk."""
        try:
            if not self.storage_path.exists():
                logger.info("No TradeManager state file found, attempting CSV replay")
                if not self._replay_from_csv():
                    logger.info("CSV replay not possible; starting fresh")
                return

            # Load state normally - TradeManager is single source of truth
            with open(self.storage_path, "r") as f:
                state = json.load(f)

            # Load trades - filter out test positions
            valid_trades = []
            filtered_trade_count = 0
            for trade_data in state.get("trades", []):
                symbol = trade_data.get("symbol", "")
                if not is_test_position(symbol):
                    try:
                        valid_trades.append(Trade.from_dict(trade_data))
                    except ValueError as e:
                        logger.warning(
                            f"Skipping invalid trade during state loading: {e}"
                        )
                        filtered_trade_count += 1
                else:
                    logger.warning(
                        f"Filtering out test trade with symbol: {symbol}"
                    )
                    filtered_trade_count += 1
            self.trades = valid_trades

            # Load positions - filter out test positions
            self.positions = {}
            filtered_position_count = 0
            for symbol, pos_data in state.get("positions", {}).items():
                if not is_test_position(symbol):
                    try:
                        self.positions[symbol] = Position.from_dict(pos_data)
                    except ValueError as e:
                        logger.warning(
                            f"Skipping invalid position during state loading: {e}"
                        )
                        filtered_position_count += 1
                else:
                    logger.warning(
                        f"Filtering out test position with symbol: {symbol}"
                    )
                    filtered_position_count += 1

            # Load closed positions
            self.closed_positions = []
            for pos_data in state.get("closed_positions", []):
                try:
                    self.closed_positions.append(Position.from_dict(pos_data))
                except ValueError as e:
                    logger.warning(
                        f"Skipping invalid closed position during state loading: {e}"
                    )

            # Load price cache
            self.price_cache = {}
            for symbol, price in state.get("price_cache", {}).items():
                self.price_cache[symbol] = Decimal(str(price))

            # Load statistics
            stats = state.get("statistics", {})
            self.total_trades = stats.get("total_trades", 0)
            self.total_volume = Decimal(str(stats.get("total_volume", 0)))
            self.total_fees = Decimal(str(stats.get("total_fees", 0)))
            self.total_realized_pnl = Decimal(
                str(stats.get("total_realized_pnl", 0))
            )

            logger.info(
                f"Loaded TradeManager state with {len(self.trades)} trades and {len(self.positions)} positions"
            )
            logger.info(f"Loaded positions: {list(self.positions.keys())}")
            if filtered_trade_count > 0 or filtered_position_count > 0:
                logger.info(
                    f"Filtered out {filtered_trade_count} test trades and {filtered_position_count} test positions during state loading"
                )

        except Exception as e:
            logger.error(f"Failed to load TradeManager state: {e}")
            if not self._replay_from_csv():
                # Start with clean state if loading fails and replay not possible
                self.trades = []
                self.positions = {}
                self.price_cache = {}

    def shutdown(self) -> None:
        """Shutdown the trade manager and save final state."""
        self.auto_save_enabled = False

        # Wait for auto-save thread to finish if it exists
        if (
            hasattr(self, "auto_save_thread")
            and self.auto_save_thread.is_alive()
        ):
            self.auto_save_thread.join(timeout=5.0)  # Wait up to 5 seconds

        # Final save with lock to ensure thread safety
        with self.lock:
            self.save_state()
        logger.info("TradeManager shutdown complete")

    def _replay_from_csv(self) -> bool:
        """Rebuild TradeManager state by replaying trades from trades.csv.

        Returns True if any trades were successfully replayed.
        """
        try:
            import csv
            from datetime import datetime as _dt

            csv_path = LOG_DIR / "trades.csv"
            if not csv_path.exists():
                logger.info("CSV replay requested but trades.csv not found")
                return False

            rows = []
            with csv_path.open("r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row or len(row) < 5:
                        continue
                    # Skip stop records if 6th column indicates is_stop True
                    if len(row) >= 6 and str(row[5]).strip().lower() == "true":
                        continue
                    rows.append(row)

            if not rows:
                logger.info("No tradable rows found in trades.csv for replay")
                return False

            def _parse_ts(ts: str) -> _dt:
                try:
                    return _dt.fromisoformat(ts)
                except Exception:
                    return _dt.min

            # Sort trades by timestamp to preserve order
            rows.sort(key=lambda r: _parse_ts(r[4]))

            replayed = 0
            for row in rows:
                try:
                    symbol, side, amount_str, price_str, ts_str = row[:5]
                    amount = Decimal(str(amount_str))
                    price = Decimal(str(price_str))
                    timestamp = _parse_ts(ts_str)

                    trade = Trade(
                        id=str(uuid.uuid4()),
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=price,
                        timestamp=timestamp,
                        strategy="replay_csv",
                        exchange="replay",
                    )
                    self.record_trade(trade)
                    replayed += 1
                except Exception as row_err:
                    logger.warning(
                        f"Skipping CSV row during replay due to error: {row_err}"
                    )
                    continue

            self.save_state()
            logger.info(f"Replayed {replayed} trades from CSV into TradeManager")
            return replayed > 0
        except Exception as e:
            logger.error(f"CSV replay failed: {e}")
            return False


# Global instance
_trade_manager_instance: Optional[TradeManager] = None
_trade_manager_lock = threading.Lock()
_signals_registered = False


def get_trade_manager() -> TradeManager:
    """Get the global TradeManager instance."""
    global _trade_manager_instance
    global _signals_registered
    if _trade_manager_instance is None:
        with _trade_manager_lock:
            if _trade_manager_instance is None:
                # Determine the correct path to the state file
                import os

                current_dir = os.getcwd()

                # Try different possible locations for the state file
                # Prefer absolute LOG_DIR to avoid CWD mismatches
                preferred_path = str((LOG_DIR / "trade_manager_state.json").resolve())
                possible_paths = [
                    preferred_path,
                    "crypto_bot/logs/trade_manager_state.json",  # Default relative path
                    "../crypto_bot/logs/trade_manager_state.json",  # From frontend directory
                    os.path.join(
                        current_dir, "crypto_bot/logs/trade_manager_state.json"
                    ),  # Absolute from current dir
                    os.path.join(
                        current_dir,
                        "../crypto_bot/logs/trade_manager_state.json",
                    ),  # Absolute from parent dir
                ]

                # Find the first existing state file
                state_file_path = None
                for path in possible_paths:
                    if Path(path).exists():
                        state_file_path = path
                        logger.info(
                            f"Found TradeManager state file at: {path}"
                        )
                        break

                if state_file_path:
                    _trade_manager_instance = TradeManager(
                        storage_path=state_file_path
                    )
                else:
                    # Use preferred absolute path even if file doesn't exist yet
                    logger.warning(
                        "No TradeManager state file found; initializing at preferred path"
                    )
                    _trade_manager_instance = TradeManager(
                        storage_path=preferred_path
                    )
                if not _signals_registered:
                    try:
                        import signal

                        def _tm_signal_handler(signum, frame):
                            try:
                                if _trade_manager_instance is not None:
                                    _trade_manager_instance.shutdown()
                            except Exception as e:
                                logger.error(
                                    f"Error during TradeManager shutdown on signal {signum}: {e}"
                                )

                        for _sig_name in ("SIGINT", "SIGTERM"):
                            _sig = getattr(signal, _sig_name, None)
                            if _sig is not None:
                                try:
                                    signal.signal(_sig, _tm_signal_handler)
                                except Exception:
                                    pass
                        _signals_registered = True
                    except Exception as e:
                        logger.warning(f"Failed to register signal handlers: {e}")
    return _trade_manager_instance


def reset_trade_manager() -> None:
    """Reset the global TradeManager instance to force reload."""
    global _trade_manager_instance
    with _trade_manager_lock:
        _trade_manager_instance = None
    logger.info("TradeManager global instance reset")


def create_trade(
    symbol: str,
    side: str,
    amount: Decimal,
    price: Decimal,
    strategy: str = "",
    exchange: str = "",
    fees: Decimal = Decimal("0"),
    order_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Trade:
    """Create a new Trade object with validation."""
    return Trade(
        id=str(uuid.uuid4()),
        symbol=symbol,
        side=side,
        amount=amount,
        price=price,
        timestamp=datetime.utcnow(),
        strategy=strategy,
        exchange=exchange,
        fees=fees,
        order_id=order_id,
        client_order_id=client_order_id,
        metadata=metadata or {},
    )


class PositionSyncManager:
    """
    Manages synchronization between TradeManager and legacy position systems
    during the migration to TradeManager as single source of truth.
    """

    def __init__(self, trade_manager: TradeManager):
        self.trade_manager = trade_manager
        self.logger = logging.getLogger(__name__)

    def sync_context_positions(
        self, ctx_positions: Dict[str, dict]
    ) -> Dict[str, dict]:
        """
        Sync legacy ctx.positions with TradeManager positions.
        Returns updated ctx.positions dict for backward compatibility.
        """
        try:
            # Get all open positions from TradeManager
            tm_positions = self.trade_manager.get_all_positions()

            # Clear and rebuild ctx.positions from TradeManager
            ctx_positions.clear()

            for tm_pos in tm_positions:
                # Convert TradeManager position to legacy format
                ctx_positions[tm_pos.symbol] = {
                    "entry_price": float(tm_pos.average_price),
                    "size": float(tm_pos.total_amount),
                    "side": tm_pos.side,
                    "symbol": tm_pos.symbol,
                    "pnl": 0.0,  # Will be calculated by existing logic
                    "highest_price": (
                        float(tm_pos.highest_price)
                        if tm_pos.highest_price
                        else float(tm_pos.average_price)
                    ),
                    "lowest_price": (
                        float(tm_pos.lowest_price)
                        if tm_pos.lowest_price
                        else float(tm_pos.average_price)
                    ),
                    "trailing_stop": (
                        float(tm_pos.stop_loss_price)
                        if tm_pos.stop_loss_price
                        else 0.0
                    ),
                    "timestamp": tm_pos.entry_time.isoformat(),
                    "strategy": "migrated_from_trade_manager",
                }

            self.logger.debug(
                f"Synced {len(tm_positions)} positions from TradeManager to ctx.positions"
            )
            return ctx_positions

        except Exception as e:
            self.logger.error(f"Error syncing context positions: {e}")
            return ctx_positions

    def sync_paper_wallet_positions(
        self,
        paper_wallet_positions: Dict[str, dict],
        paper_wallet=None,
        current_price: Optional[float] = None,
    ) -> Dict[str, dict]:
        """
        Sync paper wallet positions with TradeManager.
        This ensures paper wallet stays consistent with TradeManager.
        """
        try:
            tm_positions = self.trade_manager.get_all_positions()

            # Convert TradeManager positions to dict format
            tm_positions_dict = []
            for tm_pos in tm_positions:
                tm_positions_dict.append(
                    {
                        "symbol": tm_pos.symbol,
                        "side": tm_pos.side,
                        "total_amount": float(tm_pos.total_amount),
                        "entry_price": float(
                            tm_pos.average_price
                        ),  # Use entry_price for paper wallet compatibility
                        "fees_paid": float(tm_pos.fees_paid),
                        "entry_time": tm_pos.entry_time.isoformat(),
                    }
                )

            # Create current prices dict for PnL calculation
            current_prices = {}
            if current_price is not None:
                # Use provided current price for all symbols (simplified case)
                for tm_pos in tm_positions:
                    current_prices[tm_pos.symbol] = current_price
            else:
                # Use prices from TradeManager's price cache
                for tm_pos in tm_positions:
                    current_prices[tm_pos.symbol] = float(
                        self.trade_manager.price_cache.get(
                            tm_pos.symbol, tm_pos.average_price
                        )
                    )

            # Use paper wallet's sync method if available, otherwise fallback
            if paper_wallet and hasattr(
                paper_wallet, "sync_from_trade_manager"
            ):
                paper_wallet.sync_from_trade_manager(
                    tm_positions_dict, current_prices
                )
                return paper_wallet.positions
            else:
                # Fallback to manual sync
                paper_wallet_positions.clear()
                for tm_pos_dict in tm_positions_dict:
                    symbol = tm_pos_dict["symbol"]
                    current_price_val = current_prices.get(
                        symbol, tm_pos_dict["entry_price"]
                    )

                    # Calculate PnL
                    pnl_pct = (
                        (current_price_val - tm_pos_dict["entry_price"])
                        / tm_pos_dict["entry_price"]
                    ) * (1 if tm_pos_dict["side"] == "long" else -1)

                    paper_wallet_positions[symbol] = {
                        "symbol": symbol,
                        "side": tm_pos_dict["side"],
                        "amount": tm_pos_dict["total_amount"],
                        "entry_price": tm_pos_dict["entry_price"],
                        "current_price": current_price_val,
                        "pnl": pnl_pct,
                        "fees_paid": tm_pos_dict.get("fees_paid", 0.0),
                        "timestamp": tm_pos_dict.get("entry_time", ""),
                    }

            self.logger.debug(
                f"Synced {len(tm_positions)} positions from TradeManager to paper wallet"
            )
            return paper_wallet_positions

        except Exception as e:
            self.logger.error(f"Error syncing paper wallet positions: {e}")
            return paper_wallet_positions

    def validate_consistency(
        self,
        ctx_positions: Dict[str, dict],
        paper_wallet_positions: Dict[str, dict],
    ) -> bool:
        """
        Validate that all position systems are consistent.
        Returns True if consistent, False if there are mismatches.
        """
        try:
            tm_positions = self.trade_manager.get_all_positions()
            tm_count = len(tm_positions)
            ctx_count = len(ctx_positions)
            wallet_count = len(paper_wallet_positions)

            if tm_count != ctx_count or tm_count != wallet_count:
                self.logger.warning(
                    f"Position count mismatch: TM={tm_count}, ctx={ctx_count}, wallet={wallet_count}"
                )
                return False

            # Validate that symbols match
            tm_symbols = {pos.symbol for pos in tm_positions}
            ctx_symbols = set(ctx_positions.keys())
            wallet_symbols = set(paper_wallet_positions.keys())

            if tm_symbols != ctx_symbols or tm_symbols != wallet_symbols:
                self.logger.warning(
                    f"Position symbol mismatch: TM={tm_symbols}, ctx={ctx_symbols}, wallet={wallet_symbols}"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating position consistency: {e}")
            return False
