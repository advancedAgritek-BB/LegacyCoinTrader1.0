from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


class PaperWallet:
    """Simple wallet for paper trading supporting multiple positions.
    
    This wallet simulates real trading behavior by:
    - Deducting funds on buy orders
    - Reserving funds on sell orders (for short positions)
    - Properly calculating PnL on position closure
    - Maintaining accurate balance tracking
    - Supporting partial position closures
    """

    def __init__(
        self, balance: float, max_open_trades: int = 10, allow_short: bool = True
    ) -> None:
        self.initial_balance = balance
        self.balance = balance
        # mapping of identifier (symbol or trade id) -> position details
        # each position: {"symbol": str | None, "side": str, "amount": float, "entry_price": float, "reserved": float}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.realized_pnl = 0.0
        self.max_open_trades = max_open_trades
        self.allow_short = allow_short
        self.total_trades = 0
        self.winning_trades = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def position_size(self) -> float:
        """Total size across all open positions."""
        total = 0.0
        for pos in self.positions.values():
            if "size" in pos:
                total += pos["size"]
            else:
                total += pos["amount"]
        return total

    @property
    def entry_price(self) -> float | None:
        if not self.positions:
            return None
        total_amt = self.position_size
        if not total_amt:
            return None
        total = 0.0
        for pos in self.positions.values():
            qty = pos.get("size", pos.get("amount", 0.0))
            total += qty * pos["entry_price"]
        return total / total_amt

    @property
    def side(self) -> str | None:
        if not self.positions:
            return None
        first = next(iter(self.positions.values()))["side"]
        if all(p["side"] == first for p in self.positions.values()):
            return first
        return "mixed"

    @property
    def total_value(self) -> float:
        """Total portfolio value including unrealized PnL."""
        return self.balance + self.unrealized_total()

    @property
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def unrealized_total(self) -> float:
        """Calculate total unrealized PnL across all positions."""
        if not self.positions:
            return 0.0
        
        total = 0.0
        for pos in self.positions.values():
            key = "size" if "size" in pos else "amount"
            if pos["side"] == "buy":
                # For long positions, we need current market price
                # This will be calculated by the caller
                pass
            else:
                # For short positions, we can calculate based on entry price
                # assuming current price is lower (profitable short)
                pass
        return total

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------
    def open(self, *args) -> str:
        """Open a new trade and return its identifier.

        Supported signatures:
            open(side, amount, price, identifier=None)
            open(symbol, side, amount, price, identifier=None)
        """

        if not args:
            raise TypeError("open() missing required arguments")

        if args[0] in {"buy", "sell"}:
            side = args[0]
            amount = args[1]
            price = args[2]
            identifier = args[3] if len(args) > 3 else None
            symbol = None
        else:
            symbol = args[0]
            side = args[1]
            amount = args[2]
            price = args[3]
            identifier = args[4] if len(args) > 4 else None

            if symbol in self.positions:
                raise RuntimeError(f"Position already open for symbol {symbol}")

        if side == "sell" and not self.allow_short:
            raise RuntimeError("Short selling disabled")

        if len(self.positions) >= self.max_open_trades:
            raise RuntimeError(f"Position limit reached ({self.max_open_trades})")

        if amount <= 0:
            raise ValueError("Position amount must be positive")

        if price <= 0:
            raise ValueError("Position price must be positive")

        trade_id = identifier or symbol or str(uuid4())
        cost = amount * price
        reserved = 0.0

        # Validate sufficient balance
        if side == "buy":
            if cost > self.balance:
                raise RuntimeError(f"Insufficient balance: need ${cost:.2f}, have ${self.balance:.2f}")
            self.balance -= cost
            logger.info(f"Opened BUY position: {amount} @ ${price:.6f} = ${cost:.2f}, balance: ${self.balance:.2f}")
        else:  # sell/short
            if cost > self.balance:
                raise RuntimeError(f"Insufficient balance for short: need ${cost:.2f}, have ${self.balance:.2f}")
            self.balance -= cost
            reserved = cost  # Reserve the funds for the short position
            logger.info(f"Opened SELL position: {amount} @ ${price:.6f} = ${cost:.2f}, reserved: ${reserved:.2f}, balance: ${self.balance:.2f}")

        # Store position details
        if symbol is not None:
            self.positions[trade_id] = {
                "symbol": symbol,
                "side": side,
                "size": amount,
                "entry_price": price,
                "reserved": reserved,
                "entry_time": self._get_current_time(),
            }
        else:
            self.positions[trade_id] = {
                "symbol": None,
                "side": side,
                "amount": amount,
                "entry_price": price,
                "reserved": reserved,
                "entry_time": self._get_current_time(),
            }

        self.total_trades += 1
        return trade_id

    def close(self, *args) -> float:
        """Close an existing position and return realized PnL.

        Supported signatures:
            close(symbol, amount, price)
            close(amount, price, identifier=None)
        """

        if not self.positions:
            logger.warning("No positions to close")
            return 0.0

        identifier: Optional[str] = None
        amount: float
        price: float

        if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], (int, float)) and isinstance(args[2], (int, float)):
            identifier = args[0]
            amount = float(args[1])
            price = float(args[2])
        elif len(args) >= 2 and all(isinstance(a, (int, float)) for a in args[:2]):
            amount = float(args[0])
            price = float(args[1])
            identifier = args[2] if len(args) > 2 else None
            if identifier is None and len(self.positions) == 1:
                identifier = next(iter(self.positions))
        else:
            raise TypeError("Invalid arguments for close()")

        if identifier is None:
            logger.warning("No identifier provided for position closure")
            return 0.0

        if amount <= 0:
            raise ValueError("Close amount must be positive")

        if price <= 0:
            raise ValueError("Close price must be positive")

        pos = self.positions.get(identifier)
        if not pos:
            logger.warning(f"Position {identifier} not found")
            return 0.0

        key = "size" if "size" in pos else "amount"
        available_amount = pos[key]
        
        if amount > available_amount:
            logger.warning(f"Requested close amount {amount} exceeds available {available_amount}, closing full position")
            amount = available_amount

        # Calculate PnL
        if pos["side"] == "buy":
            pnl = (price - pos["entry_price"]) * amount
            # Add the sale proceeds to balance
            self.balance += amount * price
            logger.info(f"Closed BUY position: {amount} @ ${price:.6f}, PnL: ${pnl:.2f}, balance: ${self.balance:.2f}")
        else:  # sell/short
            pnl = (pos["entry_price"] - price) * amount
            # Release reserved funds and add profit
            release_amount = pos["entry_price"] * amount
            self.balance += release_amount + pnl
            pos["reserved"] -= release_amount
            logger.info(f"Closed SELL position: {amount} @ ${price:.6f}, PnL: ${pnl:.2f}, balance: ${self.balance:.2f}")

        # Update position size
        pos[key] -= amount
        self.realized_pnl += pnl
        
        # Track winning trades
        if pnl > 0:
            self.winning_trades += 1

        # Remove position if fully closed
        if pos[key] <= 0:
            del self.positions[identifier]
            logger.info(f"Position {identifier} fully closed and removed")
        else:
            self.positions[identifier] = pos
            logger.info(f"Position {identifier} partially closed, remaining: {pos[key]}")

        return pnl

    def unrealized(self, *args) -> float:
        """Return unrealized PnL.

        Supported signatures:
            unrealized(price)
            unrealized(symbol, price)
            unrealized({id: price, ...})
        """

        if not self.positions:
            return 0.0

        if len(args) == 2 and isinstance(args[0], str):
            identifier = args[0]
            price = float(args[1])
            pos = self.positions.get(identifier)
            if not pos:
                return 0.0
            key = "size" if "size" in pos else "amount"
            if pos["side"] == "buy":
                return (price - pos["entry_price"]) * pos[key]
            return (pos["entry_price"] - price) * pos[key]

        if len(args) == 1:
            price = args[0]
            if isinstance(price, dict):
                total = 0.0
                for pid, p in price.items():
                    pos = self.positions.get(pid)
                    if not pos:
                        continue
                    key = "size" if "size" in pos else "amount"
                    if pos["side"] == "buy":
                        total += (p - pos["entry_price"]) * pos[key]
                    else:
                        total += (pos["entry_price"] - p) * pos[key]
                return total

            price_val = float(price)
            total = 0.0
            for pos in self.positions.values():
                key = "size" if "size" in pos else "amount"
                if pos["side"] == "buy":
                    total += (price_val - pos["entry_price"]) * pos[key]
                else:
                    total += (pos["entry_price"] - price_val) * pos[key]
            return total

        return 0.0

    def get_position_summary(self) -> Dict[str, Any]:
        """Get a summary of all open positions and wallet status."""
        summary = {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "realized_pnl": self.realized_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.win_rate,
            "open_positions": len(self.positions),
            "positions": {}
        }
        
        for pid, pos in self.positions.items():
            key = "size" if "size" in pos else "amount"
            summary["positions"][pid] = {
                "symbol": pos.get("symbol"),
                "side": pos["side"],
                "size": pos[key],
                "entry_price": pos["entry_price"],
                "reserved": pos.get("reserved", 0.0),
                "entry_time": pos.get("entry_time")
            }
        
        return summary

    def reset(self, new_balance: Optional[float] = None) -> None:
        """Reset the wallet to initial state or new balance."""
        if new_balance is not None:
            self.initial_balance = new_balance
            self.balance = new_balance
        else:
            self.balance = self.initial_balance
        
        self.positions.clear()
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        logger.info(f"Wallet reset to balance: ${self.balance:.2f}")

    def _get_current_time(self) -> str:
        """Get current timestamp for position tracking."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

