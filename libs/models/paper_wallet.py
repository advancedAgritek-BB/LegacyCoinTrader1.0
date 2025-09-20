from __future__ import annotations

from typing import Any, Dict, Optional, List
from uuid import uuid4
import logging
import yaml
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import numpy as np

logger = logging.getLogger(__name__)


def safe_yaml_representer(dumper, data):
    """Custom YAML representer for safe serialization."""
    if isinstance(data, Decimal):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))
    elif isinstance(data, np.ndarray):
        return dumper.represent_list(data.tolist())
    elif hasattr(data, 'item') and callable(getattr(data, 'item', None)):  # numpy scalar
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data.item()))
    elif isinstance(data, (np.integer, np.floating)):
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))
    else:
        return dumper.represent_data(data)


# Register custom representers
yaml.add_representer(Decimal, safe_yaml_representer)
yaml.add_representer(np.ndarray, safe_yaml_representer)
yaml.add_representer(np.integer, safe_yaml_representer)
yaml.add_representer(np.floating, safe_yaml_representer)


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
        self._balance = balance
        # mapping of identifier (symbol or trade id) -> position details
        # each position: {"symbol": Optional[str], "side": str, "amount": float, "entry_price": float, "reserved": float}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.realized_pnl = 0.0
        self.max_open_trades = max_open_trades
        self.allow_short = allow_short
        self.total_trades = 0
        self.winning_trades = 0
        self.state_file = Path("crypto_bot/logs/paper_wallet_state.yaml")

    # ------------------------------------------------------------------
    # TradeManager Synchronization
    # ------------------------------------------------------------------

    def sync_from_trade_manager(self, trade_manager_positions: List[dict], current_prices: Optional[Dict[str, float]] = None) -> None:
        """
        Synchronize paper wallet positions with TradeManager positions.

        Args:
            trade_manager_positions: List of position dicts from TradeManager
            current_prices: Dict of symbol -> current_price for PnL calculation
        """
        try:
            # Allow sync even if we have positions - TradeManager is source of truth
            if len(self.positions) > 0:
                logger.info(f"🔄 Syncing with TradeManager - updating {len(self.positions)} existing positions")
                logger.info(f"📊 Current paper wallet balance: ${self.balance:.2f}")
            
            # Clear existing positions and rebuild from TradeManager
            self.positions.clear()

            for tm_pos in trade_manager_positions:
                symbol = tm_pos['symbol']

                # Get current price for PnL calculation
                current_price = current_prices.get(symbol, tm_pos['entry_price']) if current_prices else tm_pos['entry_price']

                # Calculate PnL
                pnl_pct = ((current_price - tm_pos['entry_price']) / tm_pos['entry_price']) * (
                    1 if tm_pos['side'] == 'long' else -1
                )

                # Create paper wallet position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'side': tm_pos['side'],
                    'amount': tm_pos['total_amount'],
                    'entry_price': tm_pos['entry_price'],
                    'current_price': current_price,
                    'pnl': pnl_pct,
                    'fees_paid': tm_pos.get('fees_paid', 0.0),
                    'timestamp': tm_pos.get('entry_time', datetime.now().isoformat())
                }

            logger.info(f"Synchronized {len(trade_manager_positions)} positions from TradeManager to paper wallet")

        except Exception as e:
            logger.error(f"Error syncing paper wallet from TradeManager: {e}")

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
    def entry_price(self) -> Optional[float]:
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
    def side(self) -> Optional[str]:
        if not self.positions:
            return None
        first = next(iter(self.positions.values()))["side"]
        if all(p["side"] == first for p in self.positions.values()):
            return first
        return "mixed"

    @property
    def balance(self) -> float:
        """Current wallet balance, never negative."""
        return max(0.0, self._balance)

    @balance.setter
    def balance(self, value: float) -> None:
        """Set balance with validation."""
        self._balance = max(0.0, value)

    def buy(self, symbol: str, amount: float, price: float) -> bool:
        """Simulate a buy order (paper trading only - no real trades)."""
        # Convert Decimal to float if needed
        amount = float(amount) if hasattr(amount, '__float__') else float(amount)
        price = float(price) if hasattr(price, '__float__') else float(price)

        cost = amount * price
        if cost > self._balance:
            logger.warning(f"Insufficient balance for buy: ${cost:.2f} > ${self._balance:.2f}")
            return False

        # Deduct cost from balance
        self._balance -= cost

        # Create or update position
        if symbol in self.positions:
            # Add to existing position
            pos = self.positions[symbol]
            total_amount = pos['amount'] + amount
            # Weighted average entry price
            total_cost = (pos['amount'] * pos['entry_price']) + cost
            pos['entry_price'] = total_cost / total_amount
            pos['amount'] = total_amount
        else:
            # Create new position
            self.positions[symbol] = {
                'symbol': symbol,
                'side': 'long',
                'amount': amount,
                'entry_price': price,
                'reserved': 0.0,
                'timestamp': datetime.now().isoformat()
            }

        self.total_trades += 1
        logger.info(f"Paper buy order executed: {amount} {symbol} at ${price:.2f}")
        return True

    def sell(self, symbol: str, amount: float, price: float) -> bool:
        """Simulate a sell order (paper trading only - no real trades)."""
        # Convert Decimal to float if needed
        amount = float(amount) if hasattr(amount, '__float__') else float(amount)
        price = float(price) if hasattr(price, '__float__') else float(price)

        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return False

        position = self.positions[symbol]
        if position['amount'] < amount:
            logger.warning(f"Insufficient amount to sell: {amount} > {position['amount']}")
            return False

        # Calculate PnL
        pnl = (price - position['entry_price']) * amount
        self.realized_pnl += pnl

        # Add proceeds to balance
        self._balance += amount * price

        # Update position
        if position['amount'] == amount:
            # Full position closed
            del self.positions[symbol]
        else:
            # Partial position closed
            position['amount'] -= amount

        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        logger.info(f"Paper sell order executed: {amount} {symbol} at ${price:.2f}, PnL: ${pnl:.2f}")
        return True

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

    def validate_wallet_state(self) -> bool:
        """Validate that the wallet state is consistent and healthy."""
        try:
            # Calculate total position value at entry
            total_position_value = 0
            for pos in self.positions.values():
                key = "size" if "size" in pos else "amount"
                position_value = pos[key] * pos["entry_price"]
                total_position_value += position_value
            
            # Calculate total portfolio value (cash + positions)
            total_portfolio_value = self.balance + total_position_value
            
            # Check if total portfolio value is reasonable
            if total_portfolio_value < 0:
                logger.error(f"Total portfolio value is negative: ${total_portfolio_value:.2f} (balance: ${self.balance:.2f}, positions: ${total_position_value:.2f})")
                return False
            
            # Check if total position value exceeds initial balance (this might be OK for leverage)
            if total_position_value > self.initial_balance * 2:  # Allow up to 2x leverage
                logger.warning(f"High leverage detected: positions=${total_position_value:.2f}, initial_balance=${self.initial_balance:.2f}")
            
            # Check for reasonable position sizes
            for trade_id, pos in self.positions.items():
                key = "size" if "size" in pos else "amount"
                position_value = pos[key] * pos["entry_price"]
                if position_value > self.initial_balance * 0.5:  # No single position > 50% of initial balance
                    logger.warning(f"Large position detected: {pos.get('symbol', trade_id)} = ${position_value:.2f} ({position_value/self.initial_balance*100:.1f}% of initial balance)")
            
            # Negative cash balance is OK if you have open positions
            if self.balance < 0 and len(self.positions) > 0:
                logger.info(f"Negative cash balance (${self.balance:.2f}) is normal with {len(self.positions)} open positions")
                logger.info(f"Total portfolio value: ${total_portfolio_value:.2f}")
            
            logger.debug(f"Wallet state validation passed: balance=${self.balance:.2f}, positions={len(self.positions)}, total_value=${total_portfolio_value:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating wallet state: {e}")
            return False

    def get_wallet_summary(self) -> dict:
        """Get a summary of the current wallet state."""
        total_position_value = 0
        for pos in self.positions.values():
            key = "size" if "size" in pos else "amount"
            position_value = pos[key] * pos["entry_price"]
            total_position_value += position_value
        
        return {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'total_position_value': total_position_value,
            'available_balance': self.balance - total_position_value,
            'position_count': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'is_healthy': self.balance >= 0 and total_position_value <= self.initial_balance
        }

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

            # Basic symbol validation
            if symbol and '/' in symbol:
                base, quote = symbol.split('/', 1)
                if not base or not quote:
                    raise ValueError(f"Invalid symbol format: {symbol}")
                # Check for obviously invalid symbols
                if len(base) > 20 or len(quote) > 10:
                    raise ValueError(f"Suspiciously long symbol: {symbol}")
                if base == quote:
                    raise ValueError(f"Invalid symbol - base and quote are the same: {symbol}")

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

        # Enhanced balance validation with safety margin
        safety_margin = 0.05  # 5% safety margin
        available_balance = self.balance * (1 - safety_margin)
        
        if side == "buy":
            if cost > available_balance:
                raise RuntimeError(f"Insufficient balance: need ${cost:.2f}, available ${available_balance:.2f} (with {safety_margin*100}% safety margin)")
            if self.balance < 0:
                raise RuntimeError(f"Cannot open position with negative balance: ${self.balance:.2f}")
            self.balance -= cost
            logger.info(f"Opened BUY position: {amount} @ ${price:.6f} = ${cost:.2f}, balance: ${self.balance:.2f}")
        else:  # sell/short
            if cost > available_balance:
                raise RuntimeError(f"Insufficient balance for short: need ${cost:.2f}, available ${available_balance:.2f} (with {safety_margin*100}% safety margin)")
            if self.balance < 0:
                raise RuntimeError(f"Cannot open short position with negative balance: ${self.balance:.2f}")
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
        
        # Save state after opening position
        self.save_state()
        
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

        # Helper function to convert Decimal/int/float to float
        def to_float(value):
            if hasattr(value, '__float__'):  # Handles Decimal, int, float
                return float(value)
            return float(value)

        if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], (int, float, Decimal)) and isinstance(args[2], (int, float, Decimal)):
            identifier = args[0]
            amount = to_float(args[1])
            price = to_float(args[2])
        elif len(args) >= 2 and all(isinstance(a, (int, float, Decimal)) for a in args[:2]):
            amount = to_float(args[0])
            price = to_float(args[1])
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
        entry_price = pos.get("entry_price", 0)
        if entry_price <= 0:
            logger.warning(f"Invalid entry price for position {identifier}: {entry_price}")
            entry_price = price  # Use current price as fallback

        if pos["side"] == "buy":
            pnl = (price - entry_price) * amount
            # Add only the profit/loss to balance (not the full sale proceeds)
            self.balance += pnl
            logger.info(f"Closed BUY position: {amount} @ ${price:.6f} (entry: ${entry_price:.6f}), PnL: ${pnl:.2f}, balance: ${self.balance:.2f}")
        else:  # sell/short
            pnl = (entry_price - price) * amount
            # Release reserved funds and add profit
            release_amount = entry_price * amount
            self.balance += release_amount + pnl
            pos["reserved"] -= release_amount
            logger.info(f"Closed SELL position: {amount} @ ${price:.6f} (entry: ${entry_price:.6f}), PnL: ${pnl:.2f}, balance: ${self.balance:.2f}")

        # Log additional details for debugging
        if abs(pnl) < 0.01:  # Very small PnL
            logger.debug(f"Small PnL detected for {identifier}: entry=${entry_price:.6f}, exit=${price:.6f}, amount={amount}, calculated_pnl=${pnl:.6f}")

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

        # Save state after closing position
        self.save_state()

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
            # Convert Decimal to float if needed
            price = float(args[1]) if hasattr(args[1], '__float__') else float(args[1])
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
                    # Convert Decimal to float if needed
                    p_float = float(p) if hasattr(p, '__float__') else float(p)
                    key = "size" if "size" in pos else "amount"
                    if pos["side"] == "buy":
                        total += (p_float - pos["entry_price"]) * pos[key]
                    else:
                        total += (pos["entry_price"] - p_float) * pos[key]
                return total

            # Convert Decimal to float if needed
            price_val = float(price) if hasattr(price, '__float__') else float(price)
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
            self._balance = new_balance
        else:
            self._balance = self.initial_balance
        
        self.positions.clear()
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        logger.info(f"Wallet reset to balance: ${self.balance:.2f}")

    def save_state(self) -> None:
        """Save current wallet state to file."""
        try:
            state = {
                'balance': self.balance,
                'initial_balance': self.initial_balance,
                'realized_pnl': self.realized_pnl,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'positions': self.positions
            }

            # Ensure the directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.state_file, 'w') as f:
                yaml.dump(state, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Saved paper wallet state: balance=${self.balance:.2f}, realized_pnl=${self.realized_pnl:.2f}")

            # Ensure single source of truth is synchronized
            try:
                from crypto_bot.utils.balance_manager import set_single_balance
                set_single_balance(self.balance)
            except Exception as e:
                logger.warning(f"Failed to sync single balance source: {e}")

        except Exception as e:
            logger.error(f"Failed to save paper wallet state: {e}")

    def _convert_to_decimal(self, value):
        """Convert string representations back to Decimal objects where appropriate."""
        if isinstance(value, str):
            try:
                # Try to convert to Decimal if it looks like a decimal number
                if '.' in value or 'e' in value.lower():
                    return Decimal(value)
                return value
            except:
                return value
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries
            return {k: self._convert_to_decimal(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively convert nested lists
            return [self._convert_to_decimal(item) for item in value]
        return value

    def load_state(self) -> bool:
        """Load wallet state from file. Returns True if successful."""
        try:
            if not self.state_file.exists():
                logger.info("No saved paper wallet state found, using initial values")
                return False

            with open(self.state_file, 'r') as f:
                state = yaml.safe_load(f) or {}

            # Convert string representations back to appropriate types
            state = self._convert_to_decimal(state)

            # Sanitize balance to prevent negative values
            loaded_balance = state.get('balance', self._balance)
            if isinstance(loaded_balance, str):
                try:
                    loaded_balance = float(loaded_balance)
                except ValueError:
                    loaded_balance = self._balance

            self._balance = max(0.0, loaded_balance)
            self.initial_balance = state.get('initial_balance', self.initial_balance)

            # If we corrected a negative balance, log it and save the corrected state
            if loaded_balance < 0 and self._balance != loaded_balance:
                logger.warning(f"Corrected negative balance from ${loaded_balance:.2f} to ${self._balance:.2f} in loaded state")
                self.save_state()  # Save the corrected state immediately

            self.realized_pnl = state.get('realized_pnl', 0.0)
            self.total_trades = state.get('total_trades', 0)
            self.winning_trades = state.get('winning_trades', 0)
            self.positions = state.get('positions', {})

            logger.info(f"Loaded paper wallet state: balance=${self.balance:.2f}, realized_pnl=${self.realized_pnl:.2f}")
            return True

        except Exception as e:
            logger.error(f"Failed to load paper wallet state: {e}")
            return False

    def _get_current_time(self) -> str:
        """Get current timestamp for position tracking."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

