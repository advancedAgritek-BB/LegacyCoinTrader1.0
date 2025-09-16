from __future__ import annotations
from typing import Optional

"""Log active trade position and wallet balance."""


from .logger import LOG_DIR, setup_logger

LOG_FILE = str(LOG_DIR / "positions.log")
logger = setup_logger(__name__, LOG_FILE)


def log_balance(balance: float) -> None:
    """Write the current wallet balance to the log in USD."""
    # Ensure we never log negative balances - use 0.0 as minimum
    safe_balance = max(0.0, balance)
    logger.info("Wallet balance $%.2f", safe_balance)


def log_position(
    symbol: str,
    side: str,
    amount: float,
    entry_price: float,
    current_price: float,
    balance: float,
    pnl: Optional[float] = None,
    exit_reason: Optional[str] = None,
) -> None:
    """Write a log entry describing the active position.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. ``"XBT/USDT"``.
    side : str
        ``"buy"`` or ``"sell"``.
    amount : float
        Position size.
    entry_price : float
        Price when the position was opened. Logged with six decimal places.
    current_price : float
        Latest market price. Logged with six decimal places.
    balance : float
        Current wallet balance including unrealized PnL.
    pnl : float, optional
        Realized profit or loss to log instead of computing from prices.
    exit_reason : str, optional
        Reason for position exit (e.g., "stop_loss", "take_profit", "manual").
    """
    if pnl is None:
        if side == "buy":
            pnl = (current_price - entry_price) * amount
        else:  # sell/short
            pnl = (entry_price - current_price) * amount
    status = "positive" if pnl >= 0 else "negative"
    # Ensure we never log negative balances - use 0.0 as minimum
    safe_balance = max(0.0, balance)

    # Build log message based on whether this is an exit or active position
    if exit_reason:
        logger.info(
            "Position exit %s %s %.4f entry %.6f exit %.6f "
            "pnl $%.2f (%s) balance $%.2f reason: %s",
            symbol,
            side,
            amount,
            entry_price,
            current_price,
            pnl,
            status,
            safe_balance,
            exit_reason,
        )
    else:
        logger.info(
            "Active %s %s %.4f entry %.6f current %.6f "
            "pnl $%.2f (%s) balance $%.2f",
            symbol,
            side,
            amount,
            entry_price,
            current_price,
            pnl,
            status,
            safe_balance,
        )
