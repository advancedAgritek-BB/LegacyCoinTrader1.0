from __future__ import annotations

"""Utilities for sending human-readable trade summaries via Telegram."""

from typing import Optional

from .telegram import TelegramNotifier
from .logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "bot.log")


class TradeReporter:
    """Minimal reporter wrapper used by tests to ensure interface exists."""

    def __init__(self, notifier: TelegramNotifier = None, telegram_enabled: bool = True):
        self.notifier = notifier
        self.telegram_enabled = telegram_enabled

    def report_entry(self, symbol_or_trade, strategy: str = None, score: float = None, direction: str = None) -> Optional[str]:
        """Report a trade entry. Can be called with individual params or trade dict."""
        if not self.telegram_enabled or not self.notifier:
            return None
        
        if isinstance(symbol_or_trade, dict):
            # Called with trade dict - use the format method
            message = self._format_entry_message(symbol_or_trade)
            return self.notifier.notify(message)
        else:
            # Called with individual parameters - use entry_summary
            symbol = symbol_or_trade
            strategy = strategy or 'Unknown'
            score = score or 0.0
            direction = direction or 'Unknown'
            return self.notifier.notify(entry_summary(symbol, strategy, score, direction))

    def report_exit(self, symbol_or_trade, strategy: str = None, pnl: float = None, direction: str = None) -> Optional[str]:
        """Report a trade exit. Can be called with individual params or trade dict."""
        if not self.telegram_enabled or not self.notifier:
            return None
        
        if isinstance(symbol_or_trade, dict):
            # Called with trade dict - use the format method
            message = self._format_exit_message(symbol_or_trade)
            return self.notifier.notify(message)
        else:
            # Called with individual parameters - use exit_summary
            symbol = symbol_or_trade
            strategy = strategy or 'Unknown'
            pnl = pnl or 0.0
            direction = direction or 'Unknown'
            return self.notifier.notify(exit_summary(symbol, strategy, pnl, direction))

    def _format_entry_message(self, trade: dict) -> str:
        """Format entry message from trade data."""
        symbol = trade.get('symbol', 'Unknown')
        side = trade.get('side', 'Unknown').upper()
        amount = trade.get('amount', 0)
        price = trade.get('price', 0)
        return f"Entering {side} on {symbol}: {amount} @ ${price}"

    def _format_exit_message(self, trade: dict) -> str:
        """Format exit message from trade data."""
        symbol = trade.get('symbol', 'Unknown')
        side = trade.get('side', 'Unknown').upper()
        amount = trade.get('amount', 0)
        price = trade.get('price', 0)
        pnl = trade.get('pnl', 0)
        return f"Exiting {side} on {symbol}: {amount} @ ${price} | PnL: ${pnl}"


def entry_summary(symbol: str, strategy: str, score: float, direction: str) -> str:
    """Return a summary of a trade entry."""
    return (
        f"Entering {direction.upper()} on {symbol} using {strategy}. "
        f"Score: {score:.2f}"
    )


def exit_summary(symbol: str, strategy: str, pnl: float, direction: str) -> str:
    """Return a summary of a trade exit."""
    return (
        f"Exiting {direction.upper()} on {symbol} from {strategy}. "
        f"PnL: {pnl:.2f}"
    )


def report_entry(*args) -> Optional[str]:
    """Send a Telegram message summarizing a trade entry."""
    if isinstance(args[0], TelegramNotifier):
        notifier, symbol, strategy, score, direction = args
    else:
        token, chat_id, symbol, strategy, score, direction = args
        notifier = TelegramNotifier(token=token, chat_id=chat_id)
    err = notifier.notify(entry_summary(symbol, strategy, score, direction))
    if err:
        logger.error("Failed to report entry: %s", err)
    return err


def report_exit(*args) -> Optional[str]:
    """Send a Telegram message summarizing a trade exit."""
    if isinstance(args[0], TelegramNotifier):
        notifier, symbol, strategy, pnl, direction = args
    else:
        token, chat_id, symbol, strategy, pnl, direction = args
        notifier = TelegramNotifier(token=token, chat_id=chat_id)
    err = notifier.notify(exit_summary(symbol, strategy, pnl, direction))
    if err:
        logger.error("Failed to report exit: %s", err)
    return err


__all__ = [
    "TelegramNotifier",
    "TradeReporter",
    "entry_summary",
    "exit_summary",
    "report_entry",
    "report_exit",
]
