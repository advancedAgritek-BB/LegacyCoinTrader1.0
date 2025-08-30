from __future__ import annotations

"""Utilities for sending human-readable trade summaries via Telegram."""

from typing import Optional

from .telegram import TelegramNotifier
from .logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "bot.log")


class TradeReporter:
    """Minimal reporter wrapper used by tests to ensure interface exists."""

    def __init__(self, notifier: TelegramNotifier):
        self.notifier = notifier

    def report_entry(self, symbol: str, strategy: str, score: float, direction: str) -> Optional[str]:
        return report_entry(self.notifier, symbol, strategy, score, direction)

    def report_exit(self, symbol: str, strategy: str, pnl: float, direction: str) -> Optional[str]:
        return report_exit(self.notifier, symbol, strategy, pnl, direction)


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
