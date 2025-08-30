from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Any, Union
import asyncio
import inspect
import threading
import os

try:
    # For python-telegram-bot version 20.2
    from telegram import Bot
    try:
        from telegram.utils.request import Request
    except ImportError:
        Request = None
except ImportError:
    # Fallback if telegram is not available
    Bot = None
    Request = None

from .logger import LOG_DIR, setup_logger
from pathlib import Path


logger = setup_logger(__name__, LOG_DIR / "bot.log")

# Store allowed Telegram admin IDs parsed from the environment or configuration
_admin_ids: set[str] = set()


def set_admin_ids(admins: Optional[Union[Iterable[str], str, Any]]) -> None:
    """Configure allowed Telegram admin chat IDs."""
    global _admin_ids
    if admins is None:
        admins = os.getenv("TELE_CHAT_ADMINS", "")
    if isinstance(admins, str):
        parts = [a.strip() for a in admins.split(",") if a.strip()]
    elif isinstance(admins, Iterable):
        parts = [str(a).strip() for a in admins if str(a).strip()]
    else:
        parts = [str(admins).strip()] if str(admins).strip() else []
    _admin_ids = set(parts)


def is_admin(chat_id: str) -> bool:
    """Return ``True`` if ``chat_id`` is allowed to issue commands."""
    if not _admin_ids:
        return True  # No admin IDs set means no restrictions
    return str(chat_id) in _admin_ids


set_admin_ids(None)


def send_message(token: str, chat_id: str, text: str) -> Optional[str]:
    """Send ``text`` to ``chat_id`` using ``token``.

    Returns ``None`` on success or an error string on failure.
    """
    if Bot is None:
        return "Telegram library not available"
    
    try:
        if Request is not None:
            bot = Bot(token, request=Request(
                connection_pool_size=8,
                connect_timeout=30.0,
                read_timeout=30.0,
                write_timeout=30.0
            ))
        else:
            # Fallback for older versions
            bot = Bot(token)

        if bot is None:
            return "Failed to create Telegram bot instance"

        async def _send() -> None:
            try:
                if not hasattr(bot, 'send_message') or bot.send_message is None:
                    logger.error("Bot object has no send_message method")
                    return
                
                # Try with exponential backoff
                max_retries = 3
                base_timeout = 15.0
                
                for attempt in range(max_retries):
                    try:
                        timeout = base_timeout * (2 ** attempt)  # Exponential backoff: 15s, 30s, 60s
                        await asyncio.wait_for(
                            bot.send_message(chat_id=chat_id, text=text),
                            timeout=timeout
                        )
                        break  # Success, exit retry loop
                    except asyncio.TimeoutError:
                        if attempt < max_retries - 1:
                            logger.warning(
                                "Telegram message timeout (attempt %d/%d) for chat %s. Retrying with longer timeout...",
                                attempt + 1, max_retries, chat_id
                            )
                            await asyncio.sleep(1)  # Brief delay before retry
                        else:
                            logger.error(
                                "Telegram message timeout after %d attempts for chat %s. Message may be too long or network is slow.",
                                max_retries, chat_id
                            )
                    except Exception as exc:
                        if attempt < max_retries - 1:
                            logger.warning(
                                "Telegram send error (attempt %d/%d): %s. Retrying...",
                                attempt + 1, max_retries, exc
                            )
                            await asyncio.sleep(1)
                        else:
                            logger.error(
                                "Failed to send message after %d attempts: %s. Verify your Telegram token "
                                "and chat ID and ensure the bot has started a chat.",
                                max_retries, exc
                            )
                        break
            except Exception as e:
                logger.error("Unexpected error in _send: %s", e)

        if inspect.iscoroutinefunction(bot.send_message):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                loop.create_task(_send())
            else:
                asyncio.run(_send())
        else:
            bot.send_message(chat_id=chat_id, text=text)
        return None
    except Exception as e:  # pragma: no cover - network
        logger.error("Failed to send message: %s", e)
        return str(e)


class TelegramNotifier:
    """Simple notifier for sending Telegram messages."""

    def __init__(
        self,
        enabled: bool = True,
        token: str = "",
        chat_id: str = "",
        admins: Optional[Union[Iterable[str], str]] = None,
    ) -> None:
        self.enabled = enabled
        self.token = token
        self.chat_id = chat_id
        if admins:
            set_admin_ids(admins)
        # internal flag set to True after a failed send
        self._disabled = False
        # lock to serialize send attempts
        self._lock = threading.Lock()

    def notify(self, text: str) -> Optional[str]:
        """Send ``text`` if notifications are enabled and credentials exist."""
        if self._disabled or not self.enabled or not self.token or not self.chat_id:
            return None

        with self._lock:
            if self._disabled:
                return None
            # Use the local send_message function directly
            err = send_message(self.token, self.chat_id, text)
            if err is not None:
                self._disabled = True
                logger.error(
                    "Disabling Telegram notifications due to send failure: %s",
                    err,
                )
            return err

    @classmethod
    def from_config(cls, config: dict) -> "TelegramNotifier":
        """Create a notifier from a configuration dictionary."""
        admins = config.get("chat_admins") or config.get("admins")
        notifier = cls(
            token=config.get("token", ""),
            chat_id=config.get("chat_id", ""),
            enabled=config.get("enabled", True),
            admins=admins,
        )
        return notifier


def send_test_message(token: str, chat_id: str, text: str = "Test message") -> bool:
    """Send a short test message to verify Telegram configuration."""
    if not token or not chat_id:
        return False
    err = send_message(token, chat_id, text)
    return err is None


def check_telegram_health(token: str, chat_id: str) -> Dict[str, Any]:
    """Check Telegram bot health and connectivity."""
    health_info = {
        "status": "unknown",
        "response_time": None,
        "error": None,
        "recommendations": []
    }
    
    if not token or not chat_id:
        health_info["status"] = "invalid_config"
        health_info["error"] = "Missing token or chat_id"
        health_info["recommendations"].append("Check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables")
        return health_info
    
    try:
        import time
        start_time = time.time()
        err = send_message(token, chat_id, "Health check")
        response_time = time.time() - start_time
        
        if err is None:
            health_info["status"] = "healthy"
            health_info["response_time"] = response_time
            if response_time > 10:
                health_info["recommendations"].append("Response time is slow, consider increasing timeout")
        else:
            health_info["status"] = "error"
            health_info["error"] = str(err)
            health_info["recommendations"].append("Check bot token and chat ID")
            health_info["recommendations"].append("Ensure bot has permission to send messages")
            
    except Exception as e:
        health_info["status"] = "exception"
        health_info["error"] = str(e)
        health_info["recommendations"].append("Check network connectivity")
        health_info["recommendations"].append("Verify Telegram API is accessible")
    
    return health_info


def clear_paper_trading_cache(paper_wallet=None, context=None):
    """
    Clear the trade cache in paper trading mode to start fresh.
    
    This function resets:
    - Paper wallet positions and balance
    - Bot context positions
    - Trade history and statistics
    
    Args:
        paper_wallet: The paper wallet instance to reset
        context: The bot context instance to clear positions from
        
    Returns:
        str: Status message indicating what was cleared
    """
    cleared_items = []
    
    # Reset paper wallet if provided
    if paper_wallet and hasattr(paper_wallet, 'reset'):
        try:
            # Store initial balance before reset
            initial_balance = paper_wallet.initial_balance
            paper_wallet.reset()
            cleared_items.append(f"Paper wallet reset to ${initial_balance:.2f}")
        except Exception as e:
            cleared_items.append(f"Failed to reset paper wallet: {e}")
    
    # Clear context positions if provided
    if context and hasattr(context, 'positions'):
        try:
            position_count = len(context.positions)
            context.positions.clear()
            if position_count > 0:
                cleared_items.append(f"Cleared {position_count} open positions")
            else:
                cleared_items.append("No open positions to clear")
        except Exception as e:
            cleared_items.append(f"Failed to clear context positions: {e}")
    
    # Clear df_cache if available
    if context and hasattr(context, 'df_cache'):
        try:
            cache_size = sum(len(tf_cache) for tf_cache in context.df_cache.values())
            context.df_cache.clear()
            if cache_size > 0:
                cleared_items.append(f"Cleared {cache_size} cached data entries")
            else:
                cleared_items.append("No cached data to clear")
        except Exception as e:
            cleared_items.append(f"Failed to clear data cache: {e}")
    
    # Clear regime cache if available
    if context and hasattr(context, 'regime_cache'):
        try:
            regime_count = len(context.regime_cache)
            context.regime_cache.clear()
            if regime_count > 0:
                cleared_items.append(f"Cleared {regime_count} regime cache entries")
            else:
                cleared_items.append("No regime cache to clear")
        except Exception as e:
            cleared_items.append(f"Failed to clear regime cache: {e}")
    
    if not cleared_items:
        return "No cache items available to clear"
    
    return "Cache cleared successfully:\n" + "\n".join(f"• {item}" for item in cleared_items)
