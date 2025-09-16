from __future__ import annotations

from typing import Optional, Iterable, Any, Union, Dict
import asyncio
import inspect
import threading
import os
import time
from collections import deque

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


logger = setup_logger(__name__, str(LOG_DIR / "bot.log"))

# Store allowed Telegram admin IDs parsed from the environment or configuration
_admin_ids: set[str] = set()


class MessageRateLimiter:
    """Rate limiter for Telegram messages to prevent flooding."""

    def __init__(self, max_messages: int = 10, time_window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_messages: Maximum messages allowed in time window
            time_window: Time window in seconds
        """
        self.max_messages = max_messages
        self.time_window = time_window
        self.messages: deque[float] = deque()
        self._lock = threading.Lock()

    def can_send(self) -> bool:
        """Check if a message can be sent based on rate limits."""
        with self._lock:
            now = time.time()

            # Remove old messages outside the time window
            while self.messages and now - self.messages[0] > self.time_window:
                self.messages.popleft()

            # Check if we're within limits
            return len(self.messages) < self.max_messages

    def record_message(self) -> None:
        """Record that a message was sent."""
        with self._lock:
            self.messages.append(time.time())

    def get_remaining_time(self) -> float:
        """Get seconds until next message can be sent."""
        with self._lock:
            if len(self.messages) < self.max_messages:
                return 0.0

            now = time.time()
            oldest = self.messages[0]
            return max(0, self.time_window - (now - oldest))


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

    # Validate inputs first
    if not token or not chat_id or not text.strip():
        return "Invalid token, chat_id, or empty message"

    # Check if token looks valid (should start with a number followed by colon)
    if not token.count(':') == 1 or not token.split(':')[0].isdigit():
        return "Invalid Telegram bot token format"

    try:
        if Request is not None:
            bot = Bot(token, request=Request(
                connection_pool_size=8,   # Reduced to prevent exhaustion
                connect_timeout=10.0,     # Reduced from 30s
                read_timeout=15.0,        # Reduced from 30s
                write_timeout=10.0,       # Reduced from 30s
                pool_connections=8,       # Match pool size
                pool_maxsize=8,           # Match pool size
                max_retries=2,            # Reduced from 3
            ))
        else:
            # Fallback for older versions
            bot = Bot(token)

        if bot is None:
            return "Failed to create Telegram bot instance"

        async def _send() -> None:
            try:
                if not hasattr(bot, 'send_message'):
                    logger.error("Bot object has no send_message method")
                    return

                # Try with shorter timeouts and fewer retries to fail fast
                max_retries = 2
                base_timeout = 5.0  # Much shorter base timeout

                for attempt in range(max_retries):
                    try:
                        timeout = min(base_timeout + (attempt * 2), 10.0)
                        await asyncio.wait_for(
                            bot.send_message(chat_id=chat_id, text=text),
                            timeout=timeout
                        )
                        break  # Success, exit retry loop
                    except asyncio.TimeoutError:
                        if attempt < max_retries - 1:
                            logger.debug(
                                "Telegram timeout (attempt %d/%d) for %s",
                                attempt + 1, max_retries, chat_id
                            )
                            await asyncio.sleep(0.5)  # Shorter delay
                        else:
                            logger.warning(
                                "Telegram timeout after %d attempts for %s",
                                max_retries, chat_id
                            )
                            raise  # Re-raise to be caught by outer exception handler
                    except Exception as exc:
                        if attempt < max_retries - 1:
                            logger.debug(
                                "Telegram error (attempt %d/%d): %s",
                                attempt + 1, max_retries, exc
                            )
                            await asyncio.sleep(0.5)
                        else:
                            logger.warning(
                                "Failed to send message after %d attempts: %s",
                                max_retries, exc
                            )
                            raise  # Re-raise to be caught by outer exception handler
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
            # Synchronous send_message (older versions)
            try:
                bot.send_message(chat_id=chat_id, text=text)
            except Exception as e:
                logger.error("Sync send failed: %s", e)
                return str(e)
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
        # rate limiter to prevent message flooding (10 messages per minute)
        self._rate_limiter = MessageRateLimiter(
            max_messages=10, time_window=60)

    def notify(self, text: str) -> Optional[str]:
        """Send ``text`` if notifications are enabled and credentials exist."""
        if not self.enabled or not self.token or not self.chat_id:
            return None

        # Don't send if permanently disabled due to repeated failures
        if self._disabled:
            return None

        # Check rate limit
        if not self._rate_limiter.can_send():
            wait_time = self._rate_limiter.get_remaining_time()
            logger.debug(
                "Rate limited: %d messages in last 60s, waiting %.1fs",
                self._rate_limiter.max_messages, wait_time
            )
            return f"Rate limited: wait {wait_time:.1f}s"

        with self._lock:
            if self._disabled:
                return None

            # Use the local send_message function directly
            err = send_message(self.token, self.chat_id, text)

            if err is not None:
                # Only disable after multiple consecutive failures
                if not hasattr(self, '_failure_count'):
                    self._failure_count = 0
                self._failure_count += 1

                # More aggressive disabling for timeout errors
                if ("timeout" in str(err).lower() or
                        "timed out" in str(err).lower()):
                    failure_threshold = 3  # Disable faster for timeouts
                else:
                    failure_threshold = 5  # Normal threshold for other errors

                if self._failure_count >= failure_threshold:
                    self._disabled = True
                    logger.warning(
                        "Disabling Telegram notifications after %d failures. "
                        "Last error: %s",
                        self._failure_count, err,
                    )
                    logger.info(
                        "Telegram notifications disabled. To re-enable, "
                        "restart bot or call reset_notifications()"
                    )
                else:
                    logger.debug(
                        "Telegram send error (%d/%d): %s",
                        self._failure_count, failure_threshold, err,
                    )
            else:
                # Reset failure count on success and record the message
                self._failure_count = 0
                self._rate_limiter.record_message()

            return err

    def reset_notifications(self) -> None:
        """Reset notification state to re-enable after failures."""
        with self._lock:
            self._disabled = False
            self._failure_count = 0
            # Clear rate limiter history
            self._rate_limiter.messages.clear()
            logger.info("Telegram notifications reset and re-enabled")

    @classmethod
    def from_config(cls, config: dict) -> "TelegramNotifier":
        """Create a notifier from a configuration dictionary."""
        admins = config.get("chat_admins") or config.get("admins")

        # Check if enabled and has valid credentials
        enabled = config.get("enabled", True)
        token = config.get("token", "")
        chat_id = config.get("chat_id", "")

        # Auto-disable if no credentials provided
        if not token or not chat_id:
            enabled = False
            logger.info(
                "Telegram notifications disabled: missing token or chat_id")

        # Check for fail_silently option
        fail_silently = config.get("fail_silently", False)
        if fail_silently and not enabled:
            logger.debug(
                "Telegram notifications disabled silently via configuration")

        notifier = cls(
            token=token,
            chat_id=chat_id,
            enabled=enabled,
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
    health_info: Dict[str, Any] = {
        "status": "unknown",
        "response_time": None,
        "error": None,
        "recommendations": []
    }
    
    if not token or not chat_id:
        health_info["status"] = "invalid_config"
        health_info["error"] = "Missing token or chat_id"
        health_info["recommendations"].append(
            "Check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables")
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
                health_info["recommendations"].append(
                    "Response time is slow, consider increasing timeout")
        else:
            health_info["status"] = "error"
            health_info["error"] = str(err)
            health_info["recommendations"].append(
                "Check bot token and chat ID")
            health_info["recommendations"].append(
                "Ensure bot has permission to send messages")
            
    except Exception as e:
        health_info["status"] = "exception"
        health_info["error"] = str(e)
        health_info["recommendations"].append(
            "Check network connectivity")
        health_info["recommendations"].append(
            "Verify Telegram API is accessible")
    
    return health_info


def clear_paper_trading_cache(paper_wallet: Any = None, context: Any = None) -> str:
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
