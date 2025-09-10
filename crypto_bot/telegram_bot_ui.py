from __future__ import annotations

import asyncio
import threading
import time
import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Union


import schedule

try:  # pragma: no cover - optional dependency
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        CallbackQueryHandler,
        CommandHandler,
        ConversationHandler,
        MessageHandler,
        filters,
    )
except Exception:  # pragma: no cover - telegram not installed
    InlineKeyboardButton = InlineKeyboardMarkup = Update = object  # type: ignore
    Application = ApplicationBuilder = object  # type: ignore
    CallbackQueryHandler = CommandHandler = ConversationHandler = MessageHandler = object  # type: ignore
    filters = object  # type: ignore

# Remove circular import - define _paginate function locally if needed
def _paginate(items: List[str], page: int = 0, items_per_page: int = 10) -> tuple[List[str], int, int]:
    """Paginate items into pages."""
    start = page * items_per_page
    end = start + items_per_page
    total_pages = (len(items) + items_per_page - 1) // items_per_page
    return items[start:end], page, total_pages

from crypto_bot.portfolio_rotator import PortfolioRotator
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.telegram import TelegramNotifier, is_admin
from crypto_bot.log_reader import trade_summary

# Remove duplicate imports that cause circular dependencies
# from crypto_bot import log_reader, console_monitor
from .telegram_ctl import BotController
from crypto_bot.utils.open_trades import get_open_trades

START = "START"
STOP = "STOP"
STATUS = "STATUS"
LOG = "LOG"
ROTATE = "ROTATE"
TOGGLE = "TOGGLE"
MENU = "MENU"
RELOAD = "RELOAD"
SIGNALS = "SIGNALS"
BALANCE = "BALANCE"
TRADES = "TRADES"
TRADE_HISTORY = "TRADE_HISTORY"
PANIC_SELL = "PANIC_SELL"
CONFIG = "CONFIG"
EDIT_TRADE_SIZE = "EDIT_TRADE_SIZE"
EDIT_MAX_TRADES = "EDIT_MAX_TRADES"
EDIT_VALUE = 0
PNL_STATS = "PNL_STATS"

ASSET_SCORES_FILE = LOG_DIR / "asset_scores.json"
SIGNALS_FILE = LOG_DIR / "asset_scores.json"
TRADES_FILE = LOG_DIR / "trades.csv"
CONFIG_FILE = Path("crypto_bot/config.yaml")

# Text sent via ``TelegramNotifier`` when the bot starts.
MENU_TEXT = "Select a command:"


def _back_to_menu_markup() -> InlineKeyboardMarkup:
    """Return a markup with a single 'Back to Menu' button."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Back to Menu", callback_data=MENU)]]
    )


class TelegramBotUI:
    """Simple Telegram UI for controlling the trading bot."""

    def __init__(
        self,
        notifier: TelegramNotifier,
        state: Dict[str, object],
        log_file: Union[Path, str],
        rotator: Optional[PortfolioRotator] = None,
        exchange: Optional[object] = None,
        wallet: Optional[str] = None,
        command_cooldown: float = 5,
        paper_wallet: Optional[object] = None,
    ) -> None:
        self.notifier = notifier
        self.token = notifier.token
        self.chat_id = notifier.chat_id
        self.state = state
        self.log_file = Path(log_file)
        self.rotator = rotator
        self.controller = BotController(state, exchange, log_file=self.log_file, trades_file=TRADES_FILE, paper_wallet=paper_wallet)
        self.exchange = exchange
        self.wallet = wallet
        self.paper_wallet = paper_wallet
        self.command_cooldown = command_cooldown
        self._last_exec: Dict[Tuple[str, str], float] = {}
        self.logger = setup_logger(__name__, LOG_DIR / "telegram_ui.log")
        
        # Process lock to prevent multiple instances
        self.lock_file = LOG_DIR / "telegram_bot.lock"
        self.lock_fd = None
        self._acquire_lock()

        self.app = ApplicationBuilder().token(self.token).build()
        if hasattr(self.app, "bot_data"):
            self.app.bot_data["controller"] = self.controller
            self.app.bot_data["admin_id"] = self.chat_id
        self.app.add_handler(CommandHandler("start", self.start_cmd))
        self.app.add_handler(CommandHandler("stop", self.stop_cmd))
        self.app.add_handler(CommandHandler("status", self.status_cmd))
        self.app.add_handler(CommandHandler("log", self.log_cmd))
        self.app.add_handler(CommandHandler("rotate_now", self.rotate_now_cmd))
        self.app.add_handler(CommandHandler("toggle_mode", self.toggle_mode_cmd))
        self.app.add_handler(CommandHandler("reload", self.reload_cmd))
        self.app.add_handler(CommandHandler("menu", self.menu_cmd))
        self.app.add_handler(CommandHandler("signals", self.show_signals))
        self.app.add_handler(CommandHandler("balance", self.show_balance))
        self.app.add_handler(CommandHandler("trades", self.show_trades))
        self.app.add_handler(CommandHandler("trade_history", self.show_trade_history))
        self.app.add_handler(CommandHandler("config", self.show_config))
        self.app.add_handler(CommandHandler("pnl_stats", self.show_pnl_stats))
        self.app.add_handler(CommandHandler("panic_sell", self.panic_sell_cmd))
        self.app.add_handler(CallbackQueryHandler(self.start_cmd, pattern=f"^{START}$"))
        self.app.add_handler(CallbackQueryHandler(self.stop_cmd, pattern=f"^{STOP}$"))
        self.app.add_handler(CallbackQueryHandler(self.status_cmd, pattern=f"^{STATUS}$"))
        self.app.add_handler(CallbackQueryHandler(self.log_cmd, pattern=f"^{LOG}$"))
        self.app.add_handler(CallbackQueryHandler(self.rotate_now_cmd, pattern=f"^{ROTATE}$"))
        self.app.add_handler(CallbackQueryHandler(self.toggle_mode_cmd, pattern=f"^{TOGGLE}$"))
        self.app.add_handler(CallbackQueryHandler(self.reload_cmd, pattern=f"^{RELOAD}$"))
        self.app.add_handler(CallbackQueryHandler(self.menu_cmd, pattern=f"^{MENU}$"))
        self.app.add_handler(
            CallbackQueryHandler(self.show_signals, pattern=f"^{SIGNALS}$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.show_balance, pattern=f"^{BALANCE}$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.show_trades, pattern=f"^{TRADES}$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.show_trade_history, pattern=f"^{TRADE_HISTORY}$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.show_trade_history, pattern="^(next|prev)$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.panic_sell_cmd, pattern=f"^{PANIC_SELL}$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.show_config, pattern=f"^{CONFIG}$")
        )

        conv = ConversationHandler(
            entry_points=[
                CallbackQueryHandler(self.edit_trade_size, pattern=f"^{EDIT_TRADE_SIZE}$"),
                CallbackQueryHandler(self.edit_max_trades, pattern=f"^{EDIT_MAX_TRADES}$"),
            ],
            states={
                EDIT_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.set_config_value)]
            },
            fallbacks=[],
            per_message=True,  # Explicitly set to fix PTBUserWarning
        )
        self.app.add_handler(conv)
        self.app.add_handler(
            CallbackQueryHandler(self.show_pnl_stats, pattern=f"^{PNL_STATS}$")
        )
        self.app.add_handler(CommandHandler("clear_cache", self.clear_cache_cmd))
        self.app.add_handler(
            CallbackQueryHandler(self.clear_cache_cmd, pattern="^clear_cache$")
        )

        self.scheduler_thread: Optional[threading.Thread] = None

        schedule.every().day.at("00:00").do(self.send_daily_summary)
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler, daemon=True
        )
        self.scheduler_thread.start()

        self.task: Optional[asyncio.Task] = None

    async def clear_cache_cmd(self, update: Update, context: Any) -> None:
        """Clear the paper trading cache to start fresh."""
        if not await self._check_cooldown(update, "clear_cache"):
            return
        if not await self._check_admin(update):
            return

        try:
            # Import here to avoid circular imports
            try:
                from crypto_bot.utils.telegram import clear_paper_trading_cache
            except ImportError:
                await self._reply(
                    update,
                    "❌ Cache clearing functionality not available",
                    reply_markup=_back_to_menu_markup()
                )
                return

            # Get the paper wallet from the controller or direct reference
            paper_wallet = getattr(self, 'paper_wallet', None)

            # Try to get context from the controller if available
            ctx = None
            if hasattr(self.controller, 'get_context'):
                ctx = self.controller.get_context()

            # Clear the cache
            result = clear_paper_trading_cache(paper_wallet=paper_wallet, context=ctx)

            await self._reply(
                update,
                f"🔄 Cache Clear Results:\n\n{result}",
                reply_markup=_back_to_menu_markup()
            )

        except Exception as e:
            error_msg = f"❌ Failed to clear cache: {str(e)}"
            await self._reply(
                update,
                error_msg,
                reply_markup=_back_to_menu_markup()
            )

    def _acquire_lock(self) -> None:
        """Acquire a file lock to prevent multiple instances."""
        try:
            # Remove stale lock file if it exists
            if self.lock_file.exists():
                try:
                    # Try to read PID from lock file
                    pid = int(self.lock_file.read_text().strip())
                    # Check if process is still running
                    os.kill(pid, 0)
                    # If we get here, process is running
                    self.logger.warning(f"Telegram bot already running with PID {pid}")
                    # Instead of raising error, just log and continue - let user decide
                    self.logger.warning("Multiple instances detected. Consider stopping other instances.")
                    # Remove stale lock to allow this instance to take over
                    self.lock_file.unlink()
                except (ValueError, OSError, ProcessLookupError):
                    # Process not running, remove stale lock
                    self.lock_file.unlink()
                    self.logger.info("Removed stale lock file from dead process")

            # Create lock file with current PID
            self.lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.lock_fd, str(os.getpid()).encode())
            os.close(self.lock_fd)
            self.logger.info(f"Telegram bot lock acquired by PID {os.getpid()}")

        except Exception as e:
            self.logger.error(f"Failed to acquire lock: {e}")
            raise

    def _release_lock(self) -> None:
        """Release the file lock."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
                self.logger.info("Telegram bot lock released")
        except Exception as e:
            self.logger.error(f"Failed to release lock: {e}")

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            self._release_lock()
        except Exception:
            pass  # Ignore errors during cleanup

    async def shutdown(self) -> None:
        """Gracefully shutdown the Telegram bot."""
        try:
            self.logger.info("Shutting down Telegram bot...")
            if self.task and not self.task.done():
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
            await self.app.stop()
            self._release_lock()
            self.logger.info("Telegram bot shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during Telegram bot shutdown: {e}")

    def run_async(self) -> None:
        """Start polling within the current event loop."""

        async def run() -> None:
            await self.app.initialize()
            self.notifier.notify(MENU_TEXT)
            await self.app.start()
            await self.app.updater.start_polling()

        self.task = asyncio.create_task(run())

    def _get_chat_id(self, update: Update) -> str:
        if getattr(update, "effective_chat", None):
            return str(update.effective_chat.id)
        if getattr(update, "message", None) and getattr(update.message, "chat_id", None):
            return str(update.message.chat_id)
        if getattr(update, "callback_query", None):
            msg = update.callback_query.message
            if getattr(msg, "chat_id", None):
                return str(msg.chat_id)
        return str(self.chat_id)

    async def _reply(
        self,
        update: Update,
        text: str,
        reply_markup: Optional[InlineKeyboardMarkup] = None,
    ) -> None:
        if getattr(update, "callback_query", None):
            try:
                await update.callback_query.message.edit_text(text, reply_markup=reply_markup)
            except Exception as exc:
                if "Message is not modified" not in str(exc):
                    raise
        else:
            await update.message.reply_text(text, reply_markup=reply_markup)

    async def _check_cooldown(self, update: Update, command: str) -> bool:
        chat = self._get_chat_id(update)
        now = time.time()
        key = (chat, command)
        last = self._last_exec.get(key)
        if last is not None and now - last < self.command_cooldown:
            await self._reply(update, "Please wait")
            return False
        self._last_exec[key] = now
        return True

    def _run_scheduler(self) -> None:
        while True:
            schedule.run_pending()
            time.sleep(1)

    def stop(self) -> None:
        if self.task:
            self.task.cancel()
        schedule.clear()
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=2)
        self._release_lock()

    async def _check_admin(self, update: Update) -> bool:
        """Verify the update came from an authorized chat."""
        chat_id = str(getattr(getattr(update, "effective_chat", None), "id", ""))
        if not is_admin(chat_id):
            if getattr(update, "message", None):
                await update.message.reply_text("Unauthorized")
            elif getattr(update, "callback_query", None):
                await update.callback_query.answer("Unauthorized", show_alert=True)
            self.logger.warning("Ignoring unauthorized command from %s", chat_id)
            return False
        return True

    # Command handlers -------------------------------------------------
    async def start_cmd(
        self, update: Update, context: Any
    ) -> None:
        if not await self._check_cooldown(update, "start"):
            return
        if not await self._check_admin(update):
            return
        text = await self.controller.start()
        await update.message.reply_text(text)
        self.state["running"] = True
        await self._reply(update, "Trading started", reply_markup=_back_to_menu_markup())
        await self.menu_cmd(update, context)

    async def stop_cmd(
        self, update: Update, context: Any
    ) -> None:
        if not await self._check_cooldown(update, "stop"):
            return
        if not await self._check_admin(update):
            return
        text = await self.controller.stop()
        await update.message.reply_text(text)
        self.state["running"] = False
        await self._reply(update, "Trading stopped", reply_markup=_back_to_menu_markup())

    async def status_cmd(
        self, update: Update, context: Any
    ) -> None:
        if not await self._check_cooldown(update, "status"):
            return
        text = await self.controller.status()
        await self._reply(update, text, reply_markup=_back_to_menu_markup())


    async def log_cmd(self, update: Update, context: Any) -> None:
        if not await self._check_cooldown(update, "log"):
            return
        if not await self._check_admin(update):
            return
        if self.log_file.exists():
            lines = self.log_file.read_text().splitlines()[-20:]
            text = "\n".join(lines) if lines else "(no logs)"
        else:
            text = "Log file not found"
        await self._reply(update, text, reply_markup=_back_to_menu_markup())

    async def rotate_now_cmd(
        self, update: Update, context: Any
    ) -> None:
        if not await self._check_cooldown(update, "rotate_now"):
            return
        if not await self._check_admin(update):
            return
        if not (self.rotator and self.exchange and self.wallet):
            await self._reply(update, "Rotation not configured", reply_markup=_back_to_menu_markup())
            return
        try:
            if asyncio.iscoroutinefunction(getattr(self.exchange, "fetch_balance", None)):
                bal = await self.exchange.fetch_balance()
            else:
                bal = await asyncio.to_thread(self.exchange.fetch_balance)
            holdings = {
                k: (v.get("total") if isinstance(v, dict) else v)
                for k, v in bal.items()
            }
            await self.rotator.rotate(
                self.exchange,
                self.wallet,
                holdings,
            )
            await self._reply(update, "Portfolio rotated", reply_markup=_back_to_menu_markup())
        except Exception as exc:  # pragma: no cover - network
            self.logger.error("Rotation failed: %s", exc)
            await self._reply(update, "Rotation failed", reply_markup=_back_to_menu_markup())

    def send_daily_summary(self) -> None:
        stats = trade_summary(str(LOG_DIR / "trades.csv"))
        msg = (
            "Daily Summary\n"
            f"Trades: {stats['num_trades']}\n"
            f"Total PnL: {stats['total_pnl']:.2f}\n"
            f"Win rate: {stats['win_rate']*100:.1f}%\n"
            f"Active positions: {stats['active_positions']}"
        )
        err = self.notifier.notify(msg)
        if err:
            self.logger.error("Failed to send summary: %s", err)

    async def toggle_mode_cmd(
        self, update: Update, context: Any
    ) -> None:
        if not await self._check_cooldown(update, "toggle_mode"):
            return
        if not await self._check_admin(update):
            return
        mode = self.state.get("mode")
        mode = "onchain" if mode == "cex" else "cex"
        self.state["mode"] = mode
        await self._reply(update, f"Mode set to {mode}", reply_markup=_back_to_menu_markup())

    async def reload_cmd(
        self, update: Update, context: Any
    ) -> None:
        if not await self._check_cooldown(update, "reload"):
            return
        if not await self._check_admin(update):
            return
        await self.controller.reload_config()
        await self._reply(update, "Config reload scheduled", reply_markup=_back_to_menu_markup())

    async def panic_sell_cmd(
        self, update: Update, context: Any
    ) -> None:
        if not await self._check_cooldown(update, "panic_sell"):
            return
        if not await self._check_admin(update):
            return
        text = await self.controller.close_all_positions()
        await self._reply(update, text, reply_markup=_back_to_menu_markup())

    async def menu_cmd(self, update: Update, context: Any) -> None:
        if not await self._check_cooldown(update, "menu"):
            return
        if not await self._check_admin(update):
            return
        keyboard = [
            [
                InlineKeyboardButton("Start", callback_data=START),
                InlineKeyboardButton("Stop", callback_data=STOP),
                InlineKeyboardButton("Status", callback_data=STATUS),
            ],
            [
                InlineKeyboardButton("Log", callback_data=LOG),
                InlineKeyboardButton("Rotate Now", callback_data=ROTATE),
                InlineKeyboardButton("Toggle Mode", callback_data=TOGGLE),
                InlineKeyboardButton("Reload", callback_data=RELOAD),
                InlineKeyboardButton("Panic Sell", callback_data=PANIC_SELL),
            ],
            [
                InlineKeyboardButton("Signals", callback_data=SIGNALS),
                InlineKeyboardButton("Balance", callback_data=BALANCE),
                InlineKeyboardButton("Trades", callback_data=TRADES),
                InlineKeyboardButton("Trade History", callback_data=TRADE_HISTORY),
                InlineKeyboardButton("PnL Stats", callback_data=PNL_STATS),
            ],
            [
                InlineKeyboardButton("Config Settings", callback_data=CONFIG),
                InlineKeyboardButton("Clear Cache", callback_data="clear_cache"),
            ],
        ]
        markup = InlineKeyboardMarkup(keyboard)
        await self._reply(update, "Select a command:", reply_markup=markup)

    async def show_signals(self, update: Update, context: Any) -> None:
        if not await self._check_cooldown(update, "signals"):
            return
        if not await self._check_admin(update):
            return
        # Use ``ASSET_SCORES_FILE`` so tests can patch the path easily.
        if ASSET_SCORES_FILE.exists():
            try:
                data = json.loads(ASSET_SCORES_FILE.read_text())
                lines = [f"{k}: {v:.2f}" for k, v in data.items()]
                text = "\n".join(lines) if lines else "(no signals)"
            except Exception:
                text = "Invalid signals file"
        else:
            text = "No signals found"
        await self._reply(update, text, reply_markup=_back_to_menu_markup())

    async def show_balance(self, update: Update, context: Any) -> None:
        if not await self._check_cooldown(update, "balance"):
            return
        if not await self._check_admin(update):
            return
        
        # Check if we're in paper trading mode by looking for paper wallet
        paper_wallet = self.paper_wallet or getattr(self.controller, 'paper_wallet', None)
        
        try:
            # In paper trading mode, show paper wallet balance
            if paper_wallet and hasattr(paper_wallet, 'balance'):
                summary = paper_wallet.get_position_summary()
                lines = [
                    f"📄 Paper Trading Mode",
                    f"Balance: ${summary['balance']:.2f}",
                    f"Initial: ${summary['initial_balance']:.2f}",
                    f"Realized PnL: ${summary['realized_pnl']:.2f}",
                    f"Total Trades: {summary['total_trades']}",
                    f"Winning Trades: {summary['winning_trades']}",
                    f"Win Rate: {summary['win_rate']:.1f}%",
                    f"Open Positions: {summary['open_positions']}"
                ]
                
                # Add position details if any
                if summary['positions']:
                    lines.append("\n🔹 Open Positions:")
                    for pid, pos in summary['positions'].items():
                        lines.append(f"{pos['symbol'] or pid}: {pos['side']} {pos['size']:.4f} @ ${pos['entry_price']:.6f}")
                
                text = "\n".join(lines)
            else:
                # Live trading mode - fetch from exchange
                if not self.exchange:
                    await self._reply(update, "Exchange not configured", reply_markup=_back_to_menu_markup())
                    return
                
                if asyncio.iscoroutinefunction(getattr(self.exchange, "fetch_balance", None)):
                    bal = await self.exchange.fetch_balance()
                else:
                    bal = await asyncio.to_thread(self.exchange.fetch_balance)
                free_usdt = (
                    bal.get("USDT", {}).get("free")
                    if isinstance(bal.get("USDT"), dict)
                    else None
                )
                lines = [f"💰 Live Trading Mode", f"Free USDT: {free_usdt or 0}"]
                lines += [
                    f"{k}: {v.get('total') if isinstance(v, dict) else v}"
                    for k, v in bal.items()
                    if k != "USDT"  # USDT already shown above
                ]
                text = "\n".join(lines) if lines else "(no balance)"
                
        except Exception as exc:  # pragma: no cover - network
            self.logger.error("Balance fetch failed: %s", exc)
            text = "Balance fetch failed"
        await self._reply(update, text, reply_markup=_back_to_menu_markup())

    async def show_trades(self, update: Update, context: Any) -> None:
        if not await self._check_cooldown(update, "trades"):
            return
        if not await self._check_admin(update):
            return
        
        paper_wallet = self.paper_wallet or getattr(self.controller, 'paper_wallet', None)
        
        # For paper trading, show paper wallet positions and stats
        if paper_wallet and hasattr(paper_wallet, 'balance'):
            summary = paper_wallet.get_position_summary()
            lines = [
                f"📄 Paper Trading Positions",
                f"Total Trades: {summary['total_trades']}",
                f"Winning Trades: {summary['winning_trades']}",
                f"Win Rate: {summary['win_rate']:.1f}%",
                f"Realized PnL: ${summary['realized_pnl']:.2f}",
                f"Current Balance: ${summary['balance']:.2f}",
                ""
            ]
            
            if summary['positions']:
                lines.append("🔹 Active Positions:")
                for pid, pos in summary['positions'].items():
                    symbol = pos['symbol'] or pid
                    entry_price = pos['entry_price']
                    size = pos['size']
                    side = pos['side']
                    reserved = pos.get('reserved', 0.0)
                    
                    lines.append(f"{symbol}: {side.upper()} {size:.4f} @ ${entry_price:.6f}")
                    if reserved > 0:
                        lines.append(f"  Reserved: ${reserved:.2f}")
            else:
                lines.append("No active positions")
            
            text = "\n".join(lines)
        else:
            # Live trading mode - try to read trades from file
            text = _process_trades_for_display(TRADES_FILE)
                
        await self._reply(update, text, reply_markup=_back_to_menu_markup())

    async def show_config(self, update: Update, context: Any) -> None:
        if not await self._check_cooldown(update, "config"):
            return
        if not await self._check_admin(update):
            return
        cfg = {}
        if CONFIG_FILE.exists():
            try:
                cfg = yaml.safe_load(CONFIG_FILE.read_text()) or {}
            except Exception:
                cfg = {}
        text = (
            f"trade_size_pct: {cfg.get('trade_size_pct')}\n"
            f"max_open_trades: {cfg.get('max_open_trades')}"
        )
        keyboard = [
            [InlineKeyboardButton("Edit Trade Size %", callback_data=EDIT_TRADE_SIZE)],
            [InlineKeyboardButton("Edit Max Open Trades", callback_data=EDIT_MAX_TRADES)],
        ]
        markup = InlineKeyboardMarkup(keyboard)
        await self._reply(update, text, reply_markup=markup)

    async def edit_trade_size(self, update: Update, context: Any) -> int:
        if not await self._check_admin(update):
            return ConversationHandler.END
        context.user_data["config_key"] = "trade_size_pct"
        await self._reply(update, "Enter trade size percentage (0-1):")
        return EDIT_VALUE

    async def edit_max_trades(self, update: Update, context: Any) -> int:
        if not await self._check_admin(update):
            return ConversationHandler.END
        context.user_data["config_key"] = "max_open_trades"
        await self._reply(update, "Enter max open trades (integer):")
        return EDIT_VALUE

    async def set_config_value(self, update: Update, context: Any) -> int:
        if not await self._check_admin(update):
            return ConversationHandler.END
        key = context.user_data.get("config_key")
        if not key:
            return ConversationHandler.END
        value_text = update.message.text if update.message else ""
        try:
            if key == "trade_size_pct":
                val = float(value_text)
                if not 0 < val <= 1:
                    raise ValueError
            else:
                val = int(value_text)
                if val <= 0:
                    raise ValueError
        except ValueError:
            await self._reply(update, "Invalid value, try again:")
            return EDIT_VALUE
        cfg = {}
        if CONFIG_FILE.exists():
            try:
                cfg = yaml.safe_load(CONFIG_FILE.read_text()) or {}
            except Exception:
                cfg = {}
        cfg[key] = val
        if hasattr(yaml, "safe_dump"):
            CONFIG_FILE.write_text(yaml.safe_dump(cfg, sort_keys=False))
        else:
            CONFIG_FILE.write_text(json.dumps(cfg))
        await self.controller.reload_config()
        await self._reply(update, f"{key} updated to {val}")
        return ConversationHandler.END
    async def show_pnl_stats(self, update: Update, context: Any) -> None:
        if not await self._check_cooldown(update, "pnl_stats"):
            return
        if not await self._check_admin(update):
            return
        if TRADES_FILE.exists():
            stats = trade_summary(TRADES_FILE)
            text = (
                f"Total PnL: {stats['total_pnl']:.2f}\n"
                f"Win rate: {stats['win_rate']*100:.1f}%\n"
                f"Active positions: {stats['active_positions']}"
            )
        else:
            text = "No trades found"
        await self._reply(update, text)

    async def show_trade_history(
        self, update: Update, context: Any
    ) -> None:
        if not await self._check_cooldown(update, "trade_history"):
            return
        if not await self._check_admin(update):
            return
        lines: List[str] = []
        if TRADES_FILE.exists():
            lines = TRADES_FILE.read_text().splitlines()[-100:]
        text, page, total_pages = _paginate(lines, 0)
        # Join the lines into a string for display
        display_text = "\n".join(text) if text else "No trades found"
        await self._reply(update, display_text, reply_markup=_back_to_menu_markup())


def _process_trades_for_display(trades_file: Path) -> str:
    """Process trades file and return formatted display text with PnL."""
    if not trades_file.exists():
        return "No trades file found"
    
    try:
        lines = trades_file.read_text().splitlines()
        if not lines:
            return "No trades found"
        
        # Simple trade processing - assume format: symbol,side,amount,price,timestamp
        trades = []
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 4:
                symbol = parts[0]
                side = parts[1]
                amount = float(parts[2]) if parts[2].replace('.', '').replace('-', '').isdigit() else 0
                price = float(parts[3]) if parts[3].replace('.', '').replace('-', '').isdigit() else 0
                trades.append((symbol, side, amount, price))
        
        if not trades:
            return "No valid trades found"
        
        # Calculate simple PnL (this is a simplified version)
        total_pnl = 0.0
        formatted_trades = []
        positions = {}  # Track open positions
        
        for symbol, side, amount, price in trades[-10:]:  # Show last 10 trades
            if side == 'buy':
                formatted_trades.append(f"📈 {symbol}: BUY {amount:.4f} @ ${price:.2f}")
                # Track buy position
                if symbol not in positions:
                    positions[symbol] = {'amount': amount, 'avg_price': price}
                else:
                    # Update average price for existing position
                    total_cost = positions[symbol]['amount'] * positions[symbol]['avg_price'] + amount * price
                    total_amount = positions[symbol]['amount'] + amount
                    positions[symbol]['avg_price'] = total_cost / total_amount
                    positions[symbol]['amount'] += amount
                    
            elif side == 'sell':
                formatted_trades.append(f"📉 {symbol}: SELL {amount:.4f} @ ${price:.2f}")
                # Calculate PnL if we have a buy position
                if symbol in positions and positions[symbol]['amount'] > 0:
                    sell_amount = min(amount, positions[symbol]['amount'])
                    pnl = (price - positions[symbol]['avg_price']) * sell_amount
                    total_pnl += pnl
                    
                    # Update position
                    positions[symbol]['amount'] -= sell_amount
                    if positions[symbol]['amount'] <= 0:
                        del positions[symbol]
        
        text = "💰 Live Trading\n\nRecent Trades:\n" + "\n".join(formatted_trades)
        if total_pnl != 0:
            pnl_text = f"+${total_pnl:.2f}" if total_pnl > 0 else f"-${abs(total_pnl):.2f}"
            text += f"\n\nTotal PnL: {pnl_text}"
        
        return text
        
    except Exception as e:
        return f"Error processing trades: {e}"
