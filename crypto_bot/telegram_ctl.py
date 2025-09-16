from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Optional

try:  # pragma: no cover - optional dependency
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import ContextTypes
except Exception:  # pragma: no cover - telegram not installed
    InlineKeyboardButton = InlineKeyboardMarkup = Update = object  # type: ignore
    ContextTypes = object  # type: ignore

from .utils.logger import LOG_DIR, setup_logger
from .utils.open_trades import get_open_trades
from .utils.telegram import TelegramNotifier
from .config import resolve_config_path

logger = setup_logger(__name__, LOG_DIR / "telegram_ctl.log")

STRATEGY_FILE = LOG_DIR / "strategy_stats.json"
TRADES_FILE = LOG_DIR / "trades.csv"
LOG_FILE = LOG_DIR / "bot.log"
CONFIG_FILE = Path(resolve_config_path())


async def _maybe_call(func: Any) -> Any:
    """Call ``func`` which may be sync or async."""
    if asyncio.iscoroutinefunction(func):
        return await func()
    return await asyncio.to_thread(func)


async def status_loop(
    controller: Any,
    admins: Sequence[TelegramNotifier],
    update_interval: float = 60.0,
) -> None:
    """Periodically send status updates using ``controller``."""
    while True:
        try:
            status = await _maybe_call(controller.get_status)
            positions = await _maybe_call(controller.list_positions)
            lines = [str(status)]
            if positions:
                if isinstance(positions, str):
                    lines.append(positions)
                else:
                    lines.extend(str(p) for p in positions)
            message = "\n".join(lines)
            for admin in admins:
                admin.notify(message)
        except Exception as exc:  # pragma: no cover - logging only
            logger.error("Status update failed: %s", exc)
        await asyncio.sleep(update_interval)


def start(
    controller: Any,
    admins: Sequence[TelegramNotifier],
    update_interval: float = 60.0,
    enabled: bool = True,
) -> Optional[asyncio.Task]:
    """Return background task sending periodic updates when ``enabled``."""
    if not enabled:
        return None
    return asyncio.create_task(status_loop(controller, admins, update_interval))


def is_admin(update: Update, admin_id: str) -> bool:
    """Return True if the update came from the configured admin chat."""
    user_id = str(getattr(update.effective_user, "id", ""))
    return user_id == str(admin_id)


class BotController:
    """High level bot control and status retrieval."""

    def __init__(
        self,
        state: Dict[str, Any],
        exchange: Any = None,
        *,
        log_file: Path = LOG_FILE,
        trades_file: Path = TRADES_FILE,
        strategy_file: Path = STRATEGY_FILE,
        config_file: Path = CONFIG_FILE,
        paper_wallet: Any = None,
    ) -> None:
        self.state = state
        self.exchange = exchange
        self.log_file = Path(log_file)
        self.trades_file = Path(trades_file)
        self.strategy_file = Path(strategy_file)
        self.config_file = Path(config_file)
        self.paper_wallet = paper_wallet

    async def start(self) -> str:
        self.state["running"] = True
        return "Trading started"

    async def stop(self) -> str:
        self.state["running"] = False
        return "Trading stopped"

    async def status(self) -> str:
        running = self.state.get("running", False)
        mode = self.state.get("mode")
        
        # Add paper trading info if available
        status_lines = [f"Running: {running}, mode: {mode}"]
        
        if self.paper_wallet and hasattr(self.paper_wallet, 'balance'):
            status_lines.append(f"📄 Paper Trading: ${self.paper_wallet.balance:.2f}")
            status_lines.append(f"Realized PnL: ${self.paper_wallet.realized_pnl:.2f}")
            status_lines.append(f"Open positions: {len(self.paper_wallet.positions)}")
        
        return "\n".join(status_lines)

    async def strategies(self) -> str:
        if self.strategy_file.exists():
            try:
                data = json.loads(self.strategy_file.read_text())
                lines = [f"{k}: {v}" for k, v in data.items()]
                return "\n".join(lines) if lines else "(no strategies)"
            except Exception:
                return "Invalid strategy file"
        return "No strategies found"

    async def positions(self) -> str:
        try:
            from . import console_monitor
            lines = await console_monitor.trade_stats_lines(self.exchange, self.trades_file)
            return "\n".join(lines) if lines else "(no positions)"
        except ImportError:
            return "Console monitor not available"

    async def logs(self) -> str:
        if self.log_file.exists():
            lines = self.log_file.read_text().splitlines()[-20:]
            return "\n".join(lines) if lines else "(no logs)"
        return "Log file not found"

    async def settings(self) -> str:
        if self.config_file.exists():
            return self.config_file.read_text()
        return "Config not found"

    async def get_status(self) -> str:
        return await self.status()

    async def list_positions(self) -> str:
        return await self.positions()

    async def close_all_positions(self) -> str:
        """Signal the trading bot to liquidate all open positions."""
        self.state["liquidate_all"] = True
        return "Liquidation requested"

    async def reload_config(self) -> str:
        """Signal the trading bot to reload configuration."""
        self.state["reload"] = True
        return "Config reload scheduled"


async def panic_sell_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Close all open positions immediately."""
    if not is_admin(update, context.bot_data.get("admin_id")):
        return
    text = await context.bot_data["controller"].close_all_positions()
    await update.message.reply_text(text)


class TelegramController:
    """Telegram bot controller for managing trading operations."""
    
    def __init__(self, bot_token: str, admin_id: str, controller: BotController = None):
        self.bot_token = bot_token
        self.admin_id = admin_id
        self.controller = controller or BotController({})
        self.running = False
    
    async def start(self) -> str:
        """Start the Telegram bot."""
        self.running = True
        return "Telegram bot started"
    
    async def stop(self) -> str:
        """Stop the Telegram bot."""
        self.running = False
        return "Telegram bot stopped"
    
    async def get_status(self) -> str:
        """Get bot status."""
        return await self.controller.get_status()
    
    async def list_positions(self) -> str:
        """List current positions."""
        return await self.controller.list_positions()
    
    async def close_all_positions(self) -> str:
        """Close all positions."""
        return await self.controller.close_all_positions()
    
    async def reload_config(self) -> str:
        """Reload configuration."""
        return await self.controller.reload_config()
    
    def is_admin(self, user_id: str) -> bool:
        """Check if user is admin."""
        return str(user_id) == str(self.admin_id)


