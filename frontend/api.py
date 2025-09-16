from __future__ import annotations

from fastapi import FastAPI
import json
from typing import TYPE_CHECKING, List, Optional

from crypto_bot.utils.logger import LOG_DIR

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from crypto_bot.bot_controller import TradingBotController

app = FastAPI()
CONTROLLER: "Optional[TradingBotController]" = None


def get_controller() -> "TradingBotController":
    global CONTROLLER
    if CONTROLLER is None:
        from crypto_bot.bot_controller import TradingBotController
        CONTROLLER = TradingBotController()
    return CONTROLLER
SIGNALS_FILE = LOG_DIR / "asset_scores.json"
PERFORMANCE_FILE = LOG_DIR / "strategy_performance.json"
SCORES_FILE = LOG_DIR / "strategy_scores.json"


@app.get("/live-signals")
def live_signals() -> dict:
    """Return latest signal scores as a mapping of symbol to score."""
    if SIGNALS_FILE.exists():
        try:
            return json.loads(SIGNALS_FILE.read_text())
        except Exception:
            return {}
    return {}


@app.get("/positions")
def positions() -> List[dict]:
    """Return open positions via the shared frontend helper."""
    try:
        from frontend.app import get_open_positions

        entries = get_open_positions()
        print(f"Returning {len(entries)} open positions from TradeManager")
        return entries
    except Exception as exc:
        print(f"Error getting positions from TradeManager helper: {exc}")
        return []


@app.get("/wallet-status")
def wallet_status() -> dict:
    """Return comprehensive paper wallet status including real-time PnL."""
    try:
        controller = get_controller()
        if hasattr(controller, '_ctx') and controller._ctx:
            from crypto_bot.main import get_paper_wallet_status
            status = get_paper_wallet_status(controller._ctx)
            if status:
                return status
    except Exception as e:
        return {"error": f"Failed to get wallet status: {e}"}
    
    return {"error": "No wallet status available"}


@app.get("/strategy-performance")
def strategy_performance() -> dict:
    """Return raw strategy performance data grouped by regime and strategy."""
    if PERFORMANCE_FILE.exists():
        try:
            return json.loads(PERFORMANCE_FILE.read_text())
        except Exception:
            return {}
    return {}


@app.get("/strategy-scores")
def strategy_scores() -> dict:
    """Return computed strategy metrics."""
    if SCORES_FILE.exists():
        try:
            return json.loads(SCORES_FILE.read_text())
        except Exception:
            return {}
    return {}


@app.post("/reload-config")
async def reload_config() -> dict:
    """Reload ``crypto_bot`` configuration and return status."""
    return await get_controller().reload_config()


@app.post("/close-all")
async def close_all() -> dict:
    """Request liquidation of all open positions."""
    return await get_controller().close_all_positions()
