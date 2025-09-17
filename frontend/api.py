from __future__ import annotations

from fastapi import FastAPI
import json
import re
from typing import TYPE_CHECKING, List, Optional

from crypto_bot.utils.logger import LOG_DIR

try:
    from services.portfolio.clients.interface import PortfolioServiceClient
except Exception:  # pragma: no cover
    PortfolioServiceClient = None

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
POSITIONS_FILE = LOG_DIR / "positions.log"
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


POS_PATTERN = re.compile(
    r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
    r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+) "
    r"pnl \$(?P<pnl>[0-9.+-]+).*balance \$(?P<balance>[0-9.]+)"
)


def _service_positions() -> List[dict]:
    if PortfolioServiceClient is None:
        return []

    try:
        client = PortfolioServiceClient()
        positions = client.list_positions()
    except Exception:
        return []

    entries: List[dict] = []
    for pos in positions:
        try:
            if pos.total_amount <= 0:
                continue
            current_price = pos.mark_price or pos.average_price
            if pos.side == "long":
                pnl_value = (current_price - pos.average_price) * pos.total_amount
            else:
                pnl_value = (pos.average_price - current_price) * pos.total_amount
            pnl_pct = (
                float((pnl_value / (pos.average_price * pos.total_amount)) * 100)
                if pos.average_price > 0 and pos.total_amount > 0
                else 0.0
            )
            entries.append(
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "amount": float(pos.total_amount),
                    "entry_price": float(pos.average_price),
                    "current_price": float(current_price),
                    "position_value": float(pos.total_amount * current_price),
                    "pnl": float(pnl_value),
                    "pnl_pct": pnl_pct,
                    "balance": 0.0,
                }
            )
        except Exception:
            continue
    return entries


@app.get("/positions")
def positions() -> List[dict]:
    """Return positions from TradeManager (single source of truth)."""
    try:
        # Get positions from TradeManager (single source of truth)
        from crypto_bot.utils.trade_manager import get_trade_manager
        tm = get_trade_manager()
        positions = tm.get_all_positions()

        entries = []
        for pos in positions:
            if pos.is_open:  # Only return open positions
                # Get current price for P&L calculation
                try:
                    import ccxt
                    exchange = ccxt.kraken()
                    ticker = exchange.fetch_ticker(pos.symbol)
                    current_price = ticker['last']

                    # Calculate P&L
                    pnl, pnl_pct = pos.calculate_unrealized_pnl(current_price)

                    # Calculate current value
                    current_value = float(pos.total_amount) * current_price

                    entries.append({
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "amount": float(pos.total_amount),
                        "entry_price": float(pos.average_price),
                        "current_price": current_price,
                        "position_value": current_value,
                        "pnl": float(pnl),
                        "pnl_pct": float(pnl_pct),
                        "balance": 0.0,  # Will be calculated separately
                    })
                except Exception as e:
                    # Use cached price from TradeManager if available
                    cached_price = float(tm.price_cache.get(pos.symbol, pos.average_price))

                    # Calculate P&L with cached price
                    pnl, pnl_pct = pos.calculate_unrealized_pnl(cached_price)
                    current_value = float(pos.total_amount) * cached_price

                    entries.append({
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "amount": float(pos.total_amount),
                        "entry_price": float(pos.average_price),
                        "current_price": cached_price,
                        "position_value": current_value,
                        "pnl": float(pnl),
                        "pnl_pct": float(pnl_pct),
                        "balance": 0.0,
                    })

        # Log position summary for debugging
        print(f"Returning {len(entries)} open positions from TradeManager")
        return entries

    except Exception as e:
        print(f"Error getting positions from TradeManager: {e}")
        service_entries = _service_positions()
        if service_entries:
            return service_entries
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
