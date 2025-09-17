from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional

from fastapi import FastAPI, HTTPException

from crypto_bot.utils.logger import LOG_DIR
from frontend.gateway import ApiGatewayError, async_get_gateway_json

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

PORTFOLIO_POSITIONS_PATH = "/portfolio/positions"
PORTFOLIO_WALLET_STATUS_PATH = "/portfolio/wallet-status"
STRATEGY_PERFORMANCE_PATH = "/monitoring/strategy/performance"
STRATEGY_SCORES_PATH = "/monitoring/strategy/scores"


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
async def positions() -> List[dict]:
    """Return open positions via the portfolio API exposed by the gateway."""

    try:
        payload = await async_get_gateway_json(PORTFOLIO_POSITIONS_PATH)
    except ApiGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if payload is None:
        return []

    if isinstance(payload, list):
        return payload

    return []


@app.get("/wallet-status")
async def wallet_status() -> dict:
    """Return comprehensive wallet status from the API gateway."""

    try:
        payload = await async_get_gateway_json(PORTFOLIO_WALLET_STATUS_PATH)
    except ApiGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if isinstance(payload, dict):
        return payload

    return {"error": "No wallet status available"}


@app.get("/strategy-performance")
async def strategy_performance() -> dict:
    """Return strategy performance metrics from the monitoring service."""

    try:
        payload = await async_get_gateway_json(STRATEGY_PERFORMANCE_PATH)
    except ApiGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if isinstance(payload, dict):
        return payload
    return {}


@app.get("/strategy-scores")
async def strategy_scores() -> dict:
    """Return computed strategy scores from the monitoring service."""

    try:
        payload = await async_get_gateway_json(STRATEGY_SCORES_PATH)
    except ApiGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if isinstance(payload, dict):
        return payload
    return {}


@app.post("/reload-config")
async def reload_config() -> dict:
    """Reload ``crypto_bot`` configuration and return status."""
    return await get_controller().reload_config()


@app.post("/close-all")
async def close_all() -> dict:
    """Request liquidation of all open positions."""
    return await get_controller().close_all_positions()
