from __future__ import annotations

"""Async wrapper controlling the trading bot."""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx

from crypto_bot.utils.logger import LOG_DIR, setup_logger

from crypto_bot.config import load_config as load_bot_config, resolve_config_path
from crypto_bot.utils.symbol_utils import fix_symbol

from .execution.cex_executor import execute_trade_async, get_exchange
from .portfolio_rotator import PortfolioRotator

TRADING_ENGINE_START_PATH = "/trading-engine/cycles/start"
TRADING_ENGINE_STOP_PATH = "/trading-engine/cycles/stop"
TRADING_ENGINE_STATUS_PATH = "/trading-engine/cycles/status"
TRADING_ENGINE_CLOSE_ALL_PATH = "/trading-engine/positions/close-all"
PORTFOLIO_POSITIONS_PATH = "/portfolio/positions"


class TradingBotController:
    """High level controller exposing simple async methods."""

    def __init__(
        self,
        config_path: Union[str, Path] = resolve_config_path(),
        trades_file: Union[str, Path] = LOG_DIR / "trades.csv",
        log_file: Union[str, Path] = LOG_DIR / "bot.log",
    ) -> None:
        self.config_path = Path(config_path)
        self.trades_file = Path(trades_file)
        self.log_file = Path(log_file)
        self.config = self._load_config()
        self.rotator = PortfolioRotator()
        self.exchange, self.ws_client = get_exchange(self.config)
        self.enabled: Dict[str, bool] = {
            "trend_bot": True,
            "grid_bot": True,
            "sniper_bot": True,
            "dex_scalper": True,
            "dca_bot": True,
            "mean_bot": True,
            "breakout_bot": True,
            "micro_scalp_bot": True,
            "bounce_scalper": True,
        }
        self.state = {
            "running": False,
            "mode": self.config.get("execution_mode", "dry_run"),
            "liquidate": False,
            "liquidate_all": False,
        }
        self.logger = setup_logger(__name__, LOG_DIR / "bot_controller.log")
        self.gateway_url = os.getenv("API_GATEWAY_URL", "http://localhost:8000").rstrip("/")
        self._gateway_timeout = float(os.getenv("API_GATEWAY_TIMEOUT", "10"))


    def _load_config(self) -> dict:
        try:
            data = load_bot_config(self.config_path)
        except Exception:
            data = {}

        strat_dir = self.config_path.parent.parent / "config" / "strategies"
        trend_file = strat_dir / "trend_bot.yaml"
        if trend_file.exists():
            import yaml

            with open(trend_file) as sf:
                overrides = yaml.safe_load(sf) or {}
            trend_cfg = data.get("trend", {})
            if isinstance(trend_cfg, dict):
                trend_cfg.update(overrides)
            else:
                trend_cfg = overrides
            data["trend"] = trend_cfg

        if "symbol" in data:
            data["symbol"] = fix_symbol(data["symbol"])
        if "symbols" in data:
            data["symbols"] = [fix_symbol(s) for s in data.get("symbols", [])]
        return data

    async def _gateway_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{self.gateway_url}{path}"
        async with httpx.AsyncClient(timeout=self._gateway_timeout) as client:
            response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    async def start_trading(self) -> Dict[str, object]:
        """Signal the trading engine service to start scheduled cycles."""

        payload = {
            "immediate": True,
            "metadata": {"mode": self.config.get("execution_mode", "dry_run")},
        }
        try:
            response = await self._gateway_request(
                "POST", TRADING_ENGINE_START_PATH, json=payload
            )
            data = response.json()
            self.state["running"] = True
            return {"running": True, "status": data.get("status", "started"), "details": data}
        except httpx.HTTPError as exc:
            self.logger.error("Failed to start trading engine: %s", exc)
            return {"running": False, "status": "error", "error": str(exc)}

    async def stop_trading(self) -> Dict[str, object]:
        """Signal the trading engine service to stop scheduled cycles."""

        try:
            response = await self._gateway_request(
                "POST", TRADING_ENGINE_STOP_PATH, json={}
            )
            data = response.json()
            self.state["running"] = False
            return {"running": False, "status": data.get("status", "stopped"), "details": data}
        except httpx.HTTPError as exc:
            self.logger.error("Failed to stop trading engine: %s", exc)
            return {"running": False, "status": "error", "error": str(exc)}

    async def close(self) -> None:
        """Close exchange and WebSocket client connections."""
        if self.ws_client and hasattr(self.ws_client, 'close_async'):
            try:
                await self.ws_client.close_async()
                self.logger.info("Bot controller WebSocket client closed successfully")
            except Exception as exc:
                self.logger.error("Error closing bot controller WebSocket client: %s", exc)
        elif self.ws_client and hasattr(self.ws_client, 'close'):
            try:
                self.ws_client.close()
                self.logger.info("Bot controller WebSocket client closed successfully")
            except Exception as exc:
                self.logger.error("Error closing bot controller WebSocket client: %s", exc)
        
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                if asyncio.iscoroutinefunction(getattr(self.exchange, 'close')):
                    await self.exchange.close()
                else:
                    await asyncio.to_thread(self.exchange.close)
                self.logger.info("Bot controller exchange closed successfully")
            except Exception as exc:
                self.logger.error("Error closing bot controller exchange: %s", exc)
            finally:
                self.exchange = None
                self.ws_client = None

    async def get_status(self) -> Dict[str, object]:
        """Return current trading engine state and enabled strategies."""

        try:
            response = await self._gateway_request("GET", TRADING_ENGINE_STATUS_PATH)
            data = response.json()
            running = bool(data.get("running"))
            self.state["running"] = running
            metadata = data.get("metadata") or {}
            if "mode" in metadata:
                self.state["mode"] = metadata.get("mode")
            return {
                "running": running,
                "mode": self.state.get("mode"),
                "enabled_strategies": self.enabled.copy(),
                "details": data,
            }
        except httpx.HTTPError as exc:
            self.logger.error("Failed to fetch trading engine status: %s", exc)
            return {
                "running": False,
                "mode": self.state.get("mode"),
                "enabled_strategies": self.enabled.copy(),
                "error": str(exc),
            }

    async def list_strategies(self) -> List[str]:
        """Return names of available strategies."""
        return list(self.enabled.keys())

    async def toggle_strategy(self, name: str) -> Dict[str, object]:
        """Enable or disable ``name`` and return the new state."""
        if name not in self.enabled:
            raise ValueError(f"Unknown strategy: {name}")
        self.enabled[name] = not self.enabled[name]
        return {"strategy": name, "enabled": self.enabled[name]}

    async def list_positions(self) -> List[Dict]:
        """Return currently open positions from the portfolio service."""

        try:
            response = await self._gateway_request("GET", PORTFOLIO_POSITIONS_PATH)
            data = response.json()
            if isinstance(data, list):
                return data
        except httpx.HTTPError as exc:
            self.logger.error("Failed to fetch positions from portfolio service: %s", exc)
        except Exception as exc:  # pragma: no cover - unexpected structure
            self.logger.error("Unexpected error fetching positions: %s", exc)
        return []

    async def close_position(self, symbol: str, amount: float) -> Dict:
        """Submit a market order closing ``amount`` of ``symbol``."""
        return await execute_trade_async(
            self.exchange,
            self.ws_client,
            symbol,
            "sell",
            amount,
            dry_run=self.config.get("execution_mode") == "dry_run",
            use_websocket=self.config.get("use_websocket", False),
            config=self.config,
        )

    async def close_all_positions(self) -> Dict[str, str]:
        """Request liquidation of all open positions via the trading engine."""

        try:
            response = await self._gateway_request(
                "POST", TRADING_ENGINE_CLOSE_ALL_PATH, json={}
            )
            data = response.json()
            return {"status": data.get("status", "requested"), "details": data}
        except httpx.HTTPError as exc:
            self.logger.error("Failed to request close-all via trading engine: %s", exc)
            return {"status": "error", "error": str(exc)}

    async def fetch_logs(self, lines: int = 20) -> List[str]:
        """Return the last ``lines`` from the bot log."""
        if not self.log_file.exists():
            return []
        data = self.log_file.read_text().splitlines()
        return data[-lines:]

    async def reload_config(self) -> Dict[str, object]:
        """Reload configuration from ``self.config_path``."""
        try:
            self.config = self._load_config()
            self.exchange, self.ws_client = get_exchange(self.config)
            self.state["mode"] = self.config.get("execution_mode", "dry_run")
            return {"status": "reloaded", "mode": self.state["mode"]}
        except Exception as exc:  # pragma: no cover - unexpected
            return {"status": "error", "error": str(exc)}

