"""Utilities for loading and reloading bot configuration during startup."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import ValidationError

from schema.scanner import PythConfig, ScannerConfig, SolanaScannerConfig

from crypto_bot.config import load_config as _load_config_dict
from crypto_bot.config import resolve_config_path
from crypto_bot.cooldown_manager import configure as cooldown_configure
from libs.models.open_position_guard import OpenPositionGuard
from crypto_bot.portfolio_rotator import PortfolioRotator
from libs.risk.risk_manager import RiskConfig, RiskManager
from crypto_bot.services.adapters import (
    ExecutionAdapter,
    MarketDataAdapter,
    MonitoringAdapter,
    PortfolioAdapter,
    StrategyAdapter,
    TokenDiscoveryAdapter,
)
from libs.services.interfaces import ServiceContainer
from libs.market_data.loader import configure as market_loader_configure
from crypto_bot.utils.symbol_utils import fix_symbol

logger = logging.getLogger("bot")

CONFIG_PATH = resolve_config_path()
try:
    _LAST_CONFIG_MTIME = CONFIG_PATH.stat().st_mtime
except OSError:
    _LAST_CONFIG_MTIME = 0.0

__all__ = [
    "CONFIG_PATH",
    "flatten_config",
    "get_last_config_mtime",
    "load_config",
    "maybe_reload_config",
    "reload_config",
    "set_last_config_mtime",
    "create_service_container",
]


def load_config() -> Dict[str, Any]:
    """Load YAML configuration for the bot."""
    logger.info("Loading config from %s", CONFIG_PATH)
    data = _load_config_dict(config_path=CONFIG_PATH)

    strat_dir = CONFIG_PATH.parent.parent / "config" / "strategies"
    trend_file = strat_dir / "trend_bot.yaml"
    if trend_file.exists():
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
    if "solana_symbols" in data:
        data["solana_symbols"] = [fix_symbol(s) for s in data.get("solana_symbols", [])]

    # Validate top-level config; on error, log and continue with loaded data
    try:
        if hasattr(ScannerConfig, "model_validate"):
            ScannerConfig.model_validate(data)
        else:  # pragma: no cover - for Pydantic < 2
            ScannerConfig.parse_obj(data)
    except ValidationError as exc:
        logger.warning("Invalid configuration (non-fatal): %s", exc)

    # Validate solana scanner section; on error, disable it and continue
    try:
        raw_scanner = data.get("solana_scanner", {}) or {}
        if hasattr(SolanaScannerConfig, "model_validate"):
            scanner = SolanaScannerConfig.model_validate(raw_scanner)
        else:  # pragma: no cover - for Pydantic < 2
            scanner = SolanaScannerConfig.parse_obj(raw_scanner)
        data["solana_scanner"] = scanner.dict()
    except ValidationError as exc:
        logger.warning("Invalid configuration (solana_scanner), disabling scanner: %s", exc)
        data["solana_scanner"] = {"enabled": False}

    # Validate pyth section; on error, disable and continue
    try:
        raw_pyth = data.get("pyth", {}) or {}
        if hasattr(PythConfig, "model_validate"):
            pyth_cfg = PythConfig.model_validate(raw_pyth)
        else:  # pragma: no cover - for Pydantic < 2
            pyth_cfg = PythConfig.parse_obj(raw_pyth)
        data["pyth"] = pyth_cfg.dict()
    except ValidationError as exc:
        logger.warning("Invalid configuration (pyth), disabling pyth: %s", exc)
        data["pyth"] = {"enabled": False}

    # Provide sensible fallback for max_open_trades if missing
    if "max_open_trades" not in data:
        rot_max = (data.get("risk") or {}).get("max_positions")
        if isinstance(rot_max, int) and rot_max > 0:
            data["max_open_trades"] = rot_max
        else:
            data["max_open_trades"] = 5

    return data


def maybe_reload_config(state: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Reload configuration when ``state['reload']`` is set."""
    if state.get("reload"):
        new_cfg = load_config()
        config.clear()
        config.update(new_cfg)
        state.pop("reload", None)


def flatten_config(data: Dict[str, Any], parent: str = "") -> Dict[str, Any]:
    """Flatten nested config keys to ENV_STYLE names."""
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent}_{key}" if parent else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, new_key))
        else:
            flat[new_key.upper()] = value
    return flat


def reload_config(
    config: Dict[str, Any],
    ctx,
    risk_manager: RiskManager,
    rotator: PortfolioRotator,
    position_guard: OpenPositionGuard,
    *,
    force: bool = False,
) -> None:
    """Reload the YAML config and update dependent objects."""
    global _LAST_CONFIG_MTIME
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = _LAST_CONFIG_MTIME
    if not force and mtime == _LAST_CONFIG_MTIME:
        return

    new_config = load_config()
    _LAST_CONFIG_MTIME = mtime
    config.clear()
    config.update(new_config)
    ctx.config = config
    rotator.config = config.get("portfolio_rotation", rotator.config)
    position_guard.max_open_trades = config.get(
        "max_open_trades", position_guard.max_open_trades
    )
    cooldown_configure(config.get("min_cooldown", 0))
    market_loader_configure(
        config.get("ohlcv_timeout", 120),
        config.get("max_ohlcv_failures", 3),
        config.get("max_ws_limit", 50),
        config.get("telegram", {}).get("status_updates", True),
        max_concurrent=config.get("max_concurrent_ohlcv"),
    )

    volume_ratio = 0.01 if config.get("testing_mode") else 1.0
    risk_params = {**config.get("risk", {})}
    risk_params.update(config.get("sentiment_filter", {}))
    risk_params.update(config.get("volatility_filter", {}))
    risk_params["symbol"] = config.get("symbol", "")
    risk_params["trade_size_pct"] = config.get("trade_size_pct", 0.1)
    risk_params["strategy_allocation"] = config.get("strategy_allocation", {})
    risk_params["volume_threshold_ratio"] = config.get("risk", {}).get(
        "volume_threshold_ratio", 0.1
    )
    risk_params["atr_period"] = config.get("risk", {}).get("atr_period", 14)
    risk_params["stop_loss_atr_mult"] = config.get("risk", {}).get(
        "stop_loss_atr_mult", 2.0
    )
    risk_params["take_profit_atr_mult"] = config.get("risk", {}).get(
        "take_profit_atr_mult", 4.0
    )
    risk_params["volume_ratio"] = volume_ratio

    valid_fields = {
        "max_drawdown",
        "stop_loss_pct",
        "take_profit_pct",
        "min_fng",
        "min_sentiment",
        "bull_fng",
        "bull_sentiment",
        "min_atr_pct",
        "max_funding_rate",
        "symbol",
        "trade_size_pct",
        "risk_pct",
        "min_volume",
        "volume_threshold_ratio",
        "strategy_allocation",
        "volume_ratio",
        "atr_short_window",
        "atr_long_window",
        "max_volatility_factor",
        "min_expected_value",
        "default_expected_value",
        "atr_period",
        "stop_loss_atr_mult",
        "take_profit_atr_mult",
        "max_pair_drawdown",
        "pair_drawdown_lookback",
    }
    filtered_risk_params = {
        key: value for key, value in risk_params.items() if key in valid_fields
    }
    risk_manager.config = RiskConfig(**filtered_risk_params)


def get_last_config_mtime() -> float:
    """Return the last configuration modification time."""
    return _LAST_CONFIG_MTIME


def set_last_config_mtime(mtime: float) -> None:
    """Update the cached configuration modification time."""
    global _LAST_CONFIG_MTIME
    _LAST_CONFIG_MTIME = mtime


def create_service_container() -> ServiceContainer:
    """Instantiate the default in-process service adapters."""

    return ServiceContainer(
        market_data=MarketDataAdapter(),
        strategy=StrategyAdapter(),
        portfolio=PortfolioAdapter(),
        execution=ExecutionAdapter(),
        token_discovery=TokenDiscoveryAdapter(),
        monitoring=MonitoringAdapter(),
    )
