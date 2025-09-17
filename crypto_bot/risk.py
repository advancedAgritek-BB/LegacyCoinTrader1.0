"""Risk management utilities for the trading bot."""
from __future__ import annotations

from typing import Any, Dict

from crypto_bot.risk.risk_manager import RiskConfig, RiskManager

_VALID_RISK_FIELDS = {
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


def _merge_risk_parameters(config: Dict[str, Any], volume_ratio: float) -> Dict[str, Any]:
    """Merge risk parameters from multiple config sections."""

    risk_params: Dict[str, Any] = {**config.get("risk", {})}
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

    return {
        key: value for key, value in risk_params.items() if key in _VALID_RISK_FIELDS
    }


def build_risk_manager(config: Dict[str, Any], volume_ratio: float) -> RiskManager:
    """Create a :class:`RiskManager` instance using configuration values."""

    filtered_params = _merge_risk_parameters(config, volume_ratio)
    risk_config = RiskConfig(**filtered_params)
    return RiskManager(risk_config)


__all__ = ["build_risk_manager", "RiskManager", "RiskConfig"]
