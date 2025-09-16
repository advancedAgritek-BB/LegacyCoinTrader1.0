"""Convenience imports for strategy modules."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from .base import BaseStrategy, FunctionStrategy, StrategyProtocol, as_strategy


STRATEGY_REGISTRY: Dict[str, StrategyProtocol] = {}


def _register_strategy(name: str, module: Any) -> Optional[StrategyProtocol]:
    """Adapt ``module`` to :class:`StrategyProtocol` and store it."""

    if module is None:
        return None

    candidate = getattr(module, "Strategy", module)
    extras = {
        key: getattr(module, key)
        for key in ("regime_filter", "trigger_once")
        if hasattr(module, key)
    }
    extras["module"] = module
    strategy_name = (
        getattr(module, "NAME", None)
        or getattr(candidate, "name", None)
        or getattr(candidate, "NAME", None)
        or name
    )
    try:
        strategy = as_strategy(candidate, name=strategy_name, extras=extras)
    except TypeError as exc:  # pragma: no cover - best effort
        print(f"Warning: Failed to adapt strategy {name}: {exc}")
        return None

    if isinstance(candidate, BaseStrategy):
        strategy = candidate
    setattr(strategy, "module", module)
    STRATEGY_REGISTRY[name] = strategy
    return strategy


def _optional_import(name: str):
    """Import ``name`` from this package, returning ``None`` on failure."""

    try:  # pragma: no cover - optional dependencies
        module = importlib.import_module(f".{name}", __name__)
    except Exception as e:  # pragma: no cover - ignore any import errors
        print(f"Warning: Failed to import {name}: {e}")
        return None
    _register_strategy(name, module)
    return module


# Core strategies
bounce_scalper = _optional_import("bounce_scalper")
dca_bot = _optional_import("dca_bot")
breakout_bot = _optional_import("breakout_bot")
dex_scalper = _optional_import("dex_scalper")
grid_bot = _optional_import("grid_bot")
mean_bot = _optional_import("mean_bot")
micro_scalp_bot = _optional_import("micro_scalp_bot")
sniper_bot = _optional_import("sniper_bot")
trend_bot = _optional_import("trend_bot")

# New strategies from strategy copy folder
cross_chain_arb_bot = _optional_import("cross_chain_arb_bot")
dip_hunter = _optional_import("dip_hunter")
flash_crash_bot = _optional_import("flash_crash_bot")
hft_engine = _optional_import("hft_engine")
lstm_bot = _optional_import("lstm_bot")
maker_spread = _optional_import("maker_spread")
momentum_bot = _optional_import("momentum_bot")
range_arb_bot = _optional_import("range_arb_bot")
stat_arb_bot = _optional_import("stat_arb_bot")
meme_wave_bot = _optional_import("meme_wave_bot")

# Ultra-aggressive strategies
ultra_scalp_bot = _optional_import("ultra_scalp_bot")
momentum_exploiter = _optional_import("momentum_exploiter")
volatility_harvester = _optional_import("volatility_harvester")

try:  # Export Solana sniper strategy if available
    sniper_solana = importlib.import_module("crypto_bot.strategies.sniper_solana")
    _register_strategy("sniper_solana", sniper_solana)
except Exception as e:  # pragma: no cover - optional during tests
    print(f"Warning: Failed to import sniper_solana: {e}")
    sniper_solana = None
try:
    solana_scalping = importlib.import_module("crypto_bot.solana.scalping")
    _register_strategy("solana_scalping", solana_scalping)
except Exception as e:  # pragma: no cover - optional during tests
    print(f"Warning: Failed to import solana_scalping: {e}")
    solana_scalping = None


def _register_alias(alias: str, target: str) -> None:
    if alias in STRATEGY_REGISTRY:
        return
    strat = STRATEGY_REGISTRY.get(target)
    if strat is not None:
        STRATEGY_REGISTRY[alias] = strat


_ALIASES = {
    "trend": "trend_bot",
    "grid": "grid_bot",
    "sniper": "sniper_bot",
    "dex_scalper_bot": "dex_scalper",
    "micro_scalp": "micro_scalp_bot",
    "bounce_scalper_bot": "bounce_scalper",
    "solana_scalping_bot": "solana_scalping",
    "dca": "dca_bot",
    "momentum": "momentum_bot",
    "lstm": "lstm_bot",
    "ultra_scalp": "ultra_scalp_bot",
    "range_arb": "range_arb_bot",
    "stat_arb": "stat_arb_bot",
}

for alias, target in _ALIASES.items():
    _register_alias(alias, target)

STRATEGIES = STRATEGY_REGISTRY

__all__ = [
    name
    for name in [
        # Core strategies
        "bounce_scalper",
        "breakout_bot",
        "dex_scalper",
        "dca_bot",
        "grid_bot",
        "mean_bot",
        "micro_scalp_bot",
        "sniper_bot",
        "trend_bot",
        "sniper_solana",
        "solana_scalping",
        # New strategies
        "cross_chain_arb_bot",
        "dip_hunter",
        "flash_crash_bot",
        "hft_engine",
        "lstm_bot",
        "maker_spread",
        "momentum_bot",
        "range_arb_bot",
        "stat_arb_bot",
        "meme_wave_bot",
        # Ultra-aggressive strategies
        "ultra_scalp_bot",
        "momentum_exploiter",
        "volatility_harvester",
    ]
    if globals().get(name) is not None
]

__all__.extend(
    [
        "STRATEGIES",
        "STRATEGY_REGISTRY",
        "BaseStrategy",
        "StrategyProtocol",
        "FunctionStrategy",
        "as_strategy",
    ]
)

