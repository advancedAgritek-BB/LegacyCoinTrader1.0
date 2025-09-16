"""Unified strategy loading helpers."""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict

from .base import StrategyProtocol, coerce_to_strategy


def _import_strategy_module(path: str, *, name: str) -> ModuleType | None:
    """Import and return the module referenced by ``path``."""

    try:  # pragma: no cover - optional dependencies
        if path.startswith("."):
            return importlib.import_module(path, __name__)
        return importlib.import_module(path)
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"Warning: Failed to import {name}: {exc}")
        return None


def _load_strategy(name: str, path: str) -> StrategyProtocol | None:
    """Import ``path`` and adapt it into a :class:`StrategyProtocol`."""

    module = _import_strategy_module(path, name=name)
    if module is None:
        return None
    try:
        return coerce_to_strategy(module, name=name)
    except TypeError as exc:  # pragma: no cover - best effort logging
        print(f"Warning: Strategy {name} does not expose generate_signal: {exc}")
        return None


_RELATIVE_STRATEGIES = [
    "arbitrage_engine",
    "bounce_scalper",
    "breakout_bot",
    "cross_chain_arb_bot",
    "dca_bot",
    "dex_scalper",
    "dip_hunter",
    "flash_crash_bot",
    "grid_bot",
    "hft_engine",
    "lstm_bot",
    "maker_spread",
    "market_making_bot",
    "mean_bot",
    "meme_wave_bot",
    "micro_scalp_bot",
    "momentum_bot",
    "momentum_exploiter",
    "range_arb_bot",
    "sniper_bot",
    "stat_arb_bot",
    "trend_bot",
    "ultra_scalp_bot",
    "volatility_harvester",
]
_EXTERNAL_STRATEGIES = {
    "sniper_solana": "crypto_bot.strategies.sniper_solana",
    "solana_scalping": "crypto_bot.solana.scalping",
}
STRATEGY_SOURCES: Dict[str, str] = {
    name: f".{name}" for name in _RELATIVE_STRATEGIES
}
STRATEGY_SOURCES.update(_EXTERNAL_STRATEGIES)

_loaded = {name: _load_strategy(name, path) for name, path in STRATEGY_SOURCES.items()}
globals().update(_loaded)

STRATEGY_REGISTRY: Dict[str, StrategyProtocol] = {
    name: strategy for name, strategy in _loaded.items() if strategy is not None
}

STRATEGY_ALIASES: Dict[str, str] = {
    "trend": "trend_bot",
    "grid": "grid_bot",
    "sniper": "sniper_bot",
    "micro_scalp": "micro_scalp_bot",
    "dex_scalper_bot": "dex_scalper",
    "bounce_scalper_bot": "bounce_scalper",
    "dca": "dca_bot",
    "momentum": "momentum_bot",
    "lstm": "lstm_bot",
    "ultra_scalp": "ultra_scalp_bot",
    "solana_scalping_bot": "solana_scalping",
}


def get_strategy(name: str) -> StrategyProtocol | None:
    """Return the strategy registered under ``name`` if available."""

    canonical = STRATEGY_ALIASES.get(name, name)
    strategy = STRATEGY_REGISTRY.get(canonical)
    if strategy is not None:
        return strategy

    path = STRATEGY_SOURCES.get(canonical)
    if path is None:
        return None

    loaded = _load_strategy(canonical, path)
    if loaded is not None:
        STRATEGY_REGISTRY[canonical] = loaded
        globals()[canonical] = loaded
    return loaded


__all__ = [
    name for name, strategy in _loaded.items() if strategy is not None
]
__all__.extend(["STRATEGY_REGISTRY", "STRATEGY_ALIASES", "get_strategy"])
