"""Convenience imports for strategy modules."""

from __future__ import annotations

import importlib


def _optional_import(name: str):
    """Import ``name`` from this package, returning ``None`` on failure."""

    try:  # pragma: no cover - optional dependencies
        return importlib.import_module(f".{name}", __name__)
    except Exception:  # pragma: no cover - ignore any import errors
        return None


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

# Export Solana sniper strategy module under a unified name
sniper_solana = importlib.import_module("crypto_bot.strategies.sniper_solana")
solana_scalping = importlib.import_module("crypto_bot.solana.scalping")

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
    ]
    if globals().get(name) is not None
]

