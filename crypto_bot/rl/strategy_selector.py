from __future__ import annotations

import pandas as pd
from pathlib import Path

from crypto_bot.utils.logger import LOG_DIR
from typing import Callable, Dict

from crypto_bot.strategy import STRATEGY_ALIASES, get_strategy

# Default log file location
LOG_FILE = Path("crypto_bot/logs/strategy_pnl.csv")

# Map strategy names to generation functions
_STRATEGY_FN_MAP: Dict[str, Callable[[pd.DataFrame], tuple]] = {}
for name in [
    "trend_bot",
    "grid_bot",
    "sniper_bot",
    "dex_scalper",
    "dca_bot",
    "mean_bot",
    "breakout_bot",
    "solana_scalping",
]:
    strategy = get_strategy(name)
    if strategy is not None:
        _STRATEGY_FN_MAP[name] = strategy.generate_signal

for alias, canonical in STRATEGY_ALIASES.items():
    strategy = _STRATEGY_FN_MAP.get(canonical)
    if strategy is None:
        obj = get_strategy(alias)
        strategy = obj.generate_signal if obj is not None else None
    if strategy is not None:
        _STRATEGY_FN_MAP.setdefault(alias, strategy)


class RLStrategySelector:
    """Contextual bandit using mean PnL weighted by trade counts."""

    def __init__(self) -> None:
        # {regime: {strategy: {"mean": float, "count": int}}}
        self.regime_scores: Dict[str, Dict[str, Dict[str, float]]] = {}

    def train(self, log_file: Path = LOG_FILE) -> None:
        """Train on historical PnL log."""
        if not Path(log_file).exists():
            return
        df = pd.read_csv(log_file)
        if {"regime", "strategy", "pnl"}.issubset(df.columns):
            grouped = df.groupby(["regime", "strategy"]).agg(
                mean=("pnl", "mean"), count=("pnl", "count")
            )
            for (regime, strat), row in grouped.iterrows():
                self.regime_scores.setdefault(regime, {})[strat] = {
                    "mean": float(row["mean"]),
                    "count": int(row["count"]),
                }

    def select(self, regime: str) -> Callable[[pd.DataFrame], tuple]:
        # Lazy import to avoid circular dependency
        def _get_strategy_for():
            from ..strategy_router import strategy_for
            return strategy_for

        scores = self.regime_scores.get(regime)
        if not scores:
            return _get_strategy_for()(regime)
        best = max(
            scores.items(), key=lambda x: x[1]["mean"] * x[1]["count"]
        )[0]
        return _STRATEGY_FN_MAP.get(best, _get_strategy_for()(regime))


_selector = RLStrategySelector()


def train(log_file: Path = LOG_FILE) -> None:
    """Train the global selector."""
    _selector.train(log_file)


def select_strategy(regime: str) -> Callable[[pd.DataFrame], tuple]:
    """Return strategy for regime using the trained selector."""
    if not _selector.regime_scores:
        _selector.train()
    return _selector.select(regime)
