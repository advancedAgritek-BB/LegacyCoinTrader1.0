from __future__ import annotations

import pandas as pd
from pathlib import Path

from crypto_bot.utils.logger import LOG_DIR
from typing import Callable, Dict, Optional

from crypto_bot.strategy import STRATEGIES
from crypto_bot.strategy.base import StrategyProtocol

# Default log file location
LOG_FILE = Path("crypto_bot/logs/strategy_pnl.csv")

_STRATEGY_CACHE: Dict[str, StrategyProtocol] = {}


def _resolve_strategy(name: str) -> Optional[StrategyProtocol]:
    strategy = _STRATEGY_CACHE.get(name)
    if strategy is not None:
        return strategy

    strategy = STRATEGIES.get(name)
    if strategy is None:
        try:
            from ..strategy_router import get_strategy_by_name as router_get
        except Exception:
            router_get = None
        if router_get is not None:
            strategy = router_get(name)
    if strategy is not None:
        _STRATEGY_CACHE[name] = strategy
    return strategy


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
        strategy = _resolve_strategy(best)
        if strategy is None:
            return _get_strategy_for()(regime)
        return strategy


_selector = RLStrategySelector()


def train(log_file: Path = LOG_FILE) -> None:
    """Train the global selector."""
    _selector.train(log_file)


def select_strategy(regime: str) -> Callable[[pd.DataFrame], tuple]:
    """Return strategy for regime using the trained selector."""
    if not _selector.regime_scores:
        _selector.train()
    return _selector.select(regime)
