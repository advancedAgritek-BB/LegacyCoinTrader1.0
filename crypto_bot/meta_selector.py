import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from crypto_bot.utils.logger import LOG_DIR
from datetime import datetime

import pandas as pd

from crypto_bot.strategy import STRATEGIES
from crypto_bot.strategy.base import StrategyProtocol


LOG_FILE = LOG_DIR / "strategy_performance.json"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "meta_selector_lgbm.txt"


class MetaRegressor:
    """Wrapper for the LightGBM model predicting strategy PnL."""

    MODEL_PATH = MODEL_PATH
    _model: Optional[object] = None

    @classmethod
    def _load(cls) -> Optional[object]:
        if cls._model is None and cls.MODEL_PATH.exists():
            try:
                import lightgbm as lgb
                cls._model = lgb.Booster(model_file=str(cls.MODEL_PATH))
            except Exception:
                cls._model = None
        return cls._model

    @classmethod
    def predict_scores(
        cls, regime: str, stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Return expected PnL per strategy using the ML model."""

        model = cls._load()
        if model is None or not stats:
            return {}
        df = pd.DataFrame.from_dict(stats, orient="index")
        # Regime may be encoded in the model; include as column if supported
        features = []
        try:
            features = model.feature_name()
        except Exception:
            pass
        if "regime" in features:
            df["regime"] = regime
        try:
            preds = model.predict(df)
        except Exception:
            return {}
        return {k: float(v) for k, v in zip(df.index, preds)}


_STRATEGY_CACHE: Dict[str, StrategyProtocol] = {}


def get_strategy_by_name(name: str) -> Optional[StrategyProtocol]:
    """Return the strategy mapped to ``name`` if present."""

    strategy = _STRATEGY_CACHE.get(name)
    if strategy is not None:
        return strategy

    strategy = STRATEGIES.get(name)
    if strategy is None:
        try:
            from crypto_bot.strategy_router import get_strategy_by_name as router_get
        except Exception:
            router_get = None
        if router_get is not None:
            strategy = router_get(name)
    if strategy is not None:
        _STRATEGY_CACHE[name] = strategy
    return strategy


def _load() -> Dict[str, Dict[str, List[dict]]]:
    """Return parsed performance log data."""
    if not LOG_FILE.exists():
        return {}
    try:
        return json.loads(LOG_FILE.read_text())
    except Exception:
        return {}


def _compute_stats(trades: List[dict]) -> Optional[Dict[str, float]]:
    now = datetime.utcnow()
    pnls = [
        float(t["pnl"]) * (0.98 ** (now - datetime.fromisoformat(t["timestamp"])).days)
        for t in trades
    ]
    if not pnls:
        return None
    wins = sum(p > 0 for p in pnls)
    total = len(pnls)
    win_rate = wins / total if total else 0.0
    series = pd.Series(pnls)
    neg_returns = series[series < 0]
    downside_std = neg_returns.std(ddof=0) if not neg_returns.empty else 0.0
    max_dd = (series.cummax() - series).max()
    raw_sharpe = 0.0
    std = series.std()
    if std:
        raw_sharpe = series.mean() / std * (total ** 0.5)
    return {
        "win_rate": win_rate,
        "raw_sharpe": float(raw_sharpe),
        "downside_std": float(downside_std),
        "max_dd": float(max_dd),
        "trade_count": total,
    }


def _stats_for(regime: str) -> Dict[str, Dict[str, float]]:
    data = _load().get(regime, {})
    stats: Dict[str, Dict[str, float]] = {}
    for strat, trades in data.items():
        s = _compute_stats(trades)
        if s is not None:
            stats[strat] = s
    return stats

def _scores_for(regime: str) -> Dict[str, float]:
    """Compute score per strategy for ``regime``."""
    data = _load().get(regime, {})
    scores: Dict[str, float] = {}
    for strat, trades in data.items():
        stats = _compute_stats(trades)
        if stats is None:
            continue
        score = (
            stats["win_rate"]
            * stats["raw_sharpe"]
            / (1 + stats["downside_std"] + stats["max_dd"])
        )
        penalty = 0.5 * stats["max_dd"]
        score -= penalty
        scores[strat] = max(score, 0.0)
    return scores


def choose_best(regime: str) -> Callable[[pd.DataFrame], tuple]:
    """Return strategy with best historical score for ``regime``."""
    # Lazy import to avoid circular dependency
    def _get_strategy_for():
        from .strategy_router import strategy_for
        return strategy_for

    scores = _scores_for(regime)
    if not scores:
        return _get_strategy_for()(regime)

    if MetaRegressor.MODEL_PATH.exists():
        stats = _stats_for(regime)
        ml_scores = MetaRegressor.predict_scores(regime, stats)
        if ml_scores:
            scores = ml_scores

    best = max(scores.items(), key=lambda x: x[1])[0]
    strategy = get_strategy_by_name(best)
    if strategy is None:
        return _get_strategy_for()(regime)
    return strategy


class MetaSelector:
    """Meta strategy selector that chooses the best strategy based on historical performance."""
    
    def __init__(self):
        self.regime_stats = {}
        self.strategy_scores = {}
    
    def get_best_strategy(self, regime: str) -> Callable[[pd.DataFrame], tuple]:
        """Get the best strategy for a given regime."""
        return choose_best(regime)
    
    def get_strategy_scores(self, regime: str) -> Dict[str, float]:
        """Get strategy scores for a given regime."""
        return _scores_for(regime)
    
    def get_regime_stats(self, regime: str) -> Dict[str, Dict[str, float]]:
        """Get statistics for all strategies in a regime."""
        return _stats_for(regime)
    
    def update_performance(self, regime: str, strategy: str, trade_data: dict):
        """Update performance data for a strategy."""
        # This would typically update the performance log
        pass
