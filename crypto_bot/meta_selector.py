import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from crypto_bot.utils.logger import LOG_DIR
from datetime import datetime

import pandas as pd


# Lazy imports to avoid circular dependencies
# from crypto_bot.strategy import (
#     trend_bot,
#     grid_bot,
#     sniper_bot,
#     dex_scalper,
#     dca_bot,
#     mean_bot,
#     breakout_bot,
#     micro_scalp_bot,
#     bounce_scalper,
#     solana_scalping,
# )

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


def _resolve_signal(obj: object, alias: str) -> Optional[Callable[[pd.DataFrame], tuple]]:
    if not obj:
        return None
    globals().setdefault(alias, obj)
    return getattr(obj, "generate_signal", None)


def _get_strategy_function(strategy_name: str):
    """Lazy import strategy functions to avoid circular dependencies."""
    try:
        if strategy_name in ["trend", "trend_bot"]:
            from crypto_bot.strategy import trend_bot

            return _resolve_signal(trend_bot, "trend_bot")
        elif strategy_name in ["grid", "grid_bot"]:
            from crypto_bot.strategy import grid_bot

            return _resolve_signal(grid_bot, "grid_bot")
        elif strategy_name in ["sniper", "sniper_bot"]:
            from crypto_bot.strategy import sniper_bot

            return _resolve_signal(sniper_bot, "sniper_bot")
        elif strategy_name in ["dex_scalper", "dex_scalper_bot"]:
            from crypto_bot.strategy import dex_scalper

            return _resolve_signal(dex_scalper, "dex_scalper")
        elif strategy_name in ["mean_bot"]:
            from crypto_bot.strategy import mean_bot

            return _resolve_signal(mean_bot, "mean_bot")
        elif strategy_name in ["breakout_bot"]:
            from crypto_bot.strategy import breakout_bot

            return _resolve_signal(breakout_bot, "breakout_bot")
        elif strategy_name in ["micro_scalp", "micro_scalp_bot"]:
            from crypto_bot.strategy import micro_scalp_bot

            return _resolve_signal(micro_scalp_bot, "micro_scalp_bot")
        elif strategy_name in ["bounce_scalper", "bounce_scalper_bot"]:
            from crypto_bot.strategy import bounce_scalper

            return _resolve_signal(bounce_scalper, "bounce_scalper")
        elif strategy_name in ["solana_scalping", "solana_scalping_bot"]:
            from crypto_bot.strategy import solana_scalping

            return _resolve_signal(solana_scalping, "solana_scalping")
        elif strategy_name in ["dca", "dca_bot"]:
            from crypto_bot.strategy import dca_bot

            return _resolve_signal(dca_bot, "dca_bot")
        elif strategy_name in ["momentum", "momentum_bot"]:
            from crypto_bot.strategy import momentum_bot

            return _resolve_signal(momentum_bot, "momentum_bot")
        elif strategy_name in ["lstm", "lstm_bot"]:
            from crypto_bot.strategy import lstm_bot

            return _resolve_signal(lstm_bot, "lstm_bot")
        elif strategy_name in ["ultra_scalp", "ultra_scalp_bot"]:
            from crypto_bot.strategy import ultra_scalp_bot

            return _resolve_signal(ultra_scalp_bot, "ultra_scalp_bot")
        elif strategy_name in ["volatility_harvester"]:
            from crypto_bot.strategy import volatility_harvester

            return _resolve_signal(volatility_harvester, "volatility_harvester")
        elif strategy_name in ["hft_engine"]:
            from crypto_bot.strategy import hft_engine

            return _resolve_signal(hft_engine, "hft_engine")
        elif strategy_name in ["maker_spread"]:
            from crypto_bot.strategy import maker_spread

            return _resolve_signal(maker_spread, "maker_spread")
        elif strategy_name in ["flash_crash_bot"]:
            from crypto_bot.strategy import flash_crash_bot

            return _resolve_signal(flash_crash_bot, "flash_crash_bot")
        elif strategy_name in ["meme_wave_bot"]:
            from crypto_bot.strategy import meme_wave_bot

            return _resolve_signal(meme_wave_bot, "meme_wave_bot")
        elif strategy_name in ["cross_chain_arb_bot"]:
            from crypto_bot.strategy import cross_chain_arb_bot

            return _resolve_signal(cross_chain_arb_bot, "cross_chain_arb_bot")
        elif strategy_name in ["dip_hunter"]:
            from crypto_bot.strategy import dip_hunter

            return _resolve_signal(dip_hunter, "dip_hunter")
        elif strategy_name in ["range_arb_bot"]:
            from crypto_bot.strategy import range_arb_bot

            return _resolve_signal(range_arb_bot, "range_arb_bot")
        elif strategy_name in ["stat_arb_bot"]:
            from crypto_bot.strategy import stat_arb_bot

            return _resolve_signal(stat_arb_bot, "stat_arb_bot")
        elif strategy_name in ["momentum_exploiter"]:
            from crypto_bot.strategy import momentum_exploiter

            return _resolve_signal(momentum_exploiter, "momentum_exploiter")
        elif strategy_name in ["arbitrage_engine"]:
            from crypto_bot.strategy import arbitrage_engine

            return _resolve_signal(arbitrage_engine, "arbitrage_engine")
        else:
            return None
    except ImportError:
        return None

# Strategy map is populated on import so tests can introspect it directly
_STRATEGY_FN_MAP: Dict[str, Optional[Callable[[pd.DataFrame], tuple]]] = {}


def _ensure_strategy_map() -> None:
    if _STRATEGY_FN_MAP:
        return

    entries = {
        "trend": _get_strategy_function("trend"),
        "trend_bot": _get_strategy_function("trend_bot"),
        "grid": _get_strategy_function("grid"),
        "grid_bot": _get_strategy_function("grid_bot"),
        "sniper": _get_strategy_function("sniper"),
        "sniper_bot": _get_strategy_function("sniper_bot"),
        "dex_scalper": _get_strategy_function("dex_scalper"),
        "dex_scalper_bot": _get_strategy_function("dex_scalper_bot"),
        "mean_bot": _get_strategy_function("mean_bot"),
        "breakout_bot": _get_strategy_function("breakout_bot"),
        "micro_scalp": _get_strategy_function("micro_scalp"),
        "micro_scalp_bot": _get_strategy_function("micro_scalp_bot"),
        "bounce_scalper": _get_strategy_function("bounce_scalper"),
        "bounce_scalper_bot": _get_strategy_function("bounce_scalper_bot"),
        "solana_scalping": _get_strategy_function("solana_scalping"),
        "solana_scalping_bot": _get_strategy_function("solana_scalping_bot"),
        "dca": _get_strategy_function("dca"),
        "dca_bot": _get_strategy_function("dca_bot"),
        "momentum": _get_strategy_function("momentum"),
        "momentum_bot": _get_strategy_function("momentum_bot"),
        "lstm": _get_strategy_function("lstm"),
        "lstm_bot": _get_strategy_function("lstm_bot"),
        "ultra_scalp": _get_strategy_function("ultra_scalp"),
        "ultra_scalp_bot": _get_strategy_function("ultra_scalp_bot"),
        "volatility_harvester": _get_strategy_function("volatility_harvester"),
        "hft_engine": _get_strategy_function("hft_engine"),
        "maker_spread": _get_strategy_function("maker_spread"),
        "flash_crash_bot": _get_strategy_function("flash_crash_bot"),
        "meme_wave_bot": _get_strategy_function("meme_wave_bot"),
        "cross_chain_arb_bot": _get_strategy_function("cross_chain_arb_bot"),
        "dip_hunter": _get_strategy_function("dip_hunter"),
        "range_arb_bot": _get_strategy_function("range_arb_bot"),
        "stat_arb_bot": _get_strategy_function("stat_arb_bot"),
        "momentum_exploiter": _get_strategy_function("momentum_exploiter"),
        "arbitrage_engine": _get_strategy_function("arbitrage_engine"),
    }
    _STRATEGY_FN_MAP.update({k: v for k, v in entries.items() if v is not None})


def get_strategy_by_name(
    name: str,
) -> Optional[Callable[[pd.DataFrame], tuple]]:
    """Return the strategy function mapped to ``name`` if present."""

    _ensure_strategy_map()
    return _STRATEGY_FN_MAP.get(name)


_ensure_strategy_map()


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
    return _STRATEGY_FN_MAP.get(best, _get_strategy_for()(regime))


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
