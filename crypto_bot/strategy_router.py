"""Strategy router for selecting and executing trading strategies based on market conditions."""

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, Optional


# Debug utility to help identify DataFrame conversion issues
DEBUG_DATAFRAME_CONVERSION = False

def enable_dataframe_debug():
    """Enable debug logging for DataFrame conversions."""
    global DEBUG_DATAFRAME_CONVERSION
    DEBUG_DATAFRAME_CONVERSION = True
    logger.info("DataFrame conversion debugging enabled")

def debug_dataframe_type(obj, context=""):
    """Debug helper to log DataFrame types."""
    if DEBUG_DATAFRAME_CONVERSION:
        logger.debug(f"{context} - Object type: {type(obj)}, is DataFrame: {isinstance(obj, pd.DataFrame)}")
        if isinstance(obj, dict):
            logger.debug(f"{context} - Dict keys: {list(obj.keys())[:10]}")
        elif isinstance(obj, pd.DataFrame):
            logger.debug(f"{context} - DataFrame shape: {obj.shape}, columns: {list(obj.columns)}")

import asyncio
import logging
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union, Iterable

import pandas as pd

from dataclasses import dataclass, field, asdict
import redis

import numpy as np

from pathlib import Path
import yaml
import json
import time
from functools import lru_cache
from datetime import datetime

from crypto_bot.utils import timeframe_seconds, commit_lock

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.telemetry import telemetry
import threading
from collections import defaultdict
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.utils.cache_helpers import cache_by_id
from crypto_bot.selector import bandit

from crypto_bot.strategy import STRATEGY_ALIASES, STRATEGY_REGISTRY, get_strategy
from crypto_bot.strategy.base import StrategyProtocol


_STRATEGY_REFERENCES = [
    "trend_bot",
    "grid_bot",
    "sniper_bot",
    "sniper_solana",
    "dex_scalper",
    "mean_bot",
    "breakout_bot",
    "micro_scalp_bot",
    "bounce_scalper",
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
    "ultra_scalp_bot",
    "momentum_exploiter",
    "volatility_harvester",
    "solana_scalping",
    "dca_bot",
]
globals().update({name: get_strategy(name) for name in _STRATEGY_REFERENCES})


import random
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = setup_logger(__name__, LOG_DIR / "bot.log")
_SYMBOL_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

STRATEGY_CONFIG_TYPES: dict[str, type] = {
    "trend_bot": getattr(trend_bot, "TrendBotConfig", None),
    "mean_bot": getattr(mean_bot, "MeanBotConfig", None),
    "dex_scalper": getattr(dex_scalper, "DexScalperConfig", None),
    "breakout_bot": getattr(breakout_bot, "BreakoutBotConfig", None),
    "micro_scalp_bot": getattr(micro_scalp_bot, "MicroScalpConfig", None),
    "momentum_bot": getattr(momentum_bot, "MomentumBotConfig", None),
    "sniper_bot": getattr(sniper_bot, "SniperBotConfig", None),
}
STRATEGY_CONFIG_TYPES = {
    name: cls for name, cls in STRATEGY_CONFIG_TYPES.items() if cls is not None
}


def _resolve_strategy_name(fn: Callable[..., Any]) -> str:
    """Return the canonical strategy name for ``fn``."""

    owner = getattr(fn, "__self__", None)
    if isinstance(owner, StrategyProtocol):
        return owner.name
    if owner is not None and hasattr(owner, "name"):
        return getattr(owner, "name")

    module = getattr(fn, "__module__", "")
    if module.startswith("crypto_bot.strategy."):
        return module.rsplit(".", 1)[-1]
    return getattr(fn, "__name__", "")


def _build_strategy_config_object(
    fn: Callable[[pd.DataFrame], Tuple[float, str]],
    cfg: Optional[Union[RouterConfig, Mapping[str, Any]]],
) -> Any:
    """Return configuration object for ``fn`` derived from ``cfg``."""

    strategy_name = _resolve_strategy_name(fn)
    config_cls = STRATEGY_CONFIG_TYPES.get(strategy_name)

    raw_cfg: Optional[Mapping[str, Any]]
    if isinstance(cfg, RouterConfig):
        raw_cfg = cfg.as_dict()
    else:
        raw_cfg = cfg  # type: ignore[assignment]

    if config_cls is not None:
        return config_cls.from_dict(raw_cfg)

    if isinstance(raw_cfg, Mapping):
        specific = raw_cfg.get(strategy_name)
        if isinstance(specific, Mapping):
            merged = dict(raw_cfg)
            merged.update(specific)
            return merged
        return raw_cfg

    return raw_cfg


from crypto_bot.config import load_config as load_bot_config, resolve_config_path

CONFIG_PATH = resolve_config_path()
try:
    DEFAULT_CONFIG = load_bot_config(CONFIG_PATH)
except Exception:
    DEFAULT_CONFIG = {}

# Map symbols to asyncio locks guarding order placement
symbol_locks: Dict[str, asyncio.Lock] = {}

# Event loop captured when locks are first acquired
_LOCK_LOOP: Optional[asyncio.AbstractEventLoop] = None


async def acquire_symbol_lock(symbol: str) -> None:
    """Acquire the asyncio lock associated with ``symbol``."""
    global _LOCK_LOOP
    if _LOCK_LOOP is None:
        _LOCK_LOOP = asyncio.get_running_loop()
    lock = symbol_locks.setdefault(symbol, asyncio.Lock())
    await lock.acquire()


async def release_symbol_lock(symbol: str) -> None:
    """Release the lock for ``symbol`` if held."""
    lock = symbol_locks.get(symbol)
    if lock and lock.locked():
        lock.release()


@dataclass
class RouterConfig:
    """Configuration for routing strategies."""

    regimes: Dict[str, Iterable[str]] = field(default_factory=dict)
    min_score: float = 0.0
    fusion_method: str = "weight"
    perf_window: int = 20
    min_confidence: float = 0.0
    fusion_enabled: bool = False
    strategies: list[Tuple[str, float]] = field(default_factory=list)
    rl_selector: bool = False
    meta_selector: bool = False
    bandit_enabled: bool = False
    timeframe: str = "1h"
    timeframe_minutes: int = 60
    trending_timeframe: Optional[str] = None
    volatile_timeframe: Optional[str] = None
    sideways_timeframe: Optional[str] = None
    scalp_timeframe: Optional[str] = None
    breakout_timeframe: Optional[str] = None
    commit_lock_intervals: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouterConfig":
        """Create ``RouterConfig`` from a dictionary (e.g. YAML)."""
        router = data.get("strategy_router", {})
        fusion = data.get("signal_fusion", {})
        tf = str(data.get("timeframe", "1h"))
        return cls(
            regimes=router.get("regimes", {}),
            min_score=float(
                data.get("min_confidence_score", data.get("signal_threshold", 0.0))
            ),
            fusion_method=fusion.get("fusion_method", "weight"),
            perf_window=int(fusion.get("perf_window", 20)),
            min_confidence=float(fusion.get("min_confidence", 0.0)),
            fusion_enabled=bool(fusion.get("enabled", False)),
            strategies=[tuple(x) for x in fusion.get("strategies", [])],
            rl_selector=bool(data.get("rl_selector", {}).get("enabled", False)),
            meta_selector=bool(data.get("meta_selector", {}).get("enabled", False)),
            bandit_enabled=bool(data.get("bandit", {}).get("enabled", False)),
            timeframe=tf,
            timeframe_minutes=int(pd.Timedelta(tf).total_seconds() // 60),
            trending_timeframe=str(router.get("trending_timeframe", data.get("trending_timeframe", tf))) or None,
            volatile_timeframe=str(router.get("volatile_timeframe", data.get("volatile_timeframe", tf))) or None,
            sideways_timeframe=str(router.get("sideways_timeframe", data.get("sideways_timeframe", tf))) or None,
            scalp_timeframe=str(router.get("scalp_timeframe", data.get("scalp_timeframe", tf))) or None,
            breakout_timeframe=str(router.get("breakout_timeframe", data.get("breakout_timeframe", tf))) or None,
            commit_lock_intervals=int(router.get("commit_lock_intervals", 0)),
            raw=data,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the underlying raw dictionary."""
        if isinstance(self.raw, Mapping):
            return dict(self.raw)
        return asdict(self)


DEFAULT_ROUTER_CFG = RouterConfig.from_dict(DEFAULT_CONFIG)


@dataclass
class BotStats:
    """Performance statistics for a trading bot."""

    sharpe_30d: float = 0.0
    win_rate_30d: float = 0.0
    avg_r_multiple: float = 0.0


def load_bot_stats(name: str) -> BotStats:
    """Return statistics for ``name`` loaded from Redis.

    For new strategies (no stats available), returns neutral defaults to allow testing.
    This prevents win rate filtering from blocking new symbols/strategies.
    """
    try:
        r = redis.Redis()
        raw = r.get(f"bot-stats:{name}")
    except Exception:
        # For new strategies, give them a neutral score to allow testing
        return BotStats(
            sharpe_30d=0.5,  # Neutral Sharpe ratio
            win_rate_30d=0.5,  # 50% win rate assumption for new strategies
            avg_r_multiple=1.0,  # Neutral R-multiple
        )
    if not raw:
        # For new strategies, give them a neutral score to allow testing
        return BotStats(
            sharpe_30d=0.5,  # Neutral Sharpe ratio
            win_rate_30d=0.5,  # 50% win rate assumption for new strategies
            avg_r_multiple=1.0,  # Neutral R-multiple
        )
    try:
        if isinstance(raw, bytes):
            raw = raw.decode()
        data = json.loads(raw)
    except Exception:
        # For new strategies, give them a neutral score to allow testing
        return BotStats(
            sharpe_30d=0.5,  # Neutral Sharpe ratio
            win_rate_30d=0.5,  # 50% win rate assumption for new strategies
            avg_r_multiple=1.0,  # Neutral R-multiple
        )
    return BotStats(
        sharpe_30d=float(data.get("sharpe_30d", 0.5)),  # Default to neutral if missing
        win_rate_30d=float(data.get("win_rate_30d", 0.5)),  # Default to 50% if missing
        avg_r_multiple=float(data.get("avg_r_multiple", 1.0)),  # Default to neutral if missing
    )


def score_bot(stats: BotStats) -> float:
    """Return a ranking score for ``stats``."""
    return (
        stats.sharpe_30d * 0.4
        + stats.win_rate_30d * 0.3
        + stats.avg_r_multiple * 0.3
    )


def cfg_get(cfg: Union[Mapping[str, Any], RouterConfig], key: str, default: Optional[Any] = None) -> Any:
    """Return configuration value ``key`` from ``cfg``.

    Supports both :class:`RouterConfig` instances and plain mapping objects. For
    ``RouterConfig`` the dataclass attributes are checked first, then the
    underlying ``raw`` mapping including the ``"strategy_router"`` section. For
    mappings the lookup is performed on the top level and falls back to the
    ``"strategy_router"`` subsection.
    """
    if isinstance(cfg, RouterConfig):
        if hasattr(cfg, key):
            return getattr(cfg, key, default)
        if isinstance(cfg.raw, Mapping):
            if key in cfg.raw:
                return cfg.raw.get(key, default)
            return cfg.raw.get("strategy_router", {}).get(key, default)
        return default
    if isinstance(cfg, Mapping):
        if key in cfg:
            return cfg.get(key, default)
        return cfg.get("strategy_router", {}).get(key, default)
    return default


def wrap_with_tf(fn: Callable[[pd.DataFrame], Tuple[float, str]], tf: str):
    """Return ``fn`` wrapped to extract ``tf`` from a dataframe map."""

    def wrapped(df_or_map: Any, cfg=None):
        df = None
        if isinstance(df_or_map, Mapping):
            df = df_or_map.get(tf)
            # If df is None, try fallback timeframes
            if df is None:
                for fallback_tf in ['15m', '1h', '4h', '1d']:
                    df = df_or_map.get(fallback_tf)
                    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                        break
        else:
            df = df_or_map

        # Ensure df is a valid DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning(f"Strategy {fn.__name__} received invalid data type: {type(df)}")
            return 0.0, "none"

        if isinstance(df, pd.DataFrame) and df.empty:
            logger.debug(f"Strategy {fn.__name__} received empty DataFrame")
            return 0.0, "none"

        try:
            return fn(df, cfg)
        except TypeError:
            return fn(df)

    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapped


# Path storing the last selected regime and timestamp
LAST_REGIME_FILE = LOG_DIR / "last_regime.json"


class Selector:
    """Helper class to select a strategy callable."""

    def __init__(self, config: RouterConfig):
        self.config = config

    def select(
        self,
        df: pd.DataFrame,
        regime: str,
        mode: str,
        notifier=None,
    ) -> Callable[[pd.DataFrame], Tuple[float, str]]:
        cfg = self.config

        if (isinstance(cfg, RouterConfig) and cfg.rl_selector) or (
            not isinstance(cfg, RouterConfig)
            and cfg.get("rl_selector", {}).get("enabled")
        ):
            from .rl import strategy_selector as rl_selector

            strategy_fn = rl_selector.select_strategy(regime)
            logger.info("RL selector chose %s for %s", strategy_fn.__name__, regime)
            return strategy_fn

        if (isinstance(cfg, RouterConfig) and cfg.meta_selector) or (
            not isinstance(cfg, RouterConfig)
            and cfg.get("meta_selector", {}).get("enabled")
        ):
            from . import meta_selector

            strategy_fn = meta_selector.choose_best(regime)
            logger.info("Meta selector chose %s for %s", strategy_fn.__name__, regime)
            return strategy_fn

        if (isinstance(cfg, RouterConfig) and cfg.fusion_enabled) or (
            not isinstance(cfg, RouterConfig)
            and cfg.get("signal_fusion", {}).get("enabled")
        ):
            from . import meta_selector
            from crypto_bot.signals.signal_fusion import SignalFusionEngine

            pairs_conf = (
                cfg.strategies
                if isinstance(cfg, RouterConfig)
                else cfg.get("signal_fusion", {}).get("strategies", [])
            )
            mapping = getattr(meta_selector, "_STRATEGY_FN_MAP", {})
            strategies: list[
                tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]
            ] = []
            for name, weight in pairs_conf:
                fn = mapping.get(name)
                if fn:
                    strategies.append((fn, float(weight)))
            if not strategies:
                strategies.append((strategy_for(regime, cfg), 1.0))
            engine = SignalFusionEngine(strategies)

            def fused(df: pd.DataFrame, cfg_param=None):
                return engine.fuse(
                    df,
                    cfg.as_dict() if isinstance(cfg, RouterConfig) else cfg_param,
                )

            logger.info("Routing to signal fusion engine")
            return fused

        strategy_fn = strategy_for(regime, cfg)
        logger.info("Routing to %s (%s)", strategy_fn.__name__, mode)
        return strategy_fn


def _strategy_function_map() -> Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]]:
    mapping: Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]] = {}
    for key, strategy in STRATEGY_REGISTRY.items():
        mapping[key] = strategy.generate_signal
    for alias, canonical in STRATEGY_ALIASES.items():
        strategy = STRATEGY_REGISTRY.get(canonical)
        if strategy is not None:
            mapping.setdefault(alias, strategy.generate_signal)
    return mapping


def get_strategy_by_name(
    name: str,
) -> Optional[Callable[[pd.DataFrame], Tuple[float, str]]]:
    """Return strategy callable for ``name`` if available."""

    strategy = get_strategy(name)
    if strategy is not None:
        return strategy.generate_signal

    from . import meta_selector
    from .rl import strategy_selector as rl_selector

    mapping: Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]] = _strategy_function_map()
    mapping.update(getattr(meta_selector, "_STRATEGY_FN_MAP", {}))
    mapping.update(getattr(rl_selector, "_STRATEGY_FN_MAP", {}))

    canonical = STRATEGY_ALIASES.get(name, name)
    return mapping.get(name) or mapping.get(canonical)


@cache_by_id
def _build_mappings(config: Union[Mapping[str, Any], RouterConfig]) -> tuple[
    Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]],
    Dict[str, list[Callable[[pd.DataFrame], Tuple[float, str]]]],
]:
    """Return mapping dictionaries from configuration."""
    if isinstance(config, RouterConfig):
        regimes = config.regimes
    else:
        regimes = config.get("strategy_router", {}).get("regimes", {})
    strat_map: Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]] = {}
    regime_map: Dict[str, list[Callable[[pd.DataFrame], Tuple[float, str]]]] = {}
    for regime, names in regimes.items():
        if isinstance(names, str):
            names = [names]
        funcs = [get_strategy_by_name(n) for n in names]
        funcs = [f for f in funcs if f]
        if funcs:
            strat_map[regime] = funcs[0]
            regime_map[regime] = funcs
    return strat_map, regime_map


_CONFIG_REGISTRY: Dict[int, Union[Mapping[str, Any], RouterConfig]] = {}


def _register_config(cfg: Union[Mapping[str, Any], RouterConfig]) -> int:
    """Register config and return its id for cache lookups."""
    cid = id(cfg)
    _CONFIG_REGISTRY[cid] = cfg
    return cid


@lru_cache(maxsize=8)
def _build_mappings_cached(config_id: int) -> tuple[
    Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]],
    Dict[str, list[Callable[[pd.DataFrame], Tuple[float, str]]]],
]:
    cfg = _CONFIG_REGISTRY.get(config_id, DEFAULT_ROUTER_CFG)
    return _build_mappings(cfg)

_register_config(DEFAULT_ROUTER_CFG)
STRATEGY_MAP, REGIME_STRATEGIES = _build_mappings_cached(id(DEFAULT_ROUTER_CFG))


def strategy_for(
    regime: str, 
    df: Optional[pd.DataFrame] = None,
    config: Optional[Union[RouterConfig, Mapping[str, Any]]] = None
) -> Callable[[pd.DataFrame], Tuple[float, str]]:
    """Return strategy callable for a given regime."""
    cfg = config or DEFAULT_ROUTER_CFG
    strategies = get_strategies_for_regime(regime, cfg)
    base = strategies[0] if strategies else grid_bot.generate_signal
    tf_key = f"{regime.replace('-', '_')}_timeframe"
    tf = cfg_get(cfg, tf_key, cfg_get(cfg, "timeframe", "1h"))
    return wrap_with_tf(base, tf)


def get_strategies_for_regime(
    regime: str, config: Optional[Union[RouterConfig, Mapping[str, Any]]] = None
) -> list[Callable[[pd.DataFrame], Tuple[float, str]]]:
    """Return list of strategies mapped to ``regime``."""
    cfg = config or DEFAULT_ROUTER_CFG
    _register_config(cfg)
    if isinstance(cfg, RouterConfig):
        names = cfg.regimes.get(regime, [])
    else:
        names = cfg.get("strategy_router", {}).get("regimes", {}).get(regime, [])
    if isinstance(names, str):
        names = [names]
    pairs: list[tuple[str, Callable[[pd.DataFrame], Tuple[float, str]]]] = []
    for name in names:
        fn = get_strategy_by_name(name)
        if fn:
            pairs.append((name, fn))
    if not pairs:
        _, mapping = _build_mappings_cached(id(cfg))
        return mapping.get(regime, [grid_bot.generate_signal])
    pairs.sort(key=lambda p: score_bot(load_bot_stats(p[0])), reverse=True)
    return [fn for _, fn in pairs]


def evaluate_regime(
    regime: str,
    df: pd.DataFrame,
    config: Optional[Union[RouterConfig, Mapping[str, Any]]] = None,
) -> Tuple[float, str]:
    """Evaluate and fuse all strategies assigned to ``regime``."""
    cfg = config or DEFAULT_ROUTER_CFG
    tf_key = f"{regime.replace('-', '_')}_timeframe"
    tf = cfg_get(cfg, tf_key, cfg_get(cfg, "timeframe", "1h"))
    strategies = [wrap_with_tf(s, tf) for s in get_strategies_for_regime(regime, cfg)]

    if isinstance(cfg, RouterConfig):
        method = cfg.fusion_method
        min_conf = cfg.min_confidence
        cfg_dict = cfg.as_dict()
    else:
        fusion_cfg = cfg.get("signal_fusion", {})
        method = fusion_cfg.get("fusion_method", "weight")
        min_conf = float(fusion_cfg.get("min_confidence", 0.0))
        cfg_dict = cfg

    weights = {}
    if method == "weight":
        from crypto_bot.utils.regime_pnl_tracker import compute_weights

        weights = compute_weights(regime)

    pairs: list[Tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]] = []

    def _instrument(fn: Callable[[pd.DataFrame], Tuple[float, str]]):
        def wrapped(df_input, cfg_p=None):
            # Handle both DataFrame and dict inputs
            if isinstance(df_input, dict):
                # Extract the appropriate DataFrame from the mapping
                base_tf = cfg_dict.get("timeframe", "1h")
                df_p = df_input.get(base_tf)
                if df_p is None:
                    # Fallback to any available DataFrame
                    for tf in ['15m', '1h', '4h', '1d']:
                        df_p = df_input.get(tf)
                        if df_p is not None and isinstance(df_p, pd.DataFrame) and not df_p.empty:
                            break
                if df_p is None or not isinstance(df_p, pd.DataFrame):
                    logger.warning(f"Strategy {fn.__name__} received dict but no valid DataFrame found")
                    return 0.0, "none"
            else:
                df_p = df_input

            # Ensure df_p is a valid DataFrame
            if not isinstance(df_p, pd.DataFrame):
                logger.warning(f"Strategy {fn.__name__} received invalid data type: {type(df_p)}")
                return 0.0, "none"

            telemetry.inc("router.signals_checked")
            try:
                from crypto_bot.utils.strategy_utils import safe_strategy_execution
                timeout_seconds = cfg_dict.get("strategy_timeout_seconds", 10)
                res = safe_strategy_execution(fn, df_p, cfg_p, timeout_seconds)
            except Exception as e:
                logger.warning(f"Strategy {fn.__name__} failed: {e}")
                return 0.0, "none"
            telemetry.inc("router.signal_returned")
            return res

        wrapped.__name__ = getattr(fn, "__name__", "wrapped")
        return wrapped

    for fn in strategies:
        w = float(weights.get(fn.__name__, 1.0))
        if w < min_conf:
            continue
        pairs.append((_instrument(fn), w))

    if not pairs:
        pairs.append((strategies[0], 1.0))

    from crypto_bot.signals.signal_fusion import SignalFusionEngine

    engine = SignalFusionEngine(pairs)
    return engine.fuse(df, cfg_dict)


def _bandit_context(
    df: pd.DataFrame, regime: str, symbol: Optional[str] = None
) -> Dict[str, float]:
    """Return bandit context features for Thompson sampling."""
    context: Dict[str, float] = {}
    for r in [
        "trending",
        "sideways",
        "mean-reverting",
        "breakout",
        "volatile",
        "unknown",
    ]:
        context[f"regime_{r}"] = 1.0 if regime == r else 0.0

    # Check if DataFrame is valid and has required columns
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        return context
    
    required_columns = ["close", "volume"]
    if not all(col in df.columns for col in required_columns):
        return context

    try:
        from crypto_bot.volatility_filter import calc_atr
        from crypto_bot.utils import stats
    except Exception:
        return context

    # Continue with context calculation if imports succeeded
    try:
        price = df["close"].iloc[-1]
        if symbol:
            try:
                from crypto_bot.utils.pyth import get_pyth_price

                pyth_price = get_pyth_price(symbol)
                if pyth_price:
                    price = pyth_price
            except Exception:
                pass

        atr = calc_atr(df)
        context["atr_pct"] = atr / price if price else 0.0

        ts = df.index[-1]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.to_datetime(ts)
        hour = ts.hour + ts.minute / 60
        context["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        context["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
        vol_z = stats.zscore(df["volume"], lookback=20)
        context["liquidity_z"] = float(vol_z.iloc[-1]) if hasattr(vol_z, 'empty') and not vol_z.empty else 0.0
    except Exception:
        # If any calculation fails, return basic context
        pass

    return context


def strategy_name(regime: str, mode: str) -> str:
    """Return the name of the strategy for given regime and mode."""
    if mode == "cex":
        return "trend" if regime == "trending" else "grid"
    if mode == "onchain":
        return "sniper" if regime in {"breakout", "volatile"} else "dex_scalper"
    if regime == "trending":
        return "trend"
    if regime == "scalp":
        return "micro_scalp"
    if regime in {"breakout", "volatile"}:
        return "sniper"
    return "grid"


def route(
    regime: Union[str, Dict[str, str]],
    mode: str,
    config: Optional[Union[RouterConfig, Mapping[str, Any]]] = None,
    notifier: Optional[TelegramNotifier] = None,
    df_map: Union[Mapping[str, pd.DataFrame], Optional[pd.DataFrame]] = None,
) -> Callable[[Union[pd.DataFrame, Mapping[str, pd.DataFrame]]], Tuple[float, str]]:
    """Select a strategy based on market regime and operating mode.

    Parameters
    ----------
    regime : str
        Current market regime as classified by indicators.
    mode : str
        Trading environment, either ``cex``, ``onchain`` or ``auto``.
    config : Optional[Union[RouterConfig, dict]]
        Optional configuration object. When ``meta_selector.enabled`` is
        ``True`` the strategy choice is delegated to the meta selector.
    notifier : Optional[TelegramNotifier]
        Optional notifier used to send a message when the strategy is called.

    df_map : Mapping[str, pd.DataFrame] | Optional[pd.DataFrame]
        Optional dataframe or mapping used for fast-path checks. When provided
        the router may immediately return a strategy without additional
        context.

    Returns
    -------
    Callable[[pd.DataFrame | Mapping[str, pd.DataFrame]], Tuple[float, str]]
        Strategy function returning a score and trade direction.
    """

    def _wrap(fn: Callable[[pd.DataFrame], Tuple[float, str]]):
        async def inner(df_input: Union[pd.DataFrame, Mapping[str, pd.DataFrame]], cfg=None):
            # Handle both DataFrame and dict of DataFrames
            if isinstance(df_input, dict):
                # Extract the appropriate DataFrame from the mapping
                if hasattr(cfg, 'timeframe'):
                    timeframe = cfg.timeframe
                elif isinstance(cfg, dict):
                    timeframe = cfg.get("timeframe", "1h")
                else:
                    timeframe = "1h"

                # Try to get the DataFrame for the specified timeframe
                df = df_input.get(timeframe)
                if df is None:
                    # Fallback to any available DataFrame
                    for tf in ['15m', '1h', '4h', '1d']:
                        df = df_input.get(tf)
                        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                            break
                if df is None or not isinstance(df, pd.DataFrame):
                    logger.warning(f"Strategy {fn.__name__} could not find valid DataFrame in mapping")
                    return 0.0, "none"
            else:
                df = df_input

            # Ensure df is actually a pandas DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Strategy {fn.__name__} received invalid data type: {type(df)}")
                return 0.0, "none"

            # Additional validation to catch corrupted DataFrames
            try:
                # Test basic DataFrame operations to ensure it's not corrupted
                if isinstance(df, pd.DataFrame):
                    _ = df.empty
                    _ = df.columns
                    _ = len(df)
                else:
                    logger.error(f"Strategy {fn.__name__} received non-DataFrame type: {type(df)}")
                    return 0.0, "none"
            except (AttributeError, TypeError) as e:
                logger.error(f"Strategy {fn.__name__} received corrupted DataFrame: {e}")
                return 0.0, "none"

            # Check if DataFrame is empty
            if isinstance(df, pd.DataFrame) and df.empty:
                logger.debug(f"Strategy {fn.__name__} received empty DataFrame")
                return 0.0, "none"

            # Additional safety check - ensure DataFrame still has required attributes
            if not hasattr(df, 'empty') or not hasattr(df, 'columns'):
                logger.error(f"Strategy {fn.__name__} DataFrame missing required attributes")
                return 0.0, "none"

            # Store original DataFrame type for validation after function call
            original_df_type = type(df)
            original_df_id = id(df)

            strategy_config = _build_strategy_config_object(fn, cfg)

            try:
                # Check if the function is async
                if asyncio.iscoroutinefunction(fn):
                    # Handle async functions properly
                    try:
                        res = await fn(df, strategy_config)
                    except TypeError:
                        res = await fn(df)
                else:
                    # Handle sync functions
                    res = fn(df, strategy_config)
                    # Validate that df is still a DataFrame after strategy execution
                    if not isinstance(df, pd.DataFrame):
                        logger.error(f"Strategy {fn.__name__} corrupted DataFrame - df type after call: {type(df)}")
                        return 0.0, "none"
            except TypeError:
                # Fallback for sync functions with different signatures
                if asyncio.iscoroutinefunction(fn):
                    res = await fn(df)
                else:
                    res = fn(df)
                    # Validate that df is still a DataFrame after strategy execution
                    if not isinstance(df, pd.DataFrame):
                        logger.error(f"Strategy {fn.__name__} corrupted DataFrame after TypeError fallback - df type: {type(df)}")
                        return 0.0, "none"
            except AttributeError as e:
                if "'dict' object has no attribute" in str(e):
                    logger.error(f"Strategy {fn.__name__} failed: DataFrame was converted to dict - {e}")
                    # Try to recover by checking if df is still a DataFrame
                    if not isinstance(df, pd.DataFrame):
                        logger.error(f"Strategy {fn.__name__} DataFrame was actually converted to {type(df)}")
                        logger.error(f"Strategy {fn.__name__} df type: {type(df)}, df value: {repr(df)[:200]}...")
                    # Also check if the error is coming from within the strategy function
                    import traceback
                    logger.error(f"Strategy {fn.__name__} traceback: {traceback.format_exc()}")
                    # Add debug info about the DataFrame state
                    logger.error(f"Strategy {fn.__name__} debug - original_df_type: {original_df_type}, original_df_id: {original_df_id}")
                    logger.error(f"Strategy {fn.__name__} debug - current_df_type: {type(df)}, current_df_id: {id(df)}")
                    return 0.0, "none"
                elif "bollinger_pband" in str(e):
                    logger.error(f"Strategy {fn.__name__} failed: Bollinger Bands method 'bollinger_pband' does not exist. Use 'bollinger_wband()' and calculate upper/lower bands manually.")
                    return 0.0, "none"
                else:
                    raise
            except Exception as e:
                # Catch any other exceptions that might indicate DataFrame corruption
                logger.error(f"Strategy {fn.__name__} failed with unexpected error: {e}")
                # Check if DataFrame was modified or corrupted during execution
                if not isinstance(df, pd.DataFrame):
                    logger.error(f"Strategy {fn.__name__} DataFrame was corrupted during execution: {type(df)}")
                    return 0.0, "none"
                if not hasattr(df, 'empty') or not hasattr(df, 'columns'):
                    logger.error(f"Strategy {fn.__name__} DataFrame lost required attributes during execution")
                    return 0.0, "none"
                raise

            # Validate that DataFrame is still intact after function call
            if not isinstance(df, pd.DataFrame):
                logger.error(f"Strategy {fn.__name__} DataFrame was converted to {type(df)} during execution")
                return 0.0, "none"
            
            if not hasattr(df, 'empty') or not hasattr(df, 'columns'):
                logger.error(f"Strategy {fn.__name__} DataFrame lost required attributes during execution")
                return 0.0, "none"

            if isinstance(res, tuple):
                score, direction = res[0], res[1]
            else:
                score, direction = res, "none"
            symbol = ""
            if isinstance(cfg, dict):
                symbol = cfg.get("symbol", "")
            if direction != "none" and symbol:
                await acquire_symbol_lock(symbol)
            if notifier is not None:
                notifier.notify(
                    f"\U0001f4c8 Signal: {symbol} \u2192 {direction.upper()} | Confidence: {score:.2f}"
                )
            return score, direction

        def wrapped(df_input: Union[pd.DataFrame, Mapping[str, pd.DataFrame]], cfg=None):
            # Handle both DataFrame and dict of DataFrames
            if isinstance(df_input, dict):
                # Extract the appropriate DataFrame from the mapping
                if hasattr(cfg, 'timeframe'):
                    timeframe = cfg.timeframe
                elif isinstance(cfg, dict):
                    timeframe = cfg.get("timeframe", "1h")
                else:
                    timeframe = "1h"
                
                # Try to get the DataFrame for the specified timeframe
                df = df_input.get(timeframe)
                if df is None:
                    # Fallback to any available DataFrame
                    for tf in ['15m', '1h', '4h', '1d']:
                        df = df_input.get(tf)
                        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                            break
                
                # If still no DataFrame found, return default
                if df is None or not isinstance(df, pd.DataFrame):
                    logger.warning(f"Strategy {fn.__name__} received dict but no valid DataFrame found")
                    return 0.0, "none"
            else:
                df = df_input
            
            # Ensure df is actually a pandas DataFrame
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Strategy {fn.__name__} received invalid data type: {type(df)}")
                return 0.0, "none"
            
            coro = inner(df, cfg)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            else:
                return coro

        wrapped.__name__ = fn.__name__
        return wrapped

    cfg = config or DEFAULT_ROUTER_CFG

    # === FAST-PATH FOR STRONG SIGNALS ===
    fp = (
        cfg.raw.get("strategy_router", {}).get("fast_path", {})
        if hasattr(cfg, "raw")
        else cfg.get("strategy_router", {}).get("fast_path", {})
    )
    
    # Extract DataFrame for fast-path logic
    if isinstance(df_map, dict):
        # Get the base timeframe DataFrame for fast-path checks
        base_tf = cfg.timeframe if hasattr(cfg, 'timeframe') else cfg.get("timeframe", "1h")
        df = df_map.get(base_tf)
        if df is None or not isinstance(df, pd.DataFrame):
            # Fallback to any available DataFrame
            for tf in ['15m', '1h', '4h', '1d']:
                df = df_map.get(tf)
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    break
        if df is None or not isinstance(df, pd.DataFrame):
            df = None
    else:
        df = df_map
        
    if df is not None and isinstance(df, pd.DataFrame):
        try:
            # 1) breakout squeeze detected by Bollinger band z-score and
            #    concurrent volume spike
            # Bollinger Bands calculation manually

            window = int(fp.get("breakout_squeeze_window", 15))
            bw_z_thr = float(fp.get("breakout_bandwidth_zscore", -0.84))
            vol_mult = float(fp.get("breakout_volume_multiplier", 4))
            max_bw = float(fp.get("breakout_max_bandwidth", 0.04))

            bb = BollingerBands(df["close"], window=window)
            wband_series = bb.bollinger_wband()
            wband = wband_series.iloc[-1]
            w_mean = wband_series.rolling(window).mean().iloc[-1]
            w_std = wband_series.rolling(window).std().iloc[-1]
            z = (wband - w_mean) / w_std if w_std > 0 else float("inf")
            vol_mean = df["volume"].rolling(window).mean().iloc[-1]
            if z < bw_z_thr and df["volume"].iloc[-1] > vol_mean * vol_mult:
                logger.info(
                    "FAST-PATH: breakout_bot via bandwidth z-score and volume spike"
                )
                return _wrap(breakout_bot.generate_signal)
            z_series = (
                wband_series - wband_series.rolling(window).mean()
            ) / wband_series.rolling(window).std()
            vol_ma = df["volume"].rolling(window).mean()

            if (
                z_series.iloc[-1] < -0.84
                and wband < max_bw
                and df["volume"].iloc[-1] > vol_ma.iloc[-1] * vol_mult
            ):
                logger.info(
                    "FAST-PATH: breakout_bot via BB squeeze z-score + volume spike"
                )
                return _wrap(breakout_bot.generate_signal)

            # 2) ultra-strong trend by ADX
            # Calculate ADX manually
            adx_thr = float(fp.get("trend_adx_threshold", 25))

            # Calculate ADX manually (simplified)
            high_diff = df["high"].diff()
            low_diff = df["low"].diff()

            dm_plus = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
            dm_minus = ((low_diff > high_diff) & (low_diff > 0)) * (-low_diff)

            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs()
            ], axis=1).max(axis=1)

            atr = tr.rolling(window=window).mean()
            di_plus = 100 * (dm_plus.rolling(window=window).mean() / atr)
            di_minus = 100 * (dm_minus.rolling(window=window).mean() / atr)
            dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus))
            adx_series = dx.rolling(window=window).mean()
            adx_val = adx_series.iloc[-1] if len(adx_series) > 0 and not pd.isna(adx_series.iloc[-1]) else 0
            if adx_val > adx_thr:
                logger.info("FAST-PATH: trend_bot via ADX > %.1f", adx_thr)
                return _wrap(trend_bot.generate_signal)
        except Exception:  # pragma: no cover - safety
            pass
    # === end fast-path ===

    if isinstance(regime, dict):
        if regime.get("1m") == "breakout" and regime.get("15m") == "trending":
            regime = "breakout"
        else:
            base = (
                cfg.timeframe if isinstance(cfg, RouterConfig) else cfg.get("timeframe")
            )
            regime = regime.get(base, next(iter(regime.values())))

    tf_sec = timeframe_seconds(
        None,
        cfg.timeframe if isinstance(cfg, RouterConfig) else cfg.get("timeframe", "1h"),
    )
    regime = commit_lock(
        regime,
        tf_sec,
        cfg_get(cfg, "commit_lock_intervals", 0),
    )



    # commit lock logic is now handled by the commit_lock() function above

    tf = cfg_get(cfg, "timeframe", "1h")
    tf_minutes = (
        cfg.timeframe_minutes
        if isinstance(cfg, RouterConfig)
        else getattr(cfg, "timeframe_minutes", int(pd.Timedelta(tf).total_seconds() // 60))
    )

    # Only update regime persistence if not in commit lock mode
    intervals = int(cfg_get(cfg, "commit_lock_intervals", 0))
    if intervals == 0:  # Only run regime persistence when commit lock is disabled
        LAST_REGIME_FILE.parent.mkdir(parents=True, exist_ok=True)
        last_data = {}
        if LAST_REGIME_FILE.exists():
            try:
                last_data = json.loads(LAST_REGIME_FILE.read_text())
            except Exception:
                last_data = {}
        last_ts = last_data.get("timestamp")
        last_regime = last_data.get("regime")
        if last_ts and last_regime:
            try:
                ts = datetime.fromisoformat(last_ts)
                if (datetime.utcnow() - ts).total_seconds() < tf_minutes * 60 * 3:
                    regime = last_regime
            except Exception:
                pass
        LAST_REGIME_FILE.write_text(
            json.dumps({"timestamp": datetime.utcnow().isoformat(), "regime": regime})
        )

    symbol = ""
    chain = ""
    if isinstance(cfg, RouterConfig):
        symbol = str(cfg.raw.get("symbol", ""))
        chain = str(cfg.raw.get("chain") or cfg.raw.get("preferred_chain", ""))
        grid_cfg = cfg.raw.get("grid_bot", {})
    elif isinstance(cfg, Mapping):
        symbol = str(cfg.get("symbol", ""))
        chain = str(cfg.get("chain") or cfg.get("preferred_chain", ""))
        grid_cfg = cfg.get("grid_bot", {})
    else:
        grid_cfg = {}
    if symbol.endswith("/USDC") and regime == "breakout":
        logger.info("Routing USDC breakout to Solana sniper bot")
        return _wrap(sniper_solana.generate_signal)

    if chain.lower().startswith("sol") and mode in {"auto", "onchain"} and regime in {"breakout", "volatile"}:
        logger.info("Routing %s regime to Solana sniper bot (%s mode)", regime, mode)
        return _wrap(sniper_solana.generate_signal)

    if regime == "sideways" and grid_cfg.get("dynamic_grid") and symbol:
        logger.info("Routing dynamic grid signal to micro scalp bot")
        return _wrap(micro_scalp_bot.generate_signal)

    # Thompson sampling router
    bandit_active = (
        cfg.bandit_enabled
        if isinstance(cfg, RouterConfig)
        else bool(cfg.get("bandit", {}).get("enabled"))
    )
    if bandit_active:
        strategies = get_strategies_for_regime(regime, cfg)
        if isinstance(cfg, RouterConfig):
            arms = list(cfg.regimes.get(regime, []))
        else:
            arms = list(
                cfg.get("strategy_router", {}).get("regimes", {}).get(regime, [])
            )
        arms = [a for a in arms if get_strategy_by_name(a)]
        if not arms:
            arms = [fn.__name__ for fn in strategies]
        symbol = ""
        if isinstance(cfg, RouterConfig):
            symbol = str(cfg.raw.get("symbol", ""))
        elif isinstance(cfg, Mapping):
            symbol = str(cfg.get("symbol", ""))
        context_df = df if df is not None else pd.DataFrame()
        context = _bandit_context(context_df, regime, symbol)
        choice = bandit.select(context, arms, symbol)
        fn = get_strategy_by_name(choice)
        if fn:
            logger.info("Bandit selected %s for %s", choice, regime)
            return _wrap(fn)

    if mode == "onchain":
        if chain.lower().startswith("sol"):
            if regime in {"breakout", "volatile"}:
                logger.info("Routing to Solana sniper bot (onchain)")
                return _wrap(sniper_solana.generate_signal)
            logger.info("Routing to DEX scalper (onchain)")
            return _wrap(dex_scalper.generate_signal)

        if regime in {"breakout", "volatile"}:
            logger.info("Routing to sniper bot (onchain)")
            return _wrap(sniper_bot.generate_signal)
        logger.info("Routing to DEX scalper (onchain)")
        return _wrap(dex_scalper.generate_signal)

    select_df = df if df is not None else pd.DataFrame()
    
    # Check if strategy rotation is enabled
    rotation_enabled = cfg_get(cfg, "strategy_rotation_enabled", True)
    
    if rotation_enabled:
        strategy_fn = _select_strategy_with_rotation(regime, select_df, cfg)
    else:
        strategy_fn = Selector(cfg).select(select_df, regime, mode, notifier)
    
    return _wrap(strategy_fn)

# Strategy rotation tracking
_strategy_usage_history = defaultdict(lambda: deque(maxlen=1000))
_strategy_last_used = defaultdict(lambda: datetime.min)
_strategy_min_usage_interval = defaultdict(lambda: timedelta(minutes=5))

def _update_strategy_usage(strategy_name: str, regime: str):
    """Update strategy usage tracking."""
    now = datetime.now()
    _strategy_usage_history[strategy_name].append(now)
    _strategy_last_used[strategy_name] = now

def _get_strategy_priority(strategy_name: str, regime: str, config: Union[RouterConfig, Mapping[str, Any]]) -> float:
    """Calculate strategy priority based on usage history and performance."""
    now = datetime.now()
    last_used = _strategy_last_used[strategy_name]
    time_since_last = now - last_used
    
    # Base priority from performance weights
    if isinstance(config, RouterConfig):
        weights = config.raw.get("strategy_weights", {})
    else:
        weights = config.get("strategy_weights", {})
    
    base_priority = weights.get(strategy_name, 1.0)
    
    # Boost priority for underused strategies
    usage_count = len(_strategy_usage_history[strategy_name])
    if usage_count < 10:  # New or underused strategy
        base_priority *= 2.0
    
    # Boost priority based on time since last use
    min_interval = _strategy_min_usage_interval[strategy_name]
    if time_since_last > min_interval * 2:  # Haven't been used in a while
        base_priority *= 1.5
    elif time_since_last < min_interval:  # Used recently
        base_priority *= 0.7
    
    return base_priority

def _select_strategy_with_rotation(
    regime: str, 
    df: pd.DataFrame, 
    config: Union[RouterConfig, Mapping[str, Any]]
) -> Callable[[pd.DataFrame], Tuple[float, str]]:
    """Select strategy with rotation to ensure balanced usage."""
    strategies = get_strategies_for_regime(regime, config)
    
    if not strategies:
        logger.warning(f"No strategies found for regime: {regime}")
        return lambda df: (0.0, "none")
    
    # Calculate priorities for all strategies
    strategy_priorities = []
    for strategy in strategies:
        priority = _get_strategy_priority(strategy.__name__, regime, config)
        strategy_priorities.append((strategy, priority))
    
    # Sort by priority (highest first)
    strategy_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Select strategy with weighted random choice
    total_priority = sum(priority for _, priority in strategy_priorities)
    if total_priority == 0:
        # Fallback to random selection
        selected_strategy = random.choice(strategies)
    else:
        # Weighted random selection
        rand_val = random.uniform(0, total_priority)
        cumulative = 0
        for strategy, priority in strategy_priorities:
            cumulative += priority
            if rand_val <= cumulative:
                selected_strategy = strategy
                break
        else:
            selected_strategy = strategy_priorities[-1][0]
    
    # Update usage tracking
    _update_strategy_usage(selected_strategy.__name__, regime)
    
    # Record strategy usage for performance tracking
    try:
        from crypto_bot.utils.strategy_performance_tracker import record_strategy_usage
        # We'll record actual usage when the strategy generates a signal
    except ImportError:
        pass
    
    logger.debug(
        f"Selected strategy {selected_strategy.__name__} for regime {regime} "
        f"with priority {_get_strategy_priority(selected_strategy.__name__, regime, config):.3f}"
    )

    return selected_strategy
