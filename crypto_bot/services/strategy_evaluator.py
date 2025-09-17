from __future__ import annotations

"""Shared strategy evaluation helpers for local and remote execution."""

import asyncio
import functools
import importlib
import logging
from datetime import datetime
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd

from crypto_bot import meta_selector
from crypto_bot.services.interfaces import (
    RankedSignal,
    StrategyBatchRequest,
    StrategyBatchResponse,
    StrategyEvaluationPayload,
    StrategyEvaluationResult,
)
from crypto_bot.signals.signal_scoring import evaluate_async, evaluate_strategies
from crypto_bot.strategy import grid_bot
from crypto_bot.strategy_router import (
    RouterConfig,
    evaluate_regime,
    get_strategy_by_name,
    get_strategies_for_regime,
    route,
    strategy_for,
    strategy_name,
)
from crypto_bot.utils import perf
from crypto_bot.utils.strategy_utils import compute_strategy_weights
from crypto_bot.utils.telemetry import telemetry
from crypto_bot.volatility_filter import calc_atr

logger = logging.getLogger(__name__)


def _fn_name(fn: callable) -> str:
    """Return the underlying function name even for functools.partial."""

    if isinstance(fn, functools.partial):
        return getattr(fn.func, "__name__", str(fn))
    return getattr(fn, "__name__", str(fn))


async def _run_candidates(
    df: pd.DataFrame,
    strategies: Iterable,
    symbol: str,
    cfg: Mapping[str, Any],
    regime: Optional[str] = None,
) -> List[tuple[callable, float, str]]:
    """Evaluate ``strategies`` and rank them by score times edge."""

    strategy_list = list(strategies)
    if not strategy_list or df is None or df.empty:
        return []

    weights = compute_strategy_weights()
    try:
        evals = await evaluate_async(
            strategy_list,
            df,
            cfg,
            max_parallel=cfg.get("max_parallel", 4),
        )
    except Exception as exc:  # pragma: no cover - safety
        logger.warning("Batch evaluation failed: %s", exc)
        return []

    results: List[tuple[float, callable, float, str]] = []
    for strat, (score, direction, _atr) in zip(strategy_list, evals):
        name = _fn_name(strat)
        try:
            edge = perf.edge(name, symbol, cfg.get("drawdown_penalty_coef", 0.0))
        except Exception:  # pragma: no cover - perf failure fallback
            edge = 1.0
        weight = weights.get(name, 1.0)
        rank = score * edge * weight
        results.append((rank, strat, score, direction))

    results.sort(key=lambda x: x[0], reverse=True)
    ranked = [(s, sc, d) for (_rank, s, sc, d) in results]

    if regime is not None and ranked:
        scores = [sc for _fn, sc, _d in ranked]
        dirs = [d for _fn, _sc, d in ranked]
        if (
            len(set(scores)) == 1
            or all(s == 0.0 for s in scores)
            or all(d == "none" for d in dirs)
        ):
            for idx, (fn, _sc, _d) in enumerate(ranked):
                reg_filter = getattr(fn, "regime_filter", None)
                if reg_filter is None:
                    try:
                        module = importlib.import_module(fn.__module__)
                        reg_filter = getattr(module, "regime_filter", None)
                    except Exception:  # pragma: no cover - safety
                        reg_filter = None
                try:
                    if (
                        reg_filter
                        and hasattr(reg_filter, "matches")
                        and reg_filter.matches(regime)
                    ):
                        ranked.insert(0, ranked.pop(idx))
                        break
                except Exception:  # pragma: no cover - safety
                    pass

    return ranked


def _select_dataframe(
    df_map: Mapping[str, pd.DataFrame],
    primary_tf: str,
    fallbacks: Sequence[str] | None = None,
) -> Optional[pd.DataFrame]:
    """Return the best available DataFrame for evaluation."""

    df = df_map.get(primary_tf)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df

    for tf in fallbacks or ("15m", "1h", "4h", "1d"):
        alt = df_map.get(tf)
        if isinstance(alt, pd.DataFrame) and not alt.empty:
            return alt
    return None


def _wrap_strategy(
    fn: callable,
    base_tf: str,
    df_map: Mapping[str, pd.DataFrame],
) -> callable:
    """Wrap a strategy callable with data-frame validation helpers."""

    if fn is grid_bot.generate_signal:
        higher_df = df_map.get("1h")

        def wrapped(df_input, config=None):
            df = df_input
            if isinstance(df, MutableMapping):
                df = df.get(base_tf)
            if not isinstance(df, pd.DataFrame) or df.empty:
                return 0.0, "none"
            try:
                if config is not None:
                    return fn(df, config, higher_df=higher_df)
                return fn(df, higher_df=higher_df)
            except TypeError:
                return fn(df, higher_df=higher_df)

        wrapped.__name__ = getattr(fn, "__name__", "wrapped")
        return wrapped

    def wrapped(df_input, config=None):
        df = df_input
        if isinstance(df, MutableMapping):
            df = df.get(base_tf)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return 0.0, "none"
        try:
            if config is not None:
                return fn(df, config)
            return fn(df)
        except TypeError:
            return fn(df)

    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapped


async def evaluate_payload(
    payload: StrategyEvaluationPayload,
    *,
    notifier: Optional[object] = None,
) -> StrategyEvaluationResult:
    """Evaluate strategies for ``payload`` using router logic."""

    config = dict(payload.config or {})
    router_cfg = RouterConfig.from_dict(config)
    base_tf = router_cfg.timeframe
    df_map = {
        tf: df for tf, df in payload.timeframes.items() if isinstance(df, pd.DataFrame)
    }

    df = _select_dataframe(df_map, base_tf)
    if df is None or df.empty:
        logger.debug(
            "No OHLCV data available for %s on timeframe %s", payload.symbol, base_tf
        )
        return StrategyEvaluationResult(
            symbol=payload.symbol,
            regime=payload.regime,
            strategy=strategy_name(payload.regime, payload.mode if payload.mode != "auto" else "cex"),
            score=0.0,
            direction="none",
            ranked_signals=tuple(),
            metadata={"reason": "missing_data", **payload.metadata},
        )

    cfg = {**config, "symbol": payload.symbol}
    evaluation_mode = str(config.get("strategy_evaluation_mode", "mapped"))
    env = payload.mode if payload.mode != "auto" else "cex"
    strategy_name_value = strategy_name(payload.regime, env)
    score = 0.0
    direction = "none"
    atr_value: Optional[float] = None

    wrapped_strategies = [
        _wrap_strategy(fn, base_tf, df_map) for fn in get_strategies_for_regime(payload.regime, router_cfg)
    ]

    ranked_pairs: List[tuple[callable, float, str]] = []

    if evaluation_mode == "best" and wrapped_strategies:
        res = evaluate_strategies(wrapped_strategies, df, cfg)
        strategy_name_value = res.get("name", strategy_name_value)
        score = float(res.get("score", 0.0))
        direction = res.get("direction", "none")
        ranked_pairs = await _run_candidates(
            df,
            wrapped_strategies,
            payload.symbol,
            cfg,
            payload.regime,
        )
    elif evaluation_mode == "ensemble":
        min_conf = float(config.get("ensemble_min_conf", 0.15))
        candidates = [_wrap_strategy(strategy_for(payload.regime, router_cfg), base_tf, df_map)]
        extra = meta_selector._scores_for(payload.regime)
        for strat_name, val in extra.items():
            if val >= min_conf:
                fn = get_strategy_by_name(strat_name)
                if fn:
                    wrapped = _wrap_strategy(fn, base_tf, df_map)
                    if wrapped not in candidates:
                        candidates.append(wrapped)
        ranked_pairs = await _run_candidates(
            df,
            candidates,
            payload.symbol,
            cfg,
            payload.regime,
        )
        if ranked_pairs:
            best_fn, raw_score, raw_dir = ranked_pairs[0]
            strategy_name_value = _fn_name(best_fn)
            score = raw_score
            direction = raw_dir if raw_score >= min_conf else "none"
    else:
        strategy_callable = _wrap_strategy(
            route(payload.regime, env, router_cfg, notifier, df_map=df_map),
            base_tf,
            df_map,
        )
        df_for_strategy = _select_dataframe(df_map, base_tf)
        if df_for_strategy is None:
            score, direction = 0.0, "none"
        else:
            score, direction, _atr = (
                await evaluate_async([strategy_callable], df_for_strategy, cfg)
            )[0]
        ranked_pairs = await _run_candidates(
            df,
            wrapped_strategies or [strategy_callable],
            payload.symbol,
            cfg,
            payload.regime,
        )

    atr_period = int(config.get("risk", {}).get("atr_period", 14))
    if direction != "none":
        atr_value = calc_atr(df, window=atr_period)

    fused_score: Optional[float] = None
    fused_direction: Optional[str] = None
    try:
        fused_score, fused_direction = evaluate_regime(payload.regime, df, router_cfg)
    except Exception as exc:  # pragma: no cover - safety
        logger.debug("Signal fusion failed for %s: %s", payload.symbol, exc)

    telemetry.inc("strategy_engine.evaluated")
    if direction == "none":
        telemetry.inc("strategy_engine.direction_none")

    ranked_signals = tuple(
        RankedSignal(strategy=_fn_name(fn), score=float(sc), direction=dirn)
        for fn, sc, dirn in ranked_pairs
    )

    metadata = dict(payload.metadata)
    metadata.update(
        {
            "evaluation_mode": evaluation_mode,
            "evaluated_at": datetime.utcnow().isoformat(),
        }
    )

    return StrategyEvaluationResult(
        symbol=payload.symbol,
        regime=payload.regime,
        strategy=strategy_name_value,
        score=float(score),
        direction=direction,
        atr=atr_value,
        fused_score=fused_score,
        fused_direction=fused_direction,
        ranked_signals=ranked_signals,
        metadata=metadata,
    )


async def evaluate_batch_request(
    request: StrategyBatchRequest,
    *,
    notifier: Optional[object] = None,
) -> StrategyBatchResponse:
    """Evaluate a batch of strategy payloads."""

    tasks = [
        evaluate_payload(item, notifier=notifier) for item in request.items
    ]

    results: List[StrategyEvaluationResult] = []
    errors: List[str] = []

    for coro in asyncio.as_completed(tasks):
        try:
            res = await coro
            results.append(res)
        except Exception as exc:  # pragma: no cover - best effort
            logger.exception("Strategy evaluation failed: %s", exc)
            errors.append(str(exc))

    return StrategyBatchResponse(results=tuple(results), errors=tuple(errors))


__all__ = [
    "evaluate_payload",
    "evaluate_batch_request",
]
