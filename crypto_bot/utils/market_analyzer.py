import asyncio
import importlib
import functools
import pandas as pd
from typing import Dict, List, Optional

from .logger import LOG_DIR, setup_logger

from crypto_bot.regime.pattern_detector import detect_patterns
from crypto_bot.regime.regime_classifier import (
    classify_regime_async,
    classify_regime_cached,
)
from crypto_bot.strategy_router import RouterConfig, get_strategy_by_name
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.signals.signal_scoring import evaluate_async
from crypto_bot.strategy import grid_bot
from crypto_bot.volatility_filter import calc_atr
from ta.volatility import BollingerBands
from .stats import zscore
from crypto_bot.utils.telemetry import telemetry
from crypto_bot.services.interfaces import (
    StrategyBatchRequest,
    StrategyEvaluationPayload,
    StrategyEvaluationService,
)
from crypto_bot.services.strategy_evaluator import (
    evaluate_payload as evaluate_strategy_payload,
)


analysis_logger = setup_logger("strategy_rank", LOG_DIR / "strategy_rank.log")


async def analyze_symbol(
    symbol: str,
    df_map: Dict[str, pd.DataFrame],
    mode: str,
    config: Dict,
    notifier: Optional[TelegramNotifier] = None,
    strategy_service: Optional[StrategyEvaluationService] = None,
) -> Dict:
    """Classify the market regime and evaluate the trading signal for ``symbol``.

    Parameters
    ----------
    symbol : str
        Trading pair to analyze.
    df_map : Dict[str, pd.DataFrame]
        Mapping of timeframe to OHLCV data.
    mode : str
        Execution mode of the bot ("cex", "onchain" or "auto").
    config : Dict
        Bot configuration.
    notifier : Optional[TelegramNotifier]
        Optional notifier used to send a message when the strategy is invoked.
    strategy_service : Optional[StrategyEvaluationService]
        Remote strategy evaluation service. When provided the evaluation
        portion of the analysis is delegated to this service.
    """
    router_cfg = RouterConfig.from_dict(config)
    lookback = config.get("indicator_lookback", 10)  # Reduced from 14*2=28 to just 10
    for tf, df in df_map.items():
        if df is None or len(df) < lookback:
            analysis_logger.info(
                "Skipping analysis for %s on %s: insufficient data (%d candles)",
                symbol,
                tf,
                0 if df is None else len(df),
            )
            telemetry.inc("analysis.skipped_insufficient_history")
            return {"symbol": symbol, "skip": "insufficient_history"}
    base_tf = router_cfg.timeframe
    higher_tf = config.get("higher_timeframe", "1d")
    df = df_map.get(base_tf)
    if df is None:
        telemetry.inc("analysis.skipped_no_df")
        return {"symbol": symbol, "skip": "no_ohlcv"}

    # Check if df is actually a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        telemetry.inc("analysis.skipped_invalid_df")
        analysis_logger.warning("Skipping %s: invalid data type for %s (%s)", symbol, base_tf, type(df))
        return {"symbol": symbol, "skip": "invalid_df_type"}

    if df.empty:
        telemetry.inc("analysis.skipped_no_df")
        analysis_logger.info("Skipping %s: no data for %s", symbol, base_tf)
        return {"symbol": symbol, "skip": "empty_df"}
    baseline = float(
        config.get("min_confidence_score", config.get("signal_threshold", 0.0))
    )
    bb_z = 0.0
    if df is not None and len(df) >= 14:
        try:
            # Use the project's local Bollinger Bands implementation
            # This provides consistent calculation across the entire system
            bb = BollingerBands(df["close"].values, window=14, ndev=2)
            bb_width = bb.bollinger_wband()
            # Convert numpy array to pandas Series for zscore calculation
            bb_width_series = pd.Series(bb_width, index=df.index)
            bb_z_series = zscore(bb_width_series, 14)
            if not bb_z_series.empty and not bb_z_series.isna().iloc[-1]:
                bb_z = float(bb_z_series.iloc[-1])
        except Exception as e:
            analysis_logger.warning(f"Bollinger Bands calculation failed: {e}")
            bb_z = 0.0
    min_conf_adaptive = baseline * (1 + bb_z / 3)
    higher_df = df_map.get("1d")
    regime, probs = await classify_regime_async(df, higher_df)
    sub_regime = regime
    regime = regime.split("_")[-1]
    patterns = detect_patterns(df)
    
    # Handle different return types from classify_regime_async
    if isinstance(probs, dict):
        base_conf = float(probs.get(sub_regime, 0.0))
    elif isinstance(probs, set):
        # Convert set to dict with default confidence
        base_conf = 0.5 if sub_regime in probs else 0.0
        probs = {k: 0.5 for k in probs}
    else:
        # Fallback for other types
        base_conf = float(probs) if isinstance(probs, (int, float)) else 0.0
        probs = {sub_regime: base_conf}
    bias_cfg = config.get("sentiment_filter", {})
    try:
        from crypto_bot.sentiment_filter import boost_factor

        bias = boost_factor(bias_cfg.get("bull_fng", 50), bias_cfg.get("bull_sentiment", 50))
    except Exception:
        bias = 1.0
    if bias > 1:
        for k in list(probs.keys()):
            if k.startswith("bullish"):
                probs[k] *= bias
        total = sum(probs.values())
        if total > 0:
            probs = {kk: vv / total for kk, vv in probs.items()}
    profile = bool(config.get("profile_regime", False))
    regime, _ = await classify_regime_cached(
        symbol,
        base_tf,
        df,
        higher_df,
        profile,
    )
    regime = regime.split("_")[-1]
    higher_df = df_map.get(higher_tf)

    if df is not None:
        regime_tmp, info = await classify_regime_async(df, higher_df)
        sub_regime = regime_tmp
        regime = regime_tmp.split("_")[-1]
        if isinstance(info, dict):
            patterns = info
        elif isinstance(info, set):
            patterns = {p: 1.0 for p in info}
        else:
            base_conf = float(info)

    regime_counts: Dict[str, int] = {}
    regime_tfs = config.get("regime_timeframes", [base_tf])
    min_agree = config.get("min_consistent_agreement", 1)

    vote_map: Dict[str, pd.DataFrame] = {}
    for tf in regime_tfs:
        tf_df = df_map.get(tf)
        if tf_df is None:
            continue
        higher_df = df_map.get("1d") if tf != "1d" else None
        r, _ = await classify_regime_cached(
            symbol,
            tf,
            tf_df,
            higher_df,
            profile,
        )
        r = r.split("_")[-1]
        # Exclude 'unknown' from voting to avoid unknown dominating outcomes
        if r != "unknown":
            regime_counts[r] = regime_counts.get(r, 0) + 1
        if tf_df is not None:
            vote_map[tf] = tf_df
    if higher_tf in df_map:
        vote_map.setdefault(higher_tf, df_map[higher_tf])

    if vote_map:
        labels = await classify_regime_async(df_map=vote_map)
        if isinstance(labels, tuple):
            label_map = dict(zip(vote_map.keys(), labels))
        else:
            label_map = labels
        for tf in regime_tfs:
            r = label_map.get(tf)
            if r:
                r = r.split("_")[-1]
                if r != "unknown":
                    regime_counts[r] = regime_counts.get(r, 0) + 1

    # Seed base timeframe regime to avoid empty-vote unknowns when voting data is sparse
    base_tf_label = None
    try:
        base_tf_label, _ = await classify_regime_cached(
            symbol, base_tf, df, df_map.get(higher_tf), profile
        )
        if base_tf_label:
            base_tf_label = base_tf_label.split("_")[-1]
            if base_tf_label != "unknown":
                regime_counts[base_tf_label] = regime_counts.get(base_tf_label, 0) + 1
    except Exception:
        pass

    if regime_counts:
        regime, votes = max(regime_counts.items(), key=lambda kv: kv[1])
    else:
        regime, votes = "unknown", 0

    denom = len(regime_tfs)
    if vote_map:
        denom *= 2
    confidence = votes / max(denom, 1)
    confidence *= base_conf
    if votes < min_agree:
        regime = "unknown"

    analysis_logger.info(
        "%s regime=%s conf=%.2f votes=%d",
        symbol,
        regime,
        confidence,
        votes,
    )

    period = int(config.get("regime_return_period", 5))
    future_return = 0.0
    if len(df) > period:
        start = df["close"].iloc[-period - 1]
        end = df["close"].iloc[-1]
        future_return = (end - start) / start * 100

    result = {
        "symbol": symbol,
        "df": df,
        "regime": regime,
        "sub_regime": sub_regime,
        "patterns": patterns,
        "future_return": future_return,
        "confidence": confidence,
        "min_confidence": min_conf_adaptive,
        "probabilities": probs,
    }

    if regime != "unknown":
        env = mode if mode != "auto" else "cex"
        cfg = {**config, "symbol": symbol}
        evaluation_payload = StrategyEvaluationPayload(
            symbol=symbol,
            regime=regime,
            mode=env,
            timeframes=df_map,
            config=cfg,
            metadata={"evaluation_mode": cfg.get("strategy_evaluation_mode", "mapped")},
        )

        evaluation_result = None
        evaluation_errors: List[str] = []

        if strategy_service is not None:
            try:
                batch_response = await strategy_service.evaluate_batch(
                    StrategyBatchRequest(items=[evaluation_payload])
                )
                if batch_response.results:
                    evaluation_result = batch_response.results[0]
                if batch_response.errors:
                    evaluation_errors.extend(batch_response.errors)
            except Exception as exc:  # pragma: no cover - network failures
                analysis_logger.warning(
                    "Remote strategy evaluation failed for %s: %s", symbol, exc
                )
                evaluation_errors.append(str(exc))

        if evaluation_result is None:
            evaluation_result = await evaluate_strategy_payload(
                evaluation_payload, notifier=notifier
            )

        score = float(evaluation_result.score)
        direction = evaluation_result.direction
        atr = evaluation_result.atr
        if atr is None and direction != "none" and {"high", "low", "close"}.issubset(df.columns):
            atr = calc_atr(df)

        weights = config.get("scoring_weights", {})
        final = (
            weights.get("strategy_score", 1.0) * score
            + weights.get("regime_confidence", 0.0) * confidence
            + weights.get("volume_score", 0.0) * 1.0
            + weights.get("symbol_score", 0.0) * 1.0
            + weights.get("spread_penalty", 0.0) * 0.0
            + weights.get("strategy_regime_strength", 0.0) * 1.0
        )

        result.update(
            {
                "env": env,
                "name": evaluation_result.strategy,
                "score": final,
                "raw_score": score,
                "direction": direction,
                "atr": atr,
                "ranked_signals": [
                    {
                        "strategy": rs.strategy,
                        "score": rs.score,
                        "direction": rs.direction,
                    }
                    for rs in evaluation_result.ranked_signals
                ],
                "evaluation_metadata": dict(evaluation_result.metadata),
                "evaluation_cached": evaluation_result.cached,
            }
        )

        if evaluation_result.fused_score is not None:
            result["fused_signal"] = {
                "score": evaluation_result.fused_score,
                "direction": evaluation_result.fused_direction or "none",
            }

        if evaluation_errors:
            result.setdefault("evaluation_errors", evaluation_errors)

        telemetry.inc("analysis.evaluated")
        if direction == "none":
            telemetry.inc("analysis.direction_none")

        def wrap_for_voting(fn):
            if fn is grid_bot.generate_signal:
                return functools.partial(fn, higher_df=df_map.get("1h"))

            def wrapped(df_input, config=None):
                df_local = df_input
                if isinstance(df_local, dict):
                    df_local = df_local.get(router_cfg.timeframe, df)
                if not isinstance(df_local, pd.DataFrame) or df_local.empty:
                    return 0.0, "none"
                try:
                    if config is not None:
                        return fn(df_local, config)
                    return fn(df_local)
                except TypeError:
                    return fn(df_local)

            wrapped.__name__ = getattr(fn, "__name__", "wrapped")
            return wrapped

        votes = []
        voting = config.get("voting_strategies", [])
        if isinstance(voting, list):
            for strat_name in voting:
                fn = get_strategy_by_name(strat_name)
                if fn is None:
                    continue
                fn = wrap_for_voting(fn)
                try:
                    dir_vote = (await evaluate_async([fn], df, cfg))[0][1]
                except Exception:  # pragma: no cover - safety
                    continue
                votes.append(dir_vote)

        if votes:
            counts = {}
            for d in votes:
                counts[d] = counts.get(d, 0) + 1
            best_dir, n = max(counts.items(), key=lambda kv: kv[1])
            min_votes = int(config.get("min_agreeing_votes", 1))
            if n >= min_votes:
                result["direction"] = best_dir
            else:
                result["direction"] = "none"
    return result
