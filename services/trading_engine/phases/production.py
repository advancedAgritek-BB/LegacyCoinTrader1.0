"""Production trading engine phases built on top of service adapters."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from crypto_bot.services.interfaces import (
    CreateTradeRequest,
    ExchangeRequest,
    LoadSymbolsRequest,
    MultiTimeframeOHLCVRequest,
    RecordScannerMetricsRequest,
    RegimeCacheRequest,
    StrategyBatchRequest,
    StrategyEvaluationPayload,
    StrategyEvaluationResult,
    TokenDiscoveryRequest,
    TradeExecutionRequest,
)

logger = logging.getLogger(__name__)


def _unique(sequence: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in sequence:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_dataframe(
    cache: Mapping[str, MutableMapping[str, object]],
    symbol: str,
) -> pd.DataFrame | None:
    for timeframe_cache in cache.values():
        if not isinstance(timeframe_cache, Mapping):
            continue
        frame = timeframe_cache.get(symbol)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame
    return None


def _infer_regime(regime_cache: Mapping[str, MutableMapping[str, object]], symbol: str, default: str) -> str:
    for _, regime_map in regime_cache.items():
        if not isinstance(regime_map, Mapping):
            continue
        value = regime_map.get(symbol)
        if value is None:
            continue
        if isinstance(value, Mapping):
            regime = value.get("regime") or value.get("value")
            if regime:
                return str(regime)
        else:
            return str(value)
    return default


async def fetch_candidates(ctx) -> None:
    """Populate ``ctx.current_batch`` with candidate trading symbols."""

    phase_meta = ctx.metadata.setdefault("phases", {})
    config: Mapping[str, object] = ctx.config or {}
    manual: Sequence[str] = list(config.get("symbols", []) or [])  # type: ignore[arg-type]
    exclude: Sequence[str] = list(config.get("exclude_symbols", []) or [])  # type: ignore[arg-type]

    discovered: Sequence[str] = []
    market_data = getattr(ctx.services, "market_data", None)
    if market_data is not None:
        exchange_id = str(config.get("exchange", "kraken"))
        try:
            request = LoadSymbolsRequest(exchange_id=exchange_id, exclude=exclude, config=config)
            response = await market_data.load_symbols(request)
            discovered = response.symbols
        except Exception:  # pragma: no cover - best effort logging
            logger.exception("Failed to load symbols from market data service")
            discovered = []

    combined = _unique([*manual, *discovered])
    ctx.current_batch = combined
    phase_meta["fetch_candidates"] = {
        "manual": len(manual),
        "discovered": len(discovered),
        "total": len(combined),
    }
    logger.debug("Fetched %d candidate symbols", len(combined))


async def process_solana_candidates(ctx) -> None:
    """Incorporate Solana token discovery results into the candidate batch."""

    phase_meta = ctx.metadata.setdefault("phases", {})
    config: Mapping[str, object] = ctx.config or {}
    sol_cfg: Mapping[str, object] = config.get("solana_scanner", {}) or {}
    if not sol_cfg.get("enabled", False):
        phase_meta["process_solana_candidates"] = {"tokens": 0}
        return

    tokens: list[str] = []
    start = time.perf_counter()

    # Consume from feed if available
    feed = getattr(ctx, "solana_feed", None)
    if feed is not None:
        try:
            tokens.extend(await feed.fetch_tokens(limit=int(sol_cfg.get("max_tokens_per_scan", 20))))
        except Exception:  # pragma: no cover - optional feed failures
            logger.debug("Solana discovery feed fetch failed", exc_info=True)

    discovery = getattr(ctx.services, "token_discovery", None)
    if discovery is not None:
        try:
            request = TokenDiscoveryRequest(config=dict(sol_cfg))
            response = await discovery.discover_tokens(request)
            tokens.extend(response.tokens)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Token discovery adapter failed")

    elapsed = time.perf_counter() - start
    tokens = _unique(token for token in tokens if token)
    ctx.latest_solana_opportunities = tokens

    monitoring = getattr(ctx.services, "monitoring", None)
    if monitoring is not None and tokens:
        try:
            monitoring.record_scanner_metrics(
                RecordScannerMetricsRequest(tokens=len(tokens), latency=elapsed, config=dict(sol_cfg))
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to record scanner metrics", exc_info=True)

    solana_pairs = [token if "/" in token else f"{token}/USDC" for token in tokens]
    for pair in solana_pairs:
        if pair not in ctx.current_batch:
            ctx.current_batch.append(pair)

    phase_meta["process_solana_candidates"] = {
        "tokens": len(tokens),
        "batch": len(ctx.current_batch),
    }
    logger.debug("Processed %d Solana candidates", len(tokens))


async def update_caches(ctx) -> None:
    """Refresh OHLCV and regime caches for the current candidate batch."""

    phase_meta = ctx.metadata.setdefault("phases", {})
    if not ctx.current_batch:
        phase_meta["update_caches"] = {"symbols": 0, "timeframes": []}
        return

    config: Mapping[str, object] = ctx.config or {}
    exchange_id = str(config.get("exchange", "kraken"))
    limit = int(config.get("ohlcv_limit", 300))
    additional = config.get("additional_timeframes")
    additional_timeframes: Sequence[str] | None = None
    if isinstance(additional, Sequence):
        additional_timeframes = list(additional)

    market_data = getattr(ctx.services, "market_data", None)
    if market_data is None:
        phase_meta["update_caches"] = {"symbols": len(ctx.current_batch), "timeframes": []}
        return

    if not isinstance(ctx.df_cache, MutableMapping):
        ctx.df_cache = defaultdict(dict)
    if not isinstance(ctx.regime_cache, MutableMapping):
        ctx.regime_cache = defaultdict(dict)

    request = MultiTimeframeOHLCVRequest(
        exchange_id=exchange_id,
        cache=ctx.df_cache,
        symbols=ctx.current_batch,
        config=config,
        limit=limit,
        use_websocket=bool(config.get("use_ohlcv_websocket", False)),
        force_websocket_history=bool(config.get("force_websocket_history", False)),
        max_concurrent=config.get("max_concurrent_ohlcv"),
        additional_timeframes=additional_timeframes,
    )
    try:
        response = await market_data.update_multi_tf_cache(request)
        ctx.df_cache = response.cache
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to update OHLCV cache")

    regime_request = RegimeCacheRequest(
        exchange_id=exchange_id,
        cache=ctx.regime_cache,
        symbols=ctx.current_batch,
        config=config,
        limit=int(config.get("regime_limit", limit)),
        use_websocket=bool(config.get("use_regime_websocket", False)),
        force_websocket_history=bool(config.get("force_regime_websocket_history", False)),
        max_concurrent=config.get("max_concurrent_regime"),
        df_map=ctx.df_cache,
    )
    try:
        regime_response = await market_data.update_regime_cache(regime_request)
        ctx.regime_cache = regime_response.cache
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to update regime cache")

    timeframes = sorted(ctx.df_cache.keys()) if isinstance(ctx.df_cache, Mapping) else []
    phase_meta["update_caches"] = {
        "symbols": len(ctx.current_batch),
        "timeframes": list(timeframes),
    }
    logger.debug(
        "Updated caches for %d symbols across %s timeframes",
        len(ctx.current_batch),
        ",".join(timeframes) if timeframes else "no",
    )


async def analyse_batch(ctx) -> None:
    """Evaluate strategy payloads for the current candidate batch."""

    phase_meta = ctx.metadata.setdefault("phases", {})
    if not ctx.current_batch:
        ctx.analysis_results = []
        phase_meta["analyse_batch"] = {"payloads": 0, "results": 0, "errors": []}
        return

    config: Mapping[str, object] = ctx.config or {}
    mode = str(config.get("mode", "auto"))
    default_regime = str(config.get("default_regime", "neutral"))

    payloads: list[StrategyEvaluationPayload] = []
    for symbol in ctx.current_batch:
        frame_map: dict[str, object] = {}
        for timeframe, tf_cache in ctx.df_cache.items():
            if not isinstance(tf_cache, Mapping):
                continue
            df = tf_cache.get(symbol)
            if df is not None:
                frame_map[str(timeframe)] = df
        if not frame_map:
            continue
        regime = _infer_regime(ctx.regime_cache, symbol, default_regime)
        payloads.append(
            StrategyEvaluationPayload(
                symbol=symbol,
                regime=regime,
                mode=mode,
                timeframes=frame_map,
                config=config,
                metadata={"source": "trading-engine"},
            )
        )

    if not payloads:
        ctx.analysis_results = []
        phase_meta["analyse_batch"] = {"payloads": 0, "results": 0, "errors": []}
        return

    strategy_service = getattr(ctx.services, "strategy", None)
    if strategy_service is None:
        ctx.analysis_results = []
        phase_meta["analyse_batch"] = {"payloads": len(payloads), "results": 0, "errors": []}
        return

    try:
        response = await strategy_service.evaluate_batch(
            StrategyBatchRequest(items=tuple(payloads), metadata={"cycle": ctx.metadata})
        )
        ctx.analysis_results = list(response.results)
        errors = list(response.errors)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Strategy evaluation failed")
        ctx.analysis_results = []
        errors = ["strategy-evaluation-error"]

    phase_meta["analyse_batch"] = {
        "payloads": len(payloads),
        "results": len(ctx.analysis_results),
        "errors": errors,
    }
    logger.debug("Evaluated %d payloads (results=%d)", len(payloads), len(ctx.analysis_results))


async def execute_signals(ctx) -> None:
    """Execute actionable signals and record resulting trades."""

    phase_meta = ctx.metadata.setdefault("phases", {})
    results: Sequence[StrategyEvaluationResult] = ctx.analysis_results or []
    if not results:
        phase_meta["execute_signals"] = {"executed": 0}
        return

    executed: list[Dict[str, object]] = []
    config: Mapping[str, object] = ctx.config or {}
    exchange_id = str(config.get("exchange", ""))
    execution_mode = str(config.get("execution_mode", "dry_run"))
    dry_run = execution_mode.lower() != "live"
    execution_config = config.get(exchange_id, {}) if exchange_id else config

    if ctx.exchange is None:
        execution = getattr(ctx.services, "execution", None)
        if execution is not None:
            try:
                exchange_response = execution.create_exchange(ExchangeRequest(config=dict(execution_config)))
                ctx.exchange = exchange_response.exchange
                ctx.ws_client = exchange_response.ws_client
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Unable to initialize exchange adapter")
                ctx.exchange = None
                ctx.ws_client = None

    execution_adapter = getattr(ctx.services, "execution", None)
    portfolio = getattr(ctx.services, "portfolio", None)

    for result in results:
        direction = (result.direction or "").lower()
        if direction not in {"long", "short", "buy", "sell"}:
            continue
        df = _extract_dataframe(ctx.df_cache, result.symbol)
        if df is None:
            continue
        strategy = result.strategy or ""
        allowed, _reason = ctx.risk_manager.allow_trade(df, strategy)
        if not allowed:
            continue
        price = float(df["close"].iloc[-1])
        metadata = dict(result.metadata or {})
        confidence = float(metadata.get("confidence", result.score))
        atr = result.atr
        balance = float(getattr(ctx, "balance", 0.0))
        size = ctx.risk_manager.position_size(
            confidence,
            balance,
            df=df,
            atr=atr,
            price=price,
        )
        if size <= 0:
            continue
        if not ctx.risk_manager.can_allocate(strategy, size, balance):
            continue
        side = "buy" if direction in {"long", "buy"} else "sell"
        if execution_adapter is not None:
            try:
                exec_request = TradeExecutionRequest(
                    exchange=ctx.exchange,
                    ws_client=ctx.ws_client,
                    symbol=result.symbol,
                    side=side,
                    amount=float(size),
                    dry_run=dry_run,
                    use_websocket=bool(config.get("use_execution_websocket", False)),
                    config=dict(execution_config),
                    score=float(result.score),
                )
                exec_response = await execution_adapter.execute_trade(exec_request)
                order = dict(exec_response.order or {})
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Trade execution failed for %s", result.symbol)
                order = {}
        else:  # pragma: no cover - execution adapter missing
            order = {}

        ctx.risk_manager.allocate_capital(strategy, size)
        ctx.positions[result.symbol] = order or {
            "symbol": result.symbol,
            "side": side,
            "amount": float(size),
            "price": price,
        }
        if portfolio is not None:
            try:
                portfolio.create_trade(
                    CreateTradeRequest(
                        symbol=result.symbol,
                        side=side,
                        amount=float(size),
                        price=price,
                        strategy=strategy,
                        exchange=exchange_id,
                        metadata={"order": order, "confidence": confidence},
                    )
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to record trade for %s", result.symbol)

        executed.append(
            {
                "symbol": result.symbol,
                "side": side,
                "amount": float(size),
                "price": price,
                "strategy": strategy,
                "confidence": confidence,
            }
        )

    if executed:
        ctx.metadata["executed_trades"] = executed
    phase_meta["execute_signals"] = {"executed": len(executed)}
    logger.debug("Executed %d signals", len(executed))


async def handle_exits(ctx) -> None:
    """Perform exit management using the portfolio service."""

    phase_meta = ctx.metadata.setdefault("phases", {})
    portfolio = getattr(ctx.services, "portfolio", None)
    if portfolio is None:
        phase_meta["handle_exits"] = {"open_positions": 0, "risk_violations": []}
        return

    try:
        positions = list(portfolio.list_positions())
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to fetch open positions")
        positions = []

    try:
        risk_checks = list(portfolio.check_risk())
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Portfolio risk check failed", exc_info=True)
        risk_checks = []

    violations = [check.message for check in risk_checks if not getattr(check, "passed", True)]
    phase_meta["handle_exits"] = {
        "open_positions": len(positions),
        "risk_violations": violations,
    }
    logger.debug("Exit handling evaluated %d positions", len(positions))


async def monitor_positions_phase(ctx) -> None:
    """Record monitoring metadata for the current portfolio state."""

    phase_meta = ctx.metadata.setdefault("phases", {})
    portfolio = getattr(ctx.services, "portfolio", None)
    if portfolio is None:
        phase_meta["monitor_positions_phase"] = {"positions": 0}
        return

    try:
        positions = list(portfolio.list_positions())
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Failed to list positions for monitoring", exc_info=True)
        positions = []

    summary: Dict[str, object] = {"positions": len(positions)}
    try:
        pnl = portfolio.compute_pnl()
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Failed to compute PnL", exc_info=True)
        pnl = None
    if pnl is not None:
        summary["pnl"] = {
            "realized": float(getattr(pnl, "realized", 0.0)),
            "unrealized": float(getattr(pnl, "unrealized", 0.0)),
            "total": float(getattr(pnl, "total", 0.0)),
        }
    phase_meta["monitor_positions_phase"] = summary
    logger.debug("Monitoring summary: %s", summary)


PRODUCTION_PHASES = [
    fetch_candidates,
    process_solana_candidates,
    update_caches,
    analyse_batch,
    execute_signals,
    handle_exits,
    monitor_positions_phase,
]


__all__ = [
    "fetch_candidates",
    "process_solana_candidates",
    "update_caches",
    "analyse_batch",
    "execute_signals",
    "handle_exits",
    "monitor_positions_phase",
    "PRODUCTION_PHASES",
]
