"""Default phases executed by the trading engine service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, Mapping, MutableMapping, Optional

import pandas as pd

from libs.services.interfaces import (
    CacheUpdateResponse,
    MultiTimeframeOHLCVRequest,
    RegimeCacheRequest,
    StrategyBatchRequest,
    StrategyBatchResponse,
    StrategyEvaluationPayload,
    StrategyEvaluationResult,
    TokenDiscoveryRequest,
    TokenDiscoveryResponse,
    TradeExecutionRequest,
)

logger = logging.getLogger(__name__)


def _unique_symbols(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in symbols:
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return ordered


def _select_dataframe(df_cache: Mapping[str, Mapping[str, pd.DataFrame]], symbol: str) -> Optional[pd.DataFrame]:
    for timeframe, entries in df_cache.items():
        frame = entries.get(symbol)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            logger.debug("Using %s timeframe for %s", timeframe, symbol)
            return frame
    return None


def _infer_regime(regime_cache: Mapping[str, Mapping[str, object]], symbol: str, default: str) -> str:
    for entries in regime_cache.values():
        data = entries.get(symbol)
        if isinstance(data, Mapping):
            regime = data.get("regime") or data.get("label")
            if isinstance(regime, str) and regime:
                return regime
    return default


async def prepare_cycle(context) -> None:
    """Initialise cycle metadata and refresh cached positions."""

    context.metadata.setdefault("events", []).append("prepare")
    context.metadata["cycle_started_at"] = datetime.now(timezone.utc).isoformat()
    context.metadata.setdefault("cycle_sequence", context.state.get("cycles", 0))

    if getattr(context, "trade_manager", None) is not None:
        try:
            context.sync_positions_from_trade_manager()
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Position sync failed", exc_info=True)

    open_positions = sorted(getattr(context, "positions", {}).keys())
    context.metadata["open_positions"] = open_positions


async def discover_markets(context) -> None:
    """Build the symbol batch using static configuration and discovery results."""

    config = getattr(context, "config", {}) or {}
    static_symbols = list(config.get("symbols", []))
    discovery_cfg = (
        config.get("enhanced_scanning")
        or config.get("token_discovery")
        or {}
    )

    discovered: list[str] = []
    discovery = getattr(getattr(context, "services", None), "token_discovery", None)
    if discovery is not None:
        try:
            request = TokenDiscoveryRequest(config=discovery_cfg)
            response: TokenDiscoveryResponse = await discovery.discover_tokens(request)
            discovered = list(response.tokens or [])
        except Exception:  # pragma: no cover - discovery is optional
            logger.debug("Token discovery failed", exc_info=True)

    combined = _unique_symbols(static_symbols + discovered)
    batch_size = int(config.get("symbol_batch_size") or len(combined) or 0)
    if batch_size > 0:
        combined = combined[:batch_size]

    context.current_batch = combined
    context.metadata["symbol_batch"] = combined


async def load_market_data(context) -> None:
    """Populate OHLCV and regime caches for the current batch."""

    if not getattr(context, "current_batch", None):
        logger.debug("No symbols selected for market data update")
        return

    services = getattr(context, "services", None)
    market_data = getattr(services, "market_data", None)
    if market_data is None:
        logger.debug("Market data service unavailable; skipping cache refresh")
        return

    config = getattr(context, "config", {}) or {}
    exchange_id = str(config.get("exchange", "kraken"))
    market_cfg = config.get("market_data", {})
    primary_timeframe = market_cfg.get("timeframe", config.get("timeframe", "1h"))
    extra_timeframes = market_cfg.get("timeframes") or config.get("additional_timeframes", [])
    limit = int(market_cfg.get("limit", config.get("ohlcv_limit", 250)))

    request = MultiTimeframeOHLCVRequest(
        exchange_id=exchange_id,
        cache=context.df_cache,
        symbols=context.current_batch,
        config={
            "timeframe": primary_timeframe,
            "timeframes": [primary_timeframe, *extra_timeframes],
        },
        limit=limit,
        use_websocket=bool(config.get("use_websocket")),
        force_websocket_history=bool(config.get("force_websocket_history")),
        max_concurrent=config.get("max_concurrent_ohlcv"),
        notifier=getattr(context, "notifier", None),
    )

    try:
        response: CacheUpdateResponse = await market_data.update_multi_tf_cache(request)
        context.df_cache = response.cache
    except Exception:
        logger.exception("Failed to update OHLCV cache")

    regime_timeframes = config.get("regime_timeframes")
    if regime_timeframes:
        regime_request = RegimeCacheRequest(
            exchange_id=exchange_id,
            cache=context.regime_cache,
            symbols=context.current_batch,
            config={"timeframes": list(regime_timeframes)},
            limit=limit,
            use_websocket=bool(config.get("use_websocket")),
            force_websocket_history=bool(config.get("force_websocket_history")),
            max_concurrent=config.get("max_concurrent_ohlcv"),
            notifier=getattr(context, "notifier", None),
            df_map=context.df_cache,
        )
        try:
            regime_response: CacheUpdateResponse = await market_data.update_regime_cache(regime_request)
            context.regime_cache = regime_response.cache
        except Exception:  # pragma: no cover - advisory
            logger.debug("Regime cache update failed", exc_info=True)

    context.metadata["market_data_updated"] = len(context.current_batch)


async def evaluate_signals(context) -> None:
    """Invoke the strategy service to generate trading signals."""

    services = getattr(context, "services", None)
    strategy_service = getattr(services, "strategy", None)
    if strategy_service is None:
        logger.debug("Strategy service unavailable; no signal evaluation performed")
        context.analysis_results = []
        return

    config = getattr(context, "config", {}) or {}
    default_regime = config.get("default_regime", "neutral")
    payloads: list[StrategyEvaluationPayload] = []

    for symbol in context.current_batch:
        df = _select_dataframe(context.df_cache, symbol)
        if df is None:
            continue
        regime = _infer_regime(context.regime_cache, symbol, default_regime)
        timeframes: MutableMapping[str, pd.DataFrame] = {}
        for timeframe, entries in context.df_cache.items():
            frame = entries.get(symbol)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                timeframes[timeframe] = frame
        if not timeframes:
            continue
        payloads.append(
            StrategyEvaluationPayload(
                symbol=symbol,
                regime=regime,
                mode=str(config.get("mode", "auto")),
                timeframes=timeframes,
                config=config.get("strategy_router", config.get("strategy", {})),
                metadata={"cycle": context.metadata.get("cycle_sequence")},
            )
        )

    if not payloads:
        context.analysis_results = []
        context.metadata["analysis_results"] = 0
        return

    request = StrategyBatchRequest(items=tuple(payloads), metadata={"cycle": context.metadata.get("cycle_sequence")})
    try:
        response: StrategyBatchResponse = await strategy_service.evaluate_batch(request)
    except Exception:
        logger.exception("Strategy evaluation failed")
        context.analysis_results = []
        context.metadata["analysis_results"] = 0
        return

    results = list(response.results or [])
    context.analysis_results = results
    context.metadata["analysis_results"] = len(results)
    if response.errors:
        context.metadata["analysis_errors"] = list(response.errors)


async def execute_signals(context) -> None:
    """Run risk checks and dispatch executable orders."""

    results: Iterable[StrategyEvaluationResult] = getattr(context, "analysis_results", []) or []
    if not results:
        context.metadata["executed_trades"] = []
        return

    services = getattr(context, "services", None)
    execution = getattr(services, "execution", None)
    if execution is None:
        logger.debug("Execution service unavailable; no trades placed")
        context.metadata["executed_trades"] = []
        return

    risk_manager = getattr(context, "risk_manager", None)
    position_guard = getattr(context, "position_guard", None)
    balance = float(getattr(context, "balance", 0.0))
    execution_mode = str((getattr(context, "config", {}) or {}).get("execution_mode", "dry_run"))
    dry_run = execution_mode.lower() != "live"
    executed: list[dict[str, object]] = []

    for result in results:
        direction = (result.direction or "").lower()
        if direction not in {"long", "short"}:
            continue

        df = _select_dataframe(context.df_cache, result.symbol)
        if df is None or df.empty:
            continue
        price = float(df["close"].iloc[-1])

        if position_guard is not None:
            try:
                can_open = await position_guard.can_open(getattr(context, "positions", {}))
            except Exception:  # pragma: no cover - defensive
                logger.debug("Position guard check failed for %s", result.symbol, exc_info=True)
                can_open = True
            if not can_open:
                context.metadata.setdefault("skipped_trades", []).append(
                    {"symbol": result.symbol, "reason": "position_guard"}
                )
                continue

        allowed = True
        reason = ""
        if risk_manager is not None:
            try:
                allowed, reason = await risk_manager.allow_trade(df, result.strategy)
            except Exception:  # pragma: no cover - best effort
                logger.debug("Risk allow_trade failed for %s", result.symbol, exc_info=True)
                allowed = True
        if not allowed:
            context.metadata.setdefault("skipped_trades", []).append({"symbol": result.symbol, "reason": reason})
            continue

        size = balance * 0.0
        if risk_manager is not None:
            try:
                size = float(
                    await risk_manager.position_size(
                        float(result.score or 0.0),
                        float(balance),
                        df=df,
                        atr=result.atr,
                        price=price,
                    )
                )
            except Exception:  # pragma: no cover - fallback sizing
                logger.debug("Risk position_size failed for %s", result.symbol, exc_info=True)
                size = float(balance) * 0.05
        else:
            size = float(balance) * 0.05

        if size <= 0:
            continue

        if risk_manager is not None and hasattr(risk_manager, "can_allocate") and hasattr(
            risk_manager, "allocate_capital"
        ):
            try:
                if not await risk_manager.can_allocate(result.strategy or "", size, balance):
                    continue
                await risk_manager.allocate_capital(result.strategy or "", size)
            except Exception:  # pragma: no cover - advisory
                logger.debug("Capital allocation failed for %s", result.symbol, exc_info=True)

        amount = size / price if price else 0.0
        trade_request = TradeExecutionRequest(
            exchange=getattr(context, "exchange", None),
            ws_client=getattr(context, "ws_client", None),
            symbol=result.symbol,
            side="buy" if direction == "long" else "sell",
            amount=float(amount),
            notifier=getattr(context, "notifier", None),
            dry_run=dry_run,
            use_websocket=bool(context.config.get("use_websocket")),
            config=context.config,
            score=float(result.score or 0.0),
        )

        try:
            response = await execution.execute_trade(trade_request)
        except Exception:
            logger.exception("Trade execution failed for %s", result.symbol)
            continue

        executed.append(
            {
                "symbol": result.symbol,
                "side": trade_request.side,
                "amount": trade_request.amount,
                "order": getattr(response, "order", {}),
                "strategy": result.strategy,
            }
        )
        balance = max(balance - size, 0.0)
        context.balance = balance

        paper_wallet = getattr(context, "paper_wallet", None)
        if paper_wallet is not None:
            try:
                if direction == "long":
                    await paper_wallet.buy(result.symbol, amount, price)
                else:
                    await paper_wallet.sell(result.symbol, amount, price)
            except Exception:  # pragma: no cover - optional path
                logger.debug("Paper wallet trade failed for %s", result.symbol, exc_info=True)

    context.metadata["executed_trades"] = executed
    context.metadata["executed_trade_count"] = len(executed)


async def finalize_cycle(context) -> None:
    """Persist summary metadata and refresh derived state."""

    context.metadata["cycle_completed_at"] = datetime.now(timezone.utc).isoformat()
    context.metadata["balance"] = float(getattr(context, "balance", 0.0))

    if getattr(context, "trade_manager", None) is not None:
        try:
            context.sync_positions_from_trade_manager()
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Post-cycle position sync failed", exc_info=True)

    context.metadata["open_positions"] = sorted(getattr(context, "positions", {}).keys())


DEFAULT_PHASES = [
    prepare_cycle,
    discover_markets,
    load_market_data,
    evaluate_signals,
    execute_signals,
    finalize_cycle,
]


__all__ = [
    "prepare_cycle",
    "discover_markets",
    "load_market_data",
    "evaluate_signals",
    "execute_signals",
    "finalize_cycle",
    "DEFAULT_PHASES",
]
