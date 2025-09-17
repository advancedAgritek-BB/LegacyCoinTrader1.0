"""Core strategy evaluation engine."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import redis.asyncio as redis

from libs.services.interfaces import (
    RankedSignal,
    StrategyBatchRequest,
    StrategyBatchResponse,
    StrategyEvaluationPayload,
    StrategyEvaluationResult,
)
from libs.strategy.evaluator import evaluate_payload

from .config import Settings
from .storage import ModelRegistry

logger = logging.getLogger(__name__)


class StrategyEngine:
    """Coordinates strategy evaluation and caching."""

    def __init__(self, client: redis.Redis, settings: Settings, model_store: ModelRegistry) -> None:
        self._redis = client
        self._settings = settings
        self._model_store = model_store
        self._cache_prefix = settings.cache_prefix

    async def evaluate_batch(self, request: StrategyBatchRequest) -> StrategyBatchResponse:
        results: list[StrategyEvaluationResult] = []
        errors: list[str] = []

        for item in request.items:
            try:
                result = await self._evaluate_item(item)
                results.append(result)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Strategy evaluation failed for %s", item.symbol)
                errors.append(f"{item.symbol}: {exc}")

        return StrategyBatchResponse(results=tuple(results), errors=tuple(errors))

    async def handle_market_event(self, payload: Mapping[str, Any]) -> None:
        """Invalidate cached results when new market data is published."""

        symbol = str(payload.get("symbol", "")).strip()
        if not symbol:
            return
        await self.invalidate(symbol)

    async def invalidate(self, symbol: str) -> None:
        pattern = f"{self._cache_prefix}:{symbol}*"
        keys = []
        async for key in self._redis.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            await self._redis.delete(*keys)
            logger.debug("Invalidated %d cached results for %s", len(keys), symbol)

    async def _evaluate_item(self, payload: StrategyEvaluationPayload) -> StrategyEvaluationResult:
        cache_key = self._cache_key(payload)
        cached = await self._redis.get(cache_key)
        if cached:
            result = self._deserialize_result(cached)
            result.cached = True
            return result

        result = await evaluate_payload(payload)
        await self._redis.set(cache_key, self._serialize_result(result), ex=self._settings.evaluation_cache_ttl)
        if result.strategy:
            await self._model_store.touch(result.strategy, {"symbol": payload.symbol})
        return result

    def _serialize_result(self, result: StrategyEvaluationResult) -> str:
        return json.dumps(
            {
                "symbol": result.symbol,
                "regime": result.regime,
                "strategy": result.strategy,
                "score": result.score,
                "direction": result.direction,
                "atr": result.atr,
                "fused_score": result.fused_score,
                "fused_direction": result.fused_direction,
                "ranked_signals": [
                    {
                        "strategy": rs.strategy,
                        "score": rs.score,
                        "direction": rs.direction,
                    }
                    for rs in result.ranked_signals
                ],
                "metadata": dict(result.metadata),
                "cached": result.cached,
            }
        )

    def _deserialize_result(self, raw: bytes | str) -> StrategyEvaluationResult:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        ranked = tuple(
            RankedSignal(
                strategy=item.get("strategy", ""),
                score=float(item.get("score", 0.0)),
                direction=item.get("direction", "none"),
            )
            for item in data.get("ranked_signals", [])
        )
        return StrategyEvaluationResult(
            symbol=data.get("symbol", ""),
            regime=data.get("regime", ""),
            strategy=data.get("strategy", ""),
            score=float(data.get("score", 0.0)),
            direction=data.get("direction", "none"),
            atr=data.get("atr"),
            fused_score=data.get("fused_score"),
            fused_direction=data.get("fused_direction"),
            ranked_signals=ranked,
            metadata=data.get("metadata", {}),
            cached=bool(data.get("cached", False)),
        )

    def _cache_key(self, payload: StrategyEvaluationPayload) -> str:
        parts = [self._cache_prefix, payload.symbol, payload.regime, payload.mode]
        latest = self._latest_timestamp(payload)
        if latest:
            parts.append(latest)
        config = payload.config or {}
        if config:
            digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            parts.append(digest[:12])
        return ":".join(parts)

    def _latest_timestamp(self, payload: StrategyEvaluationPayload) -> Optional[str]:
        latest: Optional[pd.Timestamp] = None
        for df in payload.timeframes.values():
            if isinstance(df, pd.DataFrame) and not df.empty:
                ts = df.index[-1]
                try:
                    ts = pd.to_datetime(ts, utc=True)
                except Exception:  # pragma: no cover - convert best effort
                    continue
                if latest is None or ts > latest:
                    latest = ts
        if latest is None:
            return None
        return latest.isoformat()


__all__ = ["StrategyEngine"]
