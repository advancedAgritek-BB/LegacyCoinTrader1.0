"""Adapters for strategy routing and evaluation services."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Sequence

import httpx
import pandas as pd

from crypto_bot.services.interfaces import (
    RankedSignal,
    StrategyBatchRequest,
    StrategyBatchResponse,
    StrategyEvaluationPayload,
    StrategyEvaluationResult,
    StrategyEvaluationService,
    StrategyNameRequest,
    StrategyNameResponse,
    StrategyRequest,
    StrategyResponse,
)
from crypto_bot.services.strategy_evaluator import evaluate_batch_request
from crypto_bot.strategy_router import strategy_for, strategy_name
from crypto_bot.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from crypto_bot.utils.retry_handler import RetryConfig, RetryHandler, RetryStrategy

logger = logging.getLogger(__name__)


class LocalStrategyAdapter(StrategyEvaluationService):
    """Evaluate strategies within the current process."""

    def __init__(self, notifier: Any = None) -> None:
        self._notifier = notifier

    def select_strategy(self, request: StrategyRequest) -> StrategyResponse:
        strategy = strategy_for(request.regime, request.config)
        return StrategyResponse(strategy=strategy)

    def resolve_strategy_name(self, request: StrategyNameRequest) -> StrategyNameResponse:
        name = strategy_name(request.regime, request.mode)
        return StrategyNameResponse(name=name)

    async def evaluate_batch(self, request: StrategyBatchRequest) -> StrategyBatchResponse:
        return await evaluate_batch_request(request, notifier=self._notifier)


class StrategyEngineClient(StrategyEvaluationService):
    """HTTP client for the external strategy engine service."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 10.0,
        retry_config: RetryConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        notifier: Any = None,
    ) -> None:
        host = os.getenv("STRATEGY_ENGINE_HOST")
        port = os.getenv("STRATEGY_ENGINE_PORT")
        default_url = "http://localhost:8004"
        if host and port:
            default_url = f"http://{host}:{port}"
        self._base_url = base_url or os.getenv("STRATEGY_ENGINE_URL", default_url)
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self._retry = RetryHandler(
            "strategy-engine-client",
            retry_config
            or RetryConfig(
                max_retries=3,
                base_delay=0.5,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                timeout=timeout,
            ),
        )
        self._circuit_breaker = CircuitBreaker(
            "strategy-engine-client",
            circuit_config or CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30),
        )
        self._local = LocalStrategyAdapter(notifier=notifier)

    def select_strategy(self, request: StrategyRequest) -> StrategyResponse:
        return self._local.select_strategy(request)

    def resolve_strategy_name(self, request: StrategyNameRequest) -> StrategyNameResponse:
        return self._local.resolve_strategy_name(request)

    async def evaluate_batch(self, request: StrategyBatchRequest) -> StrategyBatchResponse:
        async def http_call() -> StrategyBatchResponse:
            payload = self._serialize_request(request)
            response = await self._client.post("/evaluate/batch", json=payload)
            response.raise_for_status()
            return self._deserialize_response(response.json())

        async def guarded_call() -> StrategyBatchResponse:
            return await self._circuit_breaker.call(http_call)

        try:
            return await self._retry.execute_with_retry(guarded_call)
        except Exception as exc:  # pragma: no cover - network failures
            logger.warning(
                "Remote strategy engine unavailable, falling back to local evaluation: %s",
                exc,
            )
            return await self._local.evaluate_batch(request)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _serialize_request(self, request: StrategyBatchRequest) -> Dict[str, Any]:
        return {
            "requests": [self._serialize_payload(item) for item in request.items],
            "metadata": dict(request.metadata),
        }

    def _serialize_payload(self, payload: StrategyEvaluationPayload) -> Dict[str, Any]:
        serialized_timeframes: Dict[str, Any] = {}
        for tf, df in payload.timeframes.items():
            if isinstance(df, pd.DataFrame):
                frame = df.reset_index()
                if "timestamp" not in frame.columns:
                    index_col = frame.columns[0]
                    frame = frame.rename(columns={index_col: "timestamp"})
                records: list[dict[str, Any]] = []
                for row in frame.to_dict(orient="records"):
                    ts = row.get("timestamp")
                    if ts is not None:
                        try:
                            ts = pd.to_datetime(ts, utc=True).isoformat()
                        except Exception:  # pragma: no cover - defensive
                            ts = str(ts)
                        row["timestamp"] = ts
                    records.append(row)
                serialized_timeframes[tf] = records
            elif isinstance(df, Sequence):
                serialized_timeframes[tf] = list(df)
            elif isinstance(df, Mapping):
                serialized_timeframes[tf] = dict(df)
            else:
                serialized_timeframes[tf] = df

        return {
            "symbol": payload.symbol,
            "regime": payload.regime,
            "mode": payload.mode,
            "timeframes": serialized_timeframes,
            "config": dict(payload.config or {}),
            "metadata": dict(payload.metadata),
        }

    def _deserialize_response(self, data: Mapping[str, Any]) -> StrategyBatchResponse:
        results = [self._deserialize_result(entry) for entry in data.get("results", [])]
        errors = list(data.get("errors", []))
        return StrategyBatchResponse(results=tuple(results), errors=tuple(errors))

    def _deserialize_result(self, data: Mapping[str, Any]) -> StrategyEvaluationResult:
        fused = data.get("fused_signal") or {}
        ranked_raw = data.get("ranked_signals", [])
        ranked: Sequence[RankedSignal] = tuple(
            RankedSignal(
                strategy=item.get("strategy", ""),
                score=float(item.get("score", 0.0)),
                direction=item.get("direction", "none"),
            )
            for item in ranked_raw
        )
        return StrategyEvaluationResult(
            symbol=data.get("symbol", ""),
            regime=data.get("regime", ""),
            strategy=data.get("strategy", ""),
            score=float(data.get("score", 0.0)),
            direction=data.get("direction", "none"),
            atr=data.get("atr"),
            fused_score=data.get("fused_score", fused.get("score")),
            fused_direction=data.get("fused_direction", fused.get("direction")),
            ranked_signals=ranked,
            metadata=dict(data.get("metadata", {})),
            cached=bool(data.get("cached", False)),
        )


class StrategyAdapter(StrategyEngineClient):
    """Default adapter exposed to the service container."""

    pass


__all__ = [
    "LocalStrategyAdapter",
    "StrategyAdapter",
    "StrategyEngineClient",
]
