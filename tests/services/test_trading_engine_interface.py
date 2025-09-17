from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import sys
import types

if "pydantic_settings" not in sys.modules:  # pragma: no cover - test scaffold
    module = types.ModuleType("pydantic_settings")

    class _BaseSettings:  # pragma: no cover - minimal shim
        model_config = {}

    module.BaseSettings = _BaseSettings
    module.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = module

from crypto_bot.services.interfaces import (
    CacheUpdateResponse,
    ServiceContainer,
    StrategyBatchRequest,
    StrategyBatchResponse,
    StrategyEvaluationResult,
    TokenDiscoveryRequest,
    TokenDiscoveryResponse,
    TradeExecutionRequest,
    TradeExecutionResponse,
)
from services.trading_engine.interface import TradingEngineInterface


class FakeMarketDataService:
    async def update_multi_tf_cache(self, request: Any) -> CacheUpdateResponse:
        cache = request.cache
        timeframe = request.config.get("timeframe", "1h")
        tf_cache = cache.setdefault(timeframe, {})
        index = pd.date_range("2024-01-01", periods=32, freq="H", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100 + i * 0.1 for i in range(len(index))],
                "high": [101 + i * 0.1 for i in range(len(index))],
                "low": [99 + i * 0.1 for i in range(len(index))],
                "close": [100 + i * 0.1 for i in range(len(index))],
                "volume": [50 + i for i in range(len(index))],
            },
            index=index,
        )
        for symbol in request.symbols:
            tf_cache[symbol] = frame.copy()
        return CacheUpdateResponse(cache=cache)

    async def update_regime_cache(self, request: Any) -> CacheUpdateResponse:
        cache = request.cache
        tf_cache = cache.setdefault("1h", {})
        for symbol in request.symbols:
            tf_cache[symbol] = {"regime": "trending"}
        return CacheUpdateResponse(cache=cache)


class FakeStrategyService:
    def __init__(self) -> None:
        self.requests: list[StrategyBatchRequest] = []

    async def evaluate_batch(self, request: StrategyBatchRequest) -> StrategyBatchResponse:
        self.requests.append(request)
        results: list[StrategyEvaluationResult] = []
        for payload in request.items:
            results.append(
                StrategyEvaluationResult(
                    symbol=payload.symbol,
                    regime=payload.regime,
                    strategy="trend_bot",
                    score=0.8,
                    direction="long",
                    atr=1.5,
                )
            )
        return StrategyBatchResponse(results=tuple(results), errors=tuple())


class FakeExecutionService:
    def __init__(self) -> None:
        self.requests: list[TradeExecutionRequest] = []

    async def execute_trade(self, request: TradeExecutionRequest) -> TradeExecutionResponse:
        self.requests.append(request)
        return TradeExecutionResponse(order={"id": f"order-{len(self.requests)}"})


class FakeTokenDiscoveryService:
    async def discover_tokens(self, request: TokenDiscoveryRequest) -> TokenDiscoveryResponse:
        return TokenDiscoveryResponse(tokens=["ETH/USD"])


class FakePortfolioService:
    def get_trade_manager(self):  # pragma: no cover - simple stub
        return None


class FakeMonitoringService:
    def record_scanner_metrics(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        return None


class FakeRiskClient:
    def __init__(self) -> None:
        self.checked: list[str | None] = []
        self.allocated: list[tuple[str, float]] = []

    async def allow_trade(self, df: pd.DataFrame, strategy: str | None = None):
        self.checked.append(strategy)
        return True, ""

    async def position_size(
        self,
        confidence: float,
        balance: float,
        df: pd.DataFrame | None = None,
        atr: float | None = None,
        price: float | None = None,
    ) -> float:
        return balance * 0.2

    async def can_allocate(self, strategy: str, amount: float, balance: float) -> bool:
        return True

    async def allocate_capital(self, strategy: str, amount: float) -> None:  # pragma: no cover - simple stub
        self.allocated.append((strategy, amount))


class FakePositionGuard:
    async def can_open(self, positions: dict[str, Any]) -> bool:
        return True


@pytest.mark.asyncio
async def test_trading_engine_run_cycle_executes_real_phases() -> None:
    services = ServiceContainer(
        market_data=FakeMarketDataService(),
        strategy=FakeStrategyService(),
        portfolio=FakePortfolioService(),
        execution=FakeExecutionService(),
        token_discovery=FakeTokenDiscoveryService(),
        monitoring=FakeMonitoringService(),
    )

    config = {
        "symbols": ["BTC/USD"],
        "timeframe": "1h",
        "execution_mode": "dry_run",
        "risk": {"starting_balance": 1000.0},
    }

    risk_client = FakeRiskClient()

    interface = TradingEngineInterface(
        services=services,
        config=config,
        risk_client=risk_client,
        paper_wallet=None,
        position_guard=FakePositionGuard(),
        trade_manager=None,
    )

    result = await interface.run_cycle(metadata={"trigger": "unit"})

    assert "prepare_cycle" in result.timings
    assert services.strategy.requests, "strategy service should be invoked"
    assert services.execution.requests, "execution service should be invoked"
    assert services.execution.requests[0].symbol in {"BTC/USD", "ETH/USD"}
    assert services.execution.requests[0].exchange is None
    assert services.execution.requests[0].ws_client is None
    assert interface.context.analysis_results
    assert result.metadata["executed_trade_count"] == len(services.execution.requests)

    await interface.shutdown()
