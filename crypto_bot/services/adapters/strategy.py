"""In-process adapter for strategy routing helpers."""

from __future__ import annotations

from crypto_bot.services.interfaces import (
    StrategyEvaluationService,
    StrategyNameRequest,
    StrategyNameResponse,
    StrategyRequest,
    StrategyResponse,
)
from crypto_bot.strategy_router import strategy_for, strategy_name


class StrategyAdapter(StrategyEvaluationService):
    """Adapter for :mod:`crypto_bot.strategy_router`."""

    def select_strategy(self, request: StrategyRequest) -> StrategyResponse:
        strategy = strategy_for(request.regime, request.config)
        return StrategyResponse(strategy=strategy)

    def resolve_strategy_name(self, request: StrategyNameRequest) -> StrategyNameResponse:
        name = strategy_name(request.regime, request.mode)
        return StrategyNameResponse(name=name)
