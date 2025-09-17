"""API schemas for the strategy engine service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CandleModel(BaseModel):
    """Serialized OHLCV candle."""

    timestamp: datetime | str | int | float
    open: float
    high: float
    low: float
    close: float
    volume: float


class EvaluationRequestModel(BaseModel):
    """Single symbol evaluation payload."""

    symbol: str
    regime: str
    mode: str
    timeframes: Dict[str, List[CandleModel]]
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchEvaluationRequest(BaseModel):
    """Batch of evaluation requests."""

    requests: List[EvaluationRequestModel]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RankedSignalModel(BaseModel):
    """Ranked signal information."""

    strategy: str
    score: float
    direction: str


class FusedSignalModel(BaseModel):
    """Signal fusion output."""

    score: float
    direction: str


class EvaluationResultModel(BaseModel):
    """Evaluation response for a single symbol."""

    symbol: str
    regime: str
    strategy: str
    score: float
    direction: str
    atr: Optional[float] = None
    fused_signal: Optional[FusedSignalModel] = None
    ranked_signals: List[RankedSignalModel] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    cached: bool = False


class BatchEvaluationResponse(BaseModel):
    """Batch evaluation response payload."""

    results: List[EvaluationResultModel]
    errors: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
