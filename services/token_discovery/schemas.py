"""Pydantic schemas exposed by the token discovery service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, validator


class ScanRequest(BaseModel):
    """Request payload for triggering discovery scans."""

    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=500,
        description="Maximum number of tokens to return in the response.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Optional override for the scan source identifier.",
    )


class DiscoveryResponse(BaseModel):
    """Response payload for discovery endpoints."""

    tokens: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Opportunity(BaseModel):
    """Opportunity payload returned to downstream consumers."""

    token: str = Field(..., description="Token mint or trading symbol.")
    score: float = Field(..., description="Normalized opportunity score.")
    source: str = Field(default="enhanced")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("score")
    def _normalize_score(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class OpportunityResponse(BaseModel):
    """Response payload for opportunity endpoints."""

    opportunities: List[Opportunity] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScoreRequest(BaseModel):
    """Payload for ad-hoc opportunity scoring requests."""

    tokens: Sequence[str] = Field(..., description="Tokens to score.")


class ScoreResponse(BaseModel):
    """Response from ad-hoc opportunity scoring."""

    opportunities: List[Opportunity] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    """Operational status of the token discovery service."""

    status: str = Field(default="ok")
    redis_connected: bool = Field(default=True)
    kafka_connected: bool = Field(default=False)
    last_basic_scan: Optional[datetime] = None
    last_enhanced_scan: Optional[datetime] = None
    tokens_cached: int = Field(default=0)
    opportunities_cached: int = Field(default=0)
    last_cex_scan: Optional[datetime] = None
    cex_tokens_cached: int = Field(default=0)


__all__ = [
    "DiscoveryResponse",
    "Opportunity",
    "OpportunityResponse",
    "ScanRequest",
    "ScoreRequest",
    "ScoreResponse",
    "StatusResponse",
]
