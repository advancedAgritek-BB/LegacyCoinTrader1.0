from __future__ import annotations
from typing import Optional, List

import os
from pydantic import BaseModel, Field, validator, root_validator


class ScannerConfig(BaseModel):
    """Configuration for market scanning."""

    scan_markets: bool = Field(False, description="Load all exchange pairs")
    symbols: Optional[List[str]] = Field(
        default_factory=list,
        description="Symbols to trade",
    )
    excluded_symbols: List[str] = Field(
        default_factory=list,
        description="Symbols to skip",
    )
    exchange_market_types: List[str] = Field(
        default_factory=lambda: ["spot"],
        description="Market types",
    )
    min_symbol_age_days: int = 0
    symbol_batch_size: int = 10
    scan_lookback_limit: int = 50
    cycle_lookback_limit: Optional[int] = Field(
        default=None,
        description=(
            "Override per-cycle candle load (default min(150, "
            "timeframe_minutes * 2))"
        ),
    )
    max_spread_pct: float = 1.0

    class Config:
        extra = "allow"

    @validator("symbols", pre=True)
    @classmethod
    def _default_symbols(cls, v: Optional[List[str]]) -> List[str]:
        return v or []

    @validator("exchange_market_types")
    @classmethod
    def _validate_market_type(cls, v: List[str]) -> List[str]:
        allowed = {"spot", "margin", "futures"}
        for item in v:
            if item not in allowed:
                raise ValueError(f"invalid market type: {item}")
        return v

    @validator(
        "symbol_batch_size",
        "scan_lookback_limit",
        "cycle_lookback_limit",
    )
    @classmethod
    def _positive_int(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("value must be > 0")
        return v

    @root_validator(skip_on_failure=True)
    def _enforce_symbols_when_not_scanning(cls, values: dict) -> dict:
        scan_markets = values.get("scan_markets", False)
        symbols = values.get("symbols") or []
        if not scan_markets and len(symbols) == 0:
            raise ValueError(
                "symbols must be provided when scan_markets is False"
            )
        return values


class SolanaScannerApiKeys(BaseModel):
    """API key configuration for Solana scanner."""

    moralis: str = Field(
        default_factory=lambda: os.getenv("MORALIS_KEY", "YOUR_KEY")
    )
    bitquery: str = Field(
        default_factory=lambda: os.getenv("BITQUERY_KEY", "YOUR_KEY")
    )


class SolanaScannerConfig(BaseModel):
    """Configuration for scanning Solana tokens."""

    enabled: bool = False
    interval_minutes: int = 5
    api_keys: SolanaScannerApiKeys = Field(
        default_factory=SolanaScannerApiKeys
    )
    min_volume_usd: float = 0.0
    max_tokens_per_scan: int = 20
    gecko_search: bool = True

    class Config:
        extra = "forbid"


class PythConfig(BaseModel):
    """Configuration for Pyth price feeds."""

    enabled: bool = False
    solana_endpoint: str = ""
    solana_ws_endpoint: str = ""
    program_id: str = ""

    class Config:
        extra = "forbid"
