"""Liquid pairs management module."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time

from . import get_liquid_pairs as _get_liquid_pairs


def get_liquid_pairs(exchange: str = "kraken", force_refresh: bool = False) -> List[str]:
    """Get liquid trading pairs for a specific exchange."""
    return _get_liquid_pairs(exchange, force_refresh)


def get_liquid_pairs_by_volume(exchange: str = "kraken", min_volume: float = 1000000) -> List[str]:
    """Get liquid pairs filtered by minimum volume."""
    # This would typically fetch from exchange API
    # For now, return default liquid pairs
    return get_liquid_pairs(exchange)


def get_liquid_pairs_by_market_cap(exchange: str = "kraken", min_market_cap: float = 1000000000) -> List[str]:
    """Get liquid pairs filtered by minimum market cap."""
    # This would typically fetch from exchange API
    # For now, return default liquid pairs
    return get_liquid_pairs(exchange)


__all__ = ['get_liquid_pairs', 'get_liquid_pairs_by_volume', 'get_liquid_pairs_by_market_cap']
