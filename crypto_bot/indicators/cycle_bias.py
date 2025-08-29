"""Market cycle bias derived from on-chain metrics."""

from __future__ import annotations
from typing import Optional

import os
import requests
import time

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from pathlib import Path


logger = setup_logger(__name__, LOG_DIR / "indicators.log")

# Use real API endpoints with fallbacks
DEFAULT_MVRV_URL = "https://api.alternative.me/v2/onchain/mvrv"
DEFAULT_NUPL_URL = "https://api.alternative.me/v2/onchain/nupl" 
DEFAULT_SOPR_URL = "https://api.alternative.me/v2/onchain/sopr"

# Fallback endpoints if primary ones fail
FALLBACK_MVRV_URL = "https://api.alternative.me/fng/?limit=1"
FALLBACK_NUPL_URL = "https://api.alternative.me/fng/?limit=1"
FALLBACK_SOPR_URL = "https://api.alternative.me/fng/?limit=1"


def _fetch_value(url: str, mock_env: str, fallback_url: str = None) -> float:
    """Return metric value from ``url`` using ``mock_env`` for tests."""
    mock = os.getenv(mock_env)
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0
    
    # Try primary URL first
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, dict):
            # Handle different API response formats
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                return float(data["data"][0].get("value", 0.0))
            elif "value" in data:
                return float(data["value"])
            elif "fng" in data and isinstance(data["fng"], list) and len(data["fng"]) > 0:
                # Fallback to Fear & Greed Index format
                fng_value = float(data["fng"][0].get("value", 50))
                # Convert FNG (0-100) to -1 to 1 scale
                return (fng_value - 50) / 50
            else:
                logger.warning(f"Unexpected API response format from {url}")
                return 0.0
                
    except Exception as exc:
        logger.warning(f"Failed to fetch metric from primary URL {url}: {exc}")
        
        # Try fallback URL if available
        if fallback_url:
            try:
                resp = requests.get(fallback_url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
                if "fng" in data and isinstance(data["fng"], list) and len(data["fng"]) > 0:
                    fng_value = float(data["fng"][0].get("value", 50))
                    # Convert FNG (0-100) to -1 to 1 scale
                    return (fng_value - 50) / 50
                    
            except Exception as fallback_exc:
                logger.warning(f"Fallback URL also failed {fallback_url}: {fallback_exc}")
        
        logger.error(f"Failed to fetch metric from {url}: {exc}")
    
    return 0.0


def get_cycle_bias(config: Optional[dict] = None) -> float:
    """Return bias multiplier (>1 bullish, <1 bearish)."""
    cfg = config or {}
    mvrv_url = cfg.get("mvrv_url") or os.getenv("MVRV_URL", DEFAULT_MVRV_URL)
    nupl_url = cfg.get("nupl_url") or os.getenv("NUPL_URL", DEFAULT_NUPL_URL)
    sopr_url = cfg.get("sopr_url") or os.getenv("SOPR_URL", DEFAULT_SOPR_URL)

    mvrv = _fetch_value(mvrv_url, "MOCK_MVRV", FALLBACK_MVRV_URL)
    nupl = _fetch_value(nupl_url, "MOCK_NUPL", FALLBACK_NUPL_URL)
    sopr = _fetch_value(sopr_url, "MOCK_SOPR", FALLBACK_SOPR_URL)

    # Normalize values to -1 to 1 range if they're not already
    mvrv = max(-1.0, min(1.0, mvrv))
    nupl = max(-1.0, min(1.0, nupl))
    sopr = max(-1.0, min(1.0, sopr))

    score = (mvrv + nupl + sopr) / 3
    bias = 1 + score
    
    logger.info(
        "Cycle bias %.2f from MVRV %.2f NUPL %.2f SOPR %.2f",
        bias,
        mvrv,
        nupl,
        sopr,
    )
    return bias
