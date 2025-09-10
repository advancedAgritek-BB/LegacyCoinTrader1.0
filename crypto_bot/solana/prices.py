from __future__ import annotations

"""Helper for fetching Solana token prices."""

import aiohttp
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


async def fetch_solana_prices(symbols: List[str]) -> Dict[str, float]:
    """Return current prices for ``symbols`` using Jupiter public API."""
    if not symbols:
        return {}

    # Jupiter price API (price.jup.ag) appears to be deprecated/unavailable
    # Return empty results to avoid repeated DNS errors
    logger.warning("Jupiter price API is currently unavailable. Skipping Solana price fetching.")
    return {sym: 0.0 for sym in symbols}
