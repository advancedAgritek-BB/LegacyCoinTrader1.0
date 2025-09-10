"""Price utilities module."""

from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path

from .enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher
from .market_loader import MarketLoader


def get_price_data(
    symbol: str,
    exchange: str = "kraken",
    timeframe: str = "1m",
    limit: int = 100,
    since: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """Get price data for a symbol."""
    try:
        fetcher = EnhancedOHLCVFetcher()
        data = fetcher.fetch_ohlcv(symbol, exchange, timeframe, limit, since)
        return data
    except Exception:
        return None


def get_market_data(
    symbols: List[str],
    exchange: str = "kraken",
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """Get market data for multiple symbols."""
    try:
        loader = MarketLoader()
        return loader.load_markets(symbols, exchange, **kwargs)
    except Exception:
        return {}


def get_latest_price(symbol: str, exchange: str = "kraken") -> Optional[float]:
    """Get the latest price for a symbol."""
    data = get_price_data(symbol, exchange, limit=1)
    if data is not None and not data.empty:
        return float(data['close'].iloc[-1])
    return None


__all__ = ['get_price_data', 'get_market_data', 'get_latest_price']
