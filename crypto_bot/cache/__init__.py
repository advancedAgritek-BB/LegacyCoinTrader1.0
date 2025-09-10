"""Cache management module."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time

CACHE_DIR = Path("cache")
LIQUID_PAIRS_CACHE_FILE = CACHE_DIR / "liquid_pairs.json"


def get_liquid_pairs(exchange: str = "kraken", force_refresh: bool = False) -> List[str]:
    """Get liquid trading pairs from cache or fetch them."""
    if not force_refresh and LIQUID_PAIRS_CACHE_FILE.exists():
        try:
            with open(LIQUID_PAIRS_CACHE_FILE, 'r') as f:
                data = json.load(f)
                if data.get('exchange') == exchange and data.get('timestamp', 0) > time.time() - 3600:  # 1 hour cache
                    return data.get('pairs', [])
        except Exception:
            pass
    
    # Return some default liquid pairs if cache is not available
    default_pairs = [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
        "LINK/USD", "UNI/USD", "MATIC/USD", "AVAX/USD", "ATOM/USD"
    ]
    
    # Save to cache
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_data = {
            'exchange': exchange,
            'timestamp': time.time(),
            'pairs': default_pairs
        }
        with open(LIQUID_PAIRS_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception:
        pass
    
    return default_pairs


def cache_liquid_pairs(pairs: List[str], exchange: str = "kraken") -> bool:
    """Cache liquid pairs data."""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        cache_data = {
            'exchange': exchange,
            'timestamp': time.time(),
            'pairs': pairs
        }
        with open(LIQUID_PAIRS_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
        return True
    except Exception:
        return False


def clear_cache() -> bool:
    """Clear all cached data."""
    try:
        if CACHE_DIR.exists():
            for file in CACHE_DIR.glob("*"):
                if file.is_file():
                    file.unlink()
        return True
    except Exception:
        return False


__all__ = ['get_liquid_pairs', 'cache_liquid_pairs', 'clear_cache']
