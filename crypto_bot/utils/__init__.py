from .kraken import get_ws_token
from .notifier import Notifier
from .eval_queue import compute_batches, build_priority_queue
try:
    from .market_loader import (
        load_kraken_symbols,
        fetch_ohlcv_async,
        fetch_order_book_async,
        load_ohlcv_parallel,
        update_ohlcv_cache,
        timeframe_seconds,
    )
except Exception:  # pragma: no cover - allow utils import without heavy deps
    # Provide minimal shims for symbols used in tests that import this package
    def timeframe_seconds(*args, **kwargs):  # type: ignore
        return 60
    load_kraken_symbols = fetch_ohlcv_async = fetch_order_book_async = None  # type: ignore
    load_ohlcv_parallel = update_ohlcv_cache = None  # type: ignore
from .pair_cache import load_liquid_pairs
# Symbol filtering utilities import is optional because the module has
# heavy async dependencies and some environments may not need it during
# initialization. Import it lazily where required.
try:
    from .symbol_utils import get_filtered_symbols, fix_symbol
    from .symbol_pre_filter import filter_symbols, has_enough_history
    
    # Create synchronous wrappers for tests
    def filter_symbols_sync(*args, **kwargs):
        """Synchronous wrapper for filter_symbols for testing purposes."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If we're in an async context, return the async function
                return filter_symbols
            else:
                # If no loop is running, run the async function
                return asyncio.run(filter_symbols(*args, **kwargs))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(filter_symbols(*args, **kwargs))
    
    def has_enough_history_sync(*args, **kwargs):
        """Synchronous wrapper for has_enough_history for testing purposes."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If we're in an async context, return the async function
                return has_enough_history
            else:
                # If no loop is running, run the async function
                return asyncio.run(has_enough_history(*args, **kwargs))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(has_enough_history(*args, **kwargs))
            
except Exception:  # pragma: no cover - optional during tests
    def get_filtered_symbols(*a, **k):  # type: ignore
        return []

    def fix_symbol(s: str) -> str:  # type: ignore
        return s

    def filter_symbols(*a, **k):  # type: ignore
        return {}

    def has_enough_history(*a, **k):  # type: ignore
        return True
        
    def filter_symbols_sync(*a, **k):  # type: ignore
        return {}
        
    def has_enough_history_sync(*a, **k):  # type: ignore
        return True
from .strategy_analytics import compute_metrics, write_scores, write_stats
try:
    from .regime_pnl_tracker import compute_weights
except Exception:  # pragma: no cover - optional during tests when module mocked
    def compute_weights(*_a, **_k):  # type: ignore
        return {}

# Lazy import to avoid circular dependency with strategy_router
def _get_market_analyzer_imports():
    """Lazy import function for market_analyzer to avoid circular imports."""
    from .market_analyzer import analyze_symbol, classify_regime_async, classify_regime_cached
    return analyze_symbol, classify_regime_async, classify_regime_cached

# Create lazy accessor functions
def analyze_symbol(*args, **kwargs):
    """Lazy wrapper for analyze_symbol function."""
    func, _, _ = _get_market_analyzer_imports()
    return func(*args, **kwargs)

def classify_regime_async(*args, **kwargs):
    """Lazy wrapper for classify_regime_async function."""
    _, func, _ = _get_market_analyzer_imports()
    return func(*args, **kwargs)

def classify_regime_cached(*args, **kwargs):
    """Lazy wrapper for classify_regime_cached function."""
    _, _, func = _get_market_analyzer_imports()
    return func(*args, **kwargs)
from .stats import zscore
from .commit_lock import check_and_update
commit_lock = check_and_update  # backward-compat alias used in some tests
from .telemetry import telemetry
try:
    from .solana_scanner import get_solana_new_tokens as utils_get_solana_new_tokens
except Exception:  # pragma: no cover - optional dependency
    utils_get_solana_new_tokens = None
from .pyth import get_pyth_price
from .pyth_utils import get_pyth_price
from .telegram import TelegramNotifier, send_message, is_admin
