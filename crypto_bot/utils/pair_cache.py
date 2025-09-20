import json
from pathlib import Path
from typing import Optional, Dict, List

from .logger import LOG_DIR, setup_logger

PAIR_FILE = Path(__file__).resolve().parents[2] / "cache" / "liquid_pairs.json"
logger = setup_logger(__name__, LOG_DIR / "pair_cache.log")


def load_liquid_map() -> Optional[Dict[str, float]]:
    """Return cached mapping of pair -> timestamp if available."""
    if not PAIR_FILE.exists():
        try:
            from crypto_bot.cache import get_liquid_pairs

            pairs = get_liquid_pairs()
            if pairs:
                msg = "Bootstrapped liquid pair cache with %d pairs"
                logger.info(msg, len(pairs))
                return {str(pair): 0.0 for pair in pairs}
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Failed to bootstrap liquid pair cache: %s", exc)
        return None

    # File exists, try to read it
    try:
        data: Dict[str, float] = {}
        raw_data = json.loads(PAIR_FILE.read_text())
        if isinstance(raw_data, list):
            data = {p: 0.0 for p in raw_data}
        elif isinstance(raw_data, dict):
            # Handle cache format: {"exchange": "kraken", "timestamp": 123, "pairs": [...]}  # noqa
            if "pairs" in raw_data and isinstance(raw_data["pairs"], list):
                # Use the cache timestamp if available, otherwise use 0.0
                timestamp = raw_data.get("timestamp", 0.0)
                data = {pair: timestamp for pair in raw_data["pairs"]}
            else:
                # Handle legacy format with string keys and float values
                filtered_items = {k: v for k, v in raw_data.items()
                                  if k not in ["exchange", "timestamp"]}
                data = {str(k): float(v) for k, v in filtered_items.items()}
        else:
            data = {}
        if not data:
            warning_msg = ("%s is empty. Run tasks/refresh_pairs.py or "
                           "adjust symbol_filter.uncached_volume_multiplier")
            logger.warning(warning_msg, PAIR_FILE)
            return None  # type: ignore[return-value]
        return data  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to read %s: %s", PAIR_FILE, exc)
        try:
            from crypto_bot.cache import get_liquid_pairs

            pairs = get_liquid_pairs(force_refresh=True)
            if pairs:
                logger.info("Refreshed liquid pair cache after read failure")
                return {str(pair): 0.0 for pair in pairs}  # type: ignore
        except Exception as refresh_exc:  # pragma: no cover - defensive
            logger.debug("Refresh attempt failed: %s", refresh_exc)
    return None


def load_liquid_pairs() -> Optional[List[str]]:
    """Return cached list of liquid trading pairs if available."""
    data = load_liquid_map()
    return list(data) if data else None
