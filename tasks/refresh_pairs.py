from __future__ import annotations

from typing import Union, List, Dict, Tuple
import argparse
import asyncio
import json
import time
from pathlib import Path
import logging

import ccxt.async_support as ccxt
import yaml
import aiohttp
try:
    from crypto_bot.utils import timeframe_seconds
except Exception:  # pragma: no cover - fallback for tests
    def timeframe_seconds(_exchange=None, timeframe: str = "1m"):
        import pandas as _pd
        try:
            return int(_pd.Timedelta(timeframe).total_seconds())
        except Exception:
            return 60
from crypto_bot.utils.symbol_utils import fix_symbol

CONFIG_PATH = Path(__file__).resolve().parents[1] / "crypto_bot" / "config.yaml"
CACHE_DIR = Path("cache")
PAIR_FILE = CACHE_DIR / "liquid_pairs.json"

DEFAULT_MIN_VOLUME_USD = 1_000_000
DEFAULT_TOP_K = 40
DEFAULT_REFRESH_INTERVAL = 6 * 3600  # 6 hours

logger = logging.getLogger(__name__)


def _parse_interval(value: Union[str, int, float]) -> float:
    """Return ``value`` in seconds, accepting shorthand like "6h"."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            try:
                return float(timeframe_seconds(None, value))
            except Exception:
                pass
    return float(DEFAULT_REFRESH_INTERVAL)


def load_config() -> dict:
    """Load YAML configuration if available with better error handling."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
    except Exception as exc:
        logger.warning("Failed to load main config: %s", exc)
        data = {}

    try:
        strat_dir = CONFIG_PATH.parent.parent / "config" / "strategies"
        trend_file = strat_dir / "trend_bot.yaml"
        if trend_file.exists():
            with open(trend_file) as sf:
                overrides = yaml.safe_load(sf) or {}
            trend_cfg = data.get("trend", {})
            if isinstance(trend_cfg, dict):
                trend_cfg.update(overrides)
            else:
                trend_cfg = overrides
            data["trend"] = trend_cfg
    except Exception as exc:
        logger.warning("Failed to load trend config: %s", exc)

    try:
        if "symbol" in data:
            data["symbol"] = fix_symbol(data["symbol"])
        if "symbols" in data:
            data["symbols"] = [fix_symbol(s) for s in data.get("symbols", [])]
    except Exception as exc:
        logger.warning("Failed to fix symbols: %s", exc)
    
    return data


def get_exchange(config: dict) -> ccxt.Exchange:
    """Instantiate the configured ccxt exchange."""
    name = config.get("exchange", "kraken").lower()
    if not hasattr(ccxt, name):
        raise ValueError(f"Unsupported exchange: {name}")
    return getattr(ccxt, name)({"enableRateLimit": True})


async def _fetch_tickers(exchange: ccxt.Exchange) -> dict:
    """Fetch tickers with a 10 second timeout."""
    return await asyncio.wait_for(exchange.fetch_tickers(), 10)


async def _close_exchange(exchange: ccxt.Exchange) -> None:
    """Safely close exchange connection with better error handling."""
    close = getattr(exchange, "close", None)
    if close:
        try:
            if asyncio.iscoroutinefunction(close):
                await close()
            else:
                close()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Failed to close exchange gracefully: %s", exc)
            pass


async def get_solana_liquid_pairs(min_volume: float) -> List[str]:
    """Return Raydium symbols with liquidity above ``min_volume``."""
    url = "https://api.raydium.io/v2/main/pairs"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.error("Failed to fetch Solana pairs: %s", exc)
        return []

    items = data.get("data") if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return []

    results: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        base, _, quote = name.partition("/")
        if quote.upper() != "USDC" or not base:
            continue
        vol = (
            item.get("liquidity")
            or item.get("liquidityUsd")
            or item.get("liquidity_usd")
            or item.get("liquidityUSD")
            or item.get("volumeUsd")
            or item.get("volume_usd")
            or item.get("volume24hQuote")
            or 0.0
        )
        try:
            amount = float(vol)
        except Exception:
            amount = 0.0
        if amount >= min_volume:
            results.append(f"{base.upper()}/USDC")

    return results


async def refresh_pairs_async(min_volume_usd: float, top_k: int, config: dict) -> List[str]:
    """Fetch tickers and update the cached liquid pairs list."""
    old_pairs: List[str] = []
    old_map: Dict[str, float] = {}
    if PAIR_FILE.exists():
        try:
            with open(PAIR_FILE) as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    old_pairs = loaded
                elif isinstance(loaded, dict):
                    old_pairs = list(loaded)
                    old_map = {k: float(v) for k, v in loaded.items()}
        except Exception as exc:  # pragma: no cover - corrupted cache
            logger.error("Failed to read %s: %s", PAIR_FILE, exc)

    rp_cfg = config.get("refresh_pairs", {}) if isinstance(config, dict) else {}
    allowed_quotes = {
        q.upper() for q in rp_cfg.get("allowed_quote_currencies", []) if isinstance(q, str)
    }
    blacklist = {
        a.upper() for a in rp_cfg.get("blacklist_assets", []) if isinstance(a, str)
    }

    exchange = get_exchange(config)
    sec_name = config.get("refresh_pairs", {}).get("secondary_exchange")
    secondary = get_exchange({"exchange": sec_name}) if sec_name else None
    try:
        tasks = [_fetch_tickers(exchange)]
        if secondary:
            tasks.append(_fetch_tickers(secondary))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        primary_res = results[0]
        if isinstance(primary_res, Exception):
            raise primary_res
        tickers = primary_res
        if not isinstance(tickers, dict):
            raise TypeError("fetch_tickers returned invalid data")
        if secondary:
            sec_res = results[1]
            if not isinstance(sec_res, Exception) and isinstance(sec_res, dict):
                for sym, data in sec_res.items():
                    vol2 = data.get("quoteVolume")
                    if sym in tickers:
                        vol1 = tickers[sym].get("quoteVolume")
                        if vol2 is not None and (vol1 is None or float(vol2) > float(vol1)):
                            tickers[sym] = data
                    else:
                        tickers[sym] = data
        sol_pairs = await get_solana_liquid_pairs(min_volume_usd)
        for sym in sol_pairs:
            tickers.setdefault(sym, {"quoteVolume": min_volume_usd})
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Failed to fetch tickers: %s", exc)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PAIR_FILE, "w") as f:
            json.dump(old_map or {p: time.time() for p in old_pairs}, f, indent=2)
        return old_pairs
    finally:
        await _close_exchange(exchange)
        if secondary:
            await _close_exchange(secondary)

    pairs: list[Tuple[str, float]] = []
    for symbol, data in tickers.items():
        vol = data.get("quoteVolume")
        if vol is None:
            continue

        parts = symbol.split("/")
        if len(parts) != 2:
            continue
        base, quote = parts[0].upper(), parts[1].upper()

        if allowed_quotes and quote not in allowed_quotes:
            continue
        if base in blacklist:
            continue

        pairs.append((symbol, float(vol)))

    pairs.sort(key=lambda x: x[1], reverse=True)
    top_list = [sym for sym, vol in pairs if vol >= min_volume_usd][:top_k]
    top_map = {sym: time.time() for sym in top_list}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAIR_FILE, "w") as f:
        json.dump(top_map, f, indent=2)

    return top_list


def refresh_pairs(min_volume_usd: float, top_k: int, config: dict) -> List[str]:
    """Synchronous wrapper for :func:`refresh_pairs_async`."""
    return asyncio.run(refresh_pairs_async(min_volume_usd, top_k, config))


def main() -> None:
    """Main function with improved error handling."""
    try:
        cfg = load_config()
        rp_cfg = cfg.get("refresh_pairs", {})

        parser = argparse.ArgumentParser(description="Refresh liquid trading pairs")
        parser.add_argument("--once", action="store_true", help="Run once and exit")
        parser.add_argument(
            "--min-quote-volume-usd",
            type=float,
            default=float(rp_cfg.get("min_quote_volume_usd", DEFAULT_MIN_VOLUME_USD)),
        )
        parser.add_argument("--top-k", type=int, default=int(rp_cfg.get("top_k", DEFAULT_TOP_K)))
        parser.add_argument(
            "--refresh-interval",
            type=float,
            default=_parse_interval(rp_cfg.get("refresh_interval", DEFAULT_REFRESH_INTERVAL)),
            help="Refresh interval in seconds",
        )
        args = parser.parse_args()

        while True:
            try:
                pairs = refresh_pairs(args.min_quote_volume_usd, args.top_k, cfg)
                print(f"Updated {PAIR_FILE} with {len(pairs)} pairs")
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as exc:  # pragma: no cover - network failures
                print(f"Failed to refresh pairs: {exc}")
                logger.error("Failed to refresh pairs: %s", exc)
            if args.once:
                break
            time.sleep(args.refresh_interval)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        logger.error("Fatal error in main: %s", exc)
        raise


if __name__ == "__main__":
    main()
