from __future__ import annotations

"""Fetch newly launched Solana tokens from public APIs."""

import asyncio
import logging
from typing import List, Union, Optional, Tuple
import aiohttp
import ccxt.async_support as ccxt

from . import symbol_scoring

logger = logging.getLogger(__name__)

# Global min volume filter updated by ``get_solana_new_tokens``
_MIN_VOLUME_USD = 0.0

RAYDIUM_URL = "https://api.raydium.io/pairs"
PUMP_FUN_URL = "https://client-api.prod.pump.fun/v1/launches"


async def search_geckoterminal_token(query: str) -> Optional[Tuple[str, float]]:
    """Return ``(mint, volume)`` from GeckoTerminal token search.

    The function queries ``/api/v2/search/tokens`` with ``query`` and
    ``network=solana`` and returns the first result's address and 24h
    volume in USD. ``None`` is returned when the request fails or no
    results are available.
    """

    from urllib.parse import quote_plus

    url = (
        "https://api.geckoterminal.com/api/v2/search/tokens"
        f"?query={quote_plus(query)}&network=solana"
    )

    data = await _fetch_json(url)
    if not data:
        return None

    items = data.get("data") if isinstance(data, dict) else []
    if not isinstance(items, list) or not items:
        return None

    item = items[0]
    attrs = item.get("attributes", {}) if isinstance(item, dict) else {}
    mint = str(attrs.get("address") or item.get("id") or query)
    try:
        volume = float(attrs.get("volume_usd_h24") or 0.0)
    except Exception:
        volume = 0.0

    return mint, volume


async def _fetch_json(url: str) -> Optional[Union[list, dict]]:
    """Return parsed JSON from ``url`` using ``aiohttp``."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.error("Solana scanner request failed: %s", exc)
        return None


async def _close_exchange(exchange) -> None:
    """Close ``exchange`` ignoring errors."""
    close = getattr(exchange, "close", None)
    if close:
        try:
            if asyncio.iscoroutinefunction(close):
                await close()
            else:
                close()
        except Exception:  # pragma: no cover - best effort
            pass


def _extract_tokens(data: Union[list, dict]) -> List[str]:
    """Return token mints from ``data`` respecting ``_MIN_VOLUME_USD``."""
    items = data.get("data") if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return []

    results: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mint = (
            item.get("tokenMint")
            or item.get("token_mint")
            or item.get("mint")
            or item.get("address")
        )
        if not mint:
            continue
        vol = (
            item.get("volumeUsd")
            or item.get("volume_usd")
            or item.get("liquidityUsd")
            or item.get("liquidity_usd")
            or 0.0
        )
        try:
            volume = float(vol)
        except Exception:
            volume = 0.0
        if volume >= _MIN_VOLUME_USD:
            results.append(str(mint))
    return results


async def fetch_new_raydium_pools(api_key: str, limit: int) -> List[str]:
    """Return new Raydium pool token mints."""
    url = f"{RAYDIUM_URL}?apiKey={api_key}&limit={limit}"
    data = await _fetch_json(url)
    if not data:
        return []
    tokens = _extract_tokens(data)
    return tokens[:limit]


async def fetch_pump_fun_launches(api_key: str, limit: int) -> List[str]:
    """Return recent Pump.fun launches."""
    url = f"{PUMP_FUN_URL}?api-key={api_key}&limit={limit}"
    data = await _fetch_json(url)
    if not data:
        return []
    tokens = _extract_tokens(data)
    return tokens[:limit]


async def _fallback_token_discovery(limit: int) -> List[str]:
    """Fallback token discovery when primary APIs are not available."""

    # Default popular Solana tokens to use as fallback
    fallback_tokens = [
        "So11111111111111111111111111111112",  # SOL
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
        "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",   # JUP
        "7xKXtg2CW87ZdacwQCUJLrf4VYJrFCBAcHX7ebUQWV2w",  # PYTH
        "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So", # mSOL
        "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", # BONK
        "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7AR",  # stSOL
        "HxhWkVpk5NS4Ltg5nijHJEifHodS6z4QcK5poe9JQ5LK", # HXRO (DEPRECATED)
        "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt",  # SRM
        "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E", # BTC
        "2FPyTwcZLUg1MDrwsyoP4D6s1tM7hAkHYRjkNb5w6Px", # ETH
        "MNDEFzGvMt87ueuHvVU9VcTqsAP5b3fTGPsHuuPA5ey",   # MNDE
        "ATLASXmbPQxBUYbxPsV97usA3fPQYEqzQBUHgiFCUsXx", # ATLAS
        "EP2aYBDD4WvdhnwWLUMyqU49g9k9eC8jNn7Cvf8d2AfT", # ORCA
        "EchesyfXePKdLtoiZSL8pBe8Myagyy8ZRqsACNCFGnvp", # FIDA
        "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R", # RAY
        "8HGyAAB1yoM1ttS7pXjHMa3dukTFGQggnFFH3hJZgzQh", #COPE
        "AGFEad2et2ZJif9jaGpdMixQqvW5i81aBdvKe7PHNfz3", #FTT
        "9S4t2NEAiJVMvPdRYKVrfJpBafPBLtvb6YXGbHqjDjT",   #COPE
    ]

    # Return formatted tokens
    formatted_tokens = [f"{token}/USDC" for token in fallback_tokens[:limit]]
    logger.info(f"Using {len(formatted_tokens)} fallback tokens for discovery")
    return formatted_tokens


async def get_solana_new_tokens(config: dict) -> List[str]:
    """Return deduplicated Solana token symbols from multiple sources."""

    global _MIN_VOLUME_USD

    limit = int(config.get("max_tokens_per_scan", 0)) or 20
    _MIN_VOLUME_USD = float(config.get("min_volume_usd", 0.0))
    raydium_key = str(config.get("raydium_api_key", ""))
    pump_key = str(config.get("pump_fun_api_key", ""))
    gecko_search = bool(config.get("gecko_search", True))

    tasks = []
    if raydium_key:
        coro = fetch_new_raydium_pools(raydium_key, limit)
        if not asyncio.iscoroutine(coro):
            async def _wrap(res=coro):
                return res
            coro = _wrap()
        tasks.append(coro)
    if pump_key:
        coro = fetch_pump_fun_launches(pump_key, limit)
        if not asyncio.iscoroutine(coro):
            async def _wrap(res=coro):
                return res
            coro = _wrap()
        tasks.append(coro)

    if not tasks:
        logger.info("No Solana API keys configured, using fallback token sources")
        return await _fallback_token_discovery(limit)

    try:
        # Add timeout to prevent hanging
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Solana scanner timed out after 30 seconds")
        return []
    except Exception as exc:
        logger.error("Solana scanner failed: %s", exc)
        return []
    candidates: List[str] = []
    seen_raw: set[str] = set()
    for res in results:
        for mint in res:
            if mint not in seen_raw:
                seen_raw.add(mint)
                candidates.append(mint)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break

    if not gecko_search:
        return [f"{m}/USDC" for m in candidates]

    try:
        # Add timeout to prevent hanging on GeckoTerminal searches
        search_results = await asyncio.wait_for(
            asyncio.gather(*[search_geckoterminal_token(m) for m in candidates]),
            timeout=20.0
        )
    except asyncio.TimeoutError:
        logger.warning("GeckoTerminal search timed out after 20 seconds, skipping volume filtering")
        # Filter by minimum volume if we have volume data from other sources
        filtered_candidates = []
        for mint in candidates:
            # For now, include all candidates since we can't verify volume without GeckoTerminal
            if len(filtered_candidates) < limit:
                filtered_candidates.append(mint)
        return [f"{m}/USDC" for m in filtered_candidates]
    except Exception as exc:
        logger.error("GeckoTerminal search failed: %s", exc)
        # Filter by minimum volume if we have volume data from other sources
        filtered_candidates = []
        for mint in candidates:
            # For now, include all candidates since we can't verify volume without GeckoTerminal
            if len(filtered_candidates) < limit:
                filtered_candidates.append(mint)
        return [f"{m}/USDC" for m in filtered_candidates]

    final: list[Tuple[str, float]] = []
    seen: set[str] = set()
    for res in search_results:
        if not res:
            continue
        mint, vol = res
        if vol >= _MIN_VOLUME_USD and mint not in seen:
            seen.add(mint)
            final.append((f"{mint}/USDC", vol))
        if len(final) >= limit:
            break

    if not final:
        return []

    min_score = float(config.get("min_symbol_score", 0.0))
    ex_name = str(config.get("exchange", "kraken")).lower()
    exchange_cls = getattr(ccxt, ex_name)
    exchange = exchange_cls({"enableRateLimit": True})

    try:
        scores = await asyncio.gather(
            *[
                symbol_scoring.score_symbol(
                    exchange, sym, vol, 0.0, 0.0, 1.0, config
                )
                for sym, vol in final
            ]
        )
    finally:
        await _close_exchange(exchange)

    scored = [
        (sym, score)
        for (sym, _), score in zip(final, scores)
        if score >= min_score
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in scored]
