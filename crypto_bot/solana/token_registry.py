"""Utility helpers for working with the Solana token registry."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import aiohttp

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "token_registry.log")

_JUPITER_CACHE: List[Dict[str, Any]] = []
_JUPITER_CACHE_EXPIRY: float = 0.0
_JUPITER_CACHE_TTL = 300.0  # five minutes
_JUPITER_CACHE_LOCK = asyncio.Lock()

_JUPITER_URL = "https://token.jup.ag/all"


async def get_jupiter_registry(
    *, force_refresh: bool = False, session: Optional[aiohttp.ClientSession] = None
) -> List[Dict[str, Any]]:
    """Return the cached Jupiter token registry, refreshing when stale."""

    global _JUPITER_CACHE_EXPIRY, _JUPITER_CACHE

    now = time.time()
    if (
        not force_refresh
        and _JUPITER_CACHE
        and now < _JUPITER_CACHE_EXPIRY
    ):
        return list(_JUPITER_CACHE)

    async with _JUPITER_CACHE_LOCK:
        now = time.time()
        if (
            not force_refresh
            and _JUPITER_CACHE
            and now < _JUPITER_CACHE_EXPIRY
        ):
            return list(_JUPITER_CACHE)

        close_session = False
        client = session
        if client is None:
            client = aiohttp.ClientSession()
            close_session = True

        try:
            async with client.get(_JUPITER_URL, timeout=15) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        except Exception as exc:
            logger.warning("Failed to refresh Jupiter registry: %s", exc)
            return list(_JUPITER_CACHE)
        finally:
            if close_session:
                await client.close()

        if isinstance(payload, list):
            _JUPITER_CACHE = [dict(item) for item in payload if isinstance(item, Mapping)]
        else:
            _JUPITER_CACHE = []

        _JUPITER_CACHE_EXPIRY = time.time() + _JUPITER_CACHE_TTL
        logger.info("Cached %s Jupiter tokens", len(_JUPITER_CACHE))
        return list(_JUPITER_CACHE)


def index_registry_by_symbol(
    registry: Iterable[Mapping[str, Any]], *, chain_id: Optional[int] = 101
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a case-insensitive index of registry entries by symbol."""

    index: Dict[str, List[Dict[str, Any]]] = {}
    for entry in registry:
        if not isinstance(entry, Mapping):
            continue
        if chain_id is not None:
            try:
                current_chain = int(entry.get("chainId", chain_id))
            except (TypeError, ValueError):
                continue
            if current_chain != chain_id:
                continue
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        index.setdefault(symbol, []).append(dict(entry))
    return index


def index_registry_by_mint(
    registry: Iterable[Mapping[str, Any]], *, chain_id: Optional[int] = 101
) -> Dict[str, Dict[str, Any]]:
    """Build an index keyed by mint address for fast lookups."""

    index: Dict[str, Dict[str, Any]] = {}
    for entry in registry:
        if not isinstance(entry, Mapping):
            continue
        if chain_id is not None:
            try:
                current_chain = int(entry.get("chainId", chain_id))
            except (TypeError, ValueError):
                continue
            if current_chain != chain_id:
                continue
        mint = str(entry.get("address") or "").strip()
        if not mint:
            continue
        index[mint] = dict(entry)
    return index


_MINT_INDEX: Dict[str, Dict[str, Any]] = {}
_MINT_INDEX_EXPIRY: float = 0.0
_MINT_INDEX_TTL = 300.0


async def get_symbol_for_mint(mint: str) -> Optional[str]:
    """Return the symbol for a given mint address using the cached registry."""

    global _MINT_INDEX, _MINT_INDEX_EXPIRY

    mint = mint.strip()
    if not mint:
        return None

    now = time.time()
    if _MINT_INDEX and now < _MINT_INDEX_EXPIRY:
        entry = _MINT_INDEX.get(mint)
        return str(entry.get("symbol")) if entry else None

    registry = await get_jupiter_registry()
    _MINT_INDEX = index_registry_by_mint(registry)
    _MINT_INDEX_EXPIRY = time.time() + _MINT_INDEX_TTL
    entry = _MINT_INDEX.get(mint)
    if entry:
        symbol = entry.get("symbol")
        return str(symbol) if symbol else None
    return None


__all__ = [
    "get_jupiter_registry",
    "index_registry_by_symbol",
    "index_registry_by_mint",
    "get_symbol_for_mint",
]
