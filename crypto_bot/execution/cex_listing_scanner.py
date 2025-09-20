"""Discovery utilities for centrally listed trading pairs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import aiohttp

from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "cex_scanner.log")


@dataclass
class Listing:
    """Representation of a discovered exchange listing."""

    symbol: str
    base: str
    quote: str
    status: str
    raw: Dict[str, Any]


class CexListingScanner:
    """Detect new CEX trading pairs and persist the seen universe."""

    def __init__(
        self,
        *,
        exchange: str = "kraken",
        state_file: Optional[Union[str, Path]] = None,
        limit: int = 20,
    ) -> None:
        self.exchange = exchange.lower().strip() or "kraken"
        self.limit = max(1, limit)
        path = Path(state_file) if state_file is not None else LOG_DIR / "cex_scanner_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.state_file = path

    async def discover(self) -> List[Listing]:
        """Return newly listed pairs for the configured exchange."""

        if self.exchange == "kraken":
            listings = await self._fetch_kraken_pairs()
        else:
            logger.warning("Unsupported CEX exchange '%s' for discovery", self.exchange)
            return []

        state = self._load_state()
        seen_pairs = set(state.get("seen_pairs", []))
        initialised = bool(state.get("initialised", False))

        new_listings: list[Listing] = []
        for listing in listings:
            if listing.symbol not in seen_pairs:
                if initialised:
                    new_listings.append(listing)
                seen_pairs.add(listing.symbol)

        # Persist the updated universe
        state.update(
            {
                "seen_pairs": sorted(seen_pairs),
                "last_scan": datetime.now(timezone.utc).isoformat(),
                "exchange": self.exchange,
                "initialised": True,
            }
        )
        self._save_state(state)

        if not new_listings:
            logger.debug("No new %s listings detected", self.exchange)
            return []

        limited = new_listings[: self.limit]
        logger.info(
            "Discovered %s new %s listings: %s",
            len(limited),
            self.exchange,
            ", ".join(item.symbol for item in limited),
        )
        return limited

    async def _fetch_kraken_pairs(self) -> List[Listing]:
        url = "https://api.kraken.com/0/public/AssetPairs"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    resp.raise_for_status()
                    payload = await resp.json()
        except Exception as exc:
            logger.warning("Kraken AssetPairs fetch failed: %s", exc)
            return []

        result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
        listings: list[Listing] = []
        for pair_name, pair_info in result.items():
            if not isinstance(pair_info, Mapping):
                continue
            status = str(pair_info.get("status", "online"))
            if status.lower() not in {"online", "enabled", "trading"}:
                continue
            wsname = str(pair_info.get("wsname") or pair_info.get("altname") or pair_name)
            base = str(pair_info.get("base", ""))
            quote = str(pair_info.get("quote", ""))
            listings.append(
                Listing(
                    symbol=wsname,
                    base=base,
                    quote=quote,
                    status=status,
                    raw=dict(pair_info),
                )
            )

        listings.sort(key=lambda item: item.symbol)
        return listings

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            with self.state_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, Mapping):
                    return dict(data)
        except Exception as exc:
            logger.warning("Failed to load CEX scanner state: %s", exc)
        return {}

    def _save_state(self, state: Mapping[str, Any]) -> None:
        try:
            with self.state_file.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
        except Exception as exc:
            logger.warning("Failed to persist CEX scanner state: %s", exc)


async def get_cex_new_listings(cfg: Mapping[str, object]) -> List[Dict[str, Any]]:
    """Return newly detected CEX listings as dictionaries."""

    exchange = str(cfg.get("exchange", "kraken")).lower()
    limit = int(cfg.get("limit", 20) or 20)
    state_file = cfg.get("state_file")

    scanner = CexListingScanner(
        exchange=exchange,
        state_file=state_file,
        limit=limit,
    )
    listings = await scanner.discover()
    return [
        {
            "symbol": listing.symbol,
            "base": listing.base,
            "quote": listing.quote,
            "status": listing.status,
            "exchange": exchange,
            "raw": listing.raw,
        }
        for listing in listings
    ]


__all__ = ["CexListingScanner", "Listing", "get_cex_new_listings"]
