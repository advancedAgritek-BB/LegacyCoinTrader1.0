"""In-process adapter for Solana token discovery."""

from __future__ import annotations

import logging
import os
from typing import Optional

import aiohttp

from crypto_bot.services.interfaces import (
    TokenDiscoveryRequest,
    TokenDiscoveryResponse,
    TokenDiscoveryService,
)
from crypto_bot.solana.scanner import get_solana_new_tokens

logger = logging.getLogger(__name__)


class TokenDiscoveryAdapter(TokenDiscoveryService):
    """Adapter for ``crypto_bot.solana.scanner`` with optional HTTP delegation."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self._base_url = base_url or os.getenv("TOKEN_DISCOVERY_SERVICE_URL")

    async def discover_tokens(
        self, request: TokenDiscoveryRequest
    ) -> TokenDiscoveryResponse:
        tokens: list[str] | None = None
        if self._base_url:
            # Try to get tokens from the token discovery service
            # First get Solana tokens
            solana_url = self._base_url.rstrip("/") + "/tokens/latest"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(solana_url, timeout=30) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            solana_tokens = list(data.get("tokens", []) or [])
                        else:
                            solana_tokens = []
            except Exception as exc:
                logger.warning("Solana token discovery service request failed: %s", exc)
                solana_tokens = []

            # Then get CEX tokens
            cex_url = self._base_url.rstrip("/") + "/cex/latest"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(cex_url, timeout=30) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            cex_tokens = list(data.get("tokens", []) or [])
                        else:
                            cex_tokens = []
            except Exception as exc:
                logger.warning("CEX token discovery service request failed: %s", exc)
                cex_tokens = []

            # Return separate token lists instead of combining them
            if solana_tokens or cex_tokens:
                tokens = solana_tokens + cex_tokens  # Keep combined for backward compatibility
                logger.info(f"Got {len(solana_tokens)} Solana and {len(cex_tokens)} CEX tokens from service")
                # Return response with separate lists
                return TokenDiscoveryResponse(
                    tokens=tokens,
                    dex_tokens=solana_tokens if solana_tokens else None,
                    cex_tokens=cex_tokens if cex_tokens else None
                )

        if tokens is None:
            base_config = dict(request.config or {})

            solana_cfg = dict(base_config)
            limit = int(solana_cfg.get("max_tokens_per_scan") or solana_cfg.get("limit") or 100)
            solana_cfg.setdefault("max_tokens_per_scan", limit)
            solana_cfg.setdefault("limit", limit)
            solana_cfg.setdefault("min_volume_usd", float(base_config.get("min_volume_usd", 0.0)))
            solana_cfg.setdefault("gecko_search", bool(base_config.get("gecko_search", True)))
            solana_cfg.setdefault("helius_key", solana_cfg.get("helius_key") or os.getenv("HELIUS_KEY", ""))
            solana_cfg.setdefault("raydium_api_key", solana_cfg.get("raydium_api_key") or os.getenv("RAYDIUM_API_KEY", ""))
            solana_cfg.setdefault("pump_fun_api_key", solana_cfg.get("pump_fun_api_key") or os.getenv("PUMP_FUN_API_KEY", ""))

            solana_tokens = await get_solana_new_tokens(solana_cfg)
            # Preserve order while removing duplicates
            solana_list = list(dict.fromkeys(solana_tokens or []))

            cex_limit = int(base_config.get("cex_scanner_limit") or base_config.get("cex_limit") or 200)
            try:
                from crypto_bot.execution.cex_listing_scanner import CexListingScanner

                scanner = CexListingScanner(exchange="kraken", limit=cex_limit)
                all_listings = await scanner._fetch_kraken_pairs()
                usd_listings = [listing for listing in all_listings if listing.symbol.endswith('/USD')]
                cex_tokens = [listing.symbol for listing in usd_listings[:cex_limit]]
                logger.info("Found %s Kraken USD pairs", len(cex_tokens))
            except Exception as exc:
                logger.warning("CEX token discovery failed: %s", exc)
                cex_tokens = []

            tokens = solana_list + [token for token in cex_tokens if token not in solana_list]

        return TokenDiscoveryResponse(
            tokens=tokens,
            dex_tokens=solana_list if solana_list else None,
            cex_tokens=cex_tokens if cex_tokens else None
        )
