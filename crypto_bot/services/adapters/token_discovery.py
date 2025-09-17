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
            url = self._base_url.rstrip("/") + "/scan/basic"
            payload = {
                "limit": int(dict(request.config).get("max_tokens_per_scan", 0) or 20)
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=20) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            tokens = list(data.get("tokens", []) or [])
                        else:
                            text = await resp.text()
                            logger.warning(
                                "Token discovery service returned %s: %s", resp.status, text
                            )
            except Exception as exc:  # pragma: no cover - network errors
                logger.warning("Token discovery service request failed: %s", exc)

        if tokens is None:
            raw_tokens = await get_solana_new_tokens(dict(request.config))
            tokens = list(raw_tokens or [])
        return TokenDiscoveryResponse(tokens=tokens)
