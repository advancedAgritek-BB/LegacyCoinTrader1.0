"""In-process adapter for Solana token discovery."""

from __future__ import annotations

from crypto_bot.services.interfaces import (
    TokenDiscoveryRequest,
    TokenDiscoveryResponse,
    TokenDiscoveryService,
)
from crypto_bot.solana.scanner import get_solana_new_tokens


class TokenDiscoveryAdapter(TokenDiscoveryService):
    """Adapter for :mod:`crypto_bot.solana.scanner`."""

    async def discover_tokens(
        self, request: TokenDiscoveryRequest
    ) -> TokenDiscoveryResponse:
        tokens = await get_solana_new_tokens(dict(request.config))
        return TokenDiscoveryResponse(tokens=list(tokens or []))
