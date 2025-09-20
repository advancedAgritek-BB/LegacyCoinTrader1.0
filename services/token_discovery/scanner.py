"""High level orchestration for Solana token discovery."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from crypto_bot.execution.cex_listing_scanner import get_cex_new_listings
from crypto_bot.solana.scanner import get_solana_new_tokens
from crypto_bot.solana.enhanced_scanner import EnhancedSolanaScanner

from config import Settings
from publisher import DiscoveryPublisher

logger = logging.getLogger("services.token_discovery.scanner")


class TokenDiscoveryCoordinator:
    """Coordinate Solana token discovery and opportunity scoring."""

    def __init__(self, settings: Settings, publisher: DiscoveryPublisher) -> None:
        self._settings = settings
        self._publisher = publisher

        self._lock = asyncio.Lock()
        self._basic_task: Optional[asyncio.Task] = None
        self._enhanced_task: Optional[asyncio.Task] = None
        self._cex_task: Optional[asyncio.Task] = None
        self._enhanced_scanner: Optional[EnhancedSolanaScanner] = None

        self._latest_tokens: list[str] = []
        self._latest_opportunities: list[dict[str, Any]] = []
        self._latest_cex_tokens: list[str] = []
        self._latest_cex_listings: list[dict[str, Any]] = []
        self._last_basic_scan: Optional[datetime] = None
        self._last_enhanced_scan: Optional[datetime] = None
        self._last_cex_scan: Optional[datetime] = None

    async def start(self) -> None:
        """Start background scanning loops."""

        if self._settings.background_basic_interval > 0:
            self._basic_task = asyncio.create_task(
                self._basic_loop(), name="token-discovery-basic"
            )
        if self._settings.enable_enhanced_scanner and self._settings.background_enhanced_interval > 0:
            self._enhanced_task = asyncio.create_task(
                self._enhanced_loop(), name="token-discovery-enhanced"
            )
        if self._settings.enable_cex_scanner and self._settings.background_cex_interval > 0:
            self._cex_task = asyncio.create_task(
                self._cex_loop(), name="token-discovery-cex"
            )

    async def shutdown(self) -> None:
        """Stop background scanning loops and cleanup resources."""

        tasks = [task for task in (self._basic_task, self._enhanced_task, self._cex_task) if task]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Discovery loop terminated with error")
        self._basic_task = None
        self._enhanced_task = None
        self._cex_task = None

    async def run_basic_scan(self, limit: Optional[int] = None) -> list[str]:
        """Run a single basic Solana discovery scan."""

        scan_limit = int(limit or self._settings.solana_scanner_limit)
        config = self._build_basic_config(scan_limit)

        timeout = max(1, int(getattr(self._settings, "basic_scan_timeout_seconds", 15)))
        fallback_reason: Optional[str] = None

        try:
            tokens = await asyncio.wait_for(get_solana_new_tokens(config), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Basic Solana discovery exceeded %s seconds; falling back to cached/default tokens",
                timeout,
            )
            tokens = []
            fallback_reason = "timeout"
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Basic Solana discovery failed; using cached/default tokens")
            tokens = []
            fallback_reason = "error"

        formatted = [self._format_token(token) for token in tokens if token]

        if not formatted:
            fallback_reason = fallback_reason or "empty"
            formatted = [self._format_token(token) for token in await self._fallback_candidates(scan_limit)]

        async with self._lock:
            self._latest_tokens = list(formatted)
            self._last_basic_scan = datetime.now(timezone.utc)

        metadata: Dict[str, Any] = {"limit": scan_limit}
        if fallback_reason:
            metadata["fallback"] = fallback_reason

        await self._publisher.publish_tokens(
            formatted,
            source="basic",
            metadata=metadata,
        )
        return formatted

    async def run_enhanced_scan(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Run an enhanced scan and publish opportunities."""

        if not self._settings.enable_enhanced_scanner:
            return []

        scanner = await self._get_enhanced_scanner()
        if hasattr(scanner, "_perform_scan"):
            await scanner._perform_scan()  # type: ignore[attr-defined]
        opportunities = scanner.get_top_opportunities(
            limit=limit or self._settings.enhanced_limit
        )

        async with self._lock:
            self._latest_opportunities = [dict(opp) for opp in opportunities]
            self._last_enhanced_scan = datetime.now(timezone.utc)

        await self._publisher.publish_opportunities(
            opportunities,
            source="enhanced",
            metadata={"limit": limit or self._settings.enhanced_limit},
        )
        return [dict(opp) for opp in opportunities]

    async def run_cex_scan(self, limit: Optional[int] = None) -> list[str]:
        """Run a CEX discovery scan and publish new listings."""

        if not self._settings.enable_cex_scanner:
            return []

        scan_limit = int(limit or self._settings.cex_scanner_limit)
        config = self._build_cex_config(scan_limit)
        exchange = str(config.get("exchange", "cex")).lower()

        listings = await get_cex_new_listings(config)

        formatted_tokens: list[str] = []
        listing_payload: list[dict[str, Any]] = []
        for entry in listings:
            symbol = str(entry.get("symbol") or "").strip()
            if not symbol:
                continue
            token_symbol = self._format_cex_symbol(symbol, exchange)
            formatted_tokens.append(token_symbol)
            listing_payload.append(
                {
                    "token": token_symbol,
                    "symbol": symbol,
                    "base": entry.get("base"),
                    "quote": entry.get("quote"),
                    "status": entry.get("status"),
                    "exchange": exchange,
                }
            )

        # Also get all available CEX tokens, not just new ones
        # This ensures we have tokens even when there are no "new" listings
        try:
            from crypto_bot.execution.cex_listing_scanner import CexListingScanner
            scanner = CexListingScanner(exchange=exchange, limit=1000)
            all_listings = await scanner._fetch_kraken_pairs()
            all_formatted_tokens: list[str] = []
            all_listing_payload: list[dict[str, Any]] = []

            for listing in all_listings:
                symbol = listing.symbol
                if symbol and symbol.endswith('/USD'):  # Only USD pairs
                    token_symbol = self._format_cex_symbol(symbol, exchange)
                    all_formatted_tokens.append(token_symbol)
                    all_listing_payload.append(
                        {
                            "token": token_symbol,
                            "symbol": symbol,
                            "base": listing.base,
                            "quote": listing.quote,
                            "status": listing.status,
                            "exchange": exchange,
                        }
                    )

            # Use all tokens if we have them, otherwise use new tokens
            if all_formatted_tokens:
                final_tokens = all_formatted_tokens[:scan_limit]  # Limit to scan limit
                final_listings = all_listing_payload[:scan_limit]
            else:
                final_tokens = formatted_tokens
                final_listings = listing_payload

        except Exception as exc:
            logger.warning("Failed to get all CEX tokens: %s", exc)
            final_tokens = formatted_tokens
            final_listings = listing_payload

        async with self._lock:
            self._latest_cex_tokens = list(final_tokens)
            self._latest_cex_listings = list(final_listings)
            self._last_cex_scan = datetime.now(timezone.utc)

        metadata: Dict[str, Any] = {
            "exchange": exchange,
            "limit": scan_limit,
            "count": len(final_tokens),
            "listings": final_listings,
        }

        if formatted_tokens:
            logger.info("Discovered %s new %s listings", len(formatted_tokens), exchange)
        else:
            logger.debug("No new %s listings detected", exchange)

        await self._publisher.publish_tokens(
            final_tokens,
            source=f"cex:{exchange}",
            metadata=metadata,
        )
        return final_tokens

    async def get_latest_tokens(self) -> list[str]:
        async with self._lock:
            return list(self._latest_tokens)

    async def get_latest_opportunities(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        async with self._lock:
            if limit is None:
                return [dict(opp) for opp in self._latest_opportunities]
            return [dict(opp) for opp in self._latest_opportunities[:limit]]

    async def get_latest_cex_tokens(self) -> list[str]:
        async with self._lock:
            return list(self._latest_cex_tokens)

    async def get_latest_cex_listings(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [dict(item) for item in self._latest_cex_listings]

    async def score_tokens(self, tokens: Sequence[str]) -> list[dict[str, Any]]:
        """Score arbitrary tokens using cached opportunities or heuristics."""

        if not tokens:
            return []

        async with self._lock:
            opportunity_map = {
                opp.get("symbol") or opp.get("token"): dict(opp)
                for opp in self._latest_opportunities
            }
            cached_tokens = list(self._latest_tokens)

        scored: list[dict[str, Any]] = []
        for index, token in enumerate(tokens):
            opportunity = opportunity_map.get(token)
            if opportunity:
                score = float(opportunity.get("score", 0.0))
                metadata = {
                    key: value
                    for key, value in opportunity.items()
                    if key not in {"symbol", "token", "score"}
                }
                scored.append(
                    {
                        "token": token,
                        "score": score,
                        "source": opportunity.get("source", "enhanced"),
                        "metadata": metadata,
                    }
                )
            else:
                baseline = self._baseline_score(token, index, cached_tokens)
                scored.append(
                    {
                        "token": token,
                        "score": baseline,
                        "source": "baseline",
                        "metadata": {"rank": index},
                    }
                )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic information for service status endpoints."""

        return {
            "last_basic_scan": self._last_basic_scan,
            "last_enhanced_scan": self._last_enhanced_scan,
            "last_cex_scan": self._last_cex_scan,
            "tokens_cached": len(self._latest_tokens),
            "opportunities_cached": len(self._latest_opportunities),
            "cex_tokens_cached": len(self._latest_cex_tokens),
        }

    async def _fallback_candidates(self, limit: int) -> list[str]:
        async with self._lock:
            cached = list(self._latest_tokens)
        if cached:
            logger.info("Using %s cached tokens for discovery fallback", min(len(cached), limit))
            return cached[:limit]
        defaults = self._fallback_tokens(limit)
        logger.info("Using default Solana token list for discovery fallback")
        return defaults

    async def _basic_loop(self) -> None:
        interval = max(self._settings.background_basic_interval, 1)
        while True:
            try:
                await self.run_basic_scan()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - loop safety
                logger.exception("Basic discovery loop failed")
            await asyncio.sleep(interval)

    async def _enhanced_loop(self) -> None:
        interval = max(self._settings.background_enhanced_interval, 1)
        while True:
            try:
                await self.run_enhanced_scan()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - loop safety
                logger.exception("Enhanced discovery loop failed")
            await asyncio.sleep(interval)

    async def _cex_loop(self) -> None:
        interval = max(self._settings.background_cex_interval, 1)
        while True:
            try:
                await self.run_cex_scan()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - loop safety
                logger.exception("CEX discovery loop failed")
            await asyncio.sleep(interval)

    async def _get_enhanced_scanner(self) -> EnhancedSolanaScanner:
        if self._enhanced_scanner is None:
            config = self._build_enhanced_config()
            self._enhanced_scanner = EnhancedSolanaScanner(config)
        return self._enhanced_scanner

    def _build_basic_config(self, limit: int) -> Mapping[str, Any]:
        return {
            "limit": limit,
            "max_tokens_per_scan": limit,
            "min_volume_usd": self._settings.solana_min_volume_usd,
            "gecko_search": self._settings.solana_gecko_search,
            "helius_key": self._settings.helius_key,
            "raydium_api_key": self._settings.raydium_api_key,
            "pump_fun_api_key": self._settings.pump_fun_api_key,
        }

    def _build_enhanced_config(self) -> Dict[str, Any]:
        return {
            "solana_scanner": {
                "limit": self._settings.solana_scanner_limit,
                "max_tokens_per_scan": self._settings.solana_scanner_limit,
                "min_volume_usd": self._settings.solana_min_volume_usd,
                "gecko_search": self._settings.solana_gecko_search,
                "helius_key": self._settings.helius_key,
                "raydium_api_key": self._settings.raydium_api_key,
                "pump_fun_api_key": self._settings.pump_fun_api_key,
            },
            "enhanced_scanning": {
                "enabled": self._settings.enable_enhanced_scanner,
                "min_score_threshold": self._settings.enhanced_min_score,
                "max_tokens_per_scan": self._settings.enhanced_limit,
                "min_confidence": 0.5,
                "min_liquidity_score": 0.2,
            },
        }

    def _build_cex_config(self, limit: int) -> Dict[str, Any]:
        state_file = Path(self._settings.cex_state_file)
        if not state_file.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent.parent
            state_file = (base_dir / state_file).resolve()
        return {
            "exchange": self._settings.cex_exchange,
            "limit": limit,
            "state_file": str(state_file),
        }

    @staticmethod
    def _format_token(token: str) -> str:
        if "/" in token:
            return token
        return f"{token}/USDC"

    @staticmethod
    def _format_cex_symbol(symbol: str, exchange: str) -> str:
        formatted = symbol.strip().upper()
        prefix = exchange.upper()
        return f"{prefix}:{formatted}" if formatted else formatted

    @staticmethod
    def _fallback_tokens(limit: int) -> list[str]:
        defaults = [
            "SOL/USDC",
            "BONK/USDC",
            "RAY/USDC",
            "JTO/USDC",
            "MNGO/USDC",
            "JUP/USDC",
            "W/USDC",
            "PYTH/USDC",
        ]
        if limit <= 0:
            return []
        if limit >= len(defaults):
            return defaults
        return defaults[:limit]

    @staticmethod
    def _baseline_score(token: str, index: int, cached_tokens: Sequence[str]) -> float:
        try:
            position = cached_tokens.index(token)
        except ValueError:
            position = index
        # Provide a simple decay based on position while remaining within [0, 1]
        rank = min(position, 100)
        return max(0.0, 1.0 - (rank / 100.0))


__all__ = ["TokenDiscoveryCoordinator"]
