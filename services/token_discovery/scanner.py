"""High level orchestration for Solana token discovery."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from crypto_bot.solana.scanner import get_solana_new_tokens
from crypto_bot.solana.enhanced_scanner import EnhancedSolanaScanner

from .config import Settings
from .publisher import DiscoveryPublisher

logger = logging.getLogger(__name__)


class TokenDiscoveryCoordinator:
    """Coordinate Solana token discovery and opportunity scoring."""

    def __init__(self, settings: Settings, publisher: DiscoveryPublisher) -> None:
        self._settings = settings
        self._publisher = publisher

        self._lock = asyncio.Lock()
        self._basic_task: Optional[asyncio.Task] = None
        self._enhanced_task: Optional[asyncio.Task] = None
        self._enhanced_scanner: Optional[EnhancedSolanaScanner] = None

        self._latest_tokens: list[str] = []
        self._latest_opportunities: list[dict[str, Any]] = []
        self._last_basic_scan: Optional[datetime] = None
        self._last_enhanced_scan: Optional[datetime] = None

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

    async def shutdown(self) -> None:
        """Stop background scanning loops and cleanup resources."""

        tasks = [task for task in (self._basic_task, self._enhanced_task) if task]
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

    async def run_basic_scan(self, limit: Optional[int] = None) -> list[str]:
        """Run a single basic Solana discovery scan."""

        scan_limit = int(limit or self._settings.solana_scanner_limit)
        config = self._build_basic_config(scan_limit)

        tokens = await get_solana_new_tokens(config)
        formatted = [self._format_token(token) for token in tokens]

        async with self._lock:
            self._latest_tokens = list(formatted)
            self._last_basic_scan = datetime.now(timezone.utc)

        await self._publisher.publish_tokens(
            formatted,
            source="basic",
            metadata={"limit": scan_limit},
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

    async def get_latest_tokens(self) -> list[str]:
        async with self._lock:
            return list(self._latest_tokens)

    async def get_latest_opportunities(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        async with self._lock:
            if limit is None:
                return [dict(opp) for opp in self._latest_opportunities]
            return [dict(opp) for opp in self._latest_opportunities[:limit]]

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
            "tokens_cached": len(self._latest_tokens),
            "opportunities_cached": len(self._latest_opportunities),
        }

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

    async def _get_enhanced_scanner(self) -> EnhancedSolanaScanner:
        if self._enhanced_scanner is None:
            config = self._build_enhanced_config()
            self._enhanced_scanner = EnhancedSolanaScanner(config)
        return self._enhanced_scanner

    def _build_basic_config(self, limit: int) -> Mapping[str, Any]:
        return {
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

    @staticmethod
    def _format_token(token: str) -> str:
        if "/" in token:
            return token
        return f"{token}/USDC"

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
