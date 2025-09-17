"""Client-side utilities for consuming token discovery events."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence

import redis.asyncio as redis

try:  # pragma: no cover - optional dependency
    from aiokafka import AIOKafkaConsumer
except Exception:  # pragma: no cover - aiokafka optional
    AIOKafkaConsumer = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


@dataclass
class FeedSettings:
    """Configuration for the discovery feed consumer."""

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_use_ssl: bool = False
    redis_channel_tokens: str = "trading_engine.token_candidates"
    redis_channel_opportunities: str = "trading_engine.token_opportunities"

    kafka_enabled: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_group_id: str = "token-discovery-consumer"
    kafka_topic_tokens: str = "token-discovery.tokens"
    kafka_topic_opportunities: str = "token-discovery.opportunities"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedSettings":
        params = dict(data)
        return cls(
            redis_host=str(params.get("redis_host", cls.redis_host)),
            redis_port=int(params.get("redis_port", cls.redis_port)),
            redis_db=int(params.get("redis_db", cls.redis_db)),
            redis_use_ssl=bool(params.get("redis_use_ssl", cls.redis_use_ssl)),
            redis_channel_tokens=str(
                params.get("redis_channel_tokens", cls.redis_channel_tokens)
            ),
            redis_channel_opportunities=str(
                params.get(
                    "redis_channel_opportunities",
                    cls.redis_channel_opportunities,
                )
            ),
            kafka_enabled=bool(params.get("kafka_enabled", cls.kafka_enabled)),
            kafka_bootstrap_servers=str(
                params.get("kafka_bootstrap_servers", cls.kafka_bootstrap_servers)
            ),
            kafka_group_id=str(params.get("kafka_group_id", cls.kafka_group_id)),
            kafka_topic_tokens=str(
                params.get("kafka_topic_tokens", cls.kafka_topic_tokens)
            ),
            kafka_topic_opportunities=str(
                params.get(
                    "kafka_topic_opportunities", cls.kafka_topic_opportunities
                )
            ),
        )

    def redis_dsn(self) -> str:
        protocol = "rediss" if self.redis_use_ssl else "redis"
        return f"{protocol}://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class SolanaDiscoveryFeed:
    """Asynchronous consumer for token discovery events."""

    def __init__(self, settings: FeedSettings) -> None:
        self._settings = settings
        self._redis: Optional[redis.Redis] = None
        self._redis_task: Optional[asyncio.Task] = None
        self._redis_pubsub: Optional[redis.client.PubSub] = None

        self._kafka_consumer: Optional[AIOKafkaConsumer] = None
        self._kafka_task: Optional[asyncio.Task] = None

        self._pending_tokens: Deque[str] = deque()
        self._recent_batches: Deque[List[str]] = deque(maxlen=5)
        self._opportunity_map: Dict[str, Dict[str, Any]] = {}

        self._lock = asyncio.Lock()
        self._closed = asyncio.Event()

    async def start(self) -> None:
        """Connect to messaging backends and start listeners."""

        if self._redis is None:
            self._redis = redis.from_url(
                self._settings.redis_dsn(),
                encoding="utf-8",
                decode_responses=True,
                health_check_interval=30,
            )
            try:
                await self._redis.ping()
            except Exception as exc:
                logger.warning("Unable to connect to Redis feed: %s", exc)
                await self._redis.close()
                self._redis = None
            else:
                self._redis_pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
                await self._redis_pubsub.subscribe(
                    self._settings.redis_channel_tokens,
                    self._settings.redis_channel_opportunities,
                )
                self._redis_task = asyncio.create_task(
                    self._redis_listener(), name="solana-feed-redis"
                )

        if self._settings.kafka_enabled and AIOKafkaConsumer is not None:
            self._kafka_consumer = AIOKafkaConsumer(
                self._settings.kafka_topic_tokens,
                self._settings.kafka_topic_opportunities,
                bootstrap_servers=self._settings.kafka_bootstrap_servers,
                group_id=self._settings.kafka_group_id,
                enable_auto_commit=True,
                auto_offset_reset="latest",
            )
            try:
                await self._kafka_consumer.start()
            except Exception as exc:
                logger.warning("Unable to start Kafka consumer: %s", exc)
                self._kafka_consumer = None
            else:
                self._kafka_task = asyncio.create_task(
                    self._kafka_listener(), name="solana-feed-kafka"
                )

    async def close(self) -> None:
        """Stop background listeners and close connections."""

        self._closed.set()

        if self._redis_task:
            self._redis_task.cancel()
            try:
                await self._redis_task
            except asyncio.CancelledError:
                pass
        self._redis_task = None

        if self._redis_pubsub is not None:
            with contextlib.suppress(Exception):
                await self._redis_pubsub.close()
            self._redis_pubsub = None

        if self._redis is not None:
            await self._redis.close()
            self._redis = None

        if self._kafka_task:
            self._kafka_task.cancel()
            try:
                await self._kafka_task
            except asyncio.CancelledError:
                pass
        self._kafka_task = None

        if self._kafka_consumer is not None:
            try:
                await self._kafka_consumer.stop()
            except Exception:  # pragma: no cover - best effort
                logger.debug("Failed to stop Kafka consumer", exc_info=True)
            self._kafka_consumer = None

    async def fetch_tokens(self, limit: Optional[int] = None) -> List[str]:
        """Return queued tokens from the discovery stream."""

        async with self._lock:
            tokens: List[str] = []
            while self._pending_tokens and (limit is None or len(tokens) < limit):
                tokens.append(self._pending_tokens.popleft())
            return tokens

    async def get_opportunities(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        async with self._lock:
            items = list(self._opportunity_map.values())
        items.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if limit is None:
            return [dict(item) for item in items]
        return [dict(item) for item in items[:limit]]

    async def score_tokens(self, tokens: Sequence[str]) -> List[Dict[str, Any]]:
        async with self._lock:
            opportunity_map = dict(self._opportunity_map)
            recent_flat: List[str] = [
                token
                for batch in self._recent_batches
                for token in batch
            ]

        scored: List[Dict[str, Any]] = []
        for index, token in enumerate(tokens):
            opportunity = opportunity_map.get(token)
            if opportunity:
                metadata = {
                    key: value
                    for key, value in opportunity.items()
                    if key not in {"symbol", "token", "score", "source"}
                }
                scored.append(
                    {
                        "token": token,
                        "score": float(opportunity.get("score", 0.0)),
                        "source": opportunity.get("source", "enhanced"),
                        "metadata": metadata,
                    }
                )
            else:
                try:
                    rank = recent_flat.index(token)
                except ValueError:
                    rank = index
                baseline = max(0.0, 1.0 - min(rank, 100) / 100.0)
                scored.append(
                    {
                        "token": token,
                        "score": baseline,
                        "source": "baseline",
                        "metadata": {"rank": rank},
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored

    async def _redis_listener(self) -> None:
        assert self._redis_pubsub is not None
        try:
            async for message in self._redis_pubsub.listen():
                if message is None:
                    continue
                data = message.get("data")
                if data is None:
                    continue
                await self._handle_payload(data)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Redis discovery listener failed")
        finally:
            if self._redis_pubsub is not None:
                with contextlib.suppress(Exception):
                    await self._redis_pubsub.close()
                self._redis_pubsub = None

    async def _kafka_listener(self) -> None:
        assert self._kafka_consumer is not None
        try:
            async for msg in self._kafka_consumer:
                await self._handle_payload(msg.value.decode("utf-8"))
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Kafka discovery listener failed")

    async def _handle_payload(self, payload: Any) -> None:
        try:
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode("utf-8")
            if isinstance(payload, str):
                data = json.loads(payload)
            elif isinstance(payload, dict):
                data = payload
            else:
                return
        except (json.JSONDecodeError, TypeError):
            logger.debug("Ignoring malformed discovery payload: %r", payload)
            return

        payload_type = data.get("type")
        if payload_type == "tokens":
            tokens = [str(token) for token in data.get("tokens", []) if token]
            if not tokens:
                return
            async with self._lock:
                for token in tokens:
                    self._pending_tokens.append(token)
                self._recent_batches.append(tokens)
        elif payload_type == "opportunities":
            opportunities = data.get("opportunities", [])
            if not isinstance(opportunities, list):
                return
            async with self._lock:
                for opportunity in opportunities:
                    if not isinstance(opportunity, dict):
                        continue
                    symbol = str(
                        opportunity.get("symbol")
                        or opportunity.get("token")
                        or ""
                    )
                    if not symbol:
                        continue
                    normalized = dict(opportunity)
                    normalized.setdefault("source", data.get("source", "enhanced"))
                    self._opportunity_map[symbol] = normalized


__all__ = ["FeedSettings", "SolanaDiscoveryFeed"]
