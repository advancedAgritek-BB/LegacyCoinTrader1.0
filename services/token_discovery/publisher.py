"""Utilities for publishing discovery results to Redis and Kafka."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional, Sequence

import redis.asyncio as redis

try:  # pragma: no cover - optional dependency
    from aiokafka import AIOKafkaProducer
except Exception:  # pragma: no cover - aiokafka optional
    AIOKafkaProducer = None  # type: ignore[misc,assignment]

from .config import Settings

logger = logging.getLogger(__name__)


class DiscoveryPublisher:
    """Publish discovery results to downstream messaging backends."""

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings
        self._kafka_producer: Optional[AIOKafkaProducer] = None
        self._kafka_lock = asyncio.Lock()

    async def close(self) -> None:
        """Clean up resources held by the publisher."""

        if self._kafka_producer is not None:
            try:
                await self._kafka_producer.stop()
            except Exception:  # pragma: no cover - best effort
                logger.debug("Failed to stop Kafka producer", exc_info=True)
            self._kafka_producer = None

    async def publish_tokens(
        self,
        tokens: Sequence[str],
        *,
        source: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Publish discovered tokens to Redis and Kafka."""

        if not tokens:
            return

        payload = {
            "type": "tokens",
            "source": source,
            "tokens": list(tokens),
            "metadata": dict(metadata or {}),
            "published_at": datetime.now(timezone.utc).isoformat(),
        }

        message = json.dumps(payload)
        await self._publish_to_redis(message, self._settings.redis_channel_tokens)
        await self._publish_to_kafka(message, self._settings.kafka_topic_tokens)

    async def publish_opportunities(
        self,
        opportunities: Sequence[MutableMapping[str, object] | Mapping[str, object]],
        *,
        source: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Publish scored opportunities to Redis and Kafka."""

        if not opportunities:
            return

        serialisable: list[dict[str, object]] = []
        for item in opportunities:
            serialisable.append(dict(item))

        payload = {
            "type": "opportunities",
            "source": source,
            "opportunities": serialisable,
            "metadata": dict(metadata or {}),
            "published_at": datetime.now(timezone.utc).isoformat(),
        }

        message = json.dumps(payload)
        await self._publish_to_redis(message, self._settings.redis_channel_opportunities)
        await self._publish_to_kafka(message, self._settings.kafka_topic_opportunities)

    async def _publish_to_redis(self, message: str, channel: str) -> None:
        try:
            await self._redis.publish(channel, message)
            logger.debug("Published message to Redis channel %s", channel)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to publish to Redis channel %s: %s", channel, exc)

    async def _publish_to_kafka(self, message: str, topic: str) -> None:
        if not self._settings.kafka_enabled:
            return
        if AIOKafkaProducer is None:
            logger.warning("Kafka publishing requested but aiokafka is not installed")
            return

        producer = await self._ensure_kafka_producer()
        if producer is None:
            return
        try:
            await producer.send_and_wait(topic, message.encode("utf-8"))
            logger.debug("Published message to Kafka topic %s", topic)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to publish to Kafka topic %s: %s", topic, exc)

    async def _ensure_kafka_producer(self) -> Optional[AIOKafkaProducer]:
        if not self._settings.kafka_enabled:
            return None
        if AIOKafkaProducer is None:
            return None
        async with self._kafka_lock:
            if self._kafka_producer is not None:
                return self._kafka_producer
            producer = AIOKafkaProducer(
                bootstrap_servers=self._settings.kafka_bootstrap_servers,
                client_id=self._settings.kafka_client_id,
            )
            try:
                await producer.start()
            except Exception as exc:  # pragma: no cover - connection errors
                logger.warning("Unable to start Kafka producer: %s", exc)
                return None
            self._kafka_producer = producer
            return producer


__all__ = ["DiscoveryPublisher"]
