from __future__ import annotations

"""Redis-based event bus utilities for cross-service messaging."""

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Optional, Type, TypeVar

from pydantic import BaseModel
from redis import asyncio as redis_asyncio

from .contracts import EventEnvelope

LOGGER = logging.getLogger(__name__)

EventT = TypeVar("EventT", bound=EventEnvelope)


@dataclass(slots=True)
class RedisEventBus:
    """Publish events to Redis channels using a consistent schema."""

    client: redis_asyncio.Redis
    channel_prefix: str = "legacy"
    service_name: Optional[str] = None

    def _channel(self, channel: str) -> str:
        channel = channel.strip()
        if not channel:
            raise ValueError("channel must be provided")
        prefix = self.channel_prefix.rstrip(":")
        if not prefix:
            return channel
        if channel.startswith(f"{prefix}:"):
            return channel
        return f"{prefix}:{channel}" if not channel.startswith(":") else f"{prefix}{channel}"

    async def publish(self, channel: str, event: EventEnvelope | BaseModel | Mapping[str, Any]) -> None:
        """Publish an event payload to Redis."""

        payload = self._prepare_payload(event)
        redis_channel = self._channel(channel)
        await self.client.publish(redis_channel, payload)
        LOGGER.debug("Published event to %s", redis_channel)

    def _prepare_payload(self, event: EventEnvelope | BaseModel | Mapping[str, Any]) -> str:
        if isinstance(event, EventEnvelope):
            return event.model_dump_json()
        if isinstance(event, BaseModel):
            # Pydantic v2 exposes model_dump_json; fall back to json.dumps otherwise.
            if hasattr(event, "model_dump_json"):
                return event.model_dump_json()  # type: ignore[call-arg]
            return json.dumps(event.model_dump())
        if isinstance(event, Mapping):
            return json.dumps(dict(event))
        raise TypeError(f"Unsupported event payload type: {type(event)!r}")

    async def emit(self, channel: str, event_type: str, payload: Mapping[str, Any]) -> None:
        """Helper to publish ad-hoc events without constructing models."""

        envelope = EventEnvelope(
            event_type=event_type,
            source=self.service_name or "unknown",
            payload=dict(payload),
        )
        await self.publish(channel, envelope)

    def subscriber(self, channel: str, model: Type[EventT] | None = None) -> "RedisSubscriber[EventT]":
        return RedisSubscriber(self.client, self._channel(channel), model or EventEnvelope)  # type: ignore[arg-type]


class RedisSubscriber(AsyncIterator[EventT]):
    """Async iterator yielding events from Redis Pub/Sub channels."""

    def __init__(self, client: redis_asyncio.Redis, channel: str, model: Type[EventT]) -> None:
        self._client = client
        self._channel = channel
        self._model = model
        self._pubsub: Optional[redis_asyncio.client.PubSub] = None
        self._stopped = False

    async def __aenter__(self) -> "RedisSubscriber[EventT]":
        self._pubsub = self._client.pubsub()
        await self._pubsub.subscribe(self._channel)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - managed externally
        await self.stop()

    def __aiter__(self) -> "RedisSubscriber[EventT]":
        return self

    async def __anext__(self) -> EventT:
        if self._stopped:
            raise StopAsyncIteration
        if not self._pubsub:
            raise RuntimeError("Subscriber has not been entered")
        while not self._stopped:
            message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message is None:
                await asyncio.sleep(0.05)
                continue
            data = message.get("data")
            if data is None:
                continue
            try:
                if isinstance(data, bytes):
                    event = self._model.model_validate_json(data)
                elif isinstance(data, str):
                    event = self._model.model_validate_json(data)
                else:
                    event = self._model.model_validate(data)
                LOGGER.debug("Consumed event from %s", self._channel)
                return event
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning("Failed to decode event payload from %s: %s", self._channel, exc)
        raise StopAsyncIteration

    async def stop(self) -> None:
        self._stopped = True
        if self._pubsub:
            with contextlib.suppress(Exception):
                await self._pubsub.unsubscribe(self._channel)
                await self._pubsub.close()


__all__ = ["RedisEventBus", "RedisSubscriber"]
