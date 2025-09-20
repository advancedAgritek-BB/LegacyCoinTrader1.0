"""Asynchronous subscriber for market data events."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Mapping, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)

EventHandler = Callable[[Mapping[str, Any]], Awaitable[None] | None]


class MarketDataSubscriber:
    """Listen for market data events on a Redis pub/sub channel."""

    def __init__(self, client: redis.Redis, channel: str, handler: EventHandler) -> None:
        self._redis = client
        self._channel = channel
        self._handler = handler
        self._pubsub: Optional[redis.client.PubSub] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(self._channel)
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Market data subscriber listening on %s", self._channel)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:  # pragma: no cover - expected during shutdown
                pass
            self._task = None
        if self._pubsub:
            try:
                await self._pubsub.unsubscribe(self._channel)
            finally:
                await self._pubsub.close()
            self._pubsub = None

    async def _run(self) -> None:
        assert self._pubsub is not None
        while self._running:
            try:
                message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Market data subscriber error: %s", exc)
                await asyncio.sleep(1.0)
                continue

            if not message:
                await asyncio.sleep(0)
                continue

            data = message.get("data")
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            payload: Mapping[str, Any]
            if isinstance(data, str):
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = {"raw": data}
            elif isinstance(data, Mapping):
                payload = data
            else:
                payload = {"raw": data}

            try:
                result = self._handler(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover - handler errors shouldn't kill subscriber
                logger.debug("Market data handler raised an error", exc_info=True)


__all__ = ["MarketDataSubscriber"]
