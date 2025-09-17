"""Light-weight asynchronous message topics used for order events."""

from __future__ import annotations

import asyncio
from typing import Generic, Iterable, Optional, Set, TypeVar

T = TypeVar("T")


class TopicSubscription(Generic[T]):
    """Handle used by consumers to receive messages from a topic."""

    def __init__(self, topic: "AsyncTopic[T]", queue: "asyncio.Queue[T]") -> None:
        self._topic = topic
        self._queue = queue
        self._closed = False

    async def get(self) -> T:
        """Return the next message, blocking until one is available."""
        return await self._queue.get()

    def close(self) -> None:
        """Unsubscribe the consumer from the topic."""
        if not self._closed:
            self._topic._unsubscribe(self._queue)
            self._closed = True

    def __aiter__(self):  # pragma: no cover - convenience iterator
        return self

    async def __anext__(self):  # pragma: no cover - convenience iterator
        try:
            return await self.get()
        except asyncio.CancelledError as exc:
            self.close()
            raise StopAsyncIteration from exc


class AsyncTopic(Generic[T]):
    """Publish/subscribe primitive backed by asyncio queues."""

    def __init__(self, *, max_queue: int = 0) -> None:
        self._subscribers: Set[asyncio.Queue[T]] = set()
        self._max_queue = max_queue
        self._lock = asyncio.Lock()

    async def publish(self, message: T) -> None:
        """Publish ``message`` to all subscribers."""
        async with self._lock:
            queues: Iterable[asyncio.Queue[T]] = list(self._subscribers)
        for queue in queues:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:  # pragma: no cover - defensive branch
                await queue.put(message)

    def subscribe(self) -> TopicSubscription[T]:
        """Register a new subscriber and return its queue wrapper."""
        queue: asyncio.Queue[T] = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.add(queue)
        return TopicSubscription(self, queue)

    def _unsubscribe(self, queue: asyncio.Queue[T]) -> None:
        self._subscribers.discard(queue)

    def __len__(self) -> int:  # pragma: no cover - debug helper
        return len(self._subscribers)
