from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from redis import asyncio as redis_asyncio
from redis.exceptions import RedisError

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after: int
    limit: int
    reset_after: int


class RateLimiter:
    """Redis-backed rate limiter with in-memory fallback."""

    def __init__(
        self,
        redis_client: Optional[redis_asyncio.Redis],
        default_limit: int,
        window_seconds: int,
    ) -> None:
        self.redis = redis_client
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self._lock = asyncio.Lock()
        self._fallback_store: dict[str, tuple[int, float]] = {}

    @property
    def uses_redis(self) -> bool:
        return self.redis is not None

    async def check(
        self,
        identifier: str,
        limit: Optional[int] = None,
    ) -> RateLimitResult:
        limit_value = limit if limit is not None else self.default_limit
        if limit_value <= 0:
            return RateLimitResult(True, limit_value, 0, limit_value, self.window_seconds)

        if self.redis is not None:
            try:
                return await self._check_with_redis(identifier, limit_value)
            except RedisError as exc:
                LOGGER.warning("Redis unavailable for rate limiting, falling back: %s", exc)
                self.redis = None

        return await self._check_in_memory(identifier, limit_value)

    async def _check_with_redis(self, identifier: str, limit: int) -> RateLimitResult:
        key = f"rate-limit:{identifier}"
        assert self.redis is not None

        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, self.window_seconds)
        ttl = await self.redis.ttl(key)
        remaining = max(0, limit - current)
        allowed = current <= limit
        retry_after = max(0, ttl)
        reset_after = retry_after if retry_after > 0 else self.window_seconds
        return RateLimitResult(allowed, remaining, retry_after, limit, reset_after)

    async def _check_in_memory(self, identifier: str, limit: int) -> RateLimitResult:
        async with self._lock:
            now = time.monotonic()
            count, expiry = self._fallback_store.get(identifier, (0, 0.0))
            if expiry <= now:
                count = 0
                expiry = now + self.window_seconds
            count += 1
            self._fallback_store[identifier] = (count, expiry)
            remaining = max(0, limit - count)
            allowed = count <= limit
            retry_after = max(0, int(expiry - now))
            reset_after = retry_after if retry_after > 0 else self.window_seconds
            return RateLimitResult(allowed, remaining, retry_after, limit, reset_after)

