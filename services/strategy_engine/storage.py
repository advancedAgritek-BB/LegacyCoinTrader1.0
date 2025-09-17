"""Model registry utilities for the strategy engine."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Lightweight registry for storing model metadata in Redis."""

    def __init__(self, client: redis.Redis, key_prefix: str) -> None:
        self._redis = client
        self._key_prefix = key_prefix

    async def touch(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "name": name,
            "touched_at": datetime.utcnow().isoformat(),
        }
        if metadata:
            payload.update(metadata)
        try:
            await self._redis.hset(self._key_prefix, name, json.dumps(payload))
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to store model metadata for %s", name, exc_info=True)

    async def get(self, name: str) -> Optional[Dict[str, Any]]:
        raw = await self._redis.hget(self._key_prefix, name)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Invalid model metadata for %s", name)
            return None

    async def list(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        async for key, raw in self._redis.hscan_iter(self._key_prefix):
            entry_key = key.decode("utf-8") if isinstance(key, bytes) else key
            value = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            try:
                results[entry_key] = json.loads(value)
            except json.JSONDecodeError:
                continue
        return results


__all__ = ["ModelRegistry"]
