"""Utilities for producing monotonically increasing API nonces."""

from __future__ import annotations

import asyncio
import time
from threading import Lock


class NonceManager:
    """Generate strictly increasing millisecond precision nonces."""

    def __init__(self, buffer_ms: int = 150) -> None:
        self._buffer_ms = max(0, int(buffer_ms))
        self._lock = Lock()
        self._last_nonce = 0

    def next_nonce(self) -> int:
        """Return the next nonce, ensuring strict monotonicity."""
        with self._lock:
            candidate = int(time.time() * 1000) + self._buffer_ms
            if candidate <= self._last_nonce:
                candidate = self._last_nonce + 1
            self._last_nonce = candidate
            return candidate

    async def next_nonce_async(self) -> int:
        """Async helper delegating to :meth:`next_nonce`."""
        return await asyncio.to_thread(self.next_nonce)
