"""Utility to inspect the Solana mempool for high priority fees.

This module provides a helper that queries a priority fee API to gauge
network congestion. It falls back to the environment variable
``MOCK_PRIORITY_FEE`` so tests and offline runs can control the value.
"""

from __future__ import annotations

import os
import time
from typing import Optional, List
from collections import deque

import requests


class SolanaMempoolMonitor:
    """Simple monitor for Solana priority fees and volume tracking."""

    def __init__(self, priority_fee_url: Optional[str] = None) -> None:
        self.priority_fee_url = priority_fee_url or os.getenv(
            "SOLANA_PRIORITY_FEE_URL",
            "https://mempool.solana.com/api/v0/fees/priority_fee",
        )
        # Volume tracking
        self._volume_history: deque = deque(maxlen=100)  # Store last 100 volume entries
        self._last_volume_update = 0
        self._volume_update_interval = 60  # Update volume every 60 seconds

    def fetch_priority_fee(self) -> float:
        """Return the current priority fee per compute unit in micro lamports."""
        mock_fee = os.getenv("MOCK_PRIORITY_FEE")
        if mock_fee is not None:
            try:
                return float(mock_fee)
            except ValueError:
                return 0.0
        try:
            resp = requests.get(self.priority_fee_url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return float(data.get("priorityFee", 0.0))
        except Exception:
            pass
        return 0.0

    def is_suspicious(self, threshold: float) -> bool:
        """Return True when the priority fee exceeds ``threshold``."""
        fee = self.fetch_priority_fee()
        return fee >= threshold

    def get_recent_volume(self) -> float:
        """Return the most recent volume value."""
        # For now, return a mock value since we don't have real volume data
        # In a real implementation, this would fetch from a volume API
        mock_volume = os.getenv("MOCK_MEMPOOL_VOLUME", "1000000")
        try:
            return float(mock_volume)
        except ValueError:
            return 1000000.0

    def get_average_volume(self) -> float:
        """Return the average volume over the tracked period."""
        # For now, return a mock value since we don't have real volume data
        # In a real implementation, this would calculate from volume history
        mock_avg_volume = os.getenv("MOCK_MEMPOOL_AVG_VOLUME", "800000")
        try:
            return float(mock_avg_volume)
        except ValueError:
            return 800000.0

    def update_volume(self, volume: float) -> None:
        """Update the volume history with a new value."""
        current_time = time.time()
        if current_time - self._last_volume_update >= self._volume_update_interval:
            self._volume_history.append(volume)
            self._last_volume_update = current_time

    def get_volume_history(self) -> List[float]:
        """Return the list of tracked volume values."""
        return list(self._volume_history)
