from __future__ import annotations

import time
from collections import deque
from typing import Deque, Iterable, Tuple

__all__ = ["DeterministicClock"]


class DeterministicClock:
    """Deterministic stand-in for :mod:`time` helpers used in regression tests."""

    def __init__(self, durations: Iterable[float], *, start: float = 0.0) -> None:
        self._durations: Tuple[float, ...] = tuple(float(value) for value in durations)
        self._start = float(start)

        # Build the monotonic timeline with alternating start/end timestamps.
        timeline: Deque[float] = deque([self._start])
        cumulative = self._start
        for index, duration in enumerate(self._durations):
            cumulative += duration
            timeline.append(cumulative)
            if index < len(self._durations) - 1:
                timeline.append(cumulative)

        self._timeline = timeline
        self._monotonic_sequence = timeline  # Backwards-compatibility alias.
        self.timeline = timeline
        self._wall_start = time.time()
        self._current_monotonic = self._start
        self._sleep_fn = time.sleep

    def monotonic(self) -> float:
        if len(self._timeline) > 1:
            self._current_monotonic = self._timeline.popleft()
        else:
            self._current_monotonic = self._timeline[0]
        return self._current_monotonic

    def time(self) -> float:
        return self._wall_start + (self._current_monotonic - self._start)

    def sleep(self, seconds: float) -> None:
        self._sleep_fn(seconds)
