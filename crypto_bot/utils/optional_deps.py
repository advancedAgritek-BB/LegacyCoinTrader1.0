"""Utilities for optional third-party dependencies.

This module centralizes access to optional dependencies that may not be
available in every deployment environment. Each attribute should provide a
graceful fallback so that importing modules do not need to repeat defensive
try/except blocks.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class _NormProtocol(Protocol):
    """Protocol describing the subset of scipy.stats.norm we rely on."""

    def ppf(self, value: float) -> float:
        """Return the inverse cumulative distribution function."""


@runtime_checkable
class _StatsProtocol(Protocol):
    """Protocol describing the scipy.stats API surface we consume."""

    norm: _NormProtocol


try:  # pragma: no cover - optional dependency
    from scipy import stats as _scipy_stats  # type: ignore

    if not hasattr(_scipy_stats, "norm"):
        raise ImportError
except Exception:  # pragma: no cover - fallback when scipy missing
    class _Norm:
        @staticmethod
        def ppf(_value: float) -> float:
            return 0.0

    class _FallbackStats:
        norm: _NormProtocol = _Norm()

    scipy_stats: _StatsProtocol = _FallbackStats()
else:
    scipy_stats = _scipy_stats


__all__ = ["scipy_stats"]
