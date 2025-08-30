"""Lightweight package initializer.

Avoid importing heavy subpackages at import time to keep unit tests fast
and prevent optional dependency issues during collection. Modules should
import what they need directly rather than relying on package side effects.
"""

__all__ = []
