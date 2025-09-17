"""Compatibility wrapper for the relocated service interface contracts."""

from libs.services.interfaces import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith("_")]
