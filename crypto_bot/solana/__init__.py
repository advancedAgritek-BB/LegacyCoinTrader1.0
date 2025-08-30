"""Solana utilities package.

Avoid importing heavy modules at package import time. Tests should import
specific symbols from submodules directly as needed.
"""

from .scanner import get_solana_new_tokens

__all__ = ["get_solana_new_tokens"]
