"""Identity service package for LegacyCoinTrader."""

from .config import IdentitySettings, load_identity_settings
from .service import IdentityService

__all__ = [
    "IdentityService",
    "IdentitySettings",
    "load_identity_settings",
]
