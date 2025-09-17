"""Trading engine phase collections."""

from .production import (
    analyse_batch,
    execute_signals,
    fetch_candidates,
    handle_exits,
    monitor_positions_phase,
    process_solana_candidates,
    update_caches,
    PRODUCTION_PHASES,
)
from .minimal import DEFAULT_PHASES as MINIMAL_PHASES

DEFAULT_PHASES = PRODUCTION_PHASES

__all__ = [
    "fetch_candidates",
    "process_solana_candidates",
    "update_caches",
    "analyse_batch",
    "execute_signals",
    "handle_exits",
    "monitor_positions_phase",
    "PRODUCTION_PHASES",
    "DEFAULT_PHASES",
    "MINIMAL_PHASES",
]
