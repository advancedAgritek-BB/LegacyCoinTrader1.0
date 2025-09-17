from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Iterable, Optional, List, Any
import logging

from crypto_bot.services.interfaces import ServiceContainer

logger = logging.getLogger(__name__)


from services.interface_layer.cycle import TradingCycleInterface


@dataclass
class BotContext:
    """Shared state for bot phases with enhanced memory management."""

    positions: dict
    df_cache: dict
    regime_cache: dict
    config: dict
    exchange: Optional[object] = None
    ws_client: Optional[object] = None
    risk_manager: Optional[object] = None
    notifier: Optional[object] = None
    paper_wallet: Optional[object] = None
    position_guard: Optional[object] = None
    trade_manager: Optional[object] = None  # Centralized trade manager
    services: Optional[ServiceContainer] = None
    balance: float = 0.0
    current_batch: List[str] = field(default_factory=list)
    analysis_results: Optional[list] = field(default_factory=list)
    timing: Optional[dict] = field(default_factory=dict)
    volatility_factor: float = 1.0

    # Migration support
    position_sync_manager: Optional[object] = None
    use_trade_manager_as_source: bool = False  # When True, TradeManager is single source of truth
    
    # Enhanced memory management
    memory_manager: Optional[object] = None
    managed_caches: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize production position sync manager and memory manager if available."""
        # Initialize position sync manager
        if self.trade_manager and not self.position_sync_manager:
            try:
                from crypto_bot.utils.position_sync_manager import get_position_sync_manager
                self.position_sync_manager = get_position_sync_manager(
                    trade_manager=self.trade_manager,
                    paper_wallet=self.paper_wallet,
                    config={"enable_background_sync": True, "auto_reconcile_inconsistencies": True}
                )
                logger.info("Production PositionSyncManager initialized for TradeManager migration")
            except Exception as e:
                logger.warning(f"Failed to initialize Production PositionSyncManager: {e}")
                # Fallback to original implementation
                try:
                    from crypto_bot.utils.trade_manager import PositionSyncManager
                    self.position_sync_manager = PositionSyncManager(self.trade_manager)
                    logger.info("Fallback PositionSyncManager initialized")
                except Exception as e2:
                    logger.warning(f"Fallback PositionSyncManager also failed: {e2}")
        
        # Initialize enhanced memory manager
        if not self.memory_manager:
            try:
                from crypto_bot.utils.enhanced_memory_manager import get_enhanced_memory_manager, MemoryConfig
                
                # Get memory config from main config
                memory_config = MemoryConfig(
                    memory_threshold=self.config.get('memory_threshold', 0.8),
                    gc_threshold=self.config.get('gc_threshold', 0.7),
                    cache_size_limit_mb=self.config.get('cache_size_limit_mb', 500),
                    model_cleanup_interval=self.config.get('model_cleanup_interval', 300),
                    cache_cleanup_interval=self.config.get('cache_cleanup_interval', 600),
                    enable_background_cleanup=self.config.get('enable_background_cleanup', True)
                )
                
                self.memory_manager = get_enhanced_memory_manager(memory_config)
                
                # Register existing caches with memory manager
                self._register_existing_caches()
                
                logger.info("Enhanced memory manager initialized for BotContext")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced memory manager: {e}")

    def _register_existing_caches(self):
        """Register existing caches with the memory manager."""
        if not self.memory_manager:
            return
        
        try:
            # Register DataFrame cache
            if self.df_cache:
                df_cache = self.memory_manager.register_cache(
                    "df_cache", 
                    max_size=1000, 
                    ttl_seconds=3600
                )
                self.managed_caches["df_cache"] = df_cache
                
                # Migrate existing data
                for symbol, df in self.df_cache.items():
                    df_cache.put(symbol, df)
                
                logger.info(f"Registered df_cache with {len(self.df_cache)} symbols")
            
            # Register regime cache
            if self.regime_cache:
                regime_cache = self.memory_manager.register_cache(
                    "regime_cache", 
                    max_size=500, 
                    ttl_seconds=1800
                )
                self.managed_caches["regime_cache"] = regime_cache
                
                # Migrate existing data
                for symbol, regime_df in self.regime_cache.items():
                    regime_cache.put(symbol, regime_df)
                
                logger.info(f"Registered regime_cache with {len(self.regime_cache)} symbols")
                
        except Exception as e:
            logger.error(f"Failed to register existing caches: {e}")

    def get_managed_cache(self, cache_name: str) -> Optional[object]:
        """Get a managed cache by name."""
        return self.managed_caches.get(cache_name)

    def register_ml_model(self, name: str, model: object, size_estimate_mb: int = 100):
        """Register an ML model with the memory manager."""
        if self.memory_manager:
            self.memory_manager.register_ml_model(name, model, size_estimate_mb)
        else:
            logger.warning("Memory manager not available for ML model registration")

    def update_ml_model_usage(self, name: str):
        """Update ML model usage timestamp."""
        if self.memory_manager:
            self.memory_manager.update_ml_model_usage(name)

    def perform_memory_maintenance(self) -> Dict[str, Any]:
        """Perform memory maintenance and return results."""
        if self.memory_manager:
            return self.memory_manager.perform_maintenance()
        else:
            return {"error": "Memory manager not available"}

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if self.memory_manager:
            return self.memory_manager.get_memory_stats()
        else:
            return {"error": "Memory manager not available"}

    def sync_positions_from_trade_manager(self):
        """
        Sync legacy position systems with TradeManager.
        Call this after any trade that modifies positions.
        """
        if not self.use_trade_manager_as_source or not self.position_sync_manager:
            return

        try:
            # Sync ctx.positions
            self.positions = self.position_sync_manager.sync_context_positions(self.positions)

            # Sync paper wallet positions
            if self.paper_wallet:
                self.paper_wallet.positions = self.position_sync_manager.sync_paper_wallet_positions(
                    self.paper_wallet.positions, self.paper_wallet
                )

            logger.debug("Position sync completed from TradeManager")

        except Exception as e:
            logger.error(f"Error syncing positions from TradeManager: {e}")

    def get_position_count(self) -> int:
        """Get position count from the source of truth."""
        if self.use_trade_manager_as_source and self.trade_manager:
            return len(self.trade_manager.get_all_positions())
        else:
            return len(self.positions)

    def get_position_symbols(self) -> List[str]:
        """Get list of position symbols from the source of truth."""
        if self.use_trade_manager_as_source and self.trade_manager:
            return [pos.symbol for pos in self.trade_manager.get_all_positions()]
        else:
            return list(self.positions.keys())

    def validate_position_consistency(self) -> bool:
        """Validate that all position systems are consistent."""
        if not self.position_sync_manager:
            return True

        paper_wallet_positions = self.paper_wallet.positions if self.paper_wallet else {}
        return self.position_sync_manager.validate_consistency(
            self.positions, paper_wallet_positions
        )


class PhaseRunner:
    """Run a sequence of async phases and record timing with memory monitoring."""

    def __init__(self, phases: Iterable[Callable[[BotContext], Awaitable[None]]]):
        self.phases = list(phases)
        self._cycle_interface = TradingCycleInterface(self.phases)

    async def run(self, ctx: BotContext) -> Dict[str, float]:
        self._cycle_interface.set_phases(self.phases)
        result = await self._cycle_interface.run_cycle(ctx)
        return result.timings
