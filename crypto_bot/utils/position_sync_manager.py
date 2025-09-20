"""Position synchronisation helpers used by the production stack."""

import time
import threading
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

from .logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "position_sync.log")


@dataclass
class Position:
    """Standardized position representation."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)
    status: str = "open"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "timestamp": self.timestamp,
            "status": self.status
        }


class ProductionPositionSyncManager:
    """
    Production-grade position synchronization manager.

    Features:
    - Atomic position updates
    - Consistency validation
    - Automatic reconciliation
    - Background sync monitoring
    - Error recovery
    """

    def __init__(self, trade_manager=None, paper_wallet=None, config: Optional[Dict[str, Any]] = None):
        self.trade_manager = trade_manager
        self.paper_wallet = paper_wallet
        self.config = config or self._get_default_config()

        # Sync configuration
        self.sync_interval = self.config.get("sync_interval", 30)  # seconds
        self.max_sync_attempts = self.config.get("max_sync_attempts", 3)
        self.consistency_check_interval = self.config.get("consistency_check_interval", 60)

        # Threading and locks
        self.sync_lock = threading.RLock()
        self.last_sync_time = 0
        self.sync_in_progress = False

        # Consistency tracking
        self.inconsistencies_found = 0
        self.last_consistency_check = 0
        self.consistency_history: List[Dict[str, Any]] = []

        # Background sync task (thread based to avoid event loop coupling)
        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.running = False

        logger.info("Production position sync manager initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "sync_interval": 30,
            "max_sync_attempts": 3,
            "consistency_check_interval": 60,
            "enable_background_sync": True,
            "auto_reconcile_inconsistencies": True,
            "alert_on_inconsistencies": True
        }

    def start_background_sync(self):
        """Start background position synchronization."""
        if self.config.get("enable_background_sync", True) and not self.running:
            self.running = True
            if self._background_thread is None or not self._background_thread.is_alive():
                self._stop_event.clear()
                self._background_thread = threading.Thread(
                    target=self._run_background_sync,
                    name="position-sync-manager",
                    daemon=True,
                )
                self._background_thread.start()
            logger.info("Background position sync started")

    def stop_background_sync(self):
        """Stop background position synchronization."""
        self.running = False
        self._stop_event.set()
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=2.0)
        self._background_thread = None
        logger.info("Background position sync stopped")

    def _run_background_sync(self) -> None:
        """Periodically reconcile positions in a background thread."""

        while not self._stop_event.wait(self.sync_interval):
            try:
                # Using the latest known context (if any) to keep subsystems aligned.
                if hasattr(self.trade_manager, "context_positions"):
                    context_positions = getattr(self.trade_manager, "context_positions") or {}
                else:
                    context_positions = {}
                self.sync_context_positions(context_positions)

                if self.paper_wallet is not None and hasattr(self.paper_wallet, "positions"):
                    paper_positions = dict(getattr(self.paper_wallet, "positions", {}))
                    self.sync_paper_wallet_positions(paper_positions, self.paper_wallet)

                if context_positions or (self.paper_wallet and getattr(self.paper_wallet, "positions", None)):
                    self.validate_consistency(context_positions, getattr(self.paper_wallet, "positions", {}) or {})
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Background position sync failed: %s", exc)

    def sync_context_positions(self, context_positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync context positions with TradeManager as source of truth.

        Args:
            context_positions: Current context positions dictionary

        Returns:
            Updated context positions
        """
        with self.sync_lock:
            try:
                if not self.trade_manager:
                    logger.debug("No trade manager available, skipping sync")
                    return context_positions

                # Get authoritative positions from TradeManager
                tm_positions = self._get_trade_manager_positions()

                # Convert to standardized format
                standardized_positions = {}
                for pos in tm_positions:
                    std_pos = self._convert_to_standard_position(pos)
                    standardized_positions[std_pos.symbol] = std_pos.to_dict()

                # Update context positions
                updated_positions = dict(context_positions)
                updated_positions.update(standardized_positions)

                # Remove positions that no longer exist in TradeManager
                tm_symbols = {pos.symbol for pos in tm_positions}
                context_symbols = set(updated_positions.keys())
                symbols_to_remove = context_symbols - tm_symbols

                for symbol in symbols_to_remove:
                    if symbol in updated_positions:
                        del updated_positions[symbol]
                        logger.debug(f"Removed stale position from context: {symbol}")

                self.last_sync_time = time.time()
                logger.debug(f"Synced {len(standardized_positions)} positions from TradeManager")

                return updated_positions

            except Exception as e:
                logger.error(f"Failed to sync context positions: {e}")
                return context_positions

    def sync_paper_wallet_positions(self, paper_positions: Dict[str, Any], paper_wallet) -> Dict[str, Any]:
        """
        Sync paper wallet positions with TradeManager.

        Args:
            paper_positions: Current paper wallet positions
            paper_wallet: Paper wallet instance

        Returns:
            Updated paper wallet positions
        """
        with self.sync_lock:
            try:
                if not self.trade_manager:
                    logger.debug("No trade manager available, skipping paper wallet sync")
                    return paper_positions

                # Get authoritative positions from TradeManager
                tm_positions = self._get_trade_manager_positions()

                # Convert to paper wallet format
                updated_positions = {}
                for pos in tm_positions:
                    paper_pos = self._convert_to_paper_wallet_position(pos)
                    updated_positions[pos.symbol] = paper_pos

                # Update paper wallet
                paper_wallet.positions = updated_positions

                logger.debug(f"Synced {len(updated_positions)} positions to paper wallet")
                return updated_positions

            except Exception as e:
                logger.error(f"Failed to sync paper wallet positions: {e}")
                return paper_positions

    def _get_trade_manager_positions(self) -> List[Any]:
        """Get positions from TradeManager."""
        try:
            if hasattr(self.trade_manager, 'get_all_positions'):
                return self.trade_manager.get_all_positions()
            elif hasattr(self.trade_manager, 'positions'):
                return list(self.trade_manager.positions.values())
            else:
                logger.warning("TradeManager does not have expected position access methods")
                return []
        except Exception as e:
            logger.error(f"Failed to get positions from TradeManager: {e}")
            return []

    def _convert_to_standard_position(self, tm_position: Any) -> Position:
        """Convert TradeManager position to standard format."""
        try:
            # Handle different TradeManager position formats
            if hasattr(tm_position, 'to_dict'):
                pos_dict = tm_position.to_dict()
            elif isinstance(tm_position, dict):
                pos_dict = tm_position
            else:
                # Fallback for unknown format
                pos_dict = {
                    "symbol": getattr(tm_position, "symbol", "UNKNOWN"),
                    "quantity": getattr(tm_position, "quantity", 0),
                    "entry_price": getattr(tm_position, "entry_price", 0),
                    "current_price": getattr(tm_position, "current_price", None),
                    "pnl": getattr(tm_position, "pnl", 0),
                    "pnl_pct": getattr(tm_position, "pnl_pct", 0),
                    "timestamp": getattr(tm_position, "timestamp", time.time()),
                    "status": getattr(tm_position, "status", "open")
                }

            return Position(**pos_dict)

        except Exception as e:
            logger.error(f"Failed to convert position to standard format: {e}")
            return Position(symbol="UNKNOWN", quantity=0, entry_price=0)

    def _convert_to_paper_wallet_position(self, tm_position: Any) -> Dict[str, Any]:
        """Convert TradeManager position to paper wallet format."""
        try:
            std_pos = self._convert_to_standard_position(tm_position)
            return {
                "symbol": std_pos.symbol,
                "quantity": std_pos.quantity,
                "entry_price": std_pos.entry_price,
                "current_price": std_pos.current_price,
                "pnl": std_pos.pnl,
                "pnl_pct": std_pos.pnl_pct,
                "timestamp": std_pos.timestamp,
                "status": std_pos.status
            }
        except Exception as e:
            logger.error(f"Failed to convert position to paper wallet format: {e}")
            return {}

    def validate_consistency(self, context_positions: Dict[str, Any], paper_positions: Dict[str, Any]) -> bool:
        """
        Validate consistency between context and paper wallet positions.

        Returns:
            True if consistent, False otherwise
        """
        try:
            current_time = time.time()

            # Only check consistency periodically
            if current_time - self.last_consistency_check < self.consistency_check_interval:
                return True

            self.last_consistency_check = current_time

            # Get TradeManager positions as reference
            if not self.trade_manager:
                logger.debug("No trade manager for consistency check")
                return True

            tm_positions = self._get_trade_manager_positions()
            tm_symbols = {pos.symbol for pos in tm_positions}

            # Check context positions
            context_symbols = set(context_positions.keys())
            context_inconsistencies = context_symbols.symmetric_difference(tm_symbols)

            # Check paper wallet positions
            paper_symbols = set(paper_positions.keys())
            paper_inconsistencies = paper_symbols.symmetric_difference(tm_symbols)

            # Record consistency check
            consistency_record = {
                "timestamp": current_time,
                "context_symbols": len(context_symbols),
                "paper_symbols": len(paper_symbols),
                "tm_symbols": len(tm_symbols),
                "context_inconsistencies": len(context_inconsistencies),
                "paper_inconsistencies": len(paper_inconsistencies),
                "total_inconsistencies": len(context_inconsistencies) + len(paper_inconsistencies)
            }

            self.consistency_history.append(consistency_record)

            # Keep only recent history
            if len(self.consistency_history) > 100:
                self.consistency_history = self.consistency_history[-100:]

            # Check for inconsistencies
            total_inconsistencies = len(context_inconsistencies) + len(paper_inconsistencies)

            if total_inconsistencies > 0:
                self.inconsistencies_found += 1
                logger.warning(f"Position inconsistencies detected: {total_inconsistencies}")
                logger.warning(f"Context inconsistencies: {context_inconsistencies}")
                logger.warning(f"Paper wallet inconsistencies: {paper_inconsistencies}")

                # Auto-reconcile if enabled
                if self.config.get("auto_reconcile_inconsistencies", True):
                    self._auto_reconcile_inconsistencies(
                        context_positions, paper_positions, tm_positions
                    )

                return False

            logger.debug("Position consistency validated")
            return True

        except Exception as e:
            logger.error(f"Consistency validation failed: {e}")
            return False

    def _auto_reconcile_inconsistencies(
        self,
        context_positions: Dict[str, Any],
        paper_positions: Dict[str, Any],
        tm_positions: List[Any]
    ):
        """Automatically reconcile position inconsistencies."""
        try:
            logger.info("Starting automatic position reconciliation...")

            # Sync both systems with TradeManager
            updated_context = self.sync_context_positions(context_positions)
            if self.paper_wallet is not None:
                self.sync_paper_wallet_positions(paper_positions, self.paper_wallet)

            # Persist reconciled context positions if caller supplied a mutable mapping
            if isinstance(context_positions, dict):
                context_positions.clear()
                context_positions.update(updated_context)

            logger.info("Position reconciliation completed")

        except Exception as e:
            logger.error(f"Automatic reconciliation failed: {e}")

    def get_consistency_stats(self) -> Dict[str, Any]:
        """Get consistency statistics."""
        if not self.consistency_history:
            return {"checks_performed": 0, "inconsistencies_found": 0}

        recent_checks = self.consistency_history[-10:]  # Last 10 checks

        return {
            "checks_performed": len(self.consistency_history),
            "inconsistencies_found": self.inconsistencies_found,
            "recent_avg_inconsistencies": sum(
                check["total_inconsistencies"] for check in recent_checks
            ) / len(recent_checks) if recent_checks else 0,
            "last_check_time": self.last_consistency_check,
            "last_sync_time": self.last_sync_time
        }

    def force_sync_now(self):
        """Force immediate synchronization."""
        logger.info("Forcing immediate position synchronization...")
        self.last_sync_time = 0  # Reset to force sync
        self.last_consistency_check = 0  # Reset to force consistency check


# Global sync manager instance
_sync_manager = None

def get_position_sync_manager(trade_manager=None, paper_wallet=None, config=None):
    """Get or create global position sync manager instance."""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = ProductionPositionSyncManager(trade_manager, paper_wallet, config)
    return _sync_manager
