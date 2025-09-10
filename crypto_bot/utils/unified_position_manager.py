"""
Unified Position Manager - Single Source of Truth for Position Data

This module provides a unified interface for managing positions across all
position tracking systems (TradeManager, paper_wallet, positions.log) to
ensure data consistency and prevent conflicts.
"""

import asyncio
import threading
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PositionConflict:
    """Represents a conflict between different position systems."""
    symbol: str
    trade_manager_position: Optional[dict] = None
    paper_wallet_position: Optional[dict] = None
    log_position: Optional[dict] = None
    conflict_type: str = ""
    resolution_strategy: str = ""
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False

@dataclass
class PositionSyncStats:
    """Statistics for position synchronization."""
    total_syncs: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    last_sync_time: Optional[datetime] = None
    sync_duration_avg: float = 0.0
    error_count: int = 0

class UnifiedPositionManager:
    """
    Single source of truth for all position data across the system.
    
    This class ensures that all position tracking systems (TradeManager,
    paper_wallet, positions.log) remain synchronized and consistent.
    """
    
    def __init__(self, trade_manager=None, paper_wallet=None, config: Dict[str, Any] = None):
        self.trade_manager = trade_manager
        self.paper_wallet = paper_wallet
        self.config = config or {}
        self.lock = threading.RLock()
        
        # Synchronization settings
        self.sync_interval = self.config.get('position_sync_interval', 5)
        self.last_sync = datetime.now()
        self.sync_task = None
        self.running = False
        
        # Statistics
        self.stats = PositionSyncStats()
        
        # Conflict resolution strategies
        self.resolution_strategies = {
            'trade_manager_priority': self._resolve_trade_manager_priority,
            'paper_wallet_priority': self._resolve_paper_wallet_priority,
            'merge_positions': self._resolve_merge_positions,
            'emergency_reset': self._resolve_emergency_reset,
            'most_recent': self._resolve_most_recent
        }
        
        # Position cache for quick access
        self.position_cache: Dict[str, dict] = {}
        self.cache_timestamp = datetime.now()
        
        # Conflict history
        self.conflict_history: List[PositionConflict] = []
        self.max_conflict_history = 100
        
        logger.info("Unified Position Manager initialized")
    
    async def start_sync_monitoring(self):
        """Start continuous position synchronization monitoring."""
        if self.sync_task is None and not self.running:
            self.running = True
            self.sync_task = asyncio.create_task(self._sync_monitor_loop())
            logger.info("Started position synchronization monitoring")
    
    async def stop_sync_monitoring(self):
        """Stop position synchronization monitoring."""
        self.running = False
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
            self.sync_task = None
            logger.info("Stopped position synchronization monitoring")
    
    async def _sync_monitor_loop(self):
        """Continuous monitoring loop for position synchronization."""
        while self.running:
            try:
                start_time = datetime.now()
                conflicts = await self.sync_all_systems()
                
                # Update statistics
                self.stats.total_syncs += 1
                self.stats.last_sync_time = datetime.now()
                sync_duration = (self.stats.last_sync_time - start_time).total_seconds()
                self.stats.sync_duration_avg = (
                    (self.stats.sync_duration_avg * (self.stats.total_syncs - 1) + sync_duration) / 
                    self.stats.total_syncs
                )
                
                if conflicts:
                    self.stats.conflicts_detected += len(conflicts)
                    resolved_conflicts = [c for c in conflicts if c.resolved]
                    self.stats.conflicts_resolved += len(resolved_conflicts)
                    
                    logger.info(f"Sync completed: {len(conflicts)} conflicts detected, {len(resolved_conflicts)} resolved")
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                self.stats.error_count += 1
                logger.error(f"Position sync error: {e}")
                await asyncio.sleep(self.sync_interval * 2)  # Back off on errors
    
    async def sync_all_systems(self) -> List[PositionConflict]:
        """Ensure all position systems are synchronized."""
        with self.lock:
            try:
                start_time = datetime.now()
                
                # Get positions from all systems
                tm_positions = self._get_trade_manager_positions()
                pw_positions = self._get_paper_wallet_positions()
                log_positions = self._load_positions_from_log()
                
                # Detect conflicts
                conflicts = self._detect_conflicts(tm_positions, pw_positions, log_positions)
                
                # Resolve conflicts
                if conflicts:
                    await self._resolve_conflicts(conflicts)
                
                # Update cache
                self._update_position_cache(tm_positions, pw_positions, log_positions)
                
                # Update statistics
                self.stats.total_syncs += 1
                self.stats.last_sync_time = datetime.now()
                sync_duration = (self.stats.last_sync_time - start_time).total_seconds()
                self.stats.sync_duration_avg = (
                    (self.stats.sync_duration_avg * (self.stats.total_syncs - 1) + sync_duration) / 
                    self.stats.total_syncs
                )
                
                if conflicts:
                    self.stats.conflicts_detected += len(conflicts)
                    resolved_conflicts = [c for c in conflicts if c.resolved]
                    self.stats.conflicts_resolved += len(resolved_conflicts)
                
                self.last_sync = datetime.now()
                return conflicts
                
            except Exception as e:
                self.stats.error_count += 1
                logger.error(f"Error in sync_all_systems: {e}")
                return []
    
    def _get_trade_manager_positions(self) -> Dict[str, dict]:
        """Get positions from TradeManager."""
        try:
            if self.trade_manager and hasattr(self.trade_manager, 'get_all_positions'):
                positions = self.trade_manager.get_all_positions()
                return {pos.symbol: pos.to_dict() for pos in positions}
            return {}
        except Exception as e:
            logger.error(f"Error getting TradeManager positions: {e}")
            return {}
    
    def _get_paper_wallet_positions(self) -> Dict[str, dict]:
        """Get positions from paper wallet."""
        try:
            if self.paper_wallet and hasattr(self.paper_wallet, 'positions'):
                return self.paper_wallet.positions.copy()
            return {}
        except Exception as e:
            logger.error(f"Error getting paper wallet positions: {e}")
            return {}
    
    def _load_positions_from_log(self) -> Dict[str, dict]:
        """Load positions from positions.log file."""
        try:
            log_path = Path("crypto_bot/logs/positions.log")
            if not log_path.exists():
                return {}
            
            positions = {}
            with open(log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            # Parse position log entry
                            # Format: timestamp,symbol,side,size,entry_price,current_price,pnl
                            parts = line.split(',')
                            if len(parts) >= 7:
                                symbol = parts[1]
                                positions[symbol] = {
                                    'symbol': symbol,
                                    'side': parts[2],
                                    'size': float(parts[3]),
                                    'entry_price': float(parts[4]),
                                    'current_price': float(parts[5]),
                                    'pnl': float(parts[6]),
                                    'source': 'log'
                                }
                        except Exception as e:
                            logger.debug(f"Error parsing position log line: {e}")
                            continue
            
            return positions
            
        except Exception as e:
            logger.error(f"Error loading positions from log: {e}")
            return {}
    
    def _detect_conflicts(self, tm_positions: Dict[str, dict], 
                         pw_positions: Dict[str, dict], 
                         log_positions: Dict[str, dict]) -> List[PositionConflict]:
        """Detect conflicts between different position systems."""
        conflicts = []
        all_symbols = set(tm_positions.keys()) | set(pw_positions.keys()) | set(log_positions.keys())
        
        for symbol in all_symbols:
            tm_pos = tm_positions.get(symbol)
            pw_pos = pw_positions.get(symbol)
            log_pos = log_positions.get(symbol)
            
            # Check for conflicts
            if self._has_conflict(tm_pos, pw_pos, log_pos):
                conflict = PositionConflict(
                    symbol=symbol,
                    trade_manager_position=tm_pos,
                    paper_wallet_position=pw_pos,
                    log_position=log_pos,
                    conflict_type=self._determine_conflict_type(tm_pos, pw_pos, log_pos),
                    resolution_strategy=self._determine_resolution_strategy(tm_pos, pw_pos, log_pos)
                )
                conflicts.append(conflict)
                
                # Add to history
                self.conflict_history.append(conflict)
                if len(self.conflict_history) > self.max_conflict_history:
                    self.conflict_history.pop(0)
        
        return conflicts
    
    def _has_conflict(self, tm_pos: Optional[dict], pw_pos: Optional[dict], 
                     log_pos: Optional[dict]) -> bool:
        """Check if there's a conflict between position systems."""
        positions = [pos for pos in [tm_pos, pw_pos, log_pos] if pos is not None]
        
        if len(positions) <= 1:
            return False
        
        # Compare key fields
        first_pos = positions[0]
        for other_pos in positions[1:]:
            # Compare size (with tolerance for floating point)
            size_diff = abs(first_pos.get('size', 0) - other_pos.get('size', 0))
            if size_diff > 0.000001:  # Small tolerance
                return True
            
            # Compare entry price (with tolerance)
            entry_diff = abs(first_pos.get('entry_price', 0) - other_pos.get('entry_price', 0))
            if entry_diff > 0.000001:
                return True
            
            # Compare side
            if first_pos.get('side') != other_pos.get('side'):
                return True
        
        return False
    
    def _determine_conflict_type(self, tm_pos: Optional[dict], pw_pos: Optional[dict], 
                                log_pos: Optional[dict]) -> str:
        """Determine the type of conflict."""
        if tm_pos and pw_pos and log_pos:
            return "three_way_conflict"
        elif tm_pos and pw_pos:
            return "tm_pw_conflict"
        elif tm_pos and log_pos:
            return "tm_log_conflict"
        elif pw_pos and log_pos:
            return "pw_log_conflict"
        else:
            return "unknown_conflict"
    
    def _determine_resolution_strategy(self, tm_pos: Optional[dict], pw_pos: Optional[dict], 
                                     log_pos: Optional[dict]) -> str:
        """Determine the best resolution strategy for a conflict."""
        # Prefer TradeManager as it's the most authoritative
        if tm_pos:
            return "trade_manager_priority"
        elif pw_pos:
            return "paper_wallet_priority"
        elif log_pos:
            return "most_recent"
        else:
            return "emergency_reset"
    
    async def _resolve_conflicts(self, conflicts: List[PositionConflict]):
        """Resolve detected conflicts using appropriate strategies."""
        for conflict in conflicts:
            try:
                strategy = self.resolution_strategies.get(conflict.resolution_strategy)
                if strategy:
                    await strategy(conflict)
                    conflict.resolved = True
                    logger.info(f"Resolved conflict for {conflict.symbol} using {conflict.resolution_strategy}")
                else:
                    logger.error(f"No resolution strategy found for {conflict.symbol}")
            except Exception as e:
                logger.error(f"Failed to resolve conflict for {conflict.symbol}: {e}")
    
    async def _resolve_trade_manager_priority(self, conflict: PositionConflict):
        """Resolve conflict by prioritizing TradeManager data."""
        if conflict.trade_manager_position:
            # Update paper wallet
            if self.paper_wallet and hasattr(self.paper_wallet, 'positions'):
                self.paper_wallet.positions[conflict.symbol] = conflict.trade_manager_position
            
            # Update log
            self._update_positions_log(conflict.symbol, conflict.trade_manager_position)
    
    async def _resolve_paper_wallet_priority(self, conflict: PositionConflict):
        """Resolve conflict by prioritizing paper wallet data."""
        if conflict.paper_wallet_position:
            # Update TradeManager
            if self.trade_manager and hasattr(self.trade_manager, 'update_position'):
                self.trade_manager.update_position(conflict.symbol, conflict.paper_wallet_position)
            
            # Update log
            self._update_positions_log(conflict.symbol, conflict.paper_wallet_position)
    
    async def _resolve_merge_positions(self, conflict: PositionConflict):
        """Resolve conflict by merging position data."""
        # This is a complex operation that would merge the best data from each source
        # For now, use the most recent data
        await self._resolve_most_recent(conflict)
    
    async def _resolve_most_recent(self, conflict: PositionConflict):
        """Resolve conflict by using the most recent position data."""
        positions = [
            (conflict.trade_manager_position, 'trade_manager'),
            (conflict.paper_wallet_position, 'paper_wallet'),
            (conflict.log_position, 'log')
        ]
        
        # Find the most recent position
        most_recent = None
        most_recent_source = None
        
        for pos, source in positions:
            if pos and pos.get('timestamp'):
                if most_recent is None or pos['timestamp'] > most_recent['timestamp']:
                    most_recent = pos
                    most_recent_source = source
        
        if most_recent:
            # Update all systems with the most recent data
            if self.trade_manager and hasattr(self.trade_manager, 'update_position'):
                self.trade_manager.update_position(conflict.symbol, most_recent)
            
            if self.paper_wallet and hasattr(self.paper_wallet, 'positions'):
                self.paper_wallet.positions[conflict.symbol] = most_recent
            
            self._update_positions_log(conflict.symbol, most_recent)
    
    async def _resolve_emergency_reset(self, conflict: PositionConflict):
        """Emergency reset - remove position from all systems."""
        logger.warning(f"Emergency reset for position {conflict.symbol}")
        
        # Remove from TradeManager
        if self.trade_manager and hasattr(self.trade_manager, 'close_position'):
            self.trade_manager.close_position(conflict.symbol)
        
        # Remove from paper wallet
        if self.paper_wallet and hasattr(self.paper_wallet, 'positions'):
            self.paper_wallet.positions.pop(conflict.symbol, None)
        
        # Remove from log
        self._remove_from_positions_log(conflict.symbol)
    
    def _update_positions_log(self, symbol: str, position_data: dict):
        """Update positions.log with position data."""
        try:
            log_path = Path("crypto_bot/logs/positions.log")
            
            # Read existing entries
            entries = []
            if log_path.exists():
                with open(log_path, 'r') as f:
                    entries = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # Find and update or add entry
            new_entry = f"{datetime.now().isoformat()},{symbol},{position_data.get('side', 'unknown')}," \
                       f"{position_data.get('size', 0)},{position_data.get('entry_price', 0)}," \
                       f"{position_data.get('current_price', 0)},{position_data.get('pnl', 0)}"
            
            # Replace existing entry or add new one
            updated = False
            for i, entry in enumerate(entries):
                if entry.split(',')[1] == symbol:  # Check symbol
                    entries[i] = new_entry
                    updated = True
                    break
            
            if not updated:
                entries.append(new_entry)
            
            # Write back to file
            with open(log_path, 'w') as f:
                f.write("# timestamp,symbol,side,size,entry_price,current_price,pnl\n")
                for entry in entries:
                    f.write(entry + "\n")
                    
        except Exception as e:
            logger.error(f"Error updating positions log: {e}")
    
    def _remove_from_positions_log(self, symbol: str):
        """Remove position from positions.log."""
        try:
            log_path = Path("crypto_bot/logs/positions.log")
            
            if not log_path.exists():
                return
            
            # Read and filter out the symbol
            entries = []
            with open(log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if line.split(',')[1] != symbol:  # Keep if not the target symbol
                            entries.append(line)
                    elif line.startswith('#'):
                        entries.append(line)  # Keep comments
            
            # Write back to file
            with open(log_path, 'w') as f:
                for entry in entries:
                    f.write(entry + "\n")
                    
        except Exception as e:
            logger.error(f"Error removing from positions log: {e}")
    
    def _update_position_cache(self, tm_positions: Dict[str, dict], 
                             pw_positions: Dict[str, dict], 
                             log_positions: Dict[str, dict]):
        """Update the position cache with unified data."""
        # Use TradeManager as primary source, fallback to others
        self.position_cache = tm_positions.copy()
        
        # Add positions from other sources if not in TradeManager
        for symbol, pos in pw_positions.items():
            if symbol not in self.position_cache:
                self.position_cache[symbol] = pos
        
        for symbol, pos in log_positions.items():
            if symbol not in self.position_cache:
                self.position_cache[symbol] = pos
        
        self.cache_timestamp = datetime.now()
    
    def get_unified_positions(self) -> Dict[str, dict]:
        """Get unified position data from the authoritative source."""
        with self.lock:
            # Always get fresh data for now (cache logic can be optimized later)
            tm_positions = self._get_trade_manager_positions()
            pw_positions = self._get_paper_wallet_positions()
            log_positions = self._load_positions_from_log()
            
            self._update_position_cache(tm_positions, pw_positions, log_positions)
            return self.position_cache.copy()
    
    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol."""
        positions = self.get_unified_positions()
        return positions.get(symbol)
    
    def update_position(self, symbol: str, position_data: dict):
        """Update position through unified manager."""
        with self.lock:
            # Update TradeManager if available
            if self.trade_manager and hasattr(self.trade_manager, 'update_position'):
                self.trade_manager.update_position(symbol, position_data)
            
            # Update paper wallet if available
            if self.paper_wallet and hasattr(self.paper_wallet, 'positions'):
                self.paper_wallet.positions[symbol] = position_data
            
            # Update log
            self._update_positions_log(symbol, position_data)
            
            # Update cache
            self.position_cache[symbol] = position_data
            self.cache_timestamp = datetime.now()
    
    def validate_consistency(self) -> bool:
        """Validate that all position systems are consistent."""
        try:
            # Run a sync to check for conflicts
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, schedule the sync
                future = asyncio.create_task(self.sync_all_systems())
                conflicts = loop.run_until_complete(future)
            else:
                conflicts = loop.run_until_complete(self.sync_all_systems())
            
            return len(conflicts) == 0
        except Exception as e:
            logger.error(f"Consistency validation failed: {e}")
            return False
    
    def get_sync_stats(self) -> PositionSyncStats:
        """Get synchronization statistics."""
        return self.stats
    
    def get_conflict_history(self) -> List[PositionConflict]:
        """Get recent conflict history."""
        return self.conflict_history.copy()
    
    def force_sync(self) -> List[PositionConflict]:
        """Force an immediate synchronization."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.create_task(self.sync_all_systems())
                return loop.run_until_complete(future)
            else:
                return loop.run_until_complete(self.sync_all_systems())
        except Exception as e:
            logger.error(f"Force sync failed: {e}")
            return []
