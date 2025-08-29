"""
Momentum-Aware Position Manager

This module provides intelligent position management that allows coins with momentum
to continue running while protecting profits through dynamic take profits and
momentum-based partial exits.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from risk.exit_manager import (
    MomentumExitConfig, 
    detect_momentum_continuation,
    calculate_momentum_scaled_take_profit,
    get_momentum_based_partial_exit,
    should_delay_exit_for_momentum
)
from utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "bot.log")


@dataclass
class MomentumPosition:
    """Represents a position with momentum-aware management."""
    
    position_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    entry_time: float
    size: float
    current_price: float
    
    # Momentum tracking
    momentum_strength: float = 0.0
    momentum_continuation: Dict[str, Any] = None
    last_momentum_update: float = 0.0
    
    # Profit tracking
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    
    # Exit management
    trailing_stop: float = 0.0
    take_profit_target: float = 0.0
    partial_exits: List[Dict] = None
    
    # Momentum-based adjustments
    momentum_tp_multiplier: float = 1.0
    momentum_trail_adjustment: float = 1.0
    
    def __post_init__(self):
        """Initialize position after creation."""
        if self.side == "buy":
            self.highest_price = self.entry_price
            self.lowest_price = self.entry_price
        else:
            self.highest_price = self.entry_price
            self.lowest_price = self.entry_price
        
        if self.partial_exits is None:
            self.partial_exits = []
        
        self.update_pnl()
    
    def update_pnl(self):
        """Update profit/loss calculations."""
        if self.side == "buy":
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.size
            self.unrealized_pnl_pct = (self.current_price - self.entry_price) / self.entry_price
        else:
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.size
            self.unrealized_pnl_pct = (self.entry_price - self.current_price) / self.entry_price
    
    def update_price(self, new_price: float):
        """Update current price and related metrics."""
        self.current_price = new_price
        
        # Update high/low prices
        if self.side == "buy":
            if new_price > self.highest_price:
                self.highest_price = new_price
            if new_price < self.lowest_price:
                self.lowest_price = new_price
        else:
            if new_price < self.lowest_price:
                self.lowest_price = new_price
            if new_price > self.highest_price:
                self.highest_price = new_price
        
        self.update_pnl()
    
    def should_update_momentum(self, current_time: float, update_interval: int = 30) -> bool:
        """Check if momentum should be updated."""
        return current_time - self.last_momentum_update > update_interval


class MomentumPositionManager:
    """
    Manages positions with momentum-aware exit strategies.
    
    Features:
    - Dynamic take profit scaling based on momentum strength
    - Momentum-based partial profit taking
    - Adaptive trailing stops that tighten with momentum
    - Breakout detection and momentum continuation analysis
    - Intelligent exit delays for strong momentum
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the momentum position manager."""
        self.config = config
        self.positions: Dict[str, MomentumPosition] = {}
        self.momentum_config = MomentumExitConfig(**config.get('exit_strategy', {}).get('momentum_continuation', {}))
        
        # Performance tracking
        self.total_momentum_exits = 0
        self.total_momentum_extensions = 0
        self.total_partial_exits = 0
        
        logger.info("Momentum Position Manager initialized")
    
    async def add_position(
        self, 
        position_id: str, 
        symbol: str, 
        side: str, 
        entry_price: float, 
        size: float
    ) -> MomentumPosition:
        """Add a new position for momentum-aware management."""
        
        position = MomentumPosition(
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=time.time(),
            size=size,
            current_price=entry_price
        )
        
        # Initialize momentum tracking
        await self._update_position_momentum(position)
        
        # Set initial take profit and trailing stop
        await self._set_initial_exit_levels(position)
        
        self.positions[position_id] = position
        logger.info(f"Added position {position_id} for {symbol} with momentum-aware management")
        
        return position
    
    async def update_position(
        self, 
        position_id: str, 
        new_price: float, 
        market_data: pd.DataFrame = None
    ) -> Optional[MomentumPosition]:
        """Update position with new price and market data."""
        
        position = self.positions.get(position_id)
        if not position:
            logger.warning(f"Position {position_id} not found")
            return None
        
        # Update price and PnL
        position.update_price(new_price)
        
        # Update momentum if needed
        current_time = time.time()
        if position.should_update_momentum(current_time):
            await self._update_position_momentum(position, market_data)
        
        # Check for momentum-based adjustments
        await self._check_momentum_adjustments(position)
        
        # Check for partial exit opportunities
        await self._check_partial_exit_opportunities(position)
        
        return position
    
    async def _update_position_momentum(
        self, 
        position: MomentumPosition, 
        market_data: pd.DataFrame = None
    ):
        """Update momentum metrics for a position."""
        
        try:
            if market_data is not None and not market_data.empty:
                # Calculate momentum strength (simplified version)
                position.momentum_strength = self._calculate_simple_momentum(market_data)
                
                # Detect momentum continuation
                position.momentum_continuation = detect_momentum_continuation(
                    market_data, self.momentum_config
                )
                
                position.last_momentum_update = time.time()
                
                logger.debug(f"Updated momentum for {position.symbol}: strength={position.momentum_strength:.3f}")
        
        except Exception as e:
            logger.error(f"Error updating momentum for position {position.position_id}: {e}")
    
    def _calculate_simple_momentum(self, market_data: pd.DataFrame) -> float:
        """Calculate a simple momentum score from market data."""
        try:
            if len(market_data) < 10:
                return 0.5
            
            # Price momentum (recent price change)
            recent_change = (market_data['close'].iloc[-1] - market_data['close'].iloc[-5]) / market_data['close'].iloc[-5]
            
            # Volume momentum
            recent_volume = market_data['volume'].iloc[-3:].mean()
            avg_volume = market_data['volume'].iloc[-10:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # RSI momentum
            if 'rsi' in market_data.columns:
                rsi = market_data['rsi'].iloc[-1]
                rsi_score = max(0, min(1, (rsi - 30) / 40))  # 30-70 range
            else:
                rsi_score = 0.5
            
            # Combine metrics
            momentum_score = (
                max(0, min(1, recent_change * 10)) * 0.4 +  # Price momentum
                max(0, min(1, (volume_ratio - 1) * 0.5)) * 0.3 +  # Volume momentum
                rsi_score * 0.3  # RSI momentum
            )
            
            return momentum_score
            
        except Exception as e:
            logger.error(f"Error calculating simple momentum: {e}")
            return 0.5
    
    async def _set_initial_exit_levels(self, position: MomentumPosition):
        """Set initial take profit and trailing stop levels."""
        
        exit_cfg = self.config.get('exit_strategy', {})
        
        # Base take profit
        base_tp_pct = exit_cfg.get('take_profit_pct', 0.04)
        
        # Apply momentum scaling if enabled
        if exit_cfg.get('momentum_tp_scaling', False):
            scaled_tp = calculate_momentum_scaled_take_profit(
                base_tp_pct,
                position.momentum_strength,
                self.momentum_config,
                position.momentum_continuation
            )
            position.take_profit_target = position.entry_price * (1 + scaled_tp) if position.side == "buy" else position.entry_price * (1 - scaled_tp)
            position.momentum_tp_multiplier = scaled_tp / base_tp_pct
        else:
            position.take_profit_target = position.entry_price * (1 + base_tp_pct) if position.side == "buy" else position.entry_price * (1 - base_tp_pct)
        
        # Set initial trailing stop
        base_trail_pct = exit_cfg.get('trailing_stop_pct', 0.008)
        if exit_cfg.get('momentum_trail_adjustment', False):
            adjusted_trail_pct = base_trail_pct * self._get_momentum_trail_adjustment(position.momentum_strength)
            position.momentum_trail_adjustment = adjusted_trail_pct / base_trail_pct
        else:
            adjusted_trail_pct = base_trail_pct
        
        if position.side == "buy":
            position.trailing_stop = position.entry_price * (1 - adjusted_trail_pct)
        else:
            position.trailing_stop = position.entry_price * (1 + adjusted_trail_pct)
        
        logger.info(f"Set initial exit levels for {position.symbol}: TP={position.take_profit_target:.4f}, Trail={position.trailing_stop:.4f}")
    
    def _get_momentum_trail_adjustment(self, momentum_strength: float) -> float:
        """Get trailing stop adjustment factor based on momentum strength."""
        
        if momentum_strength < self.momentum_config.weak_momentum:
            return self.momentum_config.momentum_trailing_adjustments['weak_momentum']
        elif momentum_strength < self.momentum_config.moderate_momentum:
            return self.momentum_config.momentum_trailing_adjustments['moderate_momentum']
        elif momentum_strength < self.momentum_config.strong_momentum:
            return self.momentum_config.momentum_trailing_adjustments['strong_momentum']
        else:
            return self.momentum_config.momentum_trailing_adjustments['very_strong_momentum']
    
    async def _check_momentum_adjustments(self, position: MomentumPosition):
        """Check and apply momentum-based adjustments to exit levels."""
        
        exit_cfg = self.config.get('exit_strategy', {})
        
        if not exit_cfg.get('momentum_aware_exits', False):
            return
        
        # Check if momentum has changed significantly
        if position.momentum_continuation and position.momentum_continuation.get('breakout_detected', False):
            # Extend take profit on breakout
            if exit_cfg.get('breakout_momentum', {}).get('enabled', False):
                extension_mult = exit_cfg['breakout_momentum'].get('momentum_extension_multiplier', 2.5)
                
                if position.side == "buy":
                    new_tp = position.entry_price + (position.take_profit_target - position.entry_price) * extension_mult
                    position.take_profit_target = max(position.take_profit_target, new_tp)
                else:
                    new_tp = position.entry_price - (position.entry_price - position.take_profit_target) * extension_mult
                    position.take_profit_target = min(position.take_profit_target, new_tp)
                
                self.total_momentum_extensions += 1
                logger.info(f"Extended take profit for {position.symbol} on breakout: {position.take_profit_target:.4f}")
        
        # Adjust trailing stop based on momentum strength
        if exit_cfg.get('momentum_trail_adjustment', False):
            current_trail_pct = abs(position.trailing_stop - position.entry_price) / position.entry_price
            base_trail_pct = exit_cfg.get('trailing_stop_pct', 0.008)
            adjusted_trail_pct = base_trail_pct * self._get_momentum_trail_adjustment(position.momentum_strength)
            
            if adjusted_trail_pct != current_trail_pct:
                if position.side == "buy":
                    position.trailing_stop = position.highest_price * (1 - adjusted_trail_pct)
                else:
                    position.trailing_stop = position.lowest_price * (1 + adjusted_trail_pct)
                
                logger.debug(f"Adjusted trailing stop for {position.symbol}: {position.trailing_stop:.4f}")
    
    async def _check_partial_exit_opportunities(self, position: MomentumPosition):
        """Check for momentum-based partial exit opportunities."""
        
        exit_cfg = self.config.get('exit_strategy', {})
        
        if not exit_cfg.get('momentum_partial_exits', False):
            return
        
        # Get momentum-based partial exit recommendation
        exit_pct, profit_pct = get_momentum_based_partial_exit(
            position.unrealized_pnl_pct * 100,  # Convert to percentage
            position.momentum_strength,
            self.config
        )
        
        if exit_pct > 0 and position.unrealized_pnl_pct * 100 >= profit_pct * 100:
            # Check if we haven't already taken this partial exit
            exit_key = f"{profit_pct:.3f}"
            if not any(exit['profit_level'] == exit_key for exit in position.partial_exits):
                await self._execute_partial_exit(position, exit_pct, profit_pct)
    
    async def _execute_partial_exit(self, position: MomentumPosition, exit_pct: float, profit_pct: float):
        """Execute a partial exit based on momentum."""
        
        try:
            # Calculate exit size
            exit_size = position.size * (exit_pct / 100)
            
            # Record the partial exit
            partial_exit = {
                'timestamp': time.time(),
                'profit_level': f"{profit_pct:.3f}",
                'exit_pct': exit_pct,
                'exit_size': exit_size,
                'exit_price': position.current_price,
                'momentum_strength': position.momentum_strength
            }
            
            position.partial_exits.append(partial_exit)
            
            # Reduce position size
            position.size -= exit_size
            
            # Update PnL
            position.update_pnl()
            
            self.total_partial_exits += 1
            
            logger.info(f"Executed partial exit for {position.symbol}: {exit_pct}% at {profit_pct:.1f}% profit (momentum: {position.momentum_strength:.3f})")
            
            # Notify external systems (implement as needed)
            await self._notify_partial_exit(position, partial_exit)
            
        except Exception as e:
            logger.error(f"Error executing partial exit for {position.symbol}: {e}")
    
    async def _notify_partial_exit(self, position: MomentumPosition, partial_exit: Dict):
        """Notify external systems of partial exit (placeholder for integration)."""
        # This would integrate with your execution system
        pass
    
    def should_exit_position(self, position: MomentumPosition) -> Tuple[bool, str, float]:
        """Determine if a position should be exited."""
        
        exit_cfg = self.config.get('exit_strategy', {})
        
        # Check take profit
        if position.side == "buy":
            if position.current_price >= position.take_profit_target:
                return True, "take_profit", position.take_profit_target
        else:
            if position.current_price <= position.take_profit_target:
                return True, "take_profit", position.take_profit_target
        
        # Check trailing stop
        if position.side == "buy":
            if position.current_price <= position.trailing_stop:
                return True, "trailing_stop", position.trailing_stop
        else:
            if position.current_price >= position.trailing_stop:
                return True, "trailing_stop", position.trailing_stop
        
        # Check momentum-based exit delays
        if exit_cfg.get('momentum_exit_delays', {}).get('enabled', False):
            should_delay, delay_seconds = should_delay_exit_for_momentum(
                position.momentum_strength, self.config
            )
            if should_delay:
                logger.info(f"Delaying exit for {position.symbol} due to strong momentum for {delay_seconds}s")
                return False, "delayed", delay_seconds
        
        return False, "hold", 0.0
    
    def get_position_summary(self, position_id: str) -> Optional[Dict]:
        """Get a summary of position performance and momentum metrics."""
        
        position = self.positions.get(position_id)
        if not position:
            return None
        
        return {
            'position_id': position.position_id,
            'symbol': position.symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'size': position.size,
            'unrealized_pnl': position.unrealized_pnl,
            'unrealized_pnl_pct': position.unrealized_pnl_pct,
            'momentum_strength': position.momentum_strength,
            'take_profit_target': position.take_profit_target,
            'trailing_stop': position.trailing_stop,
            'partial_exits_count': len(position.partial_exits),
            'momentum_continuation': position.momentum_continuation
        }
    
    def get_all_positions_summary(self) -> List[Dict]:
        """Get summary of all positions."""
        return [self.get_position_summary(pid) for pid in self.positions.keys()]
    
    async def close_position(self, position_id: str, reason: str = "manual"):
        """Close a position and remove it from management."""
        
        position = self.positions.get(position_id)
        if not position:
            logger.warning(f"Position {position_id} not found for closing")
            return
        
        # Log final metrics
        logger.info(f"Closing position {position_id} for {position.symbol}: {reason}")
        logger.info(f"Final PnL: {position.unrealized_pnl:.4f} ({position.unrealized_pnl_pct*100:.2f}%)")
        logger.info(f"Partial exits taken: {len(position.partial_exits)}")
        
        # Remove from management
        del self.positions[position_id]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the momentum position manager."""
        
        return {
            'total_positions_managed': len(self.positions),
            'total_momentum_exits': self.total_momentum_exits,
            'total_momentum_extensions': self.total_momentum_extensions,
            'total_partial_exits': self.total_partial_exits,
            'average_momentum_strength': np.mean([p.momentum_strength for p in self.positions.values()]) if self.positions else 0.0,
            'positions_with_strong_momentum': len([p for p in self.positions.values() if p.momentum_strength > 0.7])
        }
