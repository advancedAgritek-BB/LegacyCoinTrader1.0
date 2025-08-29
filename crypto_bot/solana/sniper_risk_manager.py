"""
Comprehensive Risk Management System for Memecoin Sniping

This module provides advanced risk management specifically designed for high-risk,
high-reward memecoin trading with sophisticated position sizing, exposure limits,
and dynamic risk adjustment.
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

from .pump_detector import PoolAnalysis
from .pool_analyzer import PoolMetrics
from .rapid_executor import ExecutionResult
from ..utils.telemetry import telemetry

logger = logging.getLogger(__name__)


@dataclass
class RiskProfile:
    """Risk profile configuration for different market conditions."""
    
    name: str
    max_position_size_pct: float  # Max % of account per position
    max_total_exposure_pct: float  # Max % of account in memecoin positions
    max_daily_loss_pct: float    # Max daily loss tolerance
    max_positions: int           # Max concurrent positions
    min_liquidity_usd: float     # Min pool liquidity required
    max_slippage_pct: float      # Max acceptable slippage
    stop_loss_pct: float         # Default stop loss
    take_profit_pct: float       # Default take profit
    correlation_limit: float     # Max correlation between positions


@dataclass
class PositionRisk:
    """Risk metrics for a specific position."""
    
    position_id: str
    token_mint: str
    entry_price: float
    current_price: float
    unrealized_pnl_pct: float
    volatility: float
    liquidity_risk: float       # Risk of liquidity disappearing
    correlation_risk: float     # Correlation with other positions
    time_decay_risk: float      # Risk from holding duration
    overall_risk_score: float   # Composite risk score


@dataclass
class PortfolioRisk:
    """Overall portfolio risk metrics."""
    
    total_exposure_pct: float
    daily_pnl_pct: float
    max_drawdown_pct: float
    position_concentration: float  # Gini coefficient of position sizes
    correlation_risk: float        # Average correlation between positions
    liquidity_risk: float          # Weighted average liquidity risk
    overall_risk_score: float      # Composite portfolio risk


class SniperRiskManager:
    """
    Advanced risk management system for memecoin sniping.
    
    Features:
    - Dynamic position sizing based on confidence and volatility
    - Real-time portfolio risk monitoring
    - Adaptive stop losses and take profits
    - Correlation-based exposure limits
    - Liquidity risk assessment
    - Emergency position liquidation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_config = config.get("sniper_risk_manager", {})
        
        # Risk profiles for different market conditions
        self.risk_profiles = {
            "conservative": RiskProfile(
                name="conservative",
                max_position_size_pct=0.02,    # 2%
                max_total_exposure_pct=0.10,   # 10%
                max_daily_loss_pct=0.05,       # 5%
                max_positions=3,
                min_liquidity_usd=50000,
                max_slippage_pct=0.03,          # 3%
                stop_loss_pct=0.15,             # 15%
                take_profit_pct=0.50,           # 50%
                correlation_limit=0.7
            ),
            "moderate": RiskProfile(
                name="moderate",
                max_position_size_pct=0.05,    # 5%
                max_total_exposure_pct=0.20,   # 20%
                max_daily_loss_pct=0.10,       # 10%
                max_positions=5,
                min_liquidity_usd=25000,
                max_slippage_pct=0.05,          # 5%
                stop_loss_pct=0.20,             # 20%
                take_profit_pct=0.75,           # 75%
                correlation_limit=0.8
            ),
            "aggressive": RiskProfile(
                name="aggressive",
                max_position_size_pct=0.10,    # 10%
                max_total_exposure_pct=0.50,   # 50%
                max_daily_loss_pct=0.20,       # 20%
                max_positions=10,
                min_liquidity_usd=10000,
                max_slippage_pct=0.10,          # 10%
                stop_loss_pct=0.30,             # 30%
                take_profit_pct=1.00,           # 100%
                correlation_limit=0.9
            )
        }
        
        # Current risk profile
        self.current_profile_name = self.risk_config.get("default_profile", "moderate")
        self.current_profile = self.risk_profiles[self.current_profile_name]
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []
        self.daily_pnl: float = 0.0
        self.session_start_time = time.time()
        
        # Risk state
        self.risk_state_file = Path(self.risk_config.get("state_file", "crypto_bot/logs/sniper_risk_state.json"))
        self.emergency_mode = False
        self.cooling_off_until: Optional[float] = None
        
        # Monitoring
        self.price_monitors: Dict[str, asyncio.Task] = {}
        self.risk_alerts: deque = deque(maxlen=100)
        
        # Historical data
        self.volatility_cache: Dict[str, Tuple[float, float]] = {}  # token -> (volatility, timestamp)
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        
        # Statistics
        self.stats = {
            "positions_opened": 0,
            "positions_closed": 0,
            "emergency_exits": 0,
            "risk_rejections": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
        # Load previous state
        self._load_state()
        
    async def start(self):
        """Start the risk management system."""
        # Start monitoring tasks
        self._start_monitoring_tasks()
        logger.info(f"Sniper risk manager started with {self.current_profile_name} profile")
        
    async def stop(self):
        """Stop the risk management system."""
        # Stop all monitoring tasks
        for task in self.price_monitors.values():
            task.cancel()
        await asyncio.gather(*self.price_monitors.values(), return_exceptions=True)
        
        # Save state
        self._save_state()
        logger.info("Sniper risk manager stopped")
        
    def calculate_position_size(
        self,
        pump_analysis: PoolAnalysis,
        pool_metrics: PoolMetrics,
        account_balance: float
    ) -> float:
        """
        Calculate optimal position size based on risk-adjusted confidence.
        
        Args:
            pump_analysis: Pump detection analysis
            pool_metrics: Pool quality metrics
            account_balance: Current account balance
            
        Returns:
            Position size in SOL
        """
        try:
            # Base position size from risk profile
            base_size_pct = self.current_profile.max_position_size_pct
            
            # Confidence adjustment
            confidence_adj = pump_analysis.risk_adjusted_score
            
            # Volatility adjustment
            volatility = self._estimate_volatility(pump_analysis.token_mint)
            volatility_adj = max(0.5, 1.0 - (volatility - 0.5))  # Reduce size for high volatility
            
            # Liquidity adjustment
            liquidity_adj = min(1.0, pool_metrics.total_liquidity_usd / 100000)  # Scale up to 100k USD
            
            # Timing adjustment
            timing_adj = pump_analysis.timing_score
            
            # Portfolio heat adjustment
            heat_adj = self._calculate_portfolio_heat_adjustment()
            
            # Calculate final position size
            adjusted_size_pct = (
                base_size_pct * 
                confidence_adj * 
                volatility_adj * 
                liquidity_adj * 
                timing_adj * 
                heat_adj
            )
            
            position_size_sol = account_balance * adjusted_size_pct
            
            # Apply minimum and maximum limits
            min_size = self.risk_config.get("min_position_size_sol", 0.01)
            max_size = account_balance * self.current_profile.max_position_size_pct
            
            position_size_sol = max(min_size, min(position_size_sol, max_size))
            
            logger.info(
                f"Position size calculated: {position_size_sol:.4f} SOL "
                f"({adjusted_size_pct:.1%} of account) for {pump_analysis.token_mint[:8]}..."
            )
            
            return position_size_sol
            
        except Exception as exc:
            logger.error(f"Position size calculation failed: {exc}")
            return account_balance * 0.01  # 1% fallback
            
    def validate_new_position(
        self,
        pump_analysis: PoolAnalysis,
        pool_metrics: PoolMetrics,
        position_size_sol: float
    ) -> Tuple[bool, str]:
        """
        Validate if a new position meets risk criteria.
        
        Args:
            pump_analysis: Pump analysis
            pool_metrics: Pool metrics
            position_size_sol: Proposed position size
            
        Returns:
            (is_valid, reason)
        """
        try:
            # Check if in emergency mode
            if self.emergency_mode:
                return False, "Emergency mode active"
                
            # Check cooling off period
            if self.cooling_off_until and time.time() < self.cooling_off_until:
                return False, "Cooling off period active"
                
            # Check maximum positions
            if len(self.positions) >= self.current_profile.max_positions:
                return False, f"Maximum positions ({self.current_profile.max_positions}) reached"
                
            # Check daily loss limit
            if self.daily_pnl < -self.current_profile.max_daily_loss_pct:
                return False, f"Daily loss limit exceeded ({self.daily_pnl:.1%})"
                
            # Check total exposure
            current_exposure = self._calculate_total_exposure()
            new_exposure_pct = (current_exposure + position_size_sol) / self._get_account_balance()
            if new_exposure_pct > self.current_profile.max_total_exposure_pct:
                return False, f"Total exposure limit exceeded ({new_exposure_pct:.1%})"
                
            # Check pool liquidity
            if pool_metrics.total_liquidity_usd < self.current_profile.min_liquidity_usd:
                return False, f"Insufficient pool liquidity (${pool_metrics.total_liquidity_usd:,.0f})"
                
            # Check slippage tolerance
            if pool_metrics.slippage_1pct > self.current_profile.max_slippage_pct:
                return False, f"Slippage too high ({pool_metrics.slippage_1pct:.1%})"
                
            # Check rug risk
            if pump_analysis.rug_risk_score > 0.7:
                return False, f"Rug risk too high ({pump_analysis.rug_risk_score:.1%})"
                
            # Check correlation with existing positions
            if not self._check_correlation_limits(pump_analysis.token_mint):
                return False, "Correlation limits exceeded"
                
            # Check market conditions
            if not self._check_market_conditions():
                return False, "Unfavorable market conditions"
                
            return True, "Position validated"
            
        except Exception as exc:
            logger.error(f"Position validation failed: {exc}")
            return False, f"Validation error: {exc}"
            
    def add_position(self, execution_result: ExecutionResult) -> str:
        """
        Add a new position to risk tracking.
        
        Args:
            execution_result: Result from trade execution
            
        Returns:
            Position ID
        """
        try:
            position_id = execution_result.position_id or f"pos_{int(time.time())}"
            
            position = {
                "position_id": position_id,
                "token_mint": execution_result.transaction_hash,  # Placeholder - should be actual mint
                "entry_price": execution_result.executed_price,
                "entry_time": execution_result.timestamp,
                "amount_sol": execution_result.executed_amount,
                "amount_tokens": execution_result.amount_tokens,
                "stop_loss_price": execution_result.executed_price * (1 - self.current_profile.stop_loss_pct),
                "take_profit_price": execution_result.executed_price * (1 + self.current_profile.take_profit_pct),
                "current_price": execution_result.executed_price,
                "unrealized_pnl": 0.0,
                "max_profit": 0.0,
                "max_drawdown": 0.0,
                "status": "active"
            }
            
            self.positions[position_id] = position
            self.stats["positions_opened"] += 1
            
            # Start price monitoring
            self._start_position_monitoring(position_id)
            
            logger.info(f"Position added: {position_id}")
            return position_id
            
        except Exception as exc:
            logger.error(f"Failed to add position: {exc}")
            return ""
            
    async def update_position_price(self, position_id: str, current_price: float):
        """Update position with current price and check risk limits."""
        try:
            position = self.positions.get(position_id)
            if not position:
                return
                
            old_price = position["current_price"]
            position["current_price"] = current_price
            
            # Calculate P&L
            price_change = (current_price - position["entry_price"]) / position["entry_price"]
            position["unrealized_pnl"] = price_change
            
            # Update max profit and drawdown
            position["max_profit"] = max(position["max_profit"], price_change)
            if position["max_profit"] > 0:
                drawdown_from_peak = (current_price - (position["entry_price"] * (1 + position["max_profit"]))) / (position["entry_price"] * (1 + position["max_profit"]))
                position["max_drawdown"] = min(position["max_drawdown"], drawdown_from_peak)
                
            # Check exit conditions
            await self._check_exit_conditions(position_id)
            
            # Update daily P&L
            self._update_daily_pnl()
            
            # Check portfolio risk
            await self._check_portfolio_risk()
            
        except Exception as exc:
            logger.error(f"Failed to update position price: {exc}")
            
    async def close_position(self, position_id: str, exit_price: float, reason: str = "manual"):
        """Close a position and update statistics."""
        try:
            position = self.positions.get(position_id)
            if not position:
                return
                
            # Calculate final P&L
            price_change = (exit_price - position["entry_price"]) / position["entry_price"]
            realized_pnl = price_change * position["amount_sol"]
            
            # Update position
            position["exit_price"] = exit_price
            position["exit_time"] = time.time()
            position["realized_pnl"] = realized_pnl
            position["exit_reason"] = reason
            position["status"] = "closed"
            
            # Move to history
            self.position_history.append(position.copy())
            del self.positions[position_id]
            
            # Stop monitoring
            if position_id in self.price_monitors:
                self.price_monitors[position_id].cancel()
                del self.price_monitors[position_id]
                
            # Update statistics
            self.stats["positions_closed"] += 1
            self.stats["total_pnl"] += realized_pnl
            self._update_daily_pnl()
            
            # Check if emergency exit
            if reason == "emergency":
                self.stats["emergency_exits"] += 1
                
            logger.info(
                f"Position closed: {position_id} | "
                f"P&L: {price_change:.1%} | "
                f"Reason: {reason}"
            )
            
        except Exception as exc:
            logger.error(f"Failed to close position: {exc}")
            
    async def emergency_liquidation(self):
        """Emergency liquidation of all positions."""
        logger.critical("EMERGENCY LIQUIDATION TRIGGERED")
        
        self.emergency_mode = True
        self.cooling_off_until = time.time() + 3600  # 1 hour cooling off
        
        # Close all positions
        for position_id in list(self.positions.keys()):
            await self.close_position(position_id, 0.0, "emergency")
            
        # Send alert
        self._add_risk_alert("emergency_liquidation", "All positions liquidated due to emergency")
        
        telemetry.inc("sniper_risk.emergency_liquidations")
        
    def calculate_stop_loss(self, position_id: str) -> float:
        """Calculate dynamic stop loss based on position performance."""
        position = self.positions.get(position_id)
        if not position:
            return 0.0
            
        entry_price = position["entry_price"]
        current_profit = position["unrealized_pnl"]
        
        # Base stop loss
        base_stop = entry_price * (1 - self.current_profile.stop_loss_pct)
        
        # Trailing stop if in profit
        if current_profit > 0.1:  # 10% profit
            trailing_stop_pct = 0.5  # Trail by 50% of profit
            trailing_stop = position["current_price"] * (1 - trailing_stop_pct * current_profit)
            return max(base_stop, trailing_stop)
            
        return base_stop
        
    def calculate_take_profit(self, position_id: str) -> float:
        """Calculate dynamic take profit based on market conditions."""
        position = self.positions.get(position_id)
        if not position:
            return 0.0
            
        entry_price = position["entry_price"]
        
        # Base take profit
        base_tp = entry_price * (1 + self.current_profile.take_profit_pct)
        
        # Adjust based on volatility
        volatility = self._estimate_volatility(position["token_mint"])
        if volatility > 1.0:  # High volatility
            return base_tp * 1.5  # Higher target
        elif volatility < 0.3:  # Low volatility
            return base_tp * 0.8  # Lower target
            
        return base_tp
        
    def get_portfolio_risk(self) -> PortfolioRisk:
        """Calculate current portfolio risk metrics."""
        try:
            total_balance = self._get_account_balance()
            total_exposure = self._calculate_total_exposure()
            
            # Daily P&L
            daily_pnl_pct = self.daily_pnl
            
            # Max drawdown
            max_dd = self.stats["max_drawdown"]
            
            # Position concentration
            if self.positions:
                position_sizes = [pos["amount_sol"] for pos in self.positions.values()]
                concentration = self._calculate_gini_coefficient(position_sizes)
            else:
                concentration = 0.0
                
            # Correlation risk
            corr_risk = self._calculate_correlation_risk()
            
            # Liquidity risk
            liq_risk = self._calculate_liquidity_risk()
            
            # Overall risk score
            risk_factors = [
                total_exposure / total_balance,
                abs(daily_pnl_pct),
                max_dd,
                concentration,
                corr_risk,
                liq_risk
            ]
            overall_risk = np.mean(risk_factors)
            
            return PortfolioRisk(
                total_exposure_pct=total_exposure / total_balance,
                daily_pnl_pct=daily_pnl_pct,
                max_drawdown_pct=max_dd,
                position_concentration=concentration,
                correlation_risk=corr_risk,
                liquidity_risk=liq_risk,
                overall_risk_score=overall_risk
            )
            
        except Exception as exc:
            logger.error(f"Portfolio risk calculation failed: {exc}")
            return PortfolioRisk(
                total_exposure_pct=0.0,
                daily_pnl_pct=0.0,
                max_drawdown_pct=0.0,
                position_concentration=0.0,
                correlation_risk=0.0,
                liquidity_risk=0.0,
                overall_risk_score=0.0
            )
            
    # Helper methods
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total exposure across all positions."""
        return sum(pos["amount_sol"] for pos in self.positions.values())
        
    def _get_account_balance(self) -> float:
        """Get current account balance."""
        # This would integrate with your wallet balance
        return 10.0  # Placeholder: 10 SOL
        
    def _estimate_volatility(self, token_mint: str) -> float:
        """Estimate token volatility."""
        # Check cache
        if token_mint in self.volatility_cache:
            volatility, timestamp = self.volatility_cache[token_mint]
            if time.time() - timestamp < 300:  # 5 minute cache
                return volatility
                
        # Calculate or fetch volatility
        # This would integrate with price data
        volatility = 0.5  # Placeholder
        self.volatility_cache[token_mint] = (volatility, time.time())
        return volatility
        
    def _calculate_portfolio_heat_adjustment(self) -> float:
        """Calculate adjustment factor based on portfolio heat."""
        current_exposure_pct = self._calculate_total_exposure() / self._get_account_balance()
        max_exposure_pct = self.current_profile.max_total_exposure_pct
        
        heat_ratio = current_exposure_pct / max_exposure_pct
        
        if heat_ratio < 0.5:
            return 1.0  # Normal sizing
        elif heat_ratio < 0.8:
            return 0.7  # Reduce sizing
        else:
            return 0.3  # Significantly reduce sizing
            
    def _check_correlation_limits(self, new_token: str) -> bool:
        """Check if new token would exceed correlation limits."""
        if not self.positions:
            return True
            
        # Calculate correlation with existing positions
        # This would use actual price correlation calculation
        # Placeholder implementation
        return True
        
    def _check_market_conditions(self) -> bool:
        """Check if overall market conditions are suitable."""
        # This would check broader market conditions
        # Placeholder implementation
        return True
        
    async def _check_exit_conditions(self, position_id: str):
        """Check if position should be exited."""
        position = self.positions.get(position_id)
        if not position:
            return
            
        current_price = position["current_price"]
        stop_loss = self.calculate_stop_loss(position_id)
        take_profit = self.calculate_take_profit(position_id)
        
        # Check stop loss
        if current_price <= stop_loss:
            await self.close_position(position_id, current_price, "stop_loss")
            return
            
        # Check take profit
        if current_price >= take_profit:
            await self.close_position(position_id, current_price, "take_profit")
            return
            
        # Check time-based exit (positions older than 24 hours)
        if time.time() - position["entry_time"] > 86400:
            await self.close_position(position_id, current_price, "time_exit")
            return
            
    async def _check_portfolio_risk(self):
        """Check overall portfolio risk and take action if needed."""
        portfolio_risk = self.get_portfolio_risk()
        
        # Check for emergency conditions
        emergency_conditions = [
            portfolio_risk.daily_pnl_pct < -0.3,  # -30% daily loss
            portfolio_risk.overall_risk_score > 0.9,  # Extreme risk
            portfolio_risk.max_drawdown_pct < -0.5,  # -50% drawdown
        ]
        
        if any(emergency_conditions):
            await self.emergency_liquidation()
            return
            
        # Check for risk reduction needs
        if portfolio_risk.overall_risk_score > 0.7:
            await self._reduce_portfolio_risk()
            
    async def _reduce_portfolio_risk(self):
        """Reduce portfolio risk by closing worst positions."""
        if not self.positions:
            return
            
        # Sort positions by risk (worst first)
        sorted_positions = sorted(
            self.positions.items(),
            key=lambda x: x[1]["unrealized_pnl"]  # Close worst performers first
        )
        
        # Close worst 25% of positions
        close_count = max(1, len(sorted_positions) // 4)
        for i in range(close_count):
            position_id, position = sorted_positions[i]
            await self.close_position(position_id, position["current_price"], "risk_reduction")
            
    def _start_position_monitoring(self, position_id: str):
        """Start price monitoring for a position."""
        task = asyncio.create_task(self._monitor_position_price(position_id))
        self.price_monitors[position_id] = task
        
    async def _monitor_position_price(self, position_id: str):
        """Monitor position price in background."""
        while position_id in self.positions:
            try:
                # Fetch current price
                # This would integrate with price feeds
                current_price = 0.001  # Placeholder
                
                await self.update_position_price(position_id, current_price)
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Position monitoring error: {exc}")
                await asyncio.sleep(10)
                
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        # Daily P&L reset task
        asyncio.create_task(self._daily_reset_task())
        
    async def _daily_reset_task(self):
        """Reset daily metrics at start of new day."""
        while True:
            try:
                await asyncio.sleep(86400)  # 24 hours
                self.daily_pnl = 0.0
                self.session_start_time = time.time()
                logger.info("Daily metrics reset")
            except Exception as exc:
                logger.error(f"Daily reset task error: {exc}")
                
    def _update_daily_pnl(self):
        """Update daily P&L from all positions."""
        unrealized_pnl = sum(
            pos["unrealized_pnl"] * pos["amount_sol"] 
            for pos in self.positions.values()
        )
        realized_pnl = sum(
            pos.get("realized_pnl", 0) 
            for pos in self.position_history 
            if pos.get("exit_time", 0) > self.session_start_time
        )
        
        self.daily_pnl = (unrealized_pnl + realized_pnl) / self._get_account_balance()
        
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for concentration measurement."""
        if not values or len(values) <= 1:
            return 0.0
            
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
    def _calculate_correlation_risk(self) -> float:
        """Calculate average correlation risk across positions."""
        # This would calculate actual correlations between positions
        # Placeholder implementation
        return 0.5
        
    def _calculate_liquidity_risk(self) -> float:
        """Calculate weighted average liquidity risk."""
        # This would assess liquidity risk for each position
        # Placeholder implementation
        return 0.3
        
    def _add_risk_alert(self, alert_type: str, message: str):
        """Add a risk alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": time.time()
        }
        self.risk_alerts.append(alert)
        logger.warning(f"Risk Alert [{alert_type}]: {message}")
        
    def _save_state(self):
        """Save risk management state to disk."""
        try:
            state = {
                "positions": self.positions,
                "daily_pnl": self.daily_pnl,
                "session_start_time": self.session_start_time,
                "emergency_mode": self.emergency_mode,
                "cooling_off_until": self.cooling_off_until,
                "stats": self.stats
            }
            
            with open(self.risk_state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as exc:
            logger.error(f"Failed to save risk state: {exc}")
            
    def _load_state(self):
        """Load risk management state from disk."""
        try:
            if self.risk_state_file.exists():
                with open(self.risk_state_file, 'r') as f:
                    state = json.load(f)
                    
                self.positions = state.get("positions", {})
                self.daily_pnl = state.get("daily_pnl", 0.0)
                self.session_start_time = state.get("session_start_time", time.time())
                self.emergency_mode = state.get("emergency_mode", False)
                self.cooling_off_until = state.get("cooling_off_until")
                self.stats.update(state.get("stats", {}))
                
                logger.info(f"Risk state loaded: {len(self.positions)} active positions")
                
        except Exception as exc:
            logger.error(f"Failed to load risk state: {exc}")
            
    def get_statistics(self) -> Dict:
        """Get risk management statistics."""
        stats = self.stats.copy()
        stats["active_positions"] = len(self.positions)
        stats["daily_pnl_pct"] = self.daily_pnl
        stats["emergency_mode"] = self.emergency_mode
        stats["current_profile"] = self.current_profile_name
        
        if self.stats["positions_closed"] > 0:
            stats["win_rate"] = len([p for p in self.position_history if p.get("realized_pnl", 0) > 0]) / self.stats["positions_closed"]
        else:
            stats["win_rate"] = 0.0
            
        return stats
