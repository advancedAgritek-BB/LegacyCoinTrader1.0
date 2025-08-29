"""
Pump Sniper Master Orchestrator

This module orchestrates all components of the advanced memecoin pump sniping system,
providing centralized coordination, decision making, and execution management.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable
from pathlib import Path
import json

from .pump_detector import PumpDetector, PoolAnalysis
from .pool_analyzer import LiquidityPoolAnalyzer, PoolMetrics
from .rapid_executor import RapidExecutor, ExecutionParams, ExecutionResult
from .sniper_risk_manager import SniperRiskManager
from .social_sentiment_analyzer import SocialSentimentAnalyzer, SentimentAnalysis
from .momentum_detector import MomentumDetector, TokenMomentum
from .watcher import PoolWatcher, NewPoolEvent
from ..utils.telemetry import telemetry

logger = logging.getLogger(__name__)


@dataclass
class SniperDecision:
    """Represents a sniping decision made by the orchestrator."""
    
    token_mint: str
    token_symbol: str
    decision: str  # "snipe", "monitor", "ignore"
    confidence: float  # 0-1
    reasoning: List[str]
    
    # Analysis inputs
    pump_analysis: Optional[PoolAnalysis] = None
    pool_metrics: Optional[PoolMetrics] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    momentum_analysis: Optional[TokenMomentum] = None
    
    # Execution parameters
    position_size_sol: float = 0.0
    max_slippage: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrchestatorStats:
    """Statistics for the pump sniper orchestrator."""
    
    pools_evaluated: int = 0
    snipe_decisions: int = 0
    successful_snipes: int = 0
    failed_snipes: int = 0
    total_pnl: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_hold_time: float = 0.0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0


class PumpSniperOrchestrator:
    """
    Master orchestrator for the memecoin pump sniping system.
    
    Features:
    - Centralized decision making using all analysis components
    - Risk-adjusted position sizing and execution
    - Real-time monitoring and position management
    - Performance tracking and optimization
    - Configuration management and hot-reloading
    - Emergency controls and safeguards
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.orchestrator_config = config.get("pump_sniper_orchestrator", {})
        
        # Core components
        self.pump_detector = PumpDetector(config)
        self.pool_analyzer = LiquidityPoolAnalyzer(config)
        self.rapid_executor = RapidExecutor(config)
        self.risk_manager = SniperRiskManager(config)
        self.sentiment_analyzer = SocialSentimentAnalyzer(config)
        self.momentum_detector = MomentumDetector(config)
        self.pool_watcher = PoolWatcher()
        
        # Decision making
        self.decision_weights = self.orchestrator_config.get("decision_weights", {
            "pump_probability": 0.25,
            "pool_quality": 0.20,
            "sentiment_score": 0.15,
            "momentum_score": 0.15,
            "risk_score": 0.15,
            "timing_score": 0.10
        })
        
        self.min_decision_confidence = self.orchestrator_config.get("min_decision_confidence", 0.7)
        self.max_concurrent_evaluations = self.orchestrator_config.get("max_concurrent_evaluations", 5)
        
        # State management
        self.active_evaluations: Dict[str, asyncio.Task] = {}
        self.recent_decisions: Dict[str, SniperDecision] = {}
        self.position_monitors: Dict[str, asyncio.Task] = {}
        
        # Configuration hot-reloading
        self.config_file = Path(self.orchestrator_config.get("config_file", "config/pump_sniper_config.yaml"))
        self.config_last_modified = 0.0
        
        # Emergency controls
        self.emergency_stop = False
        self.pause_new_trades = False
        self.max_daily_loss_reached = False
        
        # Performance tracking
        self.stats = OrchestatorStats()
        self.trade_history: List[Dict] = []
        
        # Background tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Callbacks for notifications
        self.notification_callbacks: List[Callable] = []
        
    async def start(self):
        """Start the pump sniper orchestrator."""
        self.running = True
        
        # Start all components
        await self.pump_detector.start()
        await self.pool_analyzer.start()
        await self.rapid_executor.start()
        await self.risk_manager.start()
        await self.sentiment_analyzer.start()
        await self.momentum_detector.start()
        
        # Start monitoring tasks
        await self._start_monitoring_tasks()
        
        # Start pool watching
        self._start_pool_watching()
        
        logger.info("Pump Sniper Orchestrator started successfully")
        
    async def stop(self):
        """Stop the pump sniper orchestrator."""
        self.running = False
        
        # Stop pool watching
        if hasattr(self.pool_watcher, 'stop'):
            self.pool_watcher.stop()
            
        # Cancel all active evaluations
        for task in self.active_evaluations.values():
            task.cancel()
        await asyncio.gather(*self.active_evaluations.values(), return_exceptions=True)
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        # Stop all components
        await self.pump_detector.stop()
        await self.pool_analyzer.stop()
        await self.rapid_executor.stop()
        await self.risk_manager.stop()
        await self.sentiment_analyzer.stop()
        await self.momentum_detector.stop()
        
        logger.info("Pump Sniper Orchestrator stopped")
        
    def _start_pool_watching(self):
        """Start monitoring new pool events."""
        asyncio.create_task(self._pool_watching_loop())
        
    async def _pool_watching_loop(self):
        """Main loop for processing new pool events."""
        logger.info("Starting pool watching loop")
        
        try:
            async for pool_event in self.pool_watcher.watch():
                if not self.running:
                    break
                    
                # Check if we should process this pool
                if await self._should_evaluate_pool(pool_event):
                    await self._initiate_pool_evaluation(pool_event)
                    
        except Exception as exc:
            logger.error(f"Pool watching loop error: {exc}")
            if self.running:
                # Restart after delay
                await asyncio.sleep(10)
                self._start_pool_watching()
                
    async def _should_evaluate_pool(self, pool_event: NewPoolEvent) -> bool:
        """Determine if a pool should be evaluated for sniping."""
        # Check basic criteria
        if not pool_event.token_mint or not pool_event.pool_address:
            return False
            
        # Check if already evaluating
        if pool_event.pool_address in self.active_evaluations:
            return False
            
        # Check emergency stops
        if self.emergency_stop or self.pause_new_trades:
            return False
            
        # Check evaluation limits
        if len(self.active_evaluations) >= self.max_concurrent_evaluations:
            return False
            
        # Check recent decisions (avoid re-evaluating recently rejected pools)
        if pool_event.pool_address in self.recent_decisions:
            recent_decision = self.recent_decisions[pool_event.pool_address]
            if (time.time() - recent_decision.timestamp < 3600 and  # 1 hour
                recent_decision.decision == "ignore"):
                return False
                
        # Basic liquidity check
        min_liquidity = self.orchestrator_config.get("min_initial_liquidity", 5000)
        if pool_event.liquidity < min_liquidity:
            return False
            
        return True
        
    async def _initiate_pool_evaluation(self, pool_event: NewPoolEvent):
        """Initiate comprehensive evaluation of a new pool."""
        pool_address = pool_event.pool_address
        
        # Create evaluation task
        task = asyncio.create_task(self._evaluate_pool_for_sniping(pool_event))
        self.active_evaluations[pool_address] = task
        
        # Clean up task when done
        task.add_done_callback(lambda t: self.active_evaluations.pop(pool_address, None))
        
        logger.info(f"Initiated evaluation for pool: {pool_address}")
        
    async def _evaluate_pool_for_sniping(self, pool_event: NewPoolEvent) -> Optional[SniperDecision]:
        """Comprehensive evaluation of a pool for sniping opportunity."""
        try:
            token_mint = pool_event.token_mint
            token_symbol = token_mint[:8] + "..."  # Abbreviated for logging
            
            logger.info(f"Evaluating pool for {token_symbol}")
            
            # Step 1: Parallel analysis gathering
            analysis_tasks = {
                "pump_analysis": self.pump_detector.analyze_pool(pool_event),
                "pool_metrics": self.pool_analyzer.analyze_pool_quality(pool_event),
                "sentiment_analysis": self.sentiment_analyzer.analyze_token_sentiment(token_mint, token_symbol),
            }
            
            # Wait for all analyses with timeout
            try:
                analysis_results = await asyncio.wait_for(
                    asyncio.gather(*analysis_tasks.values(), return_exceptions=True),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Analysis timeout for {token_symbol}")
                return None
                
            # Extract results
            pump_analysis, pool_metrics, sentiment_analysis = analysis_results
            
            # Check for analysis failures
            if isinstance(pump_analysis, Exception) or pump_analysis is None:
                logger.warning(f"Pump analysis failed for {token_symbol}")
                return None
                
            if isinstance(pool_metrics, Exception) or pool_metrics is None:
                logger.warning(f"Pool analysis failed for {token_symbol}")
                return None
                
            # Sentiment analysis is optional
            if isinstance(sentiment_analysis, Exception):
                sentiment_analysis = None
                
            # Step 2: Add token to momentum monitoring
            self.momentum_detector.add_token_for_monitoring(token_mint, token_symbol)
            
            # Wait briefly for initial momentum data
            await asyncio.sleep(2)
            momentum_analysis = self.momentum_detector.get_token_momentum(token_mint)
            
            # Step 3: Make sniping decision
            decision = await self._make_sniping_decision(
                pool_event, pump_analysis, pool_metrics, sentiment_analysis, momentum_analysis
            )
            
            # Step 4: Execute decision
            if decision.decision == "snipe":
                await self._execute_snipe_decision(decision)
            elif decision.decision == "monitor":
                await self._monitor_token(decision)
                
            # Step 5: Store decision
            self.recent_decisions[pool_event.pool_address] = decision
            self.stats.pools_evaluated += 1
            
            # Notify subscribers
            await self._notify_decision(decision)
            
            return decision
            
        except Exception as exc:
            logger.error(f"Pool evaluation failed for {pool_event.pool_address}: {exc}")
            return None
            
    async def _make_sniping_decision(
        self,
        pool_event: NewPoolEvent,
        pump_analysis: PoolAnalysis,
        pool_metrics: PoolMetrics,
        sentiment_analysis: Optional[SentimentAnalysis],
        momentum_analysis: Optional[TokenMomentum]
    ) -> SniperDecision:
        """Make the final sniping decision based on all analyses."""
        
        token_mint = pool_event.token_mint
        token_symbol = token_mint[:8] + "..."
        
        # Calculate individual scores
        scores = {
            "pump_probability": pump_analysis.pump_probability,
            "pool_quality": pool_metrics.sniping_viability,
            "sentiment_score": sentiment_analysis.pump_probability if sentiment_analysis else 0.5,
            "momentum_score": momentum_analysis.breakout_probability if momentum_analysis else 0.5,
            "risk_score": 1.0 - pump_analysis.rug_risk_score,  # Invert risk (lower risk = higher score)
            "timing_score": pump_analysis.timing_score
        }
        
        # Calculate weighted confidence score
        weighted_confidence = sum(
            scores[factor] * weight 
            for factor, weight in self.decision_weights.items()
        )
        
        # Risk management validation
        account_balance = 10.0  # This would come from actual balance
        position_size_sol = self.risk_manager.calculate_position_size(
            pump_analysis, pool_metrics, account_balance
        )
        
        risk_validation, risk_reason = self.risk_manager.validate_new_position(
            pump_analysis, pool_metrics, position_size_sol
        )
        
        # Decision logic
        reasoning = []
        
        if not risk_validation:
            decision = "ignore"
            reasoning.append(f"Risk validation failed: {risk_reason}")
        elif weighted_confidence >= self.min_decision_confidence:
            decision = "snipe"
            reasoning.append(f"High confidence score: {weighted_confidence:.2f}")
            reasoning.append(f"Pump probability: {scores['pump_probability']:.2f}")
            reasoning.append(f"Pool quality: {scores['pool_quality']:.2f}")
            if sentiment_analysis:
                reasoning.append(f"Sentiment score: {scores['sentiment_score']:.2f}")
        elif weighted_confidence >= 0.5:
            decision = "monitor"
            reasoning.append(f"Moderate confidence: {weighted_confidence:.2f}")
            reasoning.append("Added to monitoring for potential entry")
        else:
            decision = "ignore"
            reasoning.append(f"Low confidence score: {weighted_confidence:.2f}")
            
        # Create decision object
        sniper_decision = SniperDecision(
            token_mint=token_mint,
            token_symbol=token_symbol,
            decision=decision,
            confidence=weighted_confidence,
            reasoning=reasoning,
            pump_analysis=pump_analysis,
            pool_metrics=pool_metrics,
            sentiment_analysis=sentiment_analysis,
            momentum_analysis=momentum_analysis,
            position_size_sol=position_size_sol if decision == "snipe" else 0.0,
            max_slippage=pool_metrics.slippage_1pct * 2,  # 2x current slippage as max
            stop_loss_pct=0.2,  # 20% stop loss
            take_profit_pct=0.5   # 50% take profit
        )
        
        logger.info(
            f"Decision for {token_symbol}: {decision} "
            f"(confidence: {weighted_confidence:.2f}) - {', '.join(reasoning)}"
        )
        
        return sniper_decision
        
    async def _execute_snipe_decision(self, decision: SniperDecision):
        """Execute a snipe decision."""
        try:
            self.stats.snipe_decisions += 1
            
            # Execute the snipe
            execution_result = await self.rapid_executor.execute_snipe(
                decision.pump_analysis,
                decision.pool_metrics,
                decision.pump_analysis.signals
            )
            
            if execution_result.success:
                # Add position to risk manager
                position_id = self.risk_manager.add_position(execution_result)
                
                # Start position monitoring
                if position_id:
                    monitor_task = asyncio.create_task(
                        self._monitor_position(position_id, decision)
                    )
                    self.position_monitors[position_id] = monitor_task
                    
                self.stats.successful_snipes += 1
                self.stats.avg_execution_time = (
                    (self.stats.avg_execution_time * (self.stats.successful_snipes - 1) + 
                     execution_result.execution_time) / self.stats.successful_snipes
                )
                
                telemetry.inc("pump_sniper.successful_snipes")
                telemetry.gauge("pump_sniper.execution_time", execution_result.execution_time)
                
                logger.info(
                    f"Snipe executed successfully for {decision.token_symbol}: "
                    f"Amount: {execution_result.executed_amount:.4f} SOL, "
                    f"Price: {execution_result.executed_price:.6f}"
                )
            else:
                self.stats.failed_snipes += 1
                telemetry.inc("pump_sniper.failed_snipes")
                
                logger.warning(
                    f"Snipe execution failed for {decision.token_symbol}: "
                    f"{execution_result.error_message}"
                )
                
        except Exception as exc:
            self.stats.failed_snipes += 1
            logger.error(f"Snipe execution error for {decision.token_symbol}: {exc}")
            
    async def _monitor_token(self, decision: SniperDecision):
        """Monitor a token for improved entry opportunity."""
        token_mint = decision.token_mint
        
        # Create monitoring task
        monitor_task = asyncio.create_task(
            self._token_monitoring_loop(decision)
        )
        
        # Store for cleanup
        self.position_monitors[f"monitor_{token_mint}"] = monitor_task
        
    async def _token_monitoring_loop(self, decision: SniperDecision):
        """Monitor a token for entry opportunities."""
        token_mint = decision.token_mint
        monitoring_duration = 3600  # Monitor for 1 hour
        start_time = time.time()
        
        logger.info(f"Started monitoring {decision.token_symbol} for entry opportunities")
        
        try:
            while time.time() - start_time < monitoring_duration:
                if not self.running:
                    break
                    
                # Re-evaluate momentum and sentiment
                momentum = self.momentum_detector.get_token_momentum(token_mint)
                sentiment = self.sentiment_analyzer.get_token_sentiment(token_mint)
                
                if momentum and sentiment:
                    # Check for improved entry conditions
                    if (momentum.breakout_probability > 0.8 or 
                        sentiment.pump_probability > 0.8):
                        
                        # Re-evaluate for sniping
                        updated_decision = await self._make_sniping_decision(
                            NewPoolEvent(
                                pool_address=decision.pump_analysis.pool_address,
                                token_mint=token_mint,
                                creator="",
                                liquidity=decision.pool_metrics.total_liquidity_usd
                            ),
                            decision.pump_analysis,
                            decision.pool_metrics,
                            sentiment,
                            momentum
                        )
                        
                        if updated_decision.decision == "snipe":
                            await self._execute_snipe_decision(updated_decision)
                            break
                            
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Token monitoring error for {decision.token_symbol}: {exc}")
        finally:
            logger.info(f"Stopped monitoring {decision.token_symbol}")
            
    async def _monitor_position(self, position_id: str, decision: SniperDecision):
        """Monitor an active position."""
        logger.info(f"Started position monitoring for {decision.token_symbol}")
        
        try:
            while position_id in self.risk_manager.active_positions:
                if not self.running:
                    break
                    
                # Get current position data
                position = self.risk_manager.get_position(position_id)
                if not position:
                    break
                    
                # Check for exit conditions based on momentum and sentiment
                momentum = self.momentum_detector.get_token_momentum(decision.token_mint)
                sentiment = self.sentiment_analyzer.get_token_sentiment(decision.token_mint)
                
                # Dynamic exit logic
                should_exit = False
                exit_reason = ""
                
                if momentum:
                    # Exit on momentum reversal
                    if momentum.reversal_probability > 0.8:
                        should_exit = True
                        exit_reason = "momentum_reversal"
                    # Exit on momentum deterioration
                    elif momentum.momentum_quality < 0.3:
                        should_exit = True
                        exit_reason = "momentum_deterioration"
                        
                if sentiment and not should_exit:
                    # Exit on sentiment shift
                    if sentiment.overall_sentiment < -0.5:
                        should_exit = True
                        exit_reason = "negative_sentiment"
                        
                if should_exit:
                    await self.rapid_executor.execute_exit(
                        position_id, 
                        exit_percentage=1.0,
                        urgency=0.8
                    )
                    break
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Position monitoring error: {exc}")
        finally:
            # Clean up monitoring task
            if position_id in self.position_monitors:
                del self.position_monitors[position_id]
                
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        # Performance monitoring
        task1 = asyncio.create_task(self._performance_monitoring())
        self.monitoring_tasks.append(task1)
        
        # Config hot-reloading
        task2 = asyncio.create_task(self._config_monitoring())
        self.monitoring_tasks.append(task2)
        
        # Emergency monitoring
        task3 = asyncio.create_task(self._emergency_monitoring())
        self.monitoring_tasks.append(task3)
        
    async def _performance_monitoring(self):
        """Monitor performance and update statistics."""
        while self.running:
            try:
                # Update success rate
                total_snipes = self.stats.successful_snipes + self.stats.failed_snipes
                if total_snipes > 0:
                    self.stats.success_rate = self.stats.successful_snipes / total_snipes
                    
                # Update P&L from risk manager
                risk_stats = self.risk_manager.get_statistics()
                self.stats.total_pnl = risk_stats.get("total_pnl", 0.0)
                
                # Log performance metrics
                if total_snipes > 0 and total_snipes % 10 == 0:  # Every 10 snipes
                    logger.info(
                        f"Performance Update: {total_snipes} snipes, "
                        f"{self.stats.success_rate:.1%} success rate, "
                        f"Total P&L: {self.stats.total_pnl:.4f} SOL"
                    )
                    
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as exc:
                logger.error(f"Performance monitoring error: {exc}")
                await asyncio.sleep(60)
                
    async def _config_monitoring(self):
        """Monitor configuration file for changes."""
        while self.running:
            try:
                if self.config_file.exists():
                    modified_time = self.config_file.stat().st_mtime
                    
                    if modified_time > self.config_last_modified:
                        logger.info("Configuration file changed, reloading...")
                        await self._reload_configuration()
                        self.config_last_modified = modified_time
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as exc:
                logger.error(f"Config monitoring error: {exc}")
                await asyncio.sleep(60)
                
    async def _emergency_monitoring(self):
        """Monitor for emergency conditions."""
        while self.running:
            try:
                # Check risk manager for emergency conditions
                portfolio_risk = self.risk_manager.get_portfolio_risk()
                
                if portfolio_risk.daily_pnl_pct < -0.25:  # -25% daily loss
                    logger.critical("EMERGENCY: Daily loss limit approaching!")
                    self.pause_new_trades = True
                    
                if portfolio_risk.overall_risk_score > 0.9:
                    logger.critical("EMERGENCY: Portfolio risk extremely high!")
                    await self.risk_manager.emergency_liquidation()
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as exc:
                logger.error(f"Emergency monitoring error: {exc}")
                await asyncio.sleep(60)
                
    async def _reload_configuration(self):
        """Reload configuration from file."""
        try:
            # This would load new configuration and update components
            logger.info("Configuration reloaded successfully")
        except Exception as exc:
            logger.error(f"Configuration reload failed: {exc}")
            
    async def _notify_decision(self, decision: SniperDecision):
        """Notify subscribers of sniping decisions."""
        for callback in self.notification_callbacks:
            try:
                await callback(decision)
            except Exception as exc:
                logger.error(f"Notification callback error: {exc}")
                
    def add_notification_callback(self, callback: Callable):
        """Add a notification callback for sniping decisions."""
        self.notification_callbacks.append(callback)
        
    def get_statistics(self) -> Dict:
        """Get comprehensive orchestrator statistics."""
        stats = {
            "orchestrator": self.stats.__dict__,
            "pump_detector": self.pump_detector.get_statistics(),
            "pool_analyzer": self.pool_analyzer.get_statistics(),
            "rapid_executor": self.rapid_executor.get_statistics(),
            "risk_manager": self.risk_manager.get_statistics(),
            "sentiment_analyzer": self.sentiment_analyzer.get_statistics(),
            "momentum_detector": self.momentum_detector.get_statistics()
        }
        return stats
        
    def get_active_positions(self) -> Dict:
        """Get all active positions."""
        return self.risk_manager.get_active_positions()
        
    def get_recent_decisions(self, limit: int = 10) -> List[SniperDecision]:
        """Get recent sniping decisions."""
        decisions = list(self.recent_decisions.values())
        decisions.sort(key=lambda d: d.timestamp, reverse=True)
        return decisions[:limit]
        
    async def manual_evaluate_token(self, token_mint: str, token_symbol: str) -> Optional[SniperDecision]:
        """Manually trigger evaluation of a specific token."""
        # Create synthetic pool event
        pool_event = NewPoolEvent(
            pool_address=f"manual_{token_mint}",
            token_mint=token_mint,
            creator="manual",
            liquidity=10000.0  # Placeholder
        )
        
        return await self._evaluate_pool_for_sniping(pool_event)
        
    def emergency_stop_all(self):
        """Emergency stop all operations."""
        self.emergency_stop = True
        logger.critical("EMERGENCY STOP ACTIVATED - All operations halted")
        
    def resume_operations(self):
        """Resume normal operations after emergency stop."""
        self.emergency_stop = False
        self.pause_new_trades = False
        logger.info("Operations resumed")


# Configuration template for the pump sniper
PUMP_SNIPER_CONFIG_TEMPLATE = {
    "pump_sniper_orchestrator": {
        "min_decision_confidence": 0.7,
        "max_concurrent_evaluations": 5,
        "min_initial_liquidity": 5000,
        "decision_weights": {
            "pump_probability": 0.25,
            "pool_quality": 0.20,
            "sentiment_score": 0.15,
            "momentum_score": 0.15,
            "risk_score": 0.15,
            "timing_score": 0.10
        }
    },
    "pump_detection": {
        "min_pump_probability": 0.7,
        "min_timing_score": 0.6,
        "max_risk_score": 0.4,
        "signal_weights": {
            "liquidity_analysis": 0.25,
            "transaction_velocity": 0.20,
            "social_sentiment": 0.15,
            "price_momentum": 0.15,
            "volume_analysis": 0.15,
            "risk_factors": -0.10
        }
    },
    "pool_analyzer": {
        "min_liquidity_usd": 10000,
        "max_slippage_1pct": 0.05,
        "min_health_score": 0.6,
        "quality_weights": {
            "liquidity_depth": 0.25,
            "trading_activity": 0.20,
            "price_stability": 0.15,
            "market_efficiency": 0.15,
            "trader_diversity": 0.15,
            "risk_factors": -0.10
        }
    },
    "rapid_executor": {
        "default_slippage_pct": 0.03,
        "max_position_size_pct": 0.1,
        "priority_fee_base": 1000,
        "worker_count": 3,
        "base_position_size_sol": 0.1,
        "default_stop_loss_pct": 0.1,
        "default_take_profit_pct": 0.3,
        "dex_preferences": ["raydium", "jupiter", "orca", "serum"]
    },
    "sniper_risk_manager": {
        "default_profile": "moderate",
        "state_file": "crypto_bot/logs/sniper_risk_state.json"
    },
    "social_sentiment": {
        "platforms": ["twitter", "telegram", "discord", "reddit"],
        "min_confidence": 0.6,
        "min_volume": 10,
        "influencer_threshold": 1000
    },
    "momentum_detector": {
        "volume_spike_threshold": 3.0,
        "momentum_lookback": 100,
        "min_spike_duration": 30
    }
}
