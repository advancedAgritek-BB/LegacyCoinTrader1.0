"""
Rapid Execution Engine for High-Probability Memecoin Trades

This module provides ultra-fast execution capabilities for sniping memecoin pumps,
with advanced order routing, slippage protection, and position management.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import json
import numpy as np

from .pump_detector import PoolAnalysis, PumpSignal
from .pool_analyzer import PoolMetrics
from .watcher import NewPoolEvent
from .wallet_context import WalletContext
from ..utils.telemetry import telemetry

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode options."""
    AGGRESSIVE = "aggressive"  # Fastest execution, higher slippage tolerance
    BALANCED = "balanced"     # Balance between speed and cost
    CONSERVATIVE = "conservative"  # Lower slippage, may be slower


class OrderType(Enum):
    """Order type options."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"


@dataclass
class ExecutionParams:
    """Parameters for trade execution."""
    
    # Basic Order Info
    token_mint: str
    side: str  # "buy" or "sell"
    amount_sol: float
    max_slippage_pct: float = 0.05  # 5% max slippage
    
    # Execution Strategy
    mode: ExecutionMode = ExecutionMode.BALANCED
    order_type: OrderType = OrderType.MARKET
    split_orders: bool = False
    split_count: int = 3
    
    # Timing
    urgency_score: float = 0.5  # 0-1, affects execution aggressiveness
    max_execution_time: float = 30.0  # seconds
    
    # Risk Management
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    position_size_pct: float = 0.02  # 2% of account
    
    # Advanced Options
    use_flashloan: bool = False
    mev_protection: bool = True
    priority_fee: Optional[int] = None  # Custom priority fee
    compute_units: Optional[int] = None  # Custom compute units


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    
    success: bool
    transaction_hash: Optional[str] = None
    executed_amount: float = 0.0
    executed_price: float = 0.0
    slippage_pct: float = 0.0
    gas_cost: float = 0.0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    
    # Position tracking
    position_id: Optional[str] = None
    entry_price: float = 0.0
    amount_tokens: float = 0.0
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    block_height: Optional[int] = None
    execution_mode: Optional[str] = None


class RapidExecutor:
    """
    Ultra-fast execution engine for memecoin sniping.
    
    Features:
    - Multi-DEX routing for best execution
    - Intelligent slippage protection
    - Order splitting and timing optimization
    - MEV protection strategies
    - Real-time position management
    - Advanced risk controls
    """
    
    def __init__(
        self,
        config: Dict,
        dry_run: bool = True,
        paper_wallet=None,
        wallet_context: Optional[WalletContext] = None,
    ):
        self.config = config
        self.executor_config = config.get("rapid_executor", {})
        self.dry_run = dry_run
        self.paper_wallet = paper_wallet
        self.wallet_context = wallet_context

        # Execution settings
        self.default_slippage = self.executor_config.get("default_slippage_pct", 0.03)
        self.max_position_size = self.executor_config.get("max_position_size_pct", 0.1)
        self.priority_fee_base = self.executor_config.get("priority_fee_base", 1000)

        # DEX preferences (in order of preference)
        self.dex_preferences = self.executor_config.get("dex_preferences", [
            "raydium",
            "jupiter",
            "orca",
            "serum"
        ])

        # Position tracking
        self.active_positions: Dict[str, Dict] = {}
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        self.execution_history: List[ExecutionResult] = []

        # Performance optimization
        self.route_cache: Dict[str, Tuple] = {}  # Cache for routing information
        self.price_cache: Dict[str, Tuple[float, float]] = {}  # token -> (price, timestamp)

        # Execution workers
        self.workers: List[asyncio.Task] = []
        self.running = False

        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_volume": 0.0,
            "avg_execution_time": 0.0,
            "avg_slippage": 0.0,
            "mev_protection_saves": 0
        }

        if not self.dry_run and (not self.wallet_context or not self.wallet_context.is_configured):
            logger.warning(
                "Rapid executor running live without a fully configured wallet context"
            )
        
    async def start(self):
        """Start the rapid execution engine."""
        self.running = True
        
        # Start execution workers
        worker_count = self.executor_config.get("worker_count", 3)
        for i in range(worker_count):
            worker = asyncio.create_task(self._execution_worker(f"worker_{i}"))
            self.workers.append(worker)
            
        logger.info(f"Rapid executor started with {worker_count} workers")
        
    async def stop(self):
        """Stop the execution engine."""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Rapid executor stopped")
        
    async def execute_snipe(
        self,
        pool_analysis: PoolAnalysis,
        pool_metrics: PoolMetrics,
        signals: List[PumpSignal]
    ) -> ExecutionResult:
        """
        Execute a snipe trade based on pump detection.
        
        Args:
            pool_analysis: Pump detection analysis
            pool_metrics: Pool quality metrics
            signals: Detected pump signals
            
        Returns:
            ExecutionResult with trade outcome
        """
        try:
            # Create execution parameters
            params = await self._create_execution_params(pool_analysis, pool_metrics, signals)
            
            # Add to execution queue with high priority
            await self.execution_queue.put((params, time.time()))
            
            # Wait for execution (with timeout)
            result = await self._wait_for_execution(params.token_mint, params.max_execution_time)
            
            if result.success:
                telemetry.inc("rapid_executor.successful_snipes")
                telemetry.gauge("rapid_executor.execution_time", result.execution_time)
                telemetry.gauge("rapid_executor.slippage", result.slippage_pct)
                
                logger.info(
                    f"Snipe executed successfully: {params.token_mint[:8]}... "
                    f"Amount: {result.executed_amount:.4f} SOL, "
                    f"Price: {result.executed_price:.6f}, "
                    f"Slippage: {result.slippage_pct:.2%}, "
                    f"Time: {result.execution_time:.2f}s"
                )
            else:
                telemetry.inc("rapid_executor.failed_snipes")
                logger.warning(f"Snipe failed: {result.error_message}")
                
            return result
            
        except Exception as exc:
            logger.error(f"Snipe execution error: {exc}")
            return ExecutionResult(
                success=False,
                error_message=str(exc)
            )
            
    async def execute_exit(
        self,
        position_id: str,
        exit_percentage: float = 1.0,
        urgency: float = 0.8
    ) -> ExecutionResult:
        """
        Execute position exit.
        
        Args:
            position_id: Position to exit
            exit_percentage: Percentage of position to exit
            urgency: Urgency score (0-1)
            
        Returns:
            ExecutionResult with exit outcome
        """
        try:
            position = self.active_positions.get(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")
                
            # Create exit parameters
            params = ExecutionParams(
                token_mint=position["token_mint"],
                side="sell",
                amount_sol=position["amount_sol"] * exit_percentage,
                max_slippage_pct=0.05,  # Slightly higher slippage tolerance for exits
                mode=ExecutionMode.AGGRESSIVE if urgency > 0.7 else ExecutionMode.BALANCED,
                urgency_score=urgency
            )
            
            # Execute immediately
            result = await self._execute_trade(params)
            
            if result.success:
                # Update position
                await self._update_position_after_exit(position_id, exit_percentage, result)
                
            return result
            
        except Exception as exc:
            logger.error(f"Exit execution error: {exc}")
            return ExecutionResult(
                success=False,
                error_message=str(exc)
            )
            
    async def _execution_worker(self, worker_id: str):
        """Worker for processing execution queue."""
        logger.info(f"Execution worker {worker_id} started")
        
        while self.running:
            try:
                # Get next execution task
                params, queued_time = await asyncio.wait_for(
                    self.execution_queue.get(),
                    timeout=1.0
                )
                
                # Check if execution is still relevant
                age = time.time() - queued_time
                if age > params.max_execution_time:
                    logger.warning(f"Execution request expired: {params.token_mint}")
                    continue
                    
                # Execute the trade
                result = await self._execute_trade(params)
                
                # Store result
                self.execution_history.append(result)
                if len(self.execution_history) > 1000:  # Keep last 1000 executions
                    self.execution_history.pop(0)
                    
                # Update statistics
                self._update_stats(result)
                
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error(f"Execution worker {worker_id} error: {exc}")
                await asyncio.sleep(1)
                
        logger.info(f"Execution worker {worker_id} stopped")
        
    async def _execute_trade(self, params: ExecutionParams) -> ExecutionResult:
        """Execute a single trade."""
        start_time = time.time()
        
        try:
            # Pre-execution checks
            if not await self._pre_execution_checks(params):
                return ExecutionResult(
                    success=False,
                    error_message="Pre-execution checks failed"
                )
                
            # Get optimal route
            route = await self._get_optimal_route(params)
            if not route:
                return ExecutionResult(
                    success=False,
                    error_message="No valid route found"
                )
                
            # Calculate dynamic fees based on urgency
            priority_fee = self._calculate_priority_fee(params)
            
            # Execute based on strategy
            if params.split_orders and params.amount_sol > 0.1:  # Split large orders
                result = await self._execute_split_order(params, route, priority_fee)
            else:
                result = await self._execute_single_order(params, route, priority_fee)
                
            # Post-execution processing
            if result.success:
                await self._post_execution_processing(params, result)
                
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error_message=str(exc),
                execution_time=time.time() - start_time
            )
            
    async def _create_execution_params(
        self,
        pool_analysis: PoolAnalysis,
        pool_metrics: PoolMetrics,
        signals: List[PumpSignal]
    ) -> ExecutionParams:
        """Create execution parameters based on analysis."""
        
        # Calculate position size based on confidence
        confidence = pool_analysis.pump_probability
        base_size = self.executor_config.get("base_position_size_sol", 0.1)
        position_size = base_size * confidence
        
        # Adjust slippage tolerance based on urgency
        urgency = max(signal.strength for signal in signals) if signals else 0.5
        max_slippage = self.default_slippage * (1 + urgency)
        
        # Choose execution mode based on timing score
        if pool_analysis.timing_score > 0.8:
            mode = ExecutionMode.AGGRESSIVE
        elif pool_analysis.timing_score > 0.6:
            mode = ExecutionMode.BALANCED
        else:
            mode = ExecutionMode.CONSERVATIVE
            
        # Set stop loss and take profit
        stop_loss = self.executor_config.get("default_stop_loss_pct", 0.1)
        take_profit = self.executor_config.get("default_take_profit_pct", 0.3)
        
        return ExecutionParams(
            token_mint=pool_analysis.token_mint,
            side="buy",
            amount_sol=position_size,
            max_slippage_pct=max_slippage,
            mode=mode,
            urgency_score=urgency,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            split_orders=position_size > 0.5,  # Split large positions
            mev_protection=True
        )
        
    async def _pre_execution_checks(self, params: ExecutionParams) -> bool:
        """Perform pre-execution validation."""
        try:
            # Check account balance
            if not await self._check_balance(params.amount_sol):
                logger.warning("Insufficient balance for execution")
                return False
                
            # Check token validity
            if not await self._validate_token(params.token_mint):
                logger.warning(f"Invalid token: {params.token_mint}")
                return False
                
            # Check current market conditions
            if not await self._check_market_conditions(params):
                logger.warning("Market conditions unfavorable")
                return False
                
            # Check position limits
            if not self._check_position_limits(params):
                logger.warning("Position limits exceeded")
                return False
                
            return True
            
        except Exception as exc:
            logger.error(f"Pre-execution check failed: {exc}")
            return False
            
    async def _get_optimal_route(self, params: ExecutionParams) -> Optional[Dict]:
        """Get optimal routing for the trade."""
        try:
            # Check cache first
            cache_key = f"{params.token_mint}_{params.side}_{int(time.time() // 60)}"
            if cache_key in self.route_cache:
                return self.route_cache[cache_key]
                
            # Query multiple DEXs for best route
            routes = []
            
            for dex in self.dex_preferences:
                try:
                    route = await self._query_dex_route(dex, params)
                    if route:
                        routes.append(route)
                except Exception as exc:
                    logger.debug(f"Route query failed for {dex}: {exc}")
                    
            if not routes:
                return None
                
            # Select best route based on price impact and gas costs
            best_route = min(routes, key=lambda r: r["total_cost"])
            
            # Cache the result
            self.route_cache[cache_key] = best_route
            
            return best_route
            
        except Exception as exc:
            logger.error(f"Route optimization failed: {exc}")
            return None
            
    async def _execute_single_order(
        self,
        params: ExecutionParams,
        route: Dict,
        priority_fee: int
    ) -> ExecutionResult:
        """Execute a single order."""
        try:
            if self.dry_run:
                # Paper trading mode - simulate execution
                return await self._execute_paper_trade(params, route)
            else:
                if not self.wallet_context or not self.wallet_context.is_configured:
                    raise RuntimeError(
                        "Pump sniper wallet is not configured for live execution"
                    )
                # Live trading mode
                # Prepare transaction
                tx_data = await self._prepare_transaction(params, route, priority_fee)

                # MEV protection
                if params.mev_protection:
                    tx_data = await self._apply_mev_protection(tx_data)

                # Submit transaction
                tx_hash = await self._submit_transaction(tx_data)

                # Wait for confirmation
                result = await self._wait_for_confirmation(tx_hash, params.max_execution_time)

                if result.success:
                    # Create position
                    position_id = await self._create_position(params, result)
                    result.position_id = position_id

                return result

        except Exception as exc:
            return ExecutionResult(
                success=False,
                error_message=str(exc)
            )
            
    async def _execute_paper_trade(
        self,
        params: ExecutionParams,
        route: Dict
    ) -> ExecutionResult:
        """Execute a paper trade simulation."""
        try:
            # Simulate execution time
            execution_time = 0.1 + (params.urgency_score * 0.5)  # 0.1-0.6 seconds
            await asyncio.sleep(execution_time)

            # Get simulated price from route or use fallback
            base_price = route.get("price", 0.001) if route else 0.001

            # Simulate slippage
            slippage_factor = 1.0 + (params.max_slippage_pct * (0.5 - np.random.random()))  # Random slippage within bounds
            executed_price = base_price * slippage_factor
            slippage_pct = abs(executed_price - base_price) / base_price

            # Calculate executed amounts
            executed_amount = params.amount_sol
            amount_tokens = executed_amount / executed_price

            # Create simulated transaction hash
            tx_hash = f"paper_tx_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"

            # Update paper wallet if available
            position_id = None
            if self.paper_wallet:
                try:
                    # For buy orders, we need to simulate the token pair
                    symbol = f"{params.token_mint[:8]}.../SOL"  # Simplified symbol for paper wallet

                    if params.side == "buy":
                        # Open a buy position in paper wallet
                        trade_id = self.paper_wallet.open(
                            symbol,
                            "buy",
                            executed_amount,
                            executed_price
                        )
                        position_id = trade_id
                        logger.info(f"Paper buy executed: {executed_amount:.4f} SOL @ ${executed_price:.6f}, slippage: {slippage_pct:.2%}")
                    else:
                        # For sell orders, close existing position
                        # This would need more complex logic for position management
                        logger.info(f"Paper sell executed: {executed_amount:.4f} SOL @ ${executed_price:.6f}, slippage: {slippage_pct:.2%}")
                except Exception as e:
                    logger.warning(f"Paper wallet update failed: {e}")

            return ExecutionResult(
                success=True,
                transaction_hash=tx_hash,
                executed_amount=executed_amount,
                executed_price=executed_price,
                slippage_pct=slippage_pct,
                execution_time=execution_time,
                amount_tokens=amount_tokens,
                position_id=position_id,
                gas_cost=0.0001,  # Simulated gas cost
                block_height=100000000 + int(time.time()),  # Simulated block height
                execution_mode="paper"
            )

        except Exception as exc:
            logger.error(f"Paper trade execution failed: {exc}")
            return ExecutionResult(
                success=False,
                error_message=str(exc),
                execution_time=time.time()
            )

    async def _execute_split_order(
        self,
        params: ExecutionParams,
        route: Dict,
        priority_fee: int
    ) -> ExecutionResult:
        """Execute order split across multiple smaller trades."""
        try:
            split_amount = params.amount_sol / params.split_count
            results = []

            for i in range(params.split_count):
                # Create split parameters
                split_params = ExecutionParams(
                    token_mint=params.token_mint,
                    side=params.side,
                    amount_sol=split_amount,
                    max_slippage_pct=params.max_slippage_pct,
                    mode=params.mode,
                    order_type=params.order_type
                )

                # Execute split
                if self.dry_run:
                    # Paper trading split
                    result = await self._execute_paper_trade(split_params, route)
                else:
                    # Live trading split
                    result = await self._execute_single_order(split_params, route, priority_fee)
                results.append(result)

                # Small delay between splits to avoid MEV (or simulate timing)
                if i < params.split_count - 1:
                    await asyncio.sleep(0.1)

            # Aggregate results
            return self._aggregate_split_results(results)
            
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error_message=str(exc)
            )
            
    # Helper methods for execution
    
    def _calculate_priority_fee(self, params: ExecutionParams) -> int:
        """Calculate priority fee based on urgency."""
        base_fee = self.priority_fee_base
        urgency_multiplier = 1 + (params.urgency_score * 2)  # 1x to 3x multiplier
        
        if params.mode == ExecutionMode.AGGRESSIVE:
            urgency_multiplier *= 1.5
        elif params.mode == ExecutionMode.CONSERVATIVE:
            urgency_multiplier *= 0.7
            
        return int(base_fee * urgency_multiplier)
        
    async def _check_balance(self, required_sol: float) -> bool:
        """Check if account has sufficient balance."""
        if self.dry_run and self.paper_wallet:
            # Check paper wallet balance
            return self.paper_wallet.balance >= required_sol
        # This would integrate with your wallet balance checking
        # Placeholder implementation for live trading
        return True
        
    async def _validate_token(self, token_mint: str) -> bool:
        """Validate token mint address."""
        # This would perform token validation
        # Placeholder implementation
        return len(token_mint) == 44  # Basic Solana address length check
        
    async def _check_market_conditions(self, params: ExecutionParams) -> bool:
        """Check if market conditions are suitable for execution."""
        # This would check network congestion, volatility, etc.
        # Placeholder implementation
        return True
        
    def _check_position_limits(self, params: ExecutionParams) -> bool:
        """Check position size limits."""
        current_exposure = sum(pos.get("amount_sol", 0) for pos in self.active_positions.values())
        max_exposure = self.max_position_size * 10  # Assuming 10 SOL account
        
        return current_exposure + params.amount_sol <= max_exposure
        
    async def _query_dex_route(self, dex: str, params: ExecutionParams) -> Optional[Dict]:
        """Query a specific DEX for routing information."""
        # This would integrate with DEX APIs (Jupiter, Raydium, etc.)
        # Placeholder implementation
        return {
            "dex": dex,
            "price": 0.001,
            "price_impact": 0.02,
            "gas_cost": 0.0001,
            "total_cost": 0.0201,
            "route_data": {}
        }
        
    async def _prepare_transaction(self, params: ExecutionParams, route: Dict, priority_fee: int) -> Dict:
        """Prepare transaction data."""
        # This would prepare Solana transaction
        # Placeholder implementation
        return {
            "instructions": [],
            "priority_fee": priority_fee,
            "compute_units": params.compute_units or 400000
        }
        
    async def _apply_mev_protection(self, tx_data: Dict) -> Dict:
        """Apply MEV protection strategies."""
        # This would implement MEV protection
        # Placeholder implementation
        self.stats["mev_protection_saves"] += 1
        return tx_data
        
    async def _submit_transaction(self, tx_data: Dict) -> str:
        """Submit transaction to the network."""
        # This would submit to Solana network
        # Placeholder implementation
        return f"tx_hash_{int(time.time())}"
        
    async def _wait_for_confirmation(self, tx_hash: str, timeout: float) -> ExecutionResult:
        """Wait for transaction confirmation."""
        # This would wait for Solana confirmation
        # Placeholder implementation
        await asyncio.sleep(2)  # Simulate confirmation time
        
        return ExecutionResult(
            success=True,
            transaction_hash=tx_hash,
            executed_amount=0.1,
            executed_price=0.001,
            slippage_pct=0.01
        )
        
    async def _create_position(self, params: ExecutionParams, result: ExecutionResult) -> str:
        """Create and track new position."""
        position_id = f"pos_{int(time.time())}"
        
        position = {
            "position_id": position_id,
            "token_mint": params.token_mint,
            "amount_sol": result.executed_amount,
            "amount_tokens": result.amount_tokens,
            "entry_price": result.executed_price,
            "stop_loss_pct": params.stop_loss_pct,
            "take_profit_pct": params.take_profit_pct,
            "created_at": time.time()
        }
        
        self.active_positions[position_id] = position
        return position_id
        
    async def _update_position_after_exit(self, position_id: str, exit_pct: float, result: ExecutionResult):
        """Update position after partial/full exit."""
        position = self.active_positions[position_id]
        
        if exit_pct >= 1.0:
            # Full exit
            del self.active_positions[position_id]
        else:
            # Partial exit
            position["amount_sol"] *= (1 - exit_pct)
            position["amount_tokens"] *= (1 - exit_pct)
            
    def _aggregate_split_results(self, results: List[ExecutionResult]) -> ExecutionResult:
        """Aggregate results from split orders."""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return ExecutionResult(
                success=False,
                error_message="All split orders failed"
            )
            
        total_amount = sum(r.executed_amount for r in successful_results)
        weighted_price = sum(r.executed_price * r.executed_amount for r in successful_results) / total_amount
        avg_slippage = sum(r.slippage_pct for r in successful_results) / len(successful_results)
        
        return ExecutionResult(
            success=True,
            executed_amount=total_amount,
            executed_price=weighted_price,
            slippage_pct=avg_slippage,
            transaction_hash=successful_results[0].transaction_hash
        )
        
    async def _wait_for_execution(self, token_mint: str, timeout: float) -> ExecutionResult:
        """Wait for execution to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if execution completed
            for result in reversed(self.execution_history):
                if (result.timestamp > start_time and 
                    token_mint in str(result.transaction_hash)):
                    return result
                    
            await asyncio.sleep(0.1)
            
        return ExecutionResult(
            success=False,
            error_message="Execution timeout"
        )
        
    async def _post_execution_processing(self, params: ExecutionParams, result: ExecutionResult):
        """Post-execution processing and monitoring setup."""
        if params.stop_loss_pct or params.take_profit_pct:
            # Set up monitoring for stop loss and take profit
            await self._setup_position_monitoring(result.position_id)
            
    async def _setup_position_monitoring(self, position_id: str):
        """Set up monitoring for position management."""
        # This would set up price monitoring and automatic exits
        # Placeholder implementation
        pass
        
    def _update_stats(self, result: ExecutionResult):
        """Update execution statistics."""
        self.stats["total_executions"] += 1
        
        if result.success:
            self.stats["successful_executions"] += 1
            self.stats["total_volume"] += result.executed_amount
            
            # Update averages
            total_successful = self.stats["successful_executions"]
            self.stats["avg_execution_time"] = (
                (self.stats["avg_execution_time"] * (total_successful - 1) + result.execution_time) / 
                total_successful
            )
            self.stats["avg_slippage"] = (
                (self.stats["avg_slippage"] * (total_successful - 1) + result.slippage_pct) / 
                total_successful
            )
        else:
            self.stats["failed_executions"] += 1
            
    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        stats = self.stats.copy()
        total_executions = self.stats["total_executions"]
        
        if total_executions > 0:
            stats["success_rate"] = self.stats["successful_executions"] / total_executions
            stats["failure_rate"] = self.stats["failed_executions"] / total_executions
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
            
        stats["active_positions"] = len(self.active_positions)
        return stats
        
    def get_active_positions(self) -> Dict[str, Dict]:
        """Get all active positions."""
        return self.active_positions.copy()
        
    def get_position(self, position_id: str) -> Optional[Dict]:
        """Get specific position details."""
        return self.active_positions.get(position_id)
