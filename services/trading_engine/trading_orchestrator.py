"""
Trading Orchestrator for managing trading cycles and operations.

This module contains the core trading logic extracted from main.py,
organized into a microservice architecture.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import aiohttp
import redis

from .config import TradingEngineConfig

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """Orchestrates trading operations and cycles."""

    def __init__(self, config: TradingEngineConfig, redis_client: redis.Redis,
                 service_urls: Dict[str, str]):
        self.config = config
        self.redis = redis_client
        self.service_urls = service_urls
        self.http_client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

        self.running = False
        self.trading_task = None
        self.symbols = []
        self.current_batch = []

        # Performance tracking
        self.cycle_count = 0
        self.last_cycle_time = None
        self.average_cycle_time = 0.0

    async def start(self):
        """Start the trading orchestrator."""
        if self.running:
            logger.warning("Trading orchestrator is already running")
            return

        self.running = True
        logger.info("Starting trading orchestrator")

        # Load initial configuration
        await self._load_symbols()

        # Start trading loop
        self.trading_task = asyncio.create_task(self._trading_loop())

    async def stop(self):
        """Stop the trading orchestrator."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping trading orchestrator")

        if self.trading_task:
            self.trading_task.cancel()
            try:
                await self.trading_task
            except asyncio.CancelledError:
                pass

    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            'running': self.running,
            'symbols_count': len(self.symbols),
            'current_batch_size': len(self.current_batch),
            'cycle_count': self.cycle_count,
            'last_cycle_time': self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            'average_cycle_time': self.average_cycle_time,
            'service_urls': self.service_urls
        }

    async def execute_trading_cycle(self) -> Dict[str, Any]:
        """Execute a single trading cycle."""
        if not self.running:
            return {'error': 'Trading orchestrator is not running'}

        start_time = time.time()
        logger.info("Executing trading cycle")

        try:
            # Get symbols for this cycle
            cycle_symbols = await self._get_cycle_symbols()

            if not cycle_symbols:
                return {'status': 'no_symbols', 'message': 'No symbols available for trading'}

            # Process symbols in batches
            results = []
            for i in range(0, len(cycle_symbols), self.config.batch_size):
                batch = cycle_symbols[i:i + self.config.batch_size]
                batch_result = await self._process_symbol_batch(batch)
                results.append(batch_result)

                # Respect rate limits
                if i + self.config.batch_size < len(cycle_symbols):
                    await asyncio.sleep(self.config.batch_delay)

            # Update performance metrics
            cycle_time = time.time() - start_time
            await self._update_performance_metrics(cycle_time)

            return {
                'status': 'success',
                'cycle_time': cycle_time,
                'symbols_processed': len(cycle_symbols),
                'batches_processed': len(results),
                'results': results
            }

        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _trading_loop(self):
        """Main trading loop."""
        logger.info(f"Starting trading loop with {self.config.cycle_interval}s interval")

        while self.running:
            try:
                # Execute trading cycle
                result = await self.execute_trading_cycle()

                # Log cycle completion
                if result.get('status') == 'success':
                    logger.info(
                        f"Trading cycle completed in {result['cycle_time']:.2f}s, "
                        f"processed {result['symbols_processed']} symbols"
                    )
                else:
                    logger.warning(f"Trading cycle result: {result}")

                # Wait for next cycle
                await asyncio.sleep(self.config.cycle_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(self.config.cycle_interval)

    async def _load_symbols(self):
        """Load trading symbols from configuration."""
        try:
            # Get symbols from market data service
            market_data_url = self.service_urls.get('market_data')
            if market_data_url:
                async with self.http_client.get(
                    f"{market_data_url}/symbols",
                    headers={'X-Service-Auth': self.config.service_auth_token}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.symbols = data.get('symbols', [])
                        logger.info(f"Loaded {len(self.symbols)} symbols from market data service")
                    else:
                        logger.warning(f"Failed to load symbols from market data service: {response.status}")
            else:
                logger.warning("Market data service URL not available")

            # Fallback to config symbols if none loaded
            if not self.symbols:
                self.symbols = self.config.default_symbols
                logger.info(f"Using default symbols: {self.symbols}")

        except Exception as e:
            logger.error(f"Failed to load symbols: {e}")
            self.symbols = self.config.default_symbols

    async def _get_cycle_symbols(self) -> List[str]:
        """Get symbols for the current trading cycle."""
        try:
            # Get filtered symbols from market data service
            market_data_url = self.service_urls.get('market_data')
            if market_data_url:
                async with self.http_client.get(
                    f"{market_data_url}/filtered-symbols",
                    headers={'X-Service-Auth': self.config.service_auth_token}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('symbols', self.symbols)

            return self.symbols

        except Exception as e:
            logger.error(f"Failed to get cycle symbols: {e}")
            return self.symbols

    async def _process_symbol_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """Process a batch of symbols through the trading pipeline."""
        batch_start = time.time()
        logger.info(f"Processing batch of {len(symbols)} symbols")

        try:
            # Get market data for symbols
            market_data = await self._get_market_data(symbols)

            # Evaluate strategies
            strategy_signals = await self._evaluate_strategies(symbols, market_data)

            # Check portfolio and risk
            portfolio_status = await self._check_portfolio_status()

            # Generate trading decisions
            trades = await self._generate_trades(strategy_signals, portfolio_status)

            # Execute trades
            execution_results = await self._execute_trades(trades)

            batch_time = time.time() - batch_start

            return {
                'status': 'success',
                'symbols': symbols,
                'signals_generated': len(strategy_signals),
                'trades_generated': len(trades),
                'trades_executed': len(execution_results),
                'processing_time': batch_time
            }

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {
                'status': 'error',
                'symbols': symbols,
                'error': str(e),
                'processing_time': time.time() - batch_start
            }

    async def _get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market data for symbols."""
        try:
            market_data_url = self.service_urls.get('market_data')
            if not market_data_url:
                return {}

            # Request market data for symbols
            async with self.http_client.post(
                f"{market_data_url}/batch-ohlcv",
                json={'symbols': symbols, 'timeframe': self.config.default_timeframe},
                headers={'X-Service-Auth': self.config.service_auth_token}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get market data: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}

    async def _evaluate_strategies(self, symbols: List[str],
                                  market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate strategies for symbols."""
        try:
            strategy_url = self.service_urls.get('strategy_engine')
            if not strategy_url:
                return []

            async with self.http_client.post(
                f"{strategy_url}/evaluate-batch",
                json={'symbols': symbols, 'market_data': market_data},
                headers={'X-Service-Auth': self.config.service_auth_token}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('signals', [])
                else:
                    logger.warning(f"Failed to evaluate strategies: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error evaluating strategies: {e}")
            return []

    async def _check_portfolio_status(self) -> Dict[str, Any]:
        """Check current portfolio status."""
        try:
            portfolio_url = self.service_urls.get('portfolio')
            if not portfolio_url:
                return {'positions': {}, 'balance': 0.0, 'risk_metrics': {}}

            async with self.http_client.get(
                f"{portfolio_url}/status",
                headers={'X-Service-Auth': self.config.service_auth_token}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get portfolio status: {response.status}")
                    return {'positions': {}, 'balance': 0.0, 'risk_metrics': {}}

        except Exception as e:
            logger.error(f"Error checking portfolio status: {e}")
            return {'positions': {}, 'balance': 0.0, 'risk_metrics': {}}

    async def _generate_trades(self, signals: List[Dict[str, Any]],
                              portfolio_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trades from signals and portfolio status."""
        trades = []

        try:
            # Apply risk management and position sizing
            for signal in signals:
                trade = await self._create_trade_from_signal(signal, portfolio_status)
                if trade:
                    trades.append(trade)

        except Exception as e:
            logger.error(f"Error generating trades: {e}")

        return trades

    async def _create_trade_from_signal(self, signal: Dict[str, Any],
                                       portfolio_status: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a trade from a strategy signal."""
        try:
            # Basic trade creation logic
            symbol = signal.get('symbol')
            direction = signal.get('direction')
            confidence = signal.get('confidence', 0.0)

            # Check risk limits
            risk_metrics = portfolio_status.get('risk_metrics', {})
            if risk_metrics.get('max_risk_exceeded', False):
                return None

            # Calculate position size
            position_size = await self._calculate_position_size(
                symbol, direction, confidence, portfolio_status
            )

            if position_size <= 0:
                return None

            return {
                'symbol': symbol,
                'direction': direction,
                'size': position_size,
                'confidence': confidence,
                'strategy': signal.get('strategy'),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error creating trade from signal: {e}")
            return None

    async def _calculate_position_size(self, symbol: str, direction: str,
                                     confidence: float, portfolio_status: Dict[str, Any]) -> float:
        """Calculate position size based on risk management."""
        try:
            # Basic position sizing logic
            balance = portfolio_status.get('balance', 0.0)
            risk_per_trade = self.config.max_risk_per_trade

            # Adjust size based on confidence
            base_size = balance * risk_per_trade * confidence

            # Check existing positions
            positions = portfolio_status.get('positions', {})
            existing_position = positions.get(symbol, {})

            # Reduce size if already have position in same direction
            if existing_position and existing_position.get('direction') == direction:
                base_size *= 0.5  # Reduce by half

            return min(base_size, balance * self.config.max_position_size_pct)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    async def _execute_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute trades via execution service."""
        executed_trades = []

        try:
            execution_url = self.service_urls.get('execution')
            if not execution_url:
                logger.warning("Execution service URL not available")
                return []

            for trade in trades:
                try:
                    async with self.http_client.post(
                        f"{execution_url}/execute",
                        json=trade,
                        headers={'X-Service-Auth': self.config.service_auth_token}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            executed_trades.append(result)
                        else:
                            logger.warning(f"Failed to execute trade: {response.status}")

                except Exception as e:
                    logger.error(f"Error executing trade: {e}")

        except Exception as e:
            logger.error(f"Error in trade execution: {e}")

        return executed_trades

    async def _update_performance_metrics(self, cycle_time: float):
        """Update performance metrics."""
        self.cycle_count += 1
        self.last_cycle_time = datetime.utcnow()

        # Update rolling average
        if self.cycle_count == 1:
            self.average_cycle_time = cycle_time
        else:
            self.average_cycle_time = (
                (self.average_cycle_time * (self.cycle_count - 1)) + cycle_time
            ) / self.cycle_count

        # Store metrics in Redis
        try:
            metrics_key = "trading_engine:metrics"
            metrics = {
                'cycle_count': self.cycle_count,
                'last_cycle_time': self.last_cycle_time.isoformat(),
                'average_cycle_time': self.average_cycle_time,
                'last_cycle_duration': cycle_time
            }

            self.redis.set(metrics_key, json.dumps(metrics))

        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")

    async def cleanup(self):
        """Cleanup resources."""
        await self.stop()
        if self.http_client:
            await self.http_client.close()
