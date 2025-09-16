"""
Ultra-Fast Arbitrage Engine for High-Frequency Trading

Detects and executes arbitrage opportunities across exchanges and within exchanges
with sub-millisecond latency.

Features:
- Cross-exchange arbitrage (CEX)
- Triangular arbitrage within exchange
- Statistical arbitrage pairs
- Latency arbitrage using order book imbalances
- Smart order routing for minimal slippage
"""

from __future__ import annotations
import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import heapq

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from .base import CallableStrategy

logger = setup_logger(__name__, LOG_DIR / "arbitrage.log")


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity."""
    opportunity_id: str
    type: str  # 'cross_exchange', 'triangular', 'latency'
    symbols: List[str]
    exchanges: List[str]
    buy_price: float
    sell_price: float
    quantity: float
    gross_profit: float
    net_profit: float
    profit_pct: float
    execution_time_ms: float
    timestamp: float


@dataclass
class ArbitrageConfig:
    """Configuration for arbitrage engine."""
    min_profit_threshold: float = 0.001  # 0.1% minimum profit
    max_execution_time_ms: int = 100  # Maximum execution time
    max_position_size: float = 10000  # Maximum position size
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_concurrent_trades: int = 5
    exchanges: List[str] = None

    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = ['kraken', 'coinbase', 'binance']


class LatencyArbitrageEngine:
    """
    Ultra-fast arbitrage engine that detects and executes opportunities
    with minimal latency.
    """

    def __init__(self, config: ArbitrageConfig, exchange_clients: Dict[str, Any]):
        self.config = config
        self.exchanges = exchange_clients
        self.opportunities_queue: List[Tuple[float, ArbitrageOpportunity]] = []
        self.active_trades: Dict[str, ArbitrageOpportunity] = {}
        self.trade_history: deque = deque(maxlen=10000)

        # Performance tracking
        self.stats = {
            'opportunities_found': 0,
            'opportunities_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'avg_execution_time_ms': 0.0
        }

    async def start_arbitrage_scanning(self) -> None:
        """Start continuous arbitrage opportunity scanning."""
        logger.info("Starting arbitrage scanning...")

        tasks = [
            self._scan_cross_exchange_arbitrage(),
            self._scan_triangular_arbitrage(),
            self._scan_latency_arbitrage(),
            self._execute_opportunities(),
            self._monitor_performance()
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _scan_cross_exchange_arbitrage(self) -> None:
        """Scan for cross-exchange arbitrage opportunities."""
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']

        while True:
            try:
                for symbol in symbols:
                    prices = {}

                    # Fetch prices from all exchanges simultaneously
                    tasks = []
                    for exchange_name, client in self.exchanges.items():
                        tasks.append(self._get_exchange_price(client, symbol))

                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for i, result in enumerate(results):
                        exchange_name = list(self.exchanges.keys())[i]
                        if not isinstance(result, Exception):
                            prices[exchange_name] = result

                    # Find arbitrage opportunities
                    if len(prices) >= 2:
                        opportunity = self._find_cross_exchange_arb(symbol, prices)
                        if opportunity:
                            heapq.heappush(self.opportunities_queue,
                                         (-opportunity.profit_pct, opportunity))  # Max heap
                            self.stats['opportunities_found'] += 1

                await asyncio.sleep(0.01)  # 10ms intervals

            except Exception as e:
                logger.error(f"Error in cross-exchange scanning: {e}")
                await asyncio.sleep(0.1)

    async def _scan_triangular_arbitrage(self) -> None:
        """Scan for triangular arbitrage within exchanges."""
        triangles = [
            ['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
            ['BTC/USDT', 'SOL/BTC', 'SOL/USDT'],
            ['ETH/USDT', 'SOL/ETH', 'SOL/USDT']
        ]

        while True:
            try:
                for exchange_name, client in self.exchanges.items():
                    for triangle in triangles:
                        # Fetch all three pairs simultaneously
                        tasks = [self._get_exchange_price(client, pair) for pair in triangle]
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        if all(not isinstance(r, Exception) for r in results):
                            prices = dict(zip(triangle, results))
                            opportunity = self._find_triangular_arb(exchange_name, triangle, prices)
                            if opportunity:
                                heapq.heappush(self.opportunities_queue,
                                             (-opportunity.profit_pct, opportunity))
                                self.stats['opportunities_found'] += 1

                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in triangular scanning: {e}")
                await asyncio.sleep(0.1)

    async def _scan_latency_arbitrage(self) -> None:
        """Scan for latency-based arbitrage using order book imbalances."""
        symbols = ['BTC/USDT', 'ETH/USDT']

        while True:
            try:
                for symbol in symbols:
                    for exchange_name, client in self.exchanges.items():
                        # Get order book with high depth
                        orderbook = await client.fetch_order_book(symbol, limit=50)

                        opportunity = self._find_latency_arb(exchange_name, symbol, orderbook)
                        if opportunity:
                            heapq.heappush(self.opportunities_queue,
                                         (-opportunity.profit_pct, opportunity))
                            self.stats['opportunities_found'] += 1

                await asyncio.sleep(0.005)  # 5ms intervals for latency arb

            except Exception as e:
                logger.error(f"Error in latency scanning: {e}")
                await asyncio.sleep(0.05)

    def _find_cross_exchange_arb(self, symbol: str, prices: Dict[str, Dict]) -> Optional[ArbitrageOpportunity]:
        """Find cross-exchange arbitrage opportunity."""
        if len(prices) < 2:
            return None

        # Find best buy and sell prices
        best_bid = max((p['bid'] for p in prices.values() if 'bid' in p), default=0)
        best_ask = min((p['ask'] for p in prices.values() if 'ask' in p), default=float('inf'))

        if best_bid >= best_ask:
            return None

        # Calculate profit
        spread = best_ask - best_bid
        profit_pct = spread / best_ask

        if profit_pct < self.config.min_profit_threshold:
            return None

        # Find exchanges for buy and sell
        buy_exchange = next(name for name, p in prices.items() if p.get('bid') == best_bid)
        sell_exchange = next(name for name, p in prices.items() if p.get('ask') == best_ask)

        # Calculate executable quantity (minimum of available volumes)
        buy_volume = next(p['bid_volume'] for p in prices.values() if p.get('bid') == best_bid)
        sell_volume = next(p['ask_volume'] for p in prices.values() if p.get('ask') == best_ask)
        quantity = min(buy_volume, sell_volume, self.config.max_position_size)

        gross_profit = spread * quantity
        # Account for fees (simplified 0.1% per trade)
        fees = gross_profit * 0.002
        net_profit = gross_profit - fees

        return ArbitrageOpportunity(
            opportunity_id=f"cross_{symbol}_{int(time.time()*1000)}",
            type="cross_exchange",
            symbols=[symbol],
            exchanges=[buy_exchange, sell_exchange],
            buy_price=best_bid,
            sell_price=best_ask,
            quantity=quantity,
            gross_profit=gross_profit,
            net_profit=net_profit,
            profit_pct=profit_pct,
            execution_time_ms=0,
            timestamp=time.time()
        )

    def _find_triangular_arb(self, exchange: str, triangle: List[str],
                               prices: Dict[str, Dict]) -> Optional[ArbitrageOpportunity]:
        """Find triangular arbitrage opportunity."""
        try:
            # Extract prices
            p1_bid, p1_ask = prices[triangle[0]]['bid'], prices[triangle[0]]['ask']
            p2_bid, p2_ask = prices[triangle[1]]['bid'], prices[triangle[1]]['ask']
            p3_bid, p3_ask = prices[triangle[2]]['bid'], prices[triangle[2]]['ask']

            # Check for arbitrage: buy P1 -> buy P2 -> sell P3
            path1_profit = (1 / p1_ask) * (1 / p2_ask) * p3_bid - 1

            # Check for arbitrage: buy P1 -> sell P2 -> sell P3
            path2_profit = (1 / p1_ask) * p2_bid * p3_bid - 1

            # Check for arbitrage: sell P1 -> buy P2 -> buy P3
            path3_profit = p1_bid * (1 / p2_ask) * (1 / p3_ask) - 1

            # Find best opportunity
            profits = [path1_profit, path2_profit, path3_profit]
            best_profit = max(profits)

            if best_profit < self.config.min_profit_threshold:
                return None

            best_path = profits.index(best_profit)

            return ArbitrageOpportunity(
                opportunity_id=f"tri_{exchange}_{triangle[0]}_{int(time.time()*1000)}",
                type="triangular",
                symbols=triangle,
                exchanges=[exchange],
                buy_price=0,  # Not applicable for triangular
                sell_price=0,
                quantity=min(1000, self.config.max_position_size),  # Conservative size
                gross_profit=best_profit * 1000,  # Approximate
                net_profit=best_profit * 1000 * 0.998,  # Account for fees
                profit_pct=best_profit,
                execution_time_ms=0,
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"Error calculating triangular arb: {e}")
            return None

    def _find_latency_arb(self, exchange: str, symbol: str,
                         orderbook: Dict) -> Optional[ArbitrageOpportunity]:
        """Find latency-based arbitrage using order book analysis."""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if not bids or not asks:
                return None

            # Look for large imbalances that suggest stale quotes
            bid_depth_10 = sum(size for price, size in bids[:10])
            ask_depth_10 = sum(size for price, size in asks[:10])

            # Calculate imbalance ratio
            total_depth = bid_depth_10 + ask_depth_10
            if total_depth == 0:
                return None

            imbalance_ratio = (bid_depth_10 - ask_depth_10) / total_depth

            # Large imbalance might indicate latency advantage
            if abs(imbalance_ratio) > 0.7:  # 70% imbalance
                best_bid = bids[0][0]
                best_ask = asks[0][0]

                # Calculate potential profit
                spread = best_ask - best_bid
                profit_pct = spread / ((best_bid + best_ask) / 2)

                if profit_pct > self.config.min_profit_threshold:
                    return ArbitrageOpportunity(
                        opportunity_id=f"latency_{exchange}_{symbol}_{int(time.time()*1000)}",
                        type="latency",
                        symbols=[symbol],
                        exchanges=[exchange],
                        buy_price=best_bid,
                        sell_price=best_ask,
                        quantity=min(bid_depth_10, ask_depth_10) * 0.1,  # 10% of available depth
                        gross_profit=spread * 100,
                        net_profit=spread * 100 * 0.998,
                        profit_pct=profit_pct,
                        execution_time_ms=0,
                        timestamp=time.time()
                    )

        except Exception as e:
            logger.error(f"Error in latency arbitrage: {e}")

        return None

    async def _execute_opportunities(self) -> None:
        """Execute arbitrage opportunities from the queue."""
        while True:
            try:
                if (self.opportunities_queue and
                    len(self.active_trades) < self.config.max_concurrent_trades):

                    # Get highest profit opportunity
                    profit_score, opportunity = heapq.heappop(self.opportunities_queue)

                    # Skip if opportunity is too old
                    if time.time() - opportunity.timestamp > 1.0:  # 1 second timeout
                        continue

                    # Execute the opportunity
                    success = await self._execute_single_opportunity(opportunity)

                    if success:
                        self.stats['successful_trades'] += 1
                        self.stats['total_profit'] += opportunity.net_profit
                    else:
                        self.stats['failed_trades'] += 1

                    self.stats['opportunities_executed'] += 1

                await asyncio.sleep(0.001)  # 1ms intervals

            except Exception as e:
                logger.error(f"Error executing opportunities: {e}")
                await asyncio.sleep(0.01)

    async def _execute_single_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Execute a single arbitrage opportunity."""
        start_time = time.time()

        try:
            if opportunity.type == "cross_exchange":
                success = await self._execute_cross_exchange_arb(opportunity)
            elif opportunity.type == "triangular":
                success = await self._execute_triangular_arb(opportunity)
            elif opportunity.type == "latency":
                success = await self._execute_latency_arb(opportunity)
            else:
                success = False

            execution_time = (time.time() - start_time) * 1000
            opportunity.execution_time_ms = execution_time

            if success:
                logger.info(f"Executed {opportunity.type} arbitrage: "
                           ".2%"
                           f"{execution_time:.1f}ms")
            else:
                logger.warning(f"Failed to execute {opportunity.type} arbitrage")

            return success

        except Exception as e:
            logger.error(f"Error executing arbitrage opportunity: {e}")
            return False

    async def _execute_cross_exchange_arb(self, opportunity: ArbitrageOpportunity) -> bool:
        """Execute cross-exchange arbitrage."""
        try:
            buy_exchange = self.exchanges[opportunity.exchanges[0]]
            sell_exchange = self.exchanges[opportunity.exchanges[1]]
            symbol = opportunity.symbols[0]

            # Execute buy and sell simultaneously
            buy_task = buy_exchange.create_limit_buy_order(
                symbol, opportunity.quantity, opportunity.buy_price
            )
            sell_task = sell_exchange.create_limit_sell_order(
                symbol, opportunity.quantity, opportunity.sell_price
            )

            results = await asyncio.gather(buy_task, sell_task, return_exceptions=True)

            return all(not isinstance(r, Exception) for r in results)

        except Exception as e:
            logger.error(f"Cross-exchange execution error: {e}")
            return False

    async def _execute_triangular_arb(self, opportunity: ArbitrageOpportunity) -> bool:
        """Execute triangular arbitrage."""
        # Simplified implementation - would need proper triangular execution logic
        logger.info(f"Triangular arbitrage execution not fully implemented: {opportunity.opportunity_id}")
        return False

    async def _execute_latency_arb(self, opportunity: ArbitrageOpportunity) -> bool:
        """Execute latency arbitrage."""
        try:
            exchange = self.exchanges[opportunity.exchanges[0]]
            symbol = opportunity.symbols[0]

            # Place limit orders to capture the spread
            buy_order = await exchange.create_limit_buy_order(
                symbol, opportunity.quantity, opportunity.buy_price
            )
            sell_order = await exchange.create_limit_sell_order(
                symbol, opportunity.quantity, opportunity.sell_price
            )

            return True

        except Exception as e:
            logger.error(f"Latency arbitrage execution error: {e}")
            return False

    async def _get_exchange_price(self, client: Any, symbol: str) -> Dict[str, float]:
        """Get price data from exchange."""
        try:
            ticker = await client.fetch_ticker(symbol)
            return {
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'bid_volume': ticker.get('bidVolume', 0),
                'ask_volume': ticker.get('askVolume', 0)
            }
        except Exception as e:
            raise e

    async def _monitor_performance(self) -> None:
        """Monitor arbitrage performance."""
        while True:
            try:
                # Update average execution time
                if self.trade_history:
                    execution_times = [t.execution_time_ms for t in self.trade_history]
                    self.stats['avg_execution_time_ms'] = np.mean(execution_times)

                # Log performance every minute
                logger.info(f"Arbitrage Stats: {self.stats}")

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    def get_stats(self) -> Dict[str, Any]:
        """Get arbitrage engine statistics."""
        return self.stats.copy()


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """
    Generate arbitrage signal based on market conditions.

    Looks for high volatility and good liquidity conditions
    that favor arbitrage opportunities.
    """
    if df.empty or len(df) < 10:
        return 0.0, "none"

    # Calculate volatility
    returns = df['close'].pct_change()
    volatility = returns.std() * np.sqrt(252)  # Annualized

    # Calculate spread (simplified)
    if 'high' in df.columns and 'low' in df.columns:
        avg_spread = ((df['high'] - df['low']) / df['close']).mean()
    else:
        avg_spread = 0.01  # Default 1%

    # Arbitrage works best in:
    # 1. High volatility (more mispricings)
    # 2. Reasonable liquidity
    # 3. Low trend strength (mean-reverting)

    score = 0.0

    # High volatility boost
    if volatility > 0.5:  # 50% annualized volatility
        score += 0.4
    elif volatility > 0.3:
        score += 0.2

    # Low spread (good liquidity)
    if avg_spread < 0.005:  # Less than 0.5% spread
        score += 0.3
    elif avg_spread < 0.01:
        score += 0.2

    # Volume confirmation
    if 'volume' in df.columns:
        vol_ma = df['volume'].rolling(20).mean()
        if df['volume'].iloc[-1] > vol_ma.iloc[-1] * 1.5:
            score += 0.3

    return min(score, 1.0), "arbitrage"


class regime_filter:
    """Match high volatility regime for arbitrage."""
    @staticmethod
    def matches(regime: str) -> bool:
        return regime in ["volatile", "high_volatility", "arbitrage"]

strategy = CallableStrategy('arbitrage_engine', generate_signal)
