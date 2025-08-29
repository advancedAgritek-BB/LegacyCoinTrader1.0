"""
Advanced Order Types for High-Frequency Trading

Implements sophisticated order execution strategies including:
- Iceberg orders for large volume execution
- TWAP (Time-Weighted Average Price) orders
- VWAP (Volume-Weighted Average Price) orders
- Smart routing for minimal slippage
"""

from __future__ import annotations
import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "advanced_orders.log")


@dataclass
class IcebergOrder:
    """Iceberg order configuration for large volume execution."""
    symbol: str
    side: str
    total_quantity: float
    display_quantity: float
    price: Optional[float] = None
    order_type: str = "limit"
    time_limit: int = 300  # seconds
    min_fill_rate: float = 0.8
    max_slippage_pct: float = 0.001


@dataclass
class TWAPOrder:
    """Time-Weighted Average Price order configuration."""
    symbol: str
    side: str
    total_quantity: float
    duration_seconds: int
    price_limit: Optional[float] = None
    slice_interval: int = 10  # seconds between slices
    participation_rate: float = 0.1  # % of volume to participate in


@dataclass
class VWAPOrder:
    """Volume-Weighted Average Price order configuration."""
    symbol: str
    side: str
    total_quantity: float
    start_time: int
    end_time: int
    price_limit: Optional[float] = None
    volume_profile_lookback: int = 20  # days for volume profile


class AdvancedOrderExecutor:
    """Executes advanced order types with minimal market impact."""

    def __init__(self, exchange_client: Any, notifier: Any = None):
        self.exchange = exchange_client
        self.notifier = notifier
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.execution_history: deque = deque(maxlen=1000)

    async def execute_iceberg_order(self, order: IcebergOrder) -> Dict[str, Any]:
        """
        Execute iceberg order by breaking large order into smaller visible chunks.

        Strategy:
        1. Place initial order with display_quantity
        2. Monitor fills and replenish as chunks are consumed
        3. Adjust price based on market conditions
        4. Complete when total_quantity is filled or time_limit reached
        """
        order_id = f"iceberg_{order.symbol}_{int(time.time())}"
        remaining_quantity = order.total_quantity
        executed_quantity = 0.0

        logger.info(f"Starting iceberg order {order_id}: {order.total_quantity} {order.symbol}")

        while remaining_quantity > 0 and time.time() < time.time() + order.time_limit:
            # Calculate current display quantity
            current_display = min(order.display_quantity, remaining_quantity)

            # Get current market price for limit orders
            if order.order_type == "limit" and order.price is None:
                ticker = await self.exchange.fetch_ticker(order.symbol)
                spread = (ticker['ask'] - ticker['bid']) / ticker['bid']
                order.price = ticker['bid'] if order.side == 'buy' else ticker['ask']

                # Add small buffer to avoid immediate execution
                buffer_pct = 0.0001  # 0.01%
                if order.side == 'buy':
                    order.price *= (1 + buffer_pct)
                else:
                    order.price *= (1 - buffer_pct)

            # Place order slice
            try:
                if order.order_type == "limit":
                    result = await self.exchange.create_limit_order(
                        order.symbol, order.side, current_display, order.price
                    )
                else:
                    result = await self.exchange.create_market_order(
                        order.symbol, order.side, current_display
                    )

                # Monitor order execution
                fill_monitor_task = asyncio.create_task(
                    self._monitor_order_fill(result['id'], order)
                )

                # Wait for partial fill or timeout
                await asyncio.sleep(min(5.0, order.time_limit / 10))

                # Check order status
                order_status = await self.exchange.fetch_order(result['id'])
                filled = order_status.get('filled', 0)
                executed_quantity += filled
                remaining_quantity -= filled

                # Cancel remaining order if we got most of what we wanted
                if filled > 0:
                    await self.exchange.cancel_order(result['id'])

                logger.info(f"Iceberg slice filled: {filled}/{current_display} for {order_id}")

            except Exception as e:
                logger.error(f"Iceberg order error: {e}")
                await asyncio.sleep(1)

        completion_rate = executed_quantity / order.total_quantity
        success = completion_rate >= order.min_fill_rate

        result = {
            'order_id': order_id,
            'executed_quantity': executed_quantity,
            'completion_rate': completion_rate,
            'success': success,
            'avg_price': self._calculate_avg_price(order_id)
        }

        self.execution_history.append(result)
        return result

    async def execute_twap_order(self, order: TWAPOrder) -> Dict[str, Any]:
        """
        Execute Time-Weighted Average Price order.

        Strategy:
        1. Calculate total slices based on duration and interval
        2. Execute equal quantity slices at regular intervals
        3. Monitor market conditions and adjust execution
        4. Complete within specified time window
        """
        order_id = f"twap_{order.symbol}_{int(time.time())}"

        # Calculate execution parameters
        total_slices = order.duration_seconds // order.slice_interval
        slice_quantity = order.total_quantity / total_slices

        executed_quantity = 0.0
        slice_prices = []
        start_time = time.time()

        logger.info(f"Starting TWAP order {order_id}: {total_slices} slices of {slice_quantity}")

        for slice_num in range(total_slices):
            try:
                # Get current market price
                ticker = await self.exchange.fetch_ticker(order.symbol)
                current_price = ticker['bid'] if order.side == 'buy' else ticker['ask']

                # Check price limit
                if order.price_limit:
                    if (order.side == 'buy' and current_price > order.price_limit) or \
                       (order.side == 'sell' and current_price < order.price_limit):
                        logger.warning(f"Price limit exceeded for TWAP {order_id}")
                        break

                # Execute slice
                result = await self.exchange.create_limit_order(
                    order.symbol, order.side, slice_quantity, current_price
                )

                slice_prices.append(current_price)
                executed_quantity += slice_quantity

                logger.info(f"TWAP slice {slice_num + 1}/{total_slices} executed at {current_price}")

                # Wait for next slice interval
                elapsed = time.time() - start_time
                expected_elapsed = (slice_num + 1) * order.slice_interval
                sleep_time = max(0, expected_elapsed - elapsed)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"TWAP slice error: {e}")
                await asyncio.sleep(order.slice_interval)

        avg_price = np.mean(slice_prices) if slice_prices else 0
        completion_rate = executed_quantity / order.total_quantity

        result = {
            'order_id': order_id,
            'executed_quantity': executed_quantity,
            'completion_rate': completion_rate,
            'avg_price': avg_price,
            'slice_count': len(slice_prices)
        }

        self.execution_history.append(result)
        return result

    async def execute_vwap_order(self, order: VWAPOrder) -> Dict[str, Any]:
        """
        Execute Volume-Weighted Average Price order.

        Strategy:
        1. Analyze historical volume profile
        2. Execute larger quantities during high volume periods
        3. Adjust execution rate based on current volume
        4. Complete within time window
        """
        order_id = f"vwap_{order.symbol}_{int(time.time())}"

        # Get volume profile (simplified - in reality would use historical data)
        volume_profile = await self._get_volume_profile(order.symbol, order.volume_profile_lookback)

        executed_quantity = 0.0
        remaining_time = order.end_time - time.time()
        remaining_quantity = order.total_quantity

        logger.info(f"Starting VWAP order {order_id} for {remaining_time:.0f} seconds")

        while remaining_quantity > 0 and time.time() < order.end_time:
            try:
                # Get current volume and price
                ticker = await self.exchange.fetch_ticker(order.symbol)
                current_volume = await self._get_current_volume(order.symbol)

                # Calculate target execution rate based on volume profile
                target_rate = self._calculate_vwap_rate(
                    volume_profile, current_volume, remaining_quantity, remaining_time
                )

                # Execute portion
                execution_qty = min(target_rate, remaining_quantity)
                current_price = ticker['bid'] if order.side == 'buy' else ticker['ask']

                if order.price_limit:
                    if (order.side == 'buy' and current_price > order.price_limit) or \
                       (order.side == 'sell' and current_price < order.price_limit):
                        logger.warning(f"Price limit exceeded for VWAP {order_id}")
                        break

                result = await self.exchange.create_limit_order(
                    order.symbol, order.side, execution_qty, current_price
                )

                executed_quantity += execution_qty
                remaining_quantity -= execution_qty

                logger.info(f"VWAP execution: {execution_qty} at {current_price}")

                # Wait before next execution
                await asyncio.sleep(5)  # 5 second intervals for VWAP

            except Exception as e:
                logger.error(f"VWAP execution error: {e}")
                await asyncio.sleep(5)

        completion_rate = executed_quantity / order.total_quantity

        result = {
            'order_id': order_id,
            'executed_quantity': executed_quantity,
            'completion_rate': completion_rate,
            'success': completion_rate > 0.9  # VWAP typically aims for 90%+ completion
        }

        self.execution_history.append(result)
        return result

    async def _monitor_order_fill(self, order_id: str, order: IcebergOrder) -> None:
        """Monitor order fill status for iceberg orders."""
        try:
            while True:
                order_status = await self.exchange.fetch_order(order_id)
                filled = order_status.get('filled', 0)
                remaining = order_status.get('remaining', 0)

                if remaining <= 0 or order_status.get('status') in ['closed', 'canceled']:
                    break

                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error monitoring order {order_id}: {e}")

    async def _get_volume_profile(self, symbol: str, lookback_days: int) -> Dict[str, Any]:
        """Get historical volume profile for VWAP orders."""
        # Simplified implementation - would need proper OHLCV data
        return {
            'peak_volume_hour': 12,  # Noon UTC
            'avg_volume': 1000000,
            'volume_distribution': [0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.05, 0.02]
        }

    async def _get_current_volume(self, symbol: str) -> float:
        """Get current trading volume."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker.get('quoteVolume', 0)
        except:
            return 0

    def _calculate_vwap_rate(self, volume_profile: Dict, current_volume: float,
                           remaining_qty: float, remaining_time: float) -> float:
        """Calculate target execution rate for VWAP."""
        # Simplified VWAP rate calculation
        target_completion = remaining_qty / remaining_time if remaining_time > 0 else remaining_qty
        volume_multiplier = current_volume / volume_profile.get('avg_volume', 1)

        return min(target_completion * volume_multiplier, remaining_qty)

    def _calculate_avg_price(self, order_id: str) -> float:
        """Calculate average execution price for an order."""
        # Would need to track individual fills
        return 0.0

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for monitoring."""
        if not self.execution_history:
            return {}

        completion_rates = [h['completion_rate'] for h in self.execution_history]
        return {
            'total_orders': len(self.execution_history),
            'avg_completion_rate': np.mean(completion_rates),
            'success_rate': sum(1 for h in self.execution_history if h.get('success', False)) / len(self.execution_history),
            'active_orders': len(self.active_orders)
        }
