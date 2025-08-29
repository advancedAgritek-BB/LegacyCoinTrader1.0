"""
Market Making Strategy for High-Frequency Trading

This strategy provides liquidity by continuously quoting bid and ask prices,
profiting from the bid-ask spread while managing inventory risk.

Key Features:
- Dynamic spread adjustment based on volatility
- Inventory management with mean-reversion
- Adaptive quote sizes based on market depth
- Risk controls for adverse selection
"""

from __future__ import annotations
import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

import pandas as pd
import ta

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.volatility import normalize_score_by_volatility

logger = setup_logger(__name__, LOG_DIR / "market_making.log")


@dataclass
class Quote:
    """Represents a market making quote."""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: float
    inventory_skew: float = 0.0


@dataclass
class MarketMakingConfig:
    """Configuration for market making strategy."""
    base_spread_bps: float = 2.0  # Base spread in basis points
    max_spread_bps: float = 10.0  # Maximum spread during high volatility
    min_spread_bps: float = 1.0   # Minimum spread during low volatility
    quote_size_base: float = 1000.0  # Base quote size in USD
    max_quote_size: float = 10000.0  # Maximum quote size
    inventory_limit: float = 50000.0  # Max inventory deviation
    skew_adjustment: float = 0.1  # Inventory skew adjustment factor
    volatility_window: int = 20  # Volatility calculation window
    max_holding_time: int = 300  # Max time to hold position (seconds)
    risk_multiplier: float = 2.0  # Risk adjustment multiplier


class MarketMakingBot:
    """
    High-frequency market making bot that provides liquidity and profits from spreads.

    Strategy:
    1. Continuously quote bid and ask prices around fair value
    2. Adjust spreads based on volatility and inventory
    3. Manage inventory through mean-reversion
    4. Handle adverse selection through dynamic sizing
    """

    def __init__(self, config: MarketMakingConfig, exchange_client: Any):
        self.config = config
        self.exchange = exchange_client
        self.active_quotes: Dict[str, Quote] = {}
        self.inventory: Dict[str, float] = {}
        self.trade_history: deque = deque(maxlen=1000)
        self.pnl_realized = 0.0
        self.pnl_unrealized = 0.0

        # Market data tracking
        self.price_history: Dict[str, deque] = {}
        self.spread_history: Dict[str, deque] = {}
        self.volatility_cache: Dict[str, float] = {}

    async def start_market_making(self, symbols: List[str]) -> None:
        """Start market making for given symbols."""
        logger.info(f"Starting market making for {len(symbols)} symbols")

        tasks = []
        for symbol in symbols:
            self.price_history[symbol] = deque(maxlen=100)
            self.spread_history[symbol] = deque(maxlen=50)
            self.inventory[symbol] = 0.0
            tasks.append(self._market_make_symbol(symbol))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _market_make_symbol(self, symbol: str) -> None:
        """Run market making loop for a single symbol."""
        logger.info(f"Starting market making for {symbol}")

        while True:
            try:
                # Get current market data
                ticker = await self.exchange.fetch_ticker(symbol)
                orderbook = await self.exchange.fetch_order_book(symbol, limit=10)

                # Update price history
                mid_price = (ticker['bid'] + ticker['ask']) / 2
                self.price_history[symbol].append(mid_price)

                # Calculate current spread
                spread_bps = (ticker['ask'] - ticker['bid']) / mid_price * 10000
                self.spread_history[symbol].append(spread_bps)

                # Calculate fair value and quotes
                fair_value = self._calculate_fair_value(symbol, mid_price)
                quote = self._calculate_quote(symbol, fair_value, orderbook)

                if quote:
                    await self._place_quotes(quote)

                # Manage inventory
                await self._manage_inventory(symbol, mid_price)

                # Update P&L
                self._update_pnl(symbol, mid_price)

                # Brief pause to avoid overwhelming exchange
                await asyncio.sleep(0.1)  # 100ms intervals

            except Exception as e:
                logger.error(f"Error in market making loop for {symbol}: {e}")
                await asyncio.sleep(1)

    def _calculate_fair_value(self, symbol: str, mid_price: float) -> float:
        """Calculate fair value using recent price history."""
        if len(self.price_history[symbol]) < 10:
            return mid_price

        prices = list(self.price_history[symbol])
        recent_trend = np.polyfit(range(len(prices)), prices, 1)[0]

        # Adjust fair value based on recent trend
        trend_adjustment = recent_trend * 5  # 5-period trend adjustment
        return mid_price + trend_adjustment

    def _calculate_quote(self, symbol: str, fair_value: float,
                        orderbook: Dict) -> Optional[Quote]:
        """Calculate optimal bid and ask quotes."""

        # Calculate dynamic spread based on volatility
        volatility = self._calculate_volatility(symbol)
        dynamic_spread_bps = self._calculate_dynamic_spread(volatility)

        # Adjust spread based on inventory
        inventory_skew = self._calculate_inventory_skew(symbol)
        adjusted_spread_bps = dynamic_spread_bps * (1 + inventory_skew * self.config.skew_adjustment)

        # Ensure spread is within bounds
        spread_bps = np.clip(adjusted_spread_bps,
                           self.config.min_spread_bps,
                           self.config.max_spread_bps)

        # Calculate bid and ask prices
        spread_amount = fair_value * spread_bps / 10000
        bid_price = fair_value - spread_amount / 2
        ask_price = fair_value + spread_amount / 2

        # Calculate quote sizes
        base_size = self._calculate_quote_size(symbol, orderbook)
        bid_size = base_size * (1 - inventory_skew)  # Reduce bid size if long
        ask_size = base_size * (1 + inventory_skew)  # Reduce ask size if short

        # Ensure minimum sizes
        min_size = 0.001  # Minimum order size
        bid_size = max(bid_size, min_size)
        ask_size = max(ask_size, min_size)

        return Quote(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=time.time(),
            inventory_skew=inventory_skew
        )

    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate recent volatility."""
        if len(self.price_history[symbol]) < self.config.volatility_window:
            return 0.01  # Default 1% volatility

        prices = list(self.price_history[symbol])
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(252)  # Annualized volatility

    def _calculate_dynamic_spread(self, volatility: float) -> float:
        """Calculate spread based on volatility."""
        # Higher volatility = wider spreads
        vol_multiplier = min(volatility * 100, 5.0)  # Cap at 5x base spread
        return self.config.base_spread_bps * (1 + vol_multiplier)

    def _calculate_inventory_skew(self, symbol: str) -> float:
        """Calculate inventory skew for position management."""
        inventory = self.inventory.get(symbol, 0.0)

        if abs(inventory) < self.config.inventory_limit * 0.1:
            return 0.0

        # Return skew between -1 and 1
        return np.clip(inventory / self.config.inventory_limit, -1.0, 1.0)

    def _calculate_quote_size(self, symbol: str, orderbook: Dict) -> float:
        """Calculate optimal quote size based on market depth."""
        try:
            # Analyze order book depth
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if not bids or not asks:
                return self.config.quote_size_base

            # Calculate market depth at different levels
            bid_depth_5pct = sum(size for price, size in bids[:5])
            ask_depth_5pct = sum(size for price, size in asks[:5])

            avg_depth = (bid_depth_5pct + ask_depth_5pct) / 2

            # Adjust quote size based on market depth
            depth_multiplier = min(avg_depth / 1000, 5.0)  # Cap at 5x

            quote_size = self.config.quote_size_base * depth_multiplier
            return min(quote_size, self.config.max_quote_size)

        except Exception:
            return self.config.quote_size_base

    async def _place_quotes(self, quote: Quote) -> None:
        """Place bid and ask quotes on the exchange."""
        try:
            # Cancel existing quotes if they exist
            if quote.symbol in self.active_quotes:
                await self._cancel_quotes(quote.symbol)

            # Place new bid quote
            bid_order = await self.exchange.create_limit_buy_order(
                quote.symbol, quote.bid_size, quote.bid_price
            )

            # Place new ask quote
            ask_order = await self.exchange.create_limit_sell_order(
                quote.symbol, quote.ask_size, quote.ask_price
            )

            # Store active quotes
            self.active_quotes[quote.symbol] = quote

            logger.debug(f"Placed quotes for {quote.symbol}: "
                        f"Bid {quote.bid_price:.6f} x {quote.bid_size:.4f}, "
                        f"Ask {quote.ask_price:.6f} x {quote.ask_size:.4f}")

        except Exception as e:
            logger.error(f"Error placing quotes for {quote.symbol}: {e}")

    async def _cancel_quotes(self, symbol: str) -> None:
        """Cancel existing quotes for a symbol."""
        try:
            # Get all open orders for symbol
            orders = await self.exchange.fetch_open_orders(symbol)

            # Cancel limit orders (our quotes)
            for order in orders:
                if order['type'] == 'limit':
                    await self.exchange.cancel_order(order['id'])

            # Remove from active quotes
            if symbol in self.active_quotes:
                del self.active_quotes[symbol]

        except Exception as e:
            logger.error(f"Error canceling quotes for {symbol}: {e}")

    async def _manage_inventory(self, symbol: str, mid_price: float) -> None:
        """Manage inventory through mean-reversion trades."""
        inventory = self.inventory.get(symbol, 0.0)

        # Check if inventory exceeds threshold
        if abs(inventory) > self.config.inventory_limit * 0.8:
            try:
                # Place mean-reversion order
                if inventory > 0:
                    # We're long, place sell order below market
                    reversion_price = mid_price * 0.9995  # 0.05% below
                    size = min(abs(inventory) * 0.1, 1000)  # Sell 10% of position
                    await self.exchange.create_limit_sell_order(symbol, size, reversion_price)
                else:
                    # We're short, place buy order above market
                    reversion_price = mid_price * 1.0005  # 0.05% above
                    size = min(abs(inventory) * 0.1, 1000)  # Buy 10% of position
                    await self.exchange.create_limit_buy_order(symbol, size, reversion_price)

                logger.info(f"Inventory management trade for {symbol}: "
                           f"inventory={inventory:.2f}, price={reversion_price:.6f}")

            except Exception as e:
                logger.error(f"Error in inventory management for {symbol}: {e}")

    def _update_pnl(self, symbol: str, mid_price: float) -> None:
        """Update realized and unrealized P&L."""
        # This would need to track actual trades
        # Simplified version for demonstration
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get market making statistics."""
        return {
            'active_symbols': len(self.active_quotes),
            'total_inventory_value': sum(abs(v) for v in self.inventory.values()),
            'avg_spread_bps': np.mean([
                np.mean(list(self.spread_history[symbol])) if self.spread_history[symbol] else 0
                for symbol in self.spread_history
            ]),
            'pnl_realized': self.pnl_realized,
            'pnl_unrealized': self.pnl_unrealized
        }

    async def stop(self) -> None:
        """Stop market making and cancel all quotes."""
        logger.info("Stopping market making...")

        # Cancel all active quotes
        cancel_tasks = []
        for symbol in self.active_quotes:
            cancel_tasks.append(self._cancel_quotes(symbol))

        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

        logger.info("Market making stopped")


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """
    Generate signal for market making regime detection.

    This strategy works best in ranging/sideways markets with good liquidity.
    """
    if df.empty or len(df) < 20:
        return 0.0, "none"

    # Calculate indicators for regime detection
    sma_20 = ta.trend.sma_indicator(df["close"], window=20)
    sma_50 = ta.trend.sma_indicator(df["close"], window=50)

    # Calculate ADX for trend strength
    adx = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    # Market making works best when:
    # 1. Low trend strength (ADX < 25)
    # 2. Price near moving averages (ranging market)
    # 3. Good volume (liquidity)

    trend_strength = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 50
    price_vs_ma = abs(df["close"].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]

    # Score based on market making suitability
    if trend_strength < 25 and price_vs_ma < 0.02:  # Less than 2% deviation from MA
        score = 0.8
    elif trend_strength < 30 and price_vs_ma < 0.05:
        score = 0.6
    else:
        score = 0.2

    # Volume confirmation
    if "volume" in df.columns:
        vol_sma = df["volume"].rolling(20).mean()
        if df["volume"].iloc[-1] > vol_sma.iloc[-1]:
            score *= 1.2  # Boost score for high volume

    return min(score, 1.0), "market_making"


class regime_filter:
    """Match sideways/ranging regime for market making."""
    @staticmethod
    def matches(regime: str) -> bool:
        return regime in ["sideways", "ranging", "low_trend"]
