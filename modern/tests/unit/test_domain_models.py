"""
Unit Tests for Domain Models

Comprehensive tests for all domain models including validation,
business logic, and edge cases.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from pydantic import ValidationError

import sys
from pathlib import Path

# Add the modern/src directory to Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from domain.models import (
    TradingSymbol,
    Order,
    Position,
    Trade,
    MarketData,
    StrategySignal,
    OrderSide,
    OrderType,
    OrderStatus
)


class TestTradingSymbol:
    """Test TradingSymbol model."""

    def test_valid_trading_symbol(self):
        """Test valid trading symbol creation."""
        symbol = TradingSymbol(
            symbol="BTC/USD",
            base_currency="BTC",
            quote_currency="USD",
            exchange="kraken",
            min_order_size=Decimal("0.0001"),
            price_precision=2,
            quantity_precision=8
        )

        assert symbol.symbol == "BTC/USD"
        assert symbol.base_currency == "BTC"
        assert symbol.quote_currency == "USD"
        assert symbol.exchange == "kraken"
        assert symbol.min_order_size == Decimal("0.0001")
        assert symbol.price_precision == 2
        assert symbol.quantity_precision == 8
        assert symbol.is_active is True

    def test_symbol_format_validation(self):
        """Test symbol format validation."""
        # Valid formats
        TradingSymbol(
            symbol="BTC/USD",
            base_currency="BTC",
            quote_currency="USD",
            exchange="kraken",
            min_order_size=Decimal("0.0001"),
            price_precision=2,
            quantity_precision=8
        )

        # Invalid format - no slash
        with pytest.raises(ValidationError, match="Symbol must be in format BASE/QUOTE"):
            TradingSymbol(
                symbol="BTCUSD",
                base_currency="BTC",
                quote_currency="USD",
                exchange="kraken",
                min_order_size=Decimal("0.0001"),
                price_precision=2,
                quantity_precision=8
            )

    def test_currency_consistency_validation(self):
        """Test currency consistency validation."""
        # Mismatched base currency
        with pytest.raises(ValidationError, match="Base currency.*doesn't match symbol"):
            TradingSymbol(
                symbol="BTC/USD",
                base_currency="ETH",  # Should be BTC
                quote_currency="USD",
                exchange="kraken",
                min_order_size=Decimal("0.0001"),
                price_precision=2,
                quantity_precision=8
            )

        # Mismatched quote currency
        with pytest.raises(ValidationError, match="Quote currency.*doesn't match symbol"):
            TradingSymbol(
                symbol="BTC/USD",
                base_currency="BTC",
                quote_currency="EUR",  # Should be USD
                exchange="kraken",
                min_order_size=Decimal("0.0001"),
                price_precision=2,
                quantity_precision=8
            )

    def test_symbol_validation_edge_cases(self):
        """Test symbol validation edge cases."""
        # Empty symbol
        with pytest.raises(ValidationError):
            TradingSymbol(
                symbol="",
                base_currency="BTC",
                quote_currency="USD",
                exchange="kraken",
                min_order_size=Decimal("0.0001"),
                price_precision=2,
                quantity_precision=8
            )

        # Symbol too long
        with pytest.raises(ValidationError):
            TradingSymbol(
                symbol="VERYVERYLONGTRADINGSYMBOL/USD",
                base_currency="VERYVERYLONGTRADINGSYMBOL",
                quote_currency="USD",
                exchange="kraken",
                min_order_size=Decimal("0.0001"),
                price_precision=2,
                quantity_precision=8
            )


class TestOrder:
    """Test Order model."""

    def test_valid_order_creation(self):
        """Test valid order creation."""
        order = Order(
            id="order_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00")
        )

        assert order.id == "order_123"
        assert order.symbol == "BTC/USD"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.quantity == Decimal("0.01")
        assert order.price == Decimal("50000.00")
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0
        assert order.remaining_quantity == Decimal("0.01")

    def test_order_properties(self):
        """Test order properties."""
        order = Order(
            id="order_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00")
        )

        # Test is_filled
        assert not order.is_filled
        assert order.fill_percentage == 0.0

        # Test is_active
        assert order.is_active

        # Mark as filled
        order.filled_quantity = Decimal("0.01")
        assert order.is_filled
        assert order.fill_percentage == 100.0
        assert not order.is_active

    def test_order_validation(self):
        """Test order validation."""
        # Missing required fields
        with pytest.raises(ValidationError):
            Order(
                symbol="BTC/USD",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                quantity=Decimal("0.01")
            )

        # Invalid quantity
        with pytest.raises(ValidationError):
            Order(
                id="order_123",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                quantity=Decimal("0"),
                price=Decimal("50000.00")
            )

        # Invalid price
        with pytest.raises(ValidationError):
            Order(
                id="order_123",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                quantity=Decimal("0.01"),
                price=Decimal("0")
            )

    def test_market_order_without_price(self):
        """Test market order without price."""
        order = Order(
            id="order_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("0.01")
            # No price specified for market order
        )

        assert order.type == OrderType.MARKET
        assert order.price is None

    def test_stop_order_validation(self):
        """Test stop order validation."""
        order = Order(
            id="order_123",
            symbol="BTC/USD",
            side=OrderSide.SELL,
            type=OrderType.STOP_LOSS,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            stop_price=Decimal("49000.00")
        )

        assert order.stop_price == Decimal("49000.00")


class TestPosition:
    """Test Position model."""

    def test_valid_position_creation(self):
        """Test valid position creation."""
        position = Position(
            id="pos_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("51000.00")
        )

        assert position.id == "pos_123"
        assert position.symbol == "BTC/USD"
        assert position.side == OrderSide.BUY
        assert position.quantity == Decimal("0.01")
        assert position.entry_price == Decimal("50000.00")
        assert position.current_price == Decimal("51000.00")

        # Test calculated fields
        assert position.unrealized_pnl == Decimal("100.00")  # (51000 - 50000) * 0.01
        assert position.pnl_percentage == 2.0  # (51000 - 50000) / 50000 * 100
        assert position.market_value == Decimal("510.00")  # 0.01 * 51000
        assert position.is_profitable is True

    def test_position_loss_scenario(self):
        """Test position in loss scenario."""
        position = Position(
            id="pos_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("49000.00")
        )

        assert position.unrealized_pnl == Decimal("-100.00")
        assert position.pnl_percentage == -2.0
        assert position.is_profitable is False

    def test_sell_position_pnl(self):
        """Test sell position P&L calculation."""
        position = Position(
            id="pos_123",
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("49000.00")
        )

        assert position.unrealized_pnl == Decimal("100.00")  # (50000 - 49000) * 0.01
        assert position.pnl_percentage == 2.0

    def test_stop_loss_triggering(self):
        """Test stop loss triggering."""
        position = Position(
            id="pos_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("49000.00"),
            stop_loss_price=Decimal("49500.00")
        )

        assert position.should_stop_loss is True

        # Test when stop loss not triggered
        position.current_price = Decimal("49600.00")
        assert position.should_stop_loss is False

    def test_take_profit_triggering(self):
        """Test take profit triggering."""
        position = Position(
            id="pos_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("52000.00"),
            take_profit_price=Decimal("51500.00")
        )

        assert position.should_take_profit is True

        # Test when take profit not triggered
        position.current_price = Decimal("51400.00")
        assert position.should_take_profit is False

    def test_sell_position_stop_loss(self):
        """Test stop loss for sell position."""
        position = Position(
            id="pos_123",
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.01"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("51000.00"),
            stop_loss_price=Decimal("50500.00")
        )

        assert position.should_stop_loss is True

        # Test when stop loss not triggered
        position.current_price = Decimal("50400.00")
        assert position.should_stop_loss is False


class TestTrade:
    """Test Trade model."""

    def test_valid_trade_creation(self):
        """Test valid trade creation."""
        trade = Trade(
            id="trade_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            value=Decimal("500.00"),
            pnl=Decimal("100.00"),
            pnl_percentage=2.0,
            commission=Decimal("0.50"),
            order_id="order_123"
        )

        assert trade.id == "trade_123"
        assert trade.symbol == "BTC/USD"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == Decimal("0.01")
        assert trade.price == Decimal("50000.00")
        assert trade.value == Decimal("500.00")
        assert trade.pnl == Decimal("100.00")
        assert trade.pnl_percentage == 2.0
        assert trade.commission == Decimal("0.50")
        assert trade.is_profitable is True

    def test_trade_properties(self):
        """Test trade properties."""
        # Profitable trade
        profitable_trade = Trade(
            id="trade_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            value=Decimal("500.00"),
            pnl=Decimal("100.00"),
            pnl_percentage=2.0,
            commission=Decimal("0.50"),
            order_id="order_123"
        )
        assert profitable_trade.is_profitable is True

        # Losing trade
        losing_trade = Trade(
            id="trade_124",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            value=Decimal("500.00"),
            pnl=Decimal("-50.00"),
            pnl_percentage=-1.0,
            commission=Decimal("0.50"),
            order_id="order_124"
        )
        assert losing_trade.is_profitable is False

    def test_effective_price_calculation(self):
        """Test effective price calculation with commission."""
        trade = Trade(
            id="trade_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=Decimal("50000.00"),
            value=Decimal("500.00"),
            pnl=Decimal("100.00"),
            pnl_percentage=2.0,
            commission=Decimal("5.00"),  # 1% commission
            order_id="order_123"
        )

        # Effective price should account for commission
        expected_effective_price = Decimal("50000.00") + (Decimal("5.00") / Decimal("0.01"))
        assert trade.effective_price == expected_effective_price


class TestMarketData:
    """Test MarketData model."""

    def test_valid_market_data(self):
        """Test valid market data creation."""
        data = MarketData(
            symbol="BTC/USD",
            open=Decimal("50000.00"),
            high=Decimal("51000.00"),
            low=Decimal("49500.00"),
            close=Decimal("50500.00"),
            volume=Decimal("100.00"),
            exchange="kraken",
            timeframe="1h"
        )

        assert data.symbol == "BTC/USD"
        assert data.open == Decimal("50000.00")
        assert data.high == Decimal("51000.00")
        assert data.low == Decimal("49500.00")
        assert data.close == Decimal("50500.00")
        assert data.volume == Decimal("100.00")
        assert data.exchange == "kraken"
        assert data.timeframe == "1h"

    def test_market_data_validation(self):
        """Test market data validation."""
        # High price lower than close
        with pytest.raises(ValidationError, match="High price.*cannot be less than close price"):
            MarketData(
                symbol="BTC/USD",
                open=Decimal("50000.00"),
                high=Decimal("50000.00"),  # Lower than close
                low=Decimal("49500.00"),
                close=Decimal("50500.00"),
                volume=Decimal("100.00"),
                exchange="kraken",
                timeframe="1h"
            )

        # Low price higher than close
        with pytest.raises(ValidationError, match="Low price.*cannot be greater than close price"):
            MarketData(
                symbol="BTC/USD",
                open=Decimal("50000.00"),
                high=Decimal("51000.00"),
                low=Decimal("50500.00"),  # Higher than close
                close=Decimal("50000.00"),
                volume=Decimal("100.00"),
                exchange="kraken",
                timeframe="1h"
            )

    def test_market_data_with_additional_fields(self):
        """Test market data with optional fields."""
        data = MarketData(
            symbol="BTC/USD",
            open=Decimal("50000.00"),
            high=Decimal("51000.00"),
            low=Decimal("49500.00"),
            close=Decimal("50500.00"),
            volume=Decimal("100.00"),
            vwap=Decimal("50250.00"),
            trades=150,
            exchange="kraken",
            timeframe="1h"
        )

        assert data.vwap == Decimal("50250.00")
        assert data.trades == 150


class TestStrategySignal:
    """Test StrategySignal model."""

    def test_valid_strategy_signal(self):
        """Test valid strategy signal creation."""
        signal = StrategySignal(
            strategy_name="test_strategy",
            symbol="BTC/USD",
            signal_type="buy",
            confidence=0.85,
            strength=80.0,
            indicators={"rsi": 30.0, "macd": -0.5}
        )

        assert signal.strategy_name == "test_strategy"
        assert signal.symbol == "BTC/USD"
        assert signal.signal_type == "buy"
        assert signal.confidence == 0.85
        assert signal.strength == 80.0
        assert signal.indicators == {"rsi": 30.0, "macd": -0.5}
        assert signal.is_strong_signal is True

    def test_signal_properties(self):
        """Test signal properties."""
        # Strong signal
        strong_signal = StrategySignal(
            strategy_name="test_strategy",
            symbol="BTC/USD",
            signal_type="buy",
            confidence=0.9,
            strength=85.0
        )
        assert strong_signal.is_strong_signal is True

        # Weak signal
        weak_signal = StrategySignal(
            strategy_name="test_strategy",
            symbol="BTC/USD",
            signal_type="buy",
            confidence=0.6,
            strength=60.0
        )
        assert weak_signal.is_strong_signal is False

    def test_risk_reward_calculation(self):
        """Test risk-reward ratio calculation."""
        signal = StrategySignal(
            strategy_name="test_strategy",
            symbol="BTC/USD",
            signal_type="buy",
            confidence=0.85,
            strength=80.0,
            indicators={"current_price": 50000.0},
            stop_loss=Decimal("49000.00"),
            take_profit=Decimal("52000.00")
        )

        rr_ratio = signal.risk_reward_ratio
        expected_ratio = 2.0  # (52000 - 50000) / (50000 - 49000)
        assert rr_ratio == expected_ratio

    def test_risk_reward_without_levels(self):
        """Test risk-reward ratio without stop loss/take profit."""
        signal = StrategySignal(
            strategy_name="test_strategy",
            symbol="BTC/USD",
            signal_type="buy",
            confidence=0.85,
            strength=80.0
        )

        assert signal.risk_reward_ratio is None


class TestEnumValidation:
    """Test enum validation for all models."""

    def test_order_side_enum(self):
        """Test OrderSide enum."""
        assert OrderSide.BUY == "buy"
        assert OrderSide.SELL == "sell"

    def test_order_type_enum(self):
        """Test OrderType enum."""
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"
        assert OrderType.STOP_LOSS == "stop_loss"
        assert OrderType.TAKE_PROFIT == "take_profit"
        assert OrderType.TRAILING_STOP == "trailing_stop"

    def test_order_status_enum(self):
        """Test OrderStatus enum."""
        assert OrderStatus.PENDING == "pending"
        assert OrderStatus.OPEN == "open"
        assert OrderStatus.PARTIAL == "partial"
        assert OrderStatus.FILLED == "filled"
        assert OrderStatus.CANCELLED == "cancelled"
        assert OrderStatus.REJECTED == "rejected"
        assert OrderStatus.EXPIRED == "expired"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
