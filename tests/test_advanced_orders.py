"""Tests for advanced orders execution module."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from decimal import Decimal
from crypto_bot.execution.advanced_orders import (
    AdvancedOrderManager,
    StopLossOrder,
    TakeProfitOrder,
    TrailingStopOrder,
    OCOOrder,
    IcebergOrder
)


class TestAdvancedOrderManager:
    """Test suite for Advanced Order Manager."""

    @pytest.fixture
    def mock_exchange(self):
        exchange = Mock()
        exchange.create_order = AsyncMock()
        exchange.cancel_order = AsyncMock()
        exchange.fetch_order = AsyncMock()
        exchange.fetch_ticker = AsyncMock(return_value={'last': 100.0})
        return exchange

    @pytest.fixture
    def order_manager(self, mock_exchange):
        return AdvancedOrderManager(mock_exchange)

    @pytest.fixture
    def sample_position(self):
        return {
            'symbol': 'BTC/USDT',
            'side': 'long',
            'amount': 1.0,
            'entry_price': 100.0,
            'current_price': 105.0
        }

    def test_order_manager_init(self, order_manager, mock_exchange):
        """Test order manager initialization."""
        assert order_manager.exchange == mock_exchange
        assert order_manager.active_orders == {}
        assert hasattr(order_manager, 'order_types')

    @pytest.mark.asyncio
    async def test_create_stop_loss_order(self, order_manager, sample_position):
        """Test stop loss order creation."""
        stop_price = 95.0
        
        order = await order_manager.create_stop_loss(
            symbol=sample_position['symbol'],
            amount=sample_position['amount'],
            stop_price=stop_price,
            side='sell'
        )
        
        assert isinstance(order, StopLossOrder)
        assert order.stop_price == stop_price
        assert order.symbol == sample_position['symbol']

    @pytest.mark.asyncio
    async def test_create_take_profit_order(self, order_manager, sample_position):
        """Test take profit order creation."""
        target_price = 110.0
        
        order = await order_manager.create_take_profit(
            symbol=sample_position['symbol'],
            amount=sample_position['amount'],
            target_price=target_price,
            side='sell'
        )
        
        assert isinstance(order, TakeProfitOrder)
        assert order.target_price == target_price
        assert order.symbol == sample_position['symbol']

    @pytest.mark.asyncio
    async def test_create_trailing_stop_order(self, order_manager, sample_position):
        """Test trailing stop order creation."""
        trail_percent = 0.05  # 5%
        
        order = await order_manager.create_trailing_stop(
            symbol=sample_position['symbol'],
            amount=sample_position['amount'],
            trail_percent=trail_percent,
            side='sell'
        )
        
        assert isinstance(order, TrailingStopOrder)
        assert order.trail_percent == trail_percent
        assert order.symbol == sample_position['symbol']

    @pytest.mark.asyncio
    async def test_create_oco_order(self, order_manager, sample_position):
        """Test OCO (One-Cancels-Other) order creation."""
        stop_price = 95.0
        target_price = 110.0
        
        order = await order_manager.create_oco(
            symbol=sample_position['symbol'],
            amount=sample_position['amount'],
            stop_price=stop_price,
            target_price=target_price,
            side='sell'
        )
        
        assert isinstance(order, OCOOrder)
        assert order.stop_price == stop_price
        assert order.target_price == target_price

    @pytest.mark.asyncio
    async def test_create_iceberg_order(self, order_manager, sample_position):
        """Test iceberg order creation."""
        total_amount = 10.0
        visible_amount = 1.0
        price = 100.0
        
        order = await order_manager.create_iceberg(
            symbol=sample_position['symbol'],
            total_amount=total_amount,
            visible_amount=visible_amount,
            price=price,
            side='buy'
        )
        
        assert isinstance(order, IcebergOrder)
        assert order.total_amount == total_amount
        assert order.visible_amount == visible_amount

    @pytest.mark.asyncio
    async def test_monitor_orders(self, order_manager, mock_exchange):
        """Test order monitoring functionality."""
        # Create a mock order
        mock_order = Mock()
        mock_order.check_trigger = AsyncMock(return_value=True)
        mock_order.execute = AsyncMock()
        mock_order.order_id = 'test_order_123'
        
        order_manager.active_orders['test_order_123'] = mock_order
        
        await order_manager.monitor_orders()
        
        mock_order.check_trigger.assert_called_once()
        mock_order.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order(self, order_manager, mock_exchange):
        """Test order cancellation."""
        order_id = 'test_order_123'
        mock_order = Mock()
        mock_order.cancel = AsyncMock()
        
        order_manager.active_orders[order_id] = mock_order
        
        await order_manager.cancel_order(order_id)
        
        mock_order.cancel.assert_called_once()
        assert order_id not in order_manager.active_orders


class TestStopLossOrder:
    """Test suite for Stop Loss Order."""

    @pytest.fixture
    def stop_loss_order(self):
        return StopLossOrder(
            symbol='BTC/USDT',
            amount=1.0,
            stop_price=95.0,
            side='sell',
            order_id='sl_123'
        )

    @pytest.mark.asyncio
    async def test_stop_loss_trigger_check(self, stop_loss_order):
        """Test stop loss trigger condition."""
        # Price above stop - should not trigger
        current_price = 100.0
        with patch.object(stop_loss_order, 'get_current_price', return_value=current_price):
            triggered = await stop_loss_order.check_trigger()
            assert triggered == False

        # Price below stop - should trigger
        current_price = 94.0
        with patch.object(stop_loss_order, 'get_current_price', return_value=current_price):
            triggered = await stop_loss_order.check_trigger()
            assert triggered == True

    @pytest.mark.asyncio
    async def test_stop_loss_execution(self, stop_loss_order):
        """Test stop loss order execution."""
        mock_exchange = Mock()
        mock_exchange.create_order = AsyncMock(return_value={'id': 'executed_123'})
        stop_loss_order.exchange = mock_exchange
        
        result = await stop_loss_order.execute()
        
        mock_exchange.create_order.assert_called_once()
        assert result['id'] == 'executed_123'


class TestTakeProfitOrder:
    """Test suite for Take Profit Order."""

    @pytest.fixture
    def take_profit_order(self):
        return TakeProfitOrder(
            symbol='BTC/USDT',
            amount=1.0,
            target_price=110.0,
            side='sell',
            order_id='tp_123'
        )

    @pytest.mark.asyncio
    async def test_take_profit_trigger_check(self, take_profit_order):
        """Test take profit trigger condition."""
        # Price below target - should not trigger
        current_price = 105.0
        with patch.object(take_profit_order, 'get_current_price', return_value=current_price):
            triggered = await take_profit_order.check_trigger()
            assert triggered == False

        # Price above target - should trigger
        current_price = 111.0
        with patch.object(take_profit_order, 'get_current_price', return_value=current_price):
            triggered = await take_profit_order.check_trigger()
            assert triggered == True


class TestTrailingStopOrder:
    """Test suite for Trailing Stop Order."""

    @pytest.fixture
    def trailing_stop_order(self):
        return TrailingStopOrder(
            symbol='BTC/USDT',
            amount=1.0,
            trail_percent=0.05,  # 5%
            side='sell',
            order_id='ts_123',
            initial_price=100.0
        )

    @pytest.mark.asyncio
    async def test_trailing_stop_updates_high(self, trailing_stop_order):
        """Test trailing stop updates highest price."""
        # New high price should update trailing stop
        new_high = 105.0
        with patch.object(trailing_stop_order, 'get_current_price', return_value=new_high):
            await trailing_stop_order.check_trigger()
            
            expected_stop = new_high * (1 - trailing_stop_order.trail_percent)
            assert trailing_stop_order.stop_price == expected_stop

    @pytest.mark.asyncio
    async def test_trailing_stop_trigger(self, trailing_stop_order):
        """Test trailing stop trigger condition."""
        # Set up scenario where price has risen then fallen
        trailing_stop_order.stop_price = 99.75  # 5% below 105
        
        # Price falls below trailing stop - should trigger
        current_price = 99.0
        with patch.object(trailing_stop_order, 'get_current_price', return_value=current_price):
            triggered = await trailing_stop_order.check_trigger()
            assert triggered == True


class TestOCOOrder:
    """Test suite for One-Cancels-Other Order."""

    @pytest.fixture
    def oco_order(self):
        return OCOOrder(
            symbol='BTC/USDT',
            amount=1.0,
            stop_price=95.0,
            target_price=110.0,
            side='sell',
            order_id='oco_123'
        )

    @pytest.mark.asyncio
    async def test_oco_stop_triggers_first(self, oco_order):
        """Test OCO when stop loss triggers first."""
        current_price = 94.0  # Below stop price
        
        with patch.object(oco_order, 'get_current_price', return_value=current_price):
            triggered = await oco_order.check_trigger()
            assert triggered == True
            assert oco_order.triggered_leg == 'stop'

    @pytest.mark.asyncio
    async def test_oco_target_triggers_first(self, oco_order):
        """Test OCO when take profit triggers first."""
        current_price = 111.0  # Above target price
        
        with patch.object(oco_order, 'get_current_price', return_value=current_price):
            triggered = await oco_order.check_trigger()
            assert triggered == True
            assert oco_order.triggered_leg == 'target'


class TestIcebergOrder:
    """Test suite for Iceberg Order."""

    @pytest.fixture
    def iceberg_order(self):
        return IcebergOrder(
            symbol='BTC/USDT',
            total_quantity=10.0,
            display_quantity=1.0,
            price=100.0,
            side='buy',
            order_id='ice_123'
        )

    @pytest.mark.asyncio
    async def test_iceberg_places_visible_chunk(self, iceberg_order):
        """Test iceberg places visible amount."""
        mock_exchange = Mock()
        mock_exchange.create_order = AsyncMock(return_value={'id': 'chunk_1'})
        iceberg_order.exchange = mock_exchange
        
        await iceberg_order.place_next_chunk()
        
        mock_exchange.create_order.assert_called_once()
        call_args = mock_exchange.create_order.call_args[1]
        assert call_args['amount'] == iceberg_order.visible_amount

    @pytest.mark.asyncio
    async def test_iceberg_completion(self, iceberg_order):
        """Test iceberg order completion."""
        # Simulate all chunks filled
        iceberg_order.filled_amount = iceberg_order.total_amount
        
        is_complete = iceberg_order.is_complete()
        assert is_complete == True

    @pytest.mark.asyncio
    async def test_iceberg_partial_fill_handling(self, iceberg_order):
        """Test iceberg handles partial fills."""
        # Simulate partial fill of current chunk
        iceberg_order.filled_amount = 0.5
        iceberg_order.current_chunk_filled = 0.5
        
        mock_exchange = Mock()
        mock_exchange.create_order = AsyncMock(return_value={'id': 'chunk_2'})
        iceberg_order.exchange = mock_exchange
        
        # Should place remaining amount of current chunk
        await iceberg_order.handle_partial_fill()
        
        call_args = mock_exchange.create_order.call_args[1]
        expected_amount = iceberg_order.visible_amount - iceberg_order.current_chunk_filled
        assert call_args['amount'] == expected_amount


@pytest.mark.integration
class TestAdvancedOrdersIntegration:
    """Integration tests for advanced orders."""

    @pytest.mark.asyncio
    async def test_full_order_lifecycle(self):
        """Test complete order lifecycle."""
        mock_exchange = Mock()
        mock_exchange.create_order = AsyncMock()
        mock_exchange.cancel_order = AsyncMock()
        mock_exchange.fetch_ticker = AsyncMock(return_value={'last': 100.0})
        
        manager = AdvancedOrderManager(mock_exchange)
        
        # Create stop loss
        stop_order = await manager.create_stop_loss(
            symbol='BTC/USDT',
            amount=1.0,
            stop_price=95.0,
            side='sell'
        )
        
        assert stop_order.order_id in manager.active_orders
        
        # Cancel order
        await manager.cancel_order(stop_order.order_id)
        
        assert stop_order.order_id not in manager.active_orders

    @pytest.mark.asyncio
    async def test_order_type_registry(self):
        """Test order type registration and retrieval."""
        mock_exchange = Mock()
        manager = AdvancedOrderManager(mock_exchange)
        
        # Check all order types are registered
        assert 'stop_loss' in manager.order_types
        assert 'take_profit' in manager.order_types
        assert 'trailing_stop' in manager.order_types
        assert 'oco' in manager.order_types
        assert 'iceberg' in manager.order_types
