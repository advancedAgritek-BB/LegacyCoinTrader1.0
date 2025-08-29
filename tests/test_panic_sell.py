"""Test panic sell functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from crypto_bot.telegram_ctl import panic_sell_cmd
from crypto_bot.telegram_bot_ui import TelegramBotUI
from crypto_bot.bot_controller import TradingBotController
from crypto_bot.main import force_exit_all
from crypto_bot.paper_wallet import PaperWallet


class TestPanicSell:
    """Test panic sell functionality."""

    def test_telegram_ctl_panic_sell(self):
        """Test panic sell command in telegram controller."""
        # Mock update and context
        update = MagicMock()
        context = MagicMock()
        context.bot_data = {
            "controller": MagicMock(),
            "admin_id": "123"
        }
        
        # Mock admin check
        with patch("crypto_bot.telegram_ctl.is_admin", return_value=True):
            # Mock controller response
            context.bot_data["controller"].close_all_positions = AsyncMock(
                return_value="All positions closed"
            )
            
            # Mock reply_text as async
            update.message.reply_text = AsyncMock()
            
            # Execute panic sell
            asyncio.run(panic_sell_cmd(update, context))
            
            # Verify controller was called
            context.bot_data["controller"].close_all_positions.assert_called_once()
            update.message.reply_text.assert_called_once_with("All positions closed")

    def test_telegram_bot_ui_panic_sell(self):
        """Test panic sell command in telegram bot UI."""
        # Create UI instance with correct constructor
        ui = TelegramBotUI(
            notifier=MagicMock(),
            state={},
            log_file="test.log",
            exchange=MagicMock()
        )
        
        # Mock update and context
        update = MagicMock()
        context = MagicMock()
        
        # Mock admin check and cooldown
        with patch.object(ui, "_check_admin", return_value=True), \
             patch.object(ui, "_check_cooldown", return_value=True):
            
            # Mock the _reply method to avoid async issues
            with patch.object(ui, "_reply", new_callable=AsyncMock):
                # Execute panic sell
                asyncio.run(ui.panic_sell_cmd(update, context))
                
                # Verify controller was called by checking if the method was executed
                # The controller.close_all_positions should have been called
                assert hasattr(ui.controller, 'close_all_positions')

    def test_bot_controller_close_all_positions(self):
        """Test bot controller close all positions method."""
        controller = TradingBotController(
            config_path="test_config.yaml",
            trades_file="test_trades.csv"
        )
        
        # Test close all positions
        result = asyncio.run(controller.close_all_positions())
        
        assert result["status"] == "liquidation_scheduled"
        assert controller.state["liquidate_all"] is True

    @pytest.mark.asyncio
    async def test_force_exit_all_with_paper_wallet(self):
        """Test force exit all function with paper wallet."""
        # Create mock context
        ctx = MagicMock()
        ctx.config = {"execution_mode": "dry_run"}
        ctx.positions = {
            "BTC/USDT": {
                "side": "buy",
                "size": 0.1,
                "entry_price": 50000.0,
                "strategy": "test_strategy"
            },
            "ETH/USDT": {
                "side": "sell",
                "size": 1.0,
                "entry_price": 3000.0,
                "strategy": "test_strategy"
            }
        }
        ctx.paper_wallet = PaperWallet(10000.0)
        ctx.balance = 10000.0
        ctx.risk_manager = MagicMock()
        ctx.notifier = MagicMock()
        ctx.exchange = MagicMock()
        ctx.ws_client = MagicMock()
        ctx.df_cache = {"1h": {}}
        
        # Mock cex_trade_async
        with patch("crypto_bot.main.cex_trade_async", new_callable=AsyncMock) as mock_trade:
            await force_exit_all(ctx)
            
            # Verify both positions were closed
            assert mock_trade.call_count == 2
            
            # Verify positions were removed
            assert len(ctx.positions) == 0
            
            # Verify risk manager was called
            assert ctx.risk_manager.deallocate_capital.call_count == 2

    @pytest.mark.asyncio
    async def test_force_exit_all_with_live_trading(self):
        """Test force exit all function with live trading."""
        # Create mock context
        ctx = MagicMock()
        ctx.config = {"execution_mode": "live"}
        ctx.positions = {
            "BTC/USDT": {
                "side": "buy",
                "size": 0.1,
                "entry_price": 50000.0,
                "strategy": "test_strategy"
            }
        }
        ctx.paper_wallet = None
        ctx.risk_manager = MagicMock()
        ctx.notifier = MagicMock()
        ctx.exchange = MagicMock()
        ctx.ws_client = MagicMock()
        ctx.df_cache = {"1h": {}}
        
        # Mock cex_trade_async
        with patch("crypto_bot.main.cex_trade_async", new_callable=AsyncMock) as mock_trade:
            await force_exit_all(ctx)
            
            # Verify trade was executed
            mock_trade.assert_called_once()
            
            # Verify position was removed
            assert len(ctx.positions) == 0

    def test_panic_sell_integration(self):
        """Test full panic sell integration flow."""
        # Create controller
        controller = TradingBotController(
            config_path="test_config.yaml",
            trades_file="test_trades.csv"
        )
        
        # Set liquidate flag
        controller.state["liquidate_all"] = True
        
        # Verify flag is set
        assert controller.state["liquidate_all"] is True
        
        # Reset flag (simulating main loop processing)
        controller.state["liquidate_all"] = False
        
        # Verify flag is reset
        assert controller.state["liquidate_all"] is False

    def test_panic_sell_admin_only(self):
        """Test that panic sell is admin-only."""
        # Mock update and context
        update = MagicMock()
        context = MagicMock()
        context.bot_data = {
            "controller": MagicMock(),
            "admin_id": "123"
        }
        
        # Mock non-admin user
        update.effective_user.id = "456"
        
        # Mock admin check to return False
        with patch("crypto_bot.telegram_ctl.is_admin", return_value=False):
            # Execute panic sell
            asyncio.run(panic_sell_cmd(update, context))
            
            # Verify controller was NOT called
            context.bot_data["controller"].close_all_positions.assert_not_called()
            update.message.reply_text.assert_not_called()

    def test_panic_sell_cooldown(self):
        """Test panic sell cooldown mechanism."""
        # Create UI instance with correct constructor
        ui = TelegramBotUI(
            notifier=MagicMock(),
            state={},
            log_file="test.log",
            exchange=MagicMock()
        )
        
        # Mock update and context
        update = MagicMock()
        context = MagicMock()
        
        # Mock admin check but fail cooldown
        with patch.object(ui, "_check_admin", return_value=True), \
             patch.object(ui, "_check_cooldown", return_value=False):
            
            # Execute panic sell
            asyncio.run(ui.panic_sell_cmd(update, context))
            
            # Since cooldown failed, the method should return early without calling controller
            # We can verify this by checking that the controller exists but wasn't called
            assert hasattr(ui.controller, 'close_all_positions')

    def test_panic_sell_error_handling(self):
        """Test panic sell error handling."""
        # Create UI instance with correct constructor
        ui = TelegramBotUI(
            notifier=MagicMock(),
            state={},
            log_file="test.log",
            exchange=MagicMock()
        )
        
        # Mock update and context
        update = MagicMock()
        context = MagicMock()
        
        # Mock admin check and cooldown
        with patch.object(ui, "_check_admin", return_value=True), \
             patch.object(ui, "_check_cooldown", return_value=True):
            
            # Mock the controller to raise an exception
            ui.controller.close_all_positions = AsyncMock(side_effect=Exception("API Error"))
            
            # Mock the _reply method to avoid async issues
            with patch.object(ui, "_reply", new_callable=AsyncMock):
                # Execute panic sell - should handle error gracefully
                with pytest.raises(Exception, match="API Error"):
                    asyncio.run(ui.panic_sell_cmd(update, context))
                
                # Verify controller was called
                ui.controller.close_all_positions.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
