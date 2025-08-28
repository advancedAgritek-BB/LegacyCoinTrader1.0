

import pytest
import asyncio
import pandas as pd
from unittest.mock import Mock, AsyncMock
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.phase_runner import BotContext


class TestPaperWalletIntegration:
    """Test that paper wallet integration works exactly like live trading."""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock bot context with paper wallet."""
        config = {
            "execution_mode": "dry_run",
            "allow_short": True,
            "max_open_trades": 5,
            "timeframe": "1h"
        }
        
        paper_wallet = PaperWallet(1000.0, max_open_trades=5, allow_short=True)
        
        ctx = BotContext(
            positions={},
            df_cache={"1h": {}},
            regime_cache={},
            config=config
        )
        ctx.paper_wallet = paper_wallet
        ctx.balance = 1000.0
        ctx.exchange = Mock()
        ctx.ws_client = None
        ctx.risk_manager = Mock()
        ctx.notifier = Mock()
        ctx.position_guard = Mock()
        
        return ctx
    
    def test_initial_balance_synchronization(self, mock_context):
        """Test that initial balance is properly synchronized."""
        assert mock_context.balance == 1000.0
        assert mock_context.paper_wallet.balance == 1000.0
        assert mock_context.balance == mock_context.paper_wallet.balance
    
    def test_buy_order_updates_wallet_correctly(self, mock_context):
        """Test that buy orders properly deduct funds from wallet."""
        initial_balance = mock_context.paper_wallet.balance
        
        # Simulate opening a buy position (0.1 BTC at $50 = $5 cost)
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "buy", 0.1, 50.0)
        
        # Check that funds were deducted
        expected_balance = initial_balance - (0.1 * 50.0)
        assert mock_context.paper_wallet.balance == expected_balance
        assert mock_context.paper_wallet.balance == 995.0
        
        # Check that position was recorded
        assert trade_id in mock_context.paper_wallet.positions
        position = mock_context.paper_wallet.positions[trade_id]
        assert position["side"] == "buy"
        assert position["size"] == 0.1
        assert position["entry_price"] == 50.0
    
    def test_sell_order_updates_wallet_correctly(self, mock_context):
        """Test that sell orders properly reserve funds in wallet."""
        initial_balance = mock_context.paper_wallet.balance
        
        # Simulate opening a sell position (0.1 BTC at $50 = $5 cost)
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "sell", 0.1, 50.0)
        
        # Check that funds were reserved
        expected_balance = initial_balance - (0.1 * 50.0)
        assert mock_context.paper_wallet.balance == expected_balance
        assert mock_context.paper_wallet.balance == 995.0
        
        # Check that position was recorded with reserved funds
        assert trade_id in mock_context.paper_wallet.positions
        position = mock_context.paper_wallet.positions[trade_id]
        assert position["side"] == "sell"
        assert position["size"] == 0.1
        assert position["entry_price"] == 50.0
        assert position["reserved"] == 5.0  # 0.1 * 50
    
    def test_position_closure_updates_wallet_correctly(self, mock_context):
        """Test that closing positions properly updates wallet balance."""
        # Open a buy position
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "buy", 0.1, 50.0)
        initial_balance = mock_context.paper_wallet.balance  # Should be 995.0
        
        # Close the position at a profit
        pnl = mock_context.paper_wallet.close(trade_id, 0.1, 55.0)
        
        # Check PnL calculation
        expected_pnl = (55.0 - 50.0) * 0.1
        assert pnl == expected_pnl
        assert pnl == 0.5
        
        # Check that balance was updated correctly
        expected_balance = initial_balance + (0.1 * 55.0)
        assert mock_context.paper_wallet.balance == expected_balance
        assert mock_context.paper_wallet.balance == 1000.5
        
        # Check that position was removed
        assert trade_id not in mock_context.paper_wallet.positions
    
    def test_short_position_closure_updates_wallet_correctly(self, mock_context):
        """Test that closing short positions properly updates wallet balance."""
        # Open a sell position
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "sell", 0.1, 50.0)
        initial_balance = mock_context.paper_wallet.balance  # Should be 995.0
        
        # Close the position at a profit (price went down)
        pnl = mock_context.paper_wallet.close(trade_id, 0.1, 45.0)
        
        # Check PnL calculation
        expected_pnl = (50.0 - 45.0) * 0.1
        assert pnl == expected_pnl
        assert pnl == 0.5
        
        # Check that balance was updated correctly
        # Should get back reserved funds + profit
        expected_balance = initial_balance + (0.1 * 50.0) + expected_pnl
        assert mock_context.paper_wallet.balance == expected_balance
        assert mock_context.paper_wallet.balance == 1000.5
        
        # Check that position was removed
        assert trade_id not in mock_context.paper_wallet.positions
    
    def test_partial_position_closure(self, mock_context):
        """Test that partial position closures work correctly."""
        # Open a buy position
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "buy", 0.2, 50.0)
        initial_balance = mock_context.paper_wallet.balance  # Should be 990.0
        
        # Close half the position at a profit
        pnl1 = mock_context.paper_wallet.close(trade_id, 0.1, 55.0)
        
        # Check first PnL
        expected_pnl1 = (55.0 - 50.0) * 0.1
        assert pnl1 == expected_pnl1
        assert pnl1 == 0.5
        
        # Check that position size was reduced
        position = mock_context.paper_wallet.positions[trade_id]
        assert position["size"] == 0.1
        
        # Check that balance was updated
        expected_balance_after_first = initial_balance + (0.1 * 55.0)
        assert mock_context.paper_wallet.balance == expected_balance_after_first
        
        # Close the remaining position at a higher profit
        pnl2 = mock_context.paper_wallet.close(trade_id, 0.1, 60.0)
        
        # Check second PnL
        expected_pnl2 = (60.0 - 50.0) * 0.1
        assert pnl2 == expected_pnl2
        assert pnl2 == 1.0
        
        # Check that position was fully closed
        assert trade_id not in mock_context.paper_wallet.positions
        
        # Check final balance
        expected_final_balance = expected_balance_after_first + (0.1 * 60.0)
        assert mock_context.paper_wallet.balance == expected_final_balance
    
    def test_insufficient_balance_prevents_trade(self, mock_context):
        """Test that insufficient balance prevents opening positions."""
        # Try to open a position larger than available balance
        with pytest.raises(RuntimeError, match="Insufficient balance"):
            mock_context.paper_wallet.open("BTC/USDT", "buy", 1.0, 5000.0)
        
        # Check that balance wasn't changed
        assert mock_context.paper_wallet.balance == 1000.0
        assert len(mock_context.paper_wallet.positions) == 0
    
    def test_position_limit_enforcement(self, mock_context):
        """Test that maximum open trades limit is enforced."""
        # Open maximum allowed positions with small amounts
        for i in range(5):
            mock_context.paper_wallet.open(f"TOKEN{i}/USDT", "buy", 0.01, 10.0)
        
        # Try to open one more position
        with pytest.raises(RuntimeError, match="Position limit reached"):
            mock_context.paper_wallet.open("EXTRA/USDT", "buy", 0.01, 10.0)
        
        # Check that only 5 positions exist
        assert len(mock_context.paper_wallet.positions) == 5
    
    def test_short_selling_disabled(self, mock_context):
        """Test that short selling can be disabled."""
        # Create wallet with short selling disabled
        mock_context.paper_wallet.allow_short = False
        
        # Try to open a sell position
        with pytest.raises(RuntimeError, match="Short selling disabled"):
            mock_context.paper_wallet.open("BTC/USDT", "sell", 0.1, 50.0)
        
        # Check that no position was opened
        assert len(mock_context.paper_wallet.positions) == 0
    
    def test_wallet_statistics_tracking(self, mock_context):
        """Test that wallet properly tracks trading statistics."""
        # Open and close a profitable trade
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "buy", 0.1, 50.0)
        mock_context.paper_wallet.close(trade_id, 0.1, 55.0)
        
        # Check statistics
        assert mock_context.paper_wallet.total_trades == 1
        assert mock_context.paper_wallet.winning_trades == 1
        assert mock_context.paper_wallet.win_rate == 100.0
        assert mock_context.paper_wallet.realized_pnl == 0.5
        
        # Open and close a losing trade
        trade_id2 = mock_context.paper_wallet.open("ETH/USDT", "buy", 0.1, 30.0)
        mock_context.paper_wallet.close(trade_id2, 0.1, 29.0)
        
        # Check updated statistics
        assert mock_context.paper_wallet.total_trades == 2
        assert mock_context.paper_wallet.winning_trades == 1
        assert mock_context.paper_wallet.win_rate == 50.0
        assert mock_context.paper_wallet.realized_pnl == 0.4  # 0.5 - 0.1
    
    def test_wallet_reset_functionality(self, mock_context):
        """Test that wallet can be reset to initial state."""
        # Open some positions and make some trades
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "buy", 0.1, 50.0)
        mock_context.paper_wallet.close(trade_id, 0.1, 55.0)
        
        # Reset wallet
        mock_context.paper_wallet.reset()
        
        # Check that wallet is back to initial state
        assert mock_context.paper_wallet.balance == 1000.0
        assert len(mock_context.paper_wallet.positions) == 0
        assert mock_context.paper_wallet.realized_pnl == 0.0
        assert mock_context.paper_wallet.total_trades == 0
        assert mock_context.paper_wallet.winning_trades == 0
    
    def test_position_summary_functionality(self, mock_context):
        """Test that position summary provides comprehensive information."""
        # Open a position
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "buy", 0.1, 50.0)
        
        # Get summary
        summary = mock_context.paper_wallet.get_position_summary()
        
        # Check summary structure
        assert "balance" in summary
        assert "initial_balance" in summary
        assert "realized_pnl" in summary
        assert "total_trades" in summary
        assert "winning_trades" in summary
        assert "win_rate" in summary
        assert "open_positions" in summary
        assert "positions" in summary
        
        # Check summary values
        assert summary["balance"] == 995.0
        assert summary["initial_balance"] == 1000.0
        assert summary["open_positions"] == 1
        assert trade_id in summary["positions"]
        
        position_info = summary["positions"][trade_id]
        assert position_info["symbol"] == "BTC/USDT"
        assert position_info["side"] == "buy"
        assert position_info["size"] == 0.1
        assert position_info["entry_price"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

