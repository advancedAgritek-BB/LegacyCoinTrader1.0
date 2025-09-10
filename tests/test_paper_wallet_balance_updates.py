import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.phase_runner import BotContext


class TestPaperWalletBalanceUpdates:
    """Test that paper wallet balances are properly updated with each transaction."""
    
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
    
    def test_initial_balance_setup(self, mock_context):
        """Test that initial balance is properly set up."""
        assert mock_context.balance == 1000.0
        assert mock_context.paper_wallet.balance == 1000.0
        assert mock_context.balance == mock_context.paper_wallet.balance
    
    def test_buy_order_balance_deduction(self, mock_context):
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
    
    def test_sell_order_balance_reservation(self, mock_context):
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
    
    def test_position_closure_balance_update(self, mock_context):
        """Test that position closure properly updates wallet balance."""
        # Open a buy position
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "buy", 0.1, 50.0)
        initial_balance = mock_context.paper_wallet.balance  # Should be 995.0
        
        # Close the position at a profit (0.1 BTC at $60 = $6 proceeds)
        pnl = mock_context.paper_wallet.close(trade_id, 0.1, 60.0)
        
        # Check PnL calculation
        expected_pnl = (60.0 - 50.0) * 0.1  # $1.0 profit
        assert pnl == expected_pnl
        
        # Check that balance was updated with proceeds
        expected_balance = initial_balance + (0.1 * 60.0)  # Add sale proceeds
        assert mock_context.paper_wallet.balance == expected_balance
        assert mock_context.paper_wallet.balance == 1001.0  # 995 + 6 = 1001
    
    def test_multiple_positions_balance_tracking(self, mock_context):
        """Test that multiple positions are properly tracked in balance."""
        # Open multiple positions
        trade_id1 = mock_context.paper_wallet.open("BTC/USDT", "buy", 0.1, 50.0)
        trade_id2 = mock_context.paper_wallet.open("ETH/USDT", "buy", 1.0, 30.0)
        
        # Check total balance deduction
        expected_balance = 1000.0 - (0.1 * 50.0) - (1.0 * 30.0)
        assert mock_context.paper_wallet.balance == expected_balance
        assert mock_context.paper_wallet.balance == 965.0
        
        # Close one position
        pnl = mock_context.paper_wallet.close(trade_id1, 0.1, 55.0)
        assert pnl == (55.0 - 50.0) * 0.1  # $0.5 profit
        
        # Check updated balance
        expected_balance = 965.0 + (0.1 * 55.0)  # Add sale proceeds
        assert mock_context.paper_wallet.balance == expected_balance
        assert mock_context.paper_wallet.balance == 970.5
    
    def test_short_position_balance_management(self, mock_context):
        """Test that short positions properly manage reserved funds."""
        # Open a short position
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "sell", 0.1, 50.0)
        initial_balance = mock_context.paper_wallet.balance  # Should be 995.0
        
        # Check that funds are reserved
        position = mock_context.paper_wallet.positions[trade_id]
        assert position["reserved"] == 5.0  # 0.1 * 50
        
        # Close short position at a profit (price dropped to $40)
        pnl = mock_context.paper_wallet.close(trade_id, 0.1, 40.0)
        
        # Check PnL calculation for short
        expected_pnl = (50.0 - 40.0) * 0.1  # $1.0 profit (sold high, bought low)
        assert pnl == expected_pnl
        
        # Check that reserved funds were released and profit added
        expected_balance = initial_balance + 5.0 + expected_pnl  # Reserved + profit
        assert mock_context.paper_wallet.balance == expected_balance
        assert mock_context.paper_wallet.balance == 1001.0  # 995 + 5 + 1
    
    def test_partial_position_closure(self, mock_context):
        """Test that partial position closures properly update balances."""
        # Open a larger position
        trade_id = mock_context.paper_wallet.open("BTC/USDT", "buy", 1.0, 50.0)
        initial_balance = mock_context.paper_wallet.balance  # Should be 950.0
        
        # Partially close the position
        pnl = mock_context.paper_wallet.close(trade_id, 0.5, 60.0)
        
        # Check PnL for partial closure
        expected_pnl = (60.0 - 50.0) * 0.5  # $5.0 profit on 0.5 BTC
        assert pnl == expected_pnl
        
        # Check that balance was updated
        expected_balance = initial_balance + (0.5 * 60.0)  # Add sale proceeds
        assert mock_context.paper_wallet.balance == expected_balance
        assert mock_context.paper_wallet.balance == 980.0  # 950 + 30
        
        # Check that position size was reduced
        position = mock_context.paper_wallet.positions[trade_id]
        assert position["size"] == 0.5  # Remaining position size
    
    def test_balance_consistency_after_operations(self, mock_context):
        """Test that balance remains consistent after multiple operations."""
        # Perform a series of operations
        trade_id1 = mock_context.paper_wallet.open("BTC/USDT", "buy", 0.1, 50.0)
        trade_id2 = mock_context.paper_wallet.open("ETH/USDT", "sell", 1.0, 30.0)
        
        # Close first position
        mock_context.paper_wallet.close(trade_id1, 0.1, 55.0)
        
        # Check final balance
        final_balance = mock_context.paper_wallet.balance
        
        # Verify balance calculation
        # Initial: 1000
        # Buy BTC: -5.0 (0.1 * 50)
        # Sell ETH: -30.0 (1.0 * 30) 
        # Close BTC: +5.5 (0.1 * 55)
        # Expected: 1000 - 5 - 30 + 5.5 = 970.5
        expected_balance = 1000.0 - 5.0 - 30.0 + 5.5
        assert final_balance == expected_balance
        assert final_balance == 970.5
        
        # Verify position count
        assert len(mock_context.paper_wallet.positions) == 1  # Only ETH position remains
    
    def test_error_handling_insufficient_balance(self, mock_context):
        """Test that insufficient balance errors are properly handled."""
        # Try to open a position larger than available balance
        with pytest.raises(RuntimeError, match="Insufficient balance"):
            mock_context.paper_wallet.open("BTC/USDT", "buy", 100.0, 50.0)  # $5000 cost, only $1000 available
        
        # Verify balance wasn't changed
        assert mock_context.paper_wallet.balance == 1000.0
        assert len(mock_context.paper_wallet.positions) == 0
    
    def test_win_rate_tracking(self, mock_context):
        """Test that win rate is properly tracked across trades."""
        # Open and close a winning trade
        trade_id1 = mock_context.paper_wallet.open("BTC/USDT", "buy", 0.1, 50.0)
        mock_context.paper_wallet.close(trade_id1, 0.1, 60.0)  # $1 profit
        
        # Open and close a losing trade
        trade_id2 = mock_context.paper_wallet.open("ETH/USDT", "buy", 1.0, 30.0)
        mock_context.paper_wallet.close(trade_id2, 1.0, 25.0)  # $5 loss
        
        # Check statistics
        assert mock_context.paper_wallet.total_trades == 2
        assert mock_context.paper_wallet.winning_trades == 1
        assert mock_context.paper_wallet.win_rate == 50.0  # 1 out of 2 trades won
        assert mock_context.paper_wallet.realized_pnl == -4.0  # $1 profit - $5 loss
