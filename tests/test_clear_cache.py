"""
Test the clear paper trading cache functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock

from crypto_bot.utils.telegram import clear_paper_trading_cache
from crypto_bot.paper_wallet import PaperWallet


class TestClearPaperTradingCache:
    """Test the clear_paper_trading_cache function."""
    
    def test_clear_paper_wallet_only(self):
        """Test clearing only the paper wallet."""
        # Create a paper wallet with some positions
        wallet = PaperWallet(1000.0)
        wallet.open("BTC/USDT", "buy", 0.01, 50000.0)  # $500 cost
        wallet.open("ETH/USDT", "buy", 0.1, 3000.0)    # $300 cost
        
        # Verify initial state
        assert len(wallet.positions) == 2
        assert wallet.balance < 1000.0
        assert wallet.total_trades == 2
        
        # Clear cache
        result = clear_paper_trading_cache(paper_wallet=wallet)
        
        # Verify wallet was reset
        assert len(wallet.positions) == 0
        assert wallet.balance == 1000.0
        assert wallet.total_trades == 0
        assert "Paper wallet reset to $1000.00" in result
        
    def test_clear_context_only(self):
        """Test clearing only the bot context."""
        # Create mock context
        context = Mock()
        context.positions = {"BTC/USDT": {"side": "buy", "size": 0.1}}
        context.df_cache = {"1h": {"BTC/USDT": "data"}}
        context.regime_cache = {"BTC/USDT": "regime_data"}
        
        # Clear cache
        result = clear_paper_trading_cache(context=context)
        
        # Verify context was cleared
        assert len(context.positions) == 0
        assert len(context.df_cache) == 0
        assert len(context.regime_cache) == 0
        assert "Cleared 1 open positions" in result
        assert "Cleared 1 cached data entries" in result
        assert "Cleared 1 regime cache entries" in result
        
    def test_clear_both_wallet_and_context(self):
        """Test clearing both paper wallet and context."""
        # Create wallet and context
        wallet = PaperWallet(1000.0)
        wallet.open("BTC/USDT", "buy", 0.01, 50000.0)  # $500 cost
        
        context = Mock()
        context.positions = {"BTC/USDT": {"side": "buy", "size": 0.1}}
        context.df_cache = {"1h": {"BTC/USDT": "data"}}
        context.regime_cache = {"BTC/USDT": "regime_data"}
        
        # Clear both
        result = clear_paper_trading_cache(paper_wallet=wallet, context=context)
        
        # Verify both were cleared
        assert len(wallet.positions) == 0
        assert wallet.balance == 1000.0
        assert len(context.positions) == 0
        assert len(context.df_cache) == 0
        assert len(context.regime_cache) == 0
        
        # Check result contains both clear messages
        assert "Paper wallet reset to $1000.00" in result
        assert "Cleared 1 open positions" in result
        
    def test_clear_nothing_provided(self):
        """Test clearing when no wallet or context provided."""
        result = clear_paper_trading_cache()
        
        assert "No cache items available to clear" in result
        
    def test_clear_empty_wallet_and_context(self):
        """Test clearing when wallet and context are already empty."""
        wallet = PaperWallet(1000.0)
        context = Mock()
        context.positions = {}
        context.df_cache = {}
        context.regime_cache = {}
        
        result = clear_paper_trading_cache(paper_wallet=wallet, context=context)
        
        # Should still show success but indicate nothing to clear
        assert "Paper wallet reset to $1000.00" in result
        assert "No open positions to clear" in result
        assert "No cached data to clear" in result
        assert "No regime cache to clear" in result
        
    def test_wallet_reset_error_handling(self):
        """Test error handling when wallet reset fails."""
        wallet = Mock()
        wallet.initial_balance = 1000.0
        wallet.reset.side_effect = Exception("Reset failed")
        
        result = clear_paper_trading_cache(paper_wallet=wallet)
        
        assert "Failed to reset paper wallet: Reset failed" in result
        
    def test_context_clear_error_handling(self):
        """Test error handling when context clearing fails."""
        context = Mock()
        # Create a mock positions dict that can have side_effect and proper len
        mock_positions = Mock()
        mock_positions.clear.side_effect = Exception("Clear failed")
        mock_positions.__len__ = Mock(return_value=1)  # Mock the len() method
        context.positions = mock_positions
        context.df_cache = {"1h": {"BTC/USDT": "data"}}
        context.regime_cache = {"BTC/USDT": "regime_data"}
        
        result = clear_paper_trading_cache(context=context)
        
        assert "Failed to clear context positions: Clear failed" in result
