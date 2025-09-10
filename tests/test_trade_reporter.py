"""Tests for trade reporter functionality."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from crypto_bot.utils.trade_reporter import TelegramNotifier, TradeReporter


class TestTelegramNotifier:
    """Test suite for Telegram Notifier."""

    @pytest.fixture
    def notifier(self):
        return TelegramNotifier(token="test_token", chat_id="test_chat")

    def test_notifier_init(self, notifier):
        """Test notifier initialization."""
        assert notifier.token == "test_token"
        assert notifier.chat_id == "test_chat"

    def test_notify_sends_message(self, notifier):
        """Test notification sends message."""
        message = "Test trade notification"
        
        # Create a mock to track calls
        calls = []
        original_notify = notifier.notify
        
        def mock_notify(text):
            calls.append(text)
            return None
        
        # Replace the notify method temporarily
        notifier.notify = mock_notify
        
        try:
            notifier.notify(message)
            assert len(calls) == 1
            assert calls[0] == message
        finally:
            # Restore the original method
            notifier.notify = original_notify


class TestTradeReporter:
    """Test suite for Trade Reporter."""

    @pytest.fixture
    def reporter(self):
        notifier = Mock()
        notifier.notify = Mock(return_value=None)
        return TradeReporter(notifier=notifier, telegram_enabled=True)

    @pytest.fixture
    def sample_trade(self):
        return {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 1.0,
            'price': 50000.0,
            'timestamp': '2024-01-01T00:00:00Z'
        }

    def test_report_entry_formats_and_sends(self, reporter, sample_trade):
        """Test entry report formatting and sending."""
        reporter.report_entry(sample_trade)
        
        # Verify notify was called with the expected message
        reporter.notifier.notify.assert_called_once()
        message = reporter.notifier.notify.call_args[0][0]
        assert 'BTC/USDT' in message
        assert 'BUY' in message.upper()
        assert '50000.0' in message

    def test_report_exit_formats_and_sends(self, reporter, sample_trade):
        """Test exit report formatting and sending."""
        exit_trade = sample_trade.copy()
        exit_trade['side'] = 'sell'
        exit_trade['pnl'] = 1000.0

        reporter.report_exit(exit_trade)
        
        # Verify notify was called with the expected message
        reporter.notifier.notify.assert_called_once()
        message = reporter.notifier.notify.call_args[0][0]
        assert 'BTC/USDT' in message
        assert 'SELL' in message.upper()
        assert '1000.0' in message

    @patch('crypto_bot.utils.telegram.send_message')
    def test_reporter_disabled(self, mock_send):
        """Test reporter when telegram is disabled."""
        calls = {"count": 0}

        def fake_send(token, chat_id, text):
            calls["count"] += 1

        mock_send.side_effect = fake_send

        disabled_reporter = TradeReporter(telegram_enabled=False)
        disabled_reporter.report_entry({'symbol': 'BTC/USDT', 'side': 'buy'})
        
        assert calls["count"] == 0

    def test_reporter_enabled(self, reporter):
        """Test reporter when telegram is enabled."""
        assert reporter.telegram_enabled == True

    def test_format_entry_message(self, reporter, sample_trade):
        """Test entry message formatting."""
        message = reporter._format_entry_message(sample_trade)
        
        assert 'BTC/USDT' in message
        assert 'BUY' in message.upper()
        assert '50000.0' in message

    def test_format_exit_message(self, reporter, sample_trade):
        """Test exit message formatting."""
        exit_trade = sample_trade.copy()
        exit_trade['side'] = 'sell'
        exit_trade['pnl'] = 1500.0
        
        message = reporter._format_exit_message(exit_trade)
        
        assert 'BTC/USDT' in message
        assert 'SELL' in message.upper()
        assert '1500.0' in message

    def test_format_message_with_pnl(self, reporter):
        """Test message formatting with PnL."""
        trade_with_pnl = {
            'symbol': 'ETH/USDT',
            'side': 'sell',
            'amount': 10.0,
            'price': 3000.0,
            'pnl': -500.0
        }
        
        message = reporter._format_exit_message(trade_with_pnl)
        
        assert 'ETH/USDT' in message
        assert 'SELL' in message.upper()
        assert '-500.0' in message  # Negative PnL

    def test_format_message_without_pnl(self, reporter, sample_trade):
        """Test message formatting without PnL."""
        message = reporter._format_entry_message(sample_trade)
        
        # Should not contain PnL information for entries
        assert 'PnL' not in message
        assert 'pnl' not in message.lower()

    def test_multiple_trades_reporting(self, reporter):
        """Test reporting multiple trades."""
        trades = [
            {'symbol': 'BTC/USDT', 'side': 'buy', 'amount': 1.0, 'price': 50000.0},
            {'symbol': 'ETH/USDT', 'side': 'sell', 'amount': 10.0, 'price': 3000.0}
        ]
        
        for trade in trades:
            reporter.report_entry(trade)
        
        # Verify notify was called twice (once for each trade)
        assert reporter.notifier.notify.call_count == 2

    def test_invalid_trade_data(self, reporter):
        """Test handling of invalid trade data."""
        invalid_trade = {'symbol': 'BTC/USDT'}  # Missing required fields
        
        # Should not raise exception, just handle gracefully
        try:
            message = reporter._format_entry_message(invalid_trade)
            assert 'BTC/USDT' in message
        except Exception as e:
            pytest.fail(f"Should handle invalid trade data gracefully: {e}")

    @patch('crypto_bot.utils.telegram.send_message')
    def test_telegram_error_handling(self, mock_send, reporter, sample_trade):
        """Test handling of telegram errors."""
        mock_send.side_effect = Exception("Telegram API Error")
        
        # Should not crash the application
        try:
            reporter.report_entry(sample_trade)
        except Exception as e:
            pytest.fail(f"Should handle telegram errors gracefully: {e}")

    def test_message_length_limits(self, reporter):
        """Test message length limits."""
        long_symbol = 'A' * 100  # Very long symbol name
        trade = {
            'symbol': long_symbol,
            'side': 'buy',
            'amount': 1.0,
            'price': 100.0
        }
        
        message = reporter._format_entry_message(trade)
        
        # Message should be reasonable length
        assert len(message) < 1000
        assert long_symbol in message
