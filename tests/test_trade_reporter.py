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

    @patch('crypto_bot.utils.telegram.send_message')
    def test_notify_sends_message(self, mock_send, notifier):
        """Test notification sends message."""
        message = "Test trade notification"
        notifier.notify(message)
        mock_send.assert_called_once_with(notifier.token, notifier.chat_id, message)


class TestTradeReporter:
    """Test suite for Trade Reporter."""

    @pytest.fixture
    def reporter(self):
        return TradeReporter(telegram_enabled=True)

    @pytest.fixture
    def sample_trade(self):
        return {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 1.0,
            'price': 50000.0,
            'timestamp': '2024-01-01T00:00:00Z'
        }

    @patch('crypto_bot.utils.telegram.send_message')
    def test_report_entry_formats_and_sends(self, mock_send, reporter, sample_trade):
        """Test entry report formatting and sending."""
        calls = {}

        def fake_send(token, chat_id, text):
            calls['token'] = token
            calls['chat_id'] = chat_id
            calls['text'] = text

        mock_send.side_effect = fake_send

        reporter.report_entry(sample_trade)
        
        assert 'BTC/USDT' in calls['text']
        assert 'BUY' in calls['text'].upper()
        assert '50000.0' in calls['text']

    @patch('crypto_bot.utils.telegram.send_message')
    def test_report_exit_formats_and_sends(self, mock_send, reporter, sample_trade):
        """Test exit report formatting and sending."""
        calls = {}

        def fake_send(token, chat_id, text):
            calls['token'] = token
            calls['chat_id'] = chat_id
            calls['text'] = text

        mock_send.side_effect = fake_send

        exit_trade = sample_trade.copy()
        exit_trade['side'] = 'sell'
        exit_trade['pnl'] = 1000.0

        reporter.report_exit(exit_trade)
        
        assert 'BTC/USDT' in calls['text']
        assert 'SELL' in calls['text'].upper()
        assert '1000.0' in calls['text']

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

    @patch('crypto_bot.utils.telegram.send_message')
    def test_multiple_trades_reporting(self, mock_send, reporter):
        """Test reporting multiple trades."""
        trades = [
            {'symbol': 'BTC/USDT', 'side': 'buy', 'amount': 1.0, 'price': 50000.0},
            {'symbol': 'ETH/USDT', 'side': 'sell', 'amount': 10.0, 'price': 3000.0}
        ]
        
        for trade in trades:
            reporter.report_entry(trade)
        
        assert mock_send.call_count == 2

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
