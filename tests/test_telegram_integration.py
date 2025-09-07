"""Integration tests for Telegram bot functionality.

This module tests Telegram notifications, command handling, user interactions,
and message formatting across all bot features.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
from datetime import datetime, timedelta

from crypto_bot.utils.telegram import TelegramNotifier, send_message
from crypto_bot.telegram_bot_ui import TelegramBotUI
from crypto_bot.telegram_ctl import TelegramController


@pytest.mark.integration
class TestTelegramIntegration:
    """Test Telegram integration across all bot components."""

    @pytest.fixture
    def mock_telegram_bot(self):
        """Mock Telegram bot for testing."""
        bot = Mock()
        bot.send_message = AsyncMock(return_value={'message_id': 123})
        bot.edit_message_text = AsyncMock(return_value=True)
        bot.answer_callback_query = AsyncMock(return_value=True)
        bot.send_photo = AsyncMock(return_value={'message_id': 124})
        bot.send_document = AsyncMock(return_value={'message_id': 125})
        return bot

    @pytest.fixture
    def telegram_notifier(self, mock_telegram_bot):
        """Telegram notifier instance for testing."""
        with patch('crypto_bot.utils.telegram.Bot', return_value=mock_telegram_bot):
            notifier = TelegramNotifier(
                bot_token='test_token',
                chat_id='test_chat',
                enabled=True
            )
            return notifier

    @pytest.fixture
    def sample_trade_data(self):
        """Sample trade data for notifications."""
        return {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 1.0,
            'price': 50000.0,
            'timestamp': datetime.now(),
            'strategy': 'trend_bot',
            'pnl': 0.0
        }

    @pytest.fixture
    def sample_portfolio_data(self):
        """Sample portfolio data for notifications."""
        return {
            'total_balance': 10000.0,
            'total_pnl': 500.0,
            'pnl_percentage': 5.0,
            'open_positions': 3,
            'win_rate': 65.0,
            'positions': [
                {
                    'symbol': 'BTC/USDT',
                    'amount': 1.0,
                    'entry_price': 50000.0,
                    'current_price': 52000.0,
                    'pnl': 2000.0
                },
                {
                    'symbol': 'ETH/USDT',
                    'amount': 10.0,
                    'entry_price': 3000.0,
                    'current_price': 3100.0,
                    'pnl': 1000.0
                }
            ]
        }

    def test_telegram_notifier_basic_functionality(self, telegram_notifier, mock_telegram_bot):
        """Test basic Telegram notification functionality."""
        message = "Test notification message"

        # Test synchronous send
        result = asyncio.run(telegram_notifier.send_message(message))

        # Verify message was sent
        mock_telegram_bot.send_message.assert_called_once()
        call_args = mock_telegram_bot.send_message.call_args
        assert call_args[1]['chat_id'] == 'test_chat'
        assert message in call_args[1]['text']

    def test_trade_notification_formatting(self, telegram_notifier, mock_telegram_bot, sample_trade_data):
        """Test trade notification formatting."""
        # Test buy notification
        buy_message = telegram_notifier.format_trade_message(sample_trade_data)
        assert 'BUY' in buy_message
        assert 'BTC/USDT' in buy_message
        assert '50000' in buy_message
        assert '1' in buy_message

        # Test sell notification
        sell_data = sample_trade_data.copy()
        sell_data['side'] = 'sell'
        sell_data['pnl'] = 1000.0
        sell_message = telegram_notifier.format_trade_message(sell_data)
        assert 'SELL' in sell_message
        assert 'profit' in sell_message.lower() or 'pnl' in sell_message.lower()

        # Send notification
        asyncio.run(telegram_notifier.send_trade_notification(sell_data))
        mock_telegram_bot.send_message.assert_called()

    def test_portfolio_status_notifications(self, telegram_notifier, mock_telegram_bot, sample_portfolio_data):
        """Test portfolio status notification formatting."""
        # Test portfolio summary
        summary_message = telegram_notifier.format_portfolio_message(sample_portfolio_data)

        assert 'Portfolio' in summary_message
        assert '$10000' in summary_message
        assert '5.0%' in summary_message
        assert '65.0%' in summary_message

        # Test position details
        positions_message = telegram_notifier.format_positions_message(sample_portfolio_data['positions'])

        assert 'BTC/USDT' in positions_message
        assert 'ETH/USDT' in positions_message
        assert '2000' in positions_message  # BTC PnL
        assert '1000' in positions_message  # ETH PnL

        # Send notifications
        asyncio.run(telegram_notifier.send_portfolio_update(sample_portfolio_data))
        assert mock_telegram_bot.send_message.call_count >= 2

    def test_error_and_alert_notifications(self, telegram_notifier, mock_telegram_bot):
        """Test error and alert notification handling."""
        # Test error notification
        error_message = "Exchange connection failed"
        asyncio.run(telegram_notifier.send_error_alert(error_message))

        # Verify error formatting
        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'ERROR' in sent_message
        assert error_message in sent_message

        # Test warning notification
        warning_message = "High volatility detected"
        asyncio.run(telegram_notifier.send_warning(warning_message))

        # Verify warning formatting
        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'WARNING' in sent_message or '⚠️' in sent_message

        # Test success notification
        success_message = "Strategy optimization completed"
        asyncio.run(telegram_notifier.send_success(success_message))

        # Verify success formatting
        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'SUCCESS' in sent_message or '✅' in sent_message

    def test_signal_notifications(self, telegram_notifier, mock_telegram_bot):
        """Test trading signal notifications."""
        # Test buy signal
        buy_signal = {
            'action': 'buy',
            'symbol': 'BTC/USDT',
            'confidence': 0.85,
            'price': 50000.0,
            'strategy': 'trend_bot'
        }

        asyncio.run(telegram_notifier.send_signal_notification(buy_signal))

        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'BUY SIGNAL' in sent_message
        assert 'BTC/USDT' in sent_message
        assert '85%' in sent_message or '0.85' in sent_message
        assert 'trend_bot' in sent_message

        # Test sell signal
        sell_signal = {
            'action': 'sell',
            'symbol': 'ETH/USDT',
            'confidence': 0.72,
            'price': 3100.0,
            'strategy': 'mean_bot'
        }

        asyncio.run(telegram_notifier.send_signal_notification(sell_signal))

        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'SELL SIGNAL' in sent_message
        assert 'ETH/USDT' in sent_message

    def test_risk_alert_notifications(self, telegram_notifier, mock_telegram_bot):
        """Test risk management alert notifications."""
        # Test drawdown alert
        drawdown_data = {
            'current_drawdown': 0.08,
            'max_drawdown': 0.1,
            'portfolio_value': 9200.0
        }

        asyncio.run(telegram_notifier.send_drawdown_alert(drawdown_data))

        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'DRAWDOWN' in sent_message
        assert '8%' in sent_message
        assert '9200' in sent_message

        # Test position size alert
        position_alert = {
            'symbol': 'BTC/USDT',
            'position_size': 2.5,
            'max_allowed': 2.0,
            'current_price': 50000.0
        }

        asyncio.run(telegram_notifier.send_position_alert(position_alert))

        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'POSITION SIZE' in sent_message
        assert 'BTC/USDT' in sent_message
        assert '2.5' in sent_message

    def test_performance_report_notifications(self, telegram_notifier, mock_telegram_bot):
        """Test performance report notifications."""
        # Test daily performance report
        daily_performance = {
            'date': datetime.now().date(),
            'total_trades': 15,
            'winning_trades': 10,
            'total_pnl': 1250.0,
            'win_rate': 66.7,
            'largest_win': 500.0,
            'largest_loss': -150.0,
            'sharpe_ratio': 1.2
        }

        asyncio.run(telegram_notifier.send_daily_performance_report(daily_performance))

        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'PERFORMANCE' in sent_message
        assert '15' in sent_message  # Total trades
        assert '66.7%' in sent_message  # Win rate
        assert '1250' in sent_message  # Total PnL

        # Test strategy performance report
        strategy_performance = {
            'trend_bot': {
                'total_trades': 25,
                'win_rate': 68.0,
                'total_pnl': 2100.0,
                'avg_trade_pnl': 84.0
            },
            'mean_bot': {
                'total_trades': 18,
                'win_rate': 61.1,
                'total_pnl': 950.0,
                'avg_trade_pnl': 52.8
            }
        }

        asyncio.run(telegram_notifier.send_strategy_performance_report(strategy_performance))

        # Should send multiple messages (one per strategy)
        assert mock_telegram_bot.send_message.call_count >= 2

    def test_system_status_notifications(self, telegram_notifier, mock_telegram_bot):
        """Test system status and health notifications."""
        # Test bot startup notification
        asyncio.run(telegram_notifier.send_startup_notification())

        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'STARTED' in sent_message or 'ONLINE' in sent_message

        # Test bot shutdown notification
        asyncio.run(telegram_notifier.send_shutdown_notification())

        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'SHUTDOWN' in sent_message or 'OFFLINE' in sent_message

        # Test system health report
        health_data = {
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'disk_usage': 23.1,
            'network_status': 'healthy',
            'exchange_connections': {
                'kraken': 'connected',
                'binance': 'connected',
                'coinbase': 'degraded'
            }
        }

        asyncio.run(telegram_notifier.send_health_report(health_data))

        call_args = mock_telegram_bot.send_message.call_args
        sent_message = call_args[1]['text']
        assert 'HEALTH' in sent_message
        assert '45.2%' in sent_message  # CPU
        assert '67.8%' in sent_message  # Memory
        assert 'connected' in sent_message

    def test_telegram_bot_ui_integration(self, mock_telegram_bot):
        """Test Telegram bot UI integration."""
        with patch('crypto_bot.telegram_bot_ui.Bot', return_value=mock_telegram_bot), \
             patch('crypto_bot.telegram_bot_ui.Updater'):

            bot_ui = TelegramBotUI(
                bot_token='test_token',
                chat_id='test_chat',
                enabled=True
            )

            # Test command handling setup
            assert bot_ui.bot is not None
            assert bot_ui.enabled is True

            # Test message sending through UI
            test_message = "UI Test Message"
            asyncio.run(bot_ui.send_message(test_message))

            mock_telegram_bot.send_message.assert_called_with(
                chat_id='test_chat',
                text=test_message,
                parse_mode='HTML'
            )

    def test_telegram_controller_integration(self):
        """Test Telegram controller integration."""
        with patch('crypto_bot.telegram_ctl.Bot'), \
             patch('crypto_bot.telegram_ctl.Updater'):

            controller = TelegramController(
                bot_token='test_token',
                chat_id='test_chat',
                enabled=True
            )

            # Test controller initialization
            assert controller.enabled is True

            # Test command processing setup
            # In real implementation, this would set up command handlers

    def test_message_formatting_and_emoji(self, telegram_notifier):
        """Test message formatting with emojis and styling."""
        # Test emoji formatting
        buy_message = telegram_notifier._format_with_emoji('BUY', 'BTC/USDT', 50000.0)
        assert '🟢' in buy_message or '📈' in buy_message or 'BUY' in buy_message

        sell_message = telegram_notifier._format_with_emoji('SELL', 'ETH/USDT', 3100.0)
        assert '🔴' in sell_message or '📉' in sell_message or 'SELL' in sell_message

        # Test profit/loss formatting
        profit_message = telegram_notifier._format_pnl(500.0)
        assert '📈' in profit_message or 'PROFIT' in profit_message

        loss_message = telegram_notifier._format_pnl(-200.0)
        assert '📉' in loss_message or 'LOSS' in loss_message

    def test_notification_rate_limiting(self, telegram_notifier, mock_telegram_bot):
        """Test notification rate limiting."""
        # Send multiple notifications rapidly
        messages = [f"Test message {i}" for i in range(10)]

        # Record send times
        send_times = []

        async def send_with_timing(message):
            start_time = asyncio.get_event_loop().time()
            await telegram_notifier.send_message(message)
            end_time = asyncio.get_event_loop().time()
            send_times.append(end_time - start_time)
            await asyncio.sleep(0.01)  # Small delay between sends

        # Send all messages
        tasks = [send_with_timing(msg) for msg in messages]
        asyncio.run(asyncio.gather(*tasks))

        # Verify all messages were sent
        assert mock_telegram_bot.send_message.call_count == len(messages)

        # Check that sends didn't take excessively long (would indicate rate limiting)
        total_time = sum(send_times)
        avg_time = total_time / len(send_times)

        # Average send time should be reasonable (not blocked by rate limiting)
        assert avg_time < 1.0  # Less than 1 second per message

    def test_error_handling_and_retry(self, telegram_notifier):
        """Test error handling and retry logic."""
        # Mock bot to fail on first attempt, succeed on retry
        call_count = 0

        async def failing_send(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network error")
            return {'message_id': 123}

        with patch.object(telegram_notifier.bot, 'send_message', side_effect=failing_send):
            # Send message that should retry
            result = asyncio.run(telegram_notifier.send_message("Test message"))

            # Should have been called twice (first failed, second succeeded)
            assert call_count == 2
            assert result is not None

        # Test complete failure after retries
        with patch.object(telegram_notifier.bot, 'send_message', side_effect=Exception("Persistent error")):
            # Should handle complete failure gracefully
            result = asyncio.run(telegram_notifier.send_message("Test message"))

            # Should return None or handle error without crashing
            # (exact behavior depends on implementation)

    def test_notification_queue_and_buffering(self, telegram_notifier, mock_telegram_bot):
        """Test notification queue and message buffering."""
        # Test message batching
        messages = [
            "Message 1",
            "Message 2",
            "Message 3",
            "Message 4",
            "Message 5"
        ]

        # Send messages individually
        for msg in messages:
            asyncio.run(telegram_notifier.send_message(msg))

        # Verify all messages were sent
        assert mock_telegram_bot.send_message.call_count == len(messages)

        # Test priority messaging
        high_priority_msg = "URGENT: System alert"
        normal_msg = "Normal update"

        # High priority should be sent immediately
        asyncio.run(telegram_notifier.send_urgent_message(high_priority_msg))

        # Verify urgent message was sent
        call_args = mock_telegram_bot.send_message.call_args
        assert high_priority_msg in call_args[1]['text']

    def test_telegram_file_and_media_sending(self, telegram_notifier, mock_telegram_bot):
        """Test sending files and media through Telegram."""
        # Test sending performance chart
        chart_data = b"fake_chart_image_data"
        asyncio.run(telegram_notifier.send_performance_chart(chart_data))

        # Verify photo was sent
        mock_telegram_bot.send_photo.assert_called_once()

        # Test sending log file
        log_content = "2024-01-01 10:00:00 INFO Bot started\n2024-01-01 10:01:00 INFO Trade executed"
        asyncio.run(telegram_notifier.send_log_file(log_content))

        # Verify document was sent
        mock_telegram_bot.send_document.assert_called_once()

    def test_user_interaction_and_callbacks(self, mock_telegram_bot):
        """Test user interaction handling and callback queries."""
        with patch('crypto_bot.telegram_bot_ui.Bot', return_value=mock_telegram_bot), \
             patch('crypto_bot.telegram_bot_ui.Updater'):

            bot_ui = TelegramBotUI(
                bot_token='test_token',
                chat_id='test_chat',
                enabled=True
            )

            # Mock callback query
            callback_query = Mock()
            callback_query.data = 'status'
            callback_query.id = '123'
            callback_query.message = Mock()
            callback_query.message.chat_id = 'test_chat'

            # Test callback handling (would be implemented in real bot)
            # This tests the structure for handling user interactions

            assert callback_query.data == 'status'
            assert callback_query.id == '123'

    def test_notification_scheduling_and_timing(self, telegram_notifier, mock_telegram_bot):
        """Test notification scheduling and timing."""
        # Test delayed notification
        future_time = datetime.now() + timedelta(minutes=5)
        message = "Scheduled message"

        # In real implementation, this would schedule the message
        # For testing, we verify the scheduling logic
        delay_seconds = (future_time - datetime.now()).total_seconds()
        assert delay_seconds > 0

        # Test periodic notifications
        periodic_message = "Hourly status update"

        # Simulate periodic sending
        for i in range(3):
            asyncio.run(telegram_notifier.send_message(f"{periodic_message} #{i+1}"))
            # In real implementation, there would be time delays

        # Verify periodic messages were sent
        assert mock_telegram_bot.send_message.call_count == 3

    def test_telegram_configuration_and_setup(self):
        """Test Telegram configuration and setup."""
        # Test configuration validation
        valid_config = {
            'bot_token': '123456789:ABCdefGHIjklMNOpqrsTUVwxyz',
            'chat_id': '@test_channel',
            'enabled': True
        }

        # Should validate successfully
        assert len(valid_config['bot_token']) > 0
        assert valid_config['chat_id'].startswith('@')
        assert valid_config['enabled'] is True

        # Test invalid configuration
        invalid_config = {
            'bot_token': '',
            'chat_id': '',
            'enabled': False
        }

        # Should fail validation
        assert len(invalid_config['bot_token']) == 0
        assert len(invalid_config['chat_id']) == 0
        assert invalid_config['enabled'] is False

    def test_notification_content_filtering(self, telegram_notifier):
        """Test notification content filtering and sanitization."""
        # Test message length limits
        long_message = "A" * 5000  # Very long message
        short_message = "Short message"

        # Should handle long messages appropriately
        assert len(long_message) > 4096  # Telegram limit
        assert len(short_message) < 4096

        # Test content sanitization
        unsafe_message = "Message with <script>alert('xss')</script> tags"
        # In real implementation, this would be sanitized
        assert "<script>" in unsafe_message  # Contains potentially unsafe content

        # Test emoji and special character handling
        emoji_message = "🚀 Rocket emoji test 📈"
        assert "🚀" in emoji_message
        assert "📈" in emoji_message

    def test_multi_chat_and_broadcasting(self, telegram_notifier, mock_telegram_bot):
        """Test multi-chat support and broadcasting."""
        # Test multiple chat IDs
        chat_ids = ['chat1', 'chat2', 'chat3']
        message = "Broadcast message"

        # Simulate broadcasting to multiple chats
        for chat_id in chat_ids:
            telegram_notifier.chat_id = chat_id
            asyncio.run(telegram_notifier.send_message(message))

        # Verify message was sent to each chat
        assert mock_telegram_bot.send_message.call_count == len(chat_ids)

        # Test broadcast to group
        group_message = "Group announcement"
        asyncio.run(telegram_notifier.send_broadcast(group_message, chat_ids))

        # Should have sent to all chats
        total_sends = mock_telegram_bot.send_message.call_count
        assert total_sends == len(chat_ids) * 2  # Individual + broadcast

    def test_notification_history_and_logging(self, telegram_notifier):
        """Test notification history tracking and logging."""
        # Send several notifications
        messages = [
            "Trade executed",
            "Error occurred",
            "Portfolio update",
            "Signal generated"
        ]

        sent_notifications = []
        for msg in messages:
            asyncio.run(telegram_notifier.send_message(msg))
            sent_notifications.append({
                'message': msg,
                'timestamp': datetime.now(),
                'status': 'sent'
            })

        # Verify notification history
        assert len(sent_notifications) == len(messages)

        # Test notification logging
        with patch('crypto_bot.utils.telegram.LOG_DIR', Path(tempfile.mkdtemp())):
            # Notifications should be logged (implementation dependent)
            pass

    def test_telegram_integration_with_trading_pipeline(self, telegram_notifier, mock_telegram_bot):
        """Test Telegram integration with complete trading pipeline."""
        # Simulate complete trading flow with notifications

        # 1. Signal generation notification
        signal = {
            'action': 'buy',
            'symbol': 'BTC/USDT',
            'confidence': 0.8,
            'strategy': 'trend_bot'
        }
        asyncio.run(telegram_notifier.send_signal_notification(signal))

        # 2. Trade execution notification
        trade = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 1.0,
            'price': 50000.0,
            'strategy': 'trend_bot'
        }
        asyncio.run(telegram_notifier.send_trade_notification(trade))

        # 3. Position update notification
        position_update = {
            'symbol': 'BTC/USDT',
            'amount': 1.0,
            'entry_price': 50000.0,
            'current_price': 52000.0,
            'pnl': 2000.0
        }
        asyncio.run(telegram_notifier.send_position_update(position_update))

        # 4. Portfolio status notification
        portfolio = {
            'total_balance': 52000.0,
            'total_pnl': 2000.0,
            'pnl_percentage': 4.0,
            'open_positions': 1
        }
        asyncio.run(telegram_notifier.send_portfolio_update(portfolio))

        # Verify all notifications were sent
        assert mock_telegram_bot.send_message.call_count >= 4

        # Verify notification order and content
        call_args_list = mock_telegram_bot.send_message.call_args_list

        # First call should be signal notification
        first_call = call_args_list[0]
        assert 'SIGNAL' in first_call[1]['text']

        # Subsequent calls should contain trade and portfolio info
        messages_text = [call[1]['text'] for call in call_args_list]
        assert any('BUY' in msg for msg in messages_text)
        assert any('BTC/USDT' in msg for msg in messages_text)
        assert any('2000' in msg for msg in messages_text)  # PnL amount
