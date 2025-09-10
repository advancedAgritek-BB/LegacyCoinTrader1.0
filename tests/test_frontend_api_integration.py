"""Integration tests for frontend API endpoints.

This module tests all API endpoints and their interactions to ensure
the frontend can properly control and monitor the trading bot.
"""

import pytest
import json
import tempfile
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import os
import threading
import time

# Import the Flask app
from frontend.app import app
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.utils.telegram import TelegramNotifier


@pytest.mark.integration
class TestFrontendAPIIntegration:
    """Test frontend API endpoints integration."""

    @pytest.fixture
    def client(self):
        """Flask test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    @pytest.fixture
    def mock_bot_process(self):
        """Mock bot process for testing."""
        process = Mock()
        process.poll = Mock(return_value=None)
        process.terminate = Mock()
        process.wait = Mock()
        process.pid = 12345
        return process

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'trading': {
                'enabled': True,
                'max_positions': 5,
                'risk_per_trade': 0.02
            },
            'solana': {
                'enabled': True,
                'rpc_url': 'https://api.mainnet-beta.solana.com'
            },
            'telegram': {
                'enabled': True,
                'bot_token': 'test_token',
                'chat_id': 'test_chat'
            },
            'enhanced_scanning': {
                'enabled': True,
                'scan_interval': 30
            }
        }

    @pytest.fixture
    def sample_trades_data(self):
        """Sample trades data for testing."""
        return [
            {
                'timestamp': '2024-01-01T10:00:00',
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'amount': 1.0,
                'price': 100.0,
                'fee': 0.1
            },
            {
                'timestamp': '2024-01-01T11:00:00',
                'symbol': 'ETH/USDT',
                'side': 'sell',
                'amount': 2.0,
                'price': 200.0,
                'fee': 0.2
            }
        ]

    @pytest.fixture
    def sample_positions_data(self):
        """Sample positions data for testing."""
        return {
            'BTC/USDT': {
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'amount': 1.0,
                'entry_price': 100.0,
                'current_price': 105.0,
                'unrealized_pnl': 5.0
            },
            'ETH/USDT': {
                'symbol': 'ETH/USDT',
                'side': 'sell',
                'amount': 2.0,
                'entry_price': 200.0,
                'current_price': 195.0,
                'unrealized_pnl': 10.0
            }
        }

    def test_bot_control_integration(self, client, mock_bot_process):
        """Test bot start/stop/pause/resume integration."""
        with patch('frontend.app.subprocess.Popen') as mock_popen, \
             patch('frontend.app.is_running', return_value=False), \
             patch('frontend.app.stop_conflicting_bots'):

            mock_popen.return_value = mock_bot_process

            # Test start bot
            response = client.post('/start_bot', json={'mode': 'paper'})
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'message' in data
            assert 'started' in data['message'].lower()

            # Verify bot process was created
            mock_popen.assert_called_once()

            # Test stop bot
            with patch('frontend.app.is_running', return_value=True), \
                 patch('frontend.app.bot_proc', mock_bot_process):

                response = client.post('/stop_bot')
                assert response.status_code == 200
                data = json.loads(response.data)
                assert 'message' in data
                assert 'stopped' in data['message'].lower()

    def test_bot_status_monitoring(self, client):
        """Test bot status monitoring endpoints."""
        with patch('frontend.app.is_running', return_value=True), \
             patch('frontend.app.load_execution_mode', return_value='paper'), \
             patch('frontend.app.get_uptime', return_value='01:30:45'):

            # Test bot status endpoint
            response = client.get('/api/bot-status')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'success' in data
            assert 'data' in data
            assert 'bot_running' in data['data']
            assert 'timestamp' in data['data']
            assert data['data']['bot_running'] is True

    def test_position_management_api(self, client, sample_positions_data):
        """Test position management API endpoints."""
        with patch('frontend.app.load_positions', return_value=sample_positions_data):

            # Test get positions
            response = client.get('/api/open-positions')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert isinstance(data, list)
            assert len(data) == 2

            # Verify position data structure
            position = data[0]
            assert 'symbol' in position
            assert 'side' in position
            assert 'amount' in position
            assert 'entry_price' in position
            assert 'current_price' in position
            assert 'unrealized_pnl' in position

    def test_trade_history_api(self, client, sample_trades_data):
        """Test trade history API endpoints."""
        with patch('frontend.app.load_trades', return_value=sample_trades_data):

            # Test get trades
            response = client.get('/trades_data')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert isinstance(data, list)
            assert len(data) == 2

            # Verify trade data structure
            trade = data[0]
            assert 'timestamp' in trade
            assert 'symbol' in trade
            assert 'side' in trade
            assert 'amount' in trade
            assert 'price' in trade

            # Test trades tail endpoint
            response = client.get('/trades_tail')
            assert response.status_code == 200

    def test_manual_position_management(self, client):
        """Test manual position management via API."""
        sell_data = {
            'symbol': 'BTC/USDT',
            'amount': 1.0,
            'price': 105.0
        }

        with patch('frontend.app.manual_sell_position', return_value=True) as mock_sell:

            # Test manual sell
            response = client.post('/api/sell-position',
                                 json=sell_data,
                                 content_type='application/json')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'message' in data

            # Verify sell function was called with correct parameters
            mock_sell.assert_called_once_with(
                sell_data['symbol'],
                sell_data['amount'],
                sell_data['price']
            )

    def test_configuration_management(self, client, sample_config):
        """Test configuration management API."""
        with patch('frontend.app.load_config', return_value=sample_config), \
             patch('frontend.app.save_config') as mock_save:

            # Test get config
            response = client.get('/api/refresh_config')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'trading' in data
            assert 'solana' in data
            assert 'telegram' in data

            # Test save config
            updated_config = sample_config.copy()
            updated_config['trading']['max_positions'] = 10

            response = client.post('/api/save_config_settings',
                                 json=updated_config,
                                 content_type='application/json')
            assert response.status_code == 200

            # Verify save was called
            mock_save.assert_called_once()

    def test_manual_price_management(self, client):
        """Test manual price override API."""
        price_data = {
            'BTC/USDT': 50000.0,
            'ETH/USDT': 3000.0
        }

        with patch('frontend.app.save_manual_prices') as mock_save, \
             patch('frontend.app.load_manual_prices', return_value=price_data):

            # Test set manual prices
            response = client.post('/api/manual-prices',
                                 json=price_data,
                                 content_type='application/json')
            assert response.status_code == 200

            # Test get manual prices
            response = client.get('/api/manual-prices')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'BTC/USDT' in data
            assert data['BTC/USDT'] == 50000.0

            # Test delete manual price
            response = client.delete('/api/manual-prices/BTC/USDT')
            assert response.status_code == 200

    def test_wallet_balance_api(self, client):
        """Test wallet balance API endpoints."""
        balance_data = {
            'USDT': {'free': 1000.0, 'total': 1000.0},
            'BTC': {'free': 1.0, 'total': 1.0}
        }

        with patch('frontend.app.get_wallet_balance', return_value=balance_data):

            # Test get wallet balance
            response = client.get('/api/wallet-balance')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'USDT' in data
            assert 'BTC' in data
            assert data['USDT']['free'] == 1000.0

    def test_paper_wallet_api(self, client):
        """Test paper wallet API endpoints."""
        paper_balance = {
            'balance': 1000.0,
            'positions': {},
            'total_value': 1000.0
        }

        with patch('frontend.app.get_paper_wallet_balance', return_value=paper_balance), \
             patch('frontend.app.update_paper_wallet_balance') as mock_update:

            # Test get paper wallet balance
            response = client.get('/api/paper-wallet-balance')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['balance'] == 1000.0

            # Test update paper wallet balance
            update_data = {'new_balance': 1500.0}
            response = client.post('/api/paper-wallet-balance',
                                 json=update_data,
                                 content_type='application/json')
            assert response.status_code == 200

            mock_update.assert_called_once_with(1500.0)

    def test_live_signals_api(self, client):
        """Test live signals API endpoint."""
        signals_data = [
            {
                'symbol': 'BTC/USDT',
                'action': 'buy',
                'confidence': 0.85,
                'strategy': 'trend_bot',
                'timestamp': '2024-01-01T10:00:00'
            },
            {
                'symbol': 'ETH/USDT',
                'action': 'sell',
                'confidence': 0.75,
                'strategy': 'mean_bot',
                'timestamp': '2024-01-01T10:05:00'
            }
        ]

        with patch('frontend.app.get_live_signals', return_value=signals_data):

            response = client.get('/api/live-signals')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert isinstance(data, list)
            assert len(data) == 2

            signal = data[0]
            assert 'symbol' in signal
            assert 'action' in signal
            assert 'confidence' in signal
            assert 'strategy' in signal

    def test_strategy_performance_api(self, client):
        """Test strategy performance API endpoint."""
        performance_data = {
            'trend_bot': {
                'total_trades': 100,
                'winning_trades': 60,
                'win_rate': 60.0,
                'total_pnl': 500.0,
                'avg_trade_pnl': 5.0
            },
            'mean_bot': {
                'total_trades': 80,
                'winning_trades': 45,
                'win_rate': 56.25,
                'total_pnl': 320.0,
                'avg_trade_pnl': 4.0
            }
        }

        with patch('frontend.app.get_strategy_performance', return_value=performance_data):

            response = client.get('/api/strategy-performance')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'trend_bot' in data
            assert 'mean_bot' in data

            trend_perf = data['trend_bot']
            assert trend_perf['win_rate'] == 60.0
            assert trend_perf['total_pnl'] == 500.0

    def test_dashboard_metrics_api(self, client):
        """Test dashboard metrics API endpoint."""
        metrics_data = {
            'total_balance': 2500.0,
            'total_pnl': 500.0,
            'pnl_percentage': 25.0,
            'open_positions': 3,
            'active_strategies': 5,
            'win_rate': 65.0,
            'daily_pnl': 25.0,
            'weekly_pnl': 150.0
        }

        with patch('frontend.app.get_dashboard_metrics', return_value=metrics_data):

            response = client.get('/api/dashboard-metrics')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['total_balance'] == 2500.0
            assert data['total_pnl'] == 500.0
            assert data['win_rate'] == 65.0

    def test_price_source_health_api(self, client):
        """Test price source health monitoring API."""
        health_data = {
            'kraken': {'status': 'healthy', 'latency': 150, 'last_update': '2024-01-01T10:00:00'},
            'binance': {'status': 'healthy', 'latency': 120, 'last_update': '2024-01-01T10:00:00'},
            'coinbase': {'status': 'degraded', 'latency': 500, 'last_update': '2024-01-01T09:59:00'}
        }

        with patch('frontend.app.check_price_source_health', return_value=health_data):

            response = client.get('/api/price-source-health')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'kraken' in data
            assert 'binance' in data
            assert data['kraken']['status'] == 'healthy'
            assert data['coinbase']['status'] == 'degraded'

    def test_current_prices_api(self, client):
        """Test current prices API endpoint."""
        prices_data = {
            'BTC/USDT': {'price': 50000.0, 'change_24h': 2.5, 'volume_24h': 1000000},
            'ETH/USDT': {'price': 3000.0, 'change_24h': -1.2, 'volume_24h': 500000},
            'ADA/USDT': {'price': 0.5, 'change_24h': 5.0, 'volume_24h': 200000}
        }

        with patch('frontend.app.get_current_prices', return_value=prices_data):

            response = client.get('/api/current-prices')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'BTC/USDT' in data
            assert 'ETH/USDT' in data
            assert data['BTC/USDT']['price'] == 50000.0
            assert data['ETH/USDT']['change_24h'] == -1.2

    def test_clear_positions_api(self, client):
        """Test clear old positions API endpoint."""
        with patch('frontend.app.clear_old_positions', return_value=5) as mock_clear:

            response = client.post('/api/clear-old-positions')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'message' in data
            assert '5' in data['message']

            mock_clear.assert_called_once()

    def test_generate_sample_data_api(self, client):
        """Test generate sample trades API endpoint."""
        with patch('frontend.app.generate_sample_trades', return_value=10) as mock_generate:

            response = client.post('/api/generate_sample_trades')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'message' in data
            assert '10' in data['message']

            mock_generate.assert_called_once()

    def test_generate_scores_api(self, client):
        """Test generate scores API endpoint."""
        with patch('frontend.app.generate_asset_scores', return_value=25) as mock_generate:

            response = client.post('/api/generate_scores')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'message' in data
            assert '25' in data['message']

            mock_generate.assert_called_once()

    def test_api_config_management(self, client):
        """Test API configuration management endpoints."""
        api_config = {
            'kraken': {'api_key': 'test_key', 'api_secret': 'test_secret'},
            'binance': {'api_key': 'binance_key', 'api_secret': 'binance_secret'}
        }

        with patch('frontend.app.load_api_config', return_value=api_config), \
             patch('frontend.app.save_api_config') as mock_save:

            # Test get API config
            response = client.get('/api_config')
            assert response.status_code == 200

            # Test save API config
            response = client.post('/api/save_api_config',
                                 json=api_config,
                                 content_type='application/json')
            assert response.status_code == 200

            mock_save.assert_called_once_with(api_config)

    def test_error_handling_api(self, client):
        """Test API error handling."""
        with patch('frontend.app.load_positions', side_effect=Exception("Database error")):

            response = client.get('/api/open-positions')
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data

    def test_concurrent_requests_api(self, client):
        """Test concurrent API requests handling."""
        import threading

        results = []
        errors = []

        def make_request(endpoint):
            try:
                response = client.get(endpoint)
                results.append((endpoint, response.status_code))
            except Exception as e:
                errors.append((endpoint, str(e)))

        # Create multiple threads making concurrent requests
        threads = []
        endpoints = ['/api/bot-status', '/api/dashboard-metrics', '/api/current-prices']

        for endpoint in endpoints:
            thread = threading.Thread(target=make_request, args=(endpoint,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all requests succeeded
        assert len(results) == len(endpoints)
        assert len(errors) == 0

        for endpoint, status_code in results:
            assert status_code == 200

    def test_rate_limiting_api(self, client):
        """Test API rate limiting (if implemented)."""
        # Make multiple rapid requests to test rate limiting
        responses = []
        for _ in range(10):
            response = client.get('/api/bot-status')
            responses.append(response.status_code)

        # In a real implementation with rate limiting, some requests might be 429
        # For now, just verify all requests are handled
        assert all(status in [200, 429] for status in responses)

    def test_session_management_api(self, client):
        """Test API session management."""
        with client.session_transaction() as sess:
            sess['user_id'] = 'test_user'

        # Test that session data persists across requests
        with client.session_transaction() as sess:
            assert sess.get('user_id') == 'test_user'

    def test_cors_headers_api(self, client):
        """Test CORS headers on API endpoints."""
        response = client.get('/api/bot-status')
        assert 'Access-Control-Allow-Origin' in response.headers
        assert 'Access-Control-Allow-Methods' in response.headers
        assert 'Access-Control-Allow-Headers' in response.headers

    def test_content_type_validation_api(self, client):
        """Test content type validation for POST endpoints."""
        # Test without proper content type
        response = client.post('/api/sell-position',
                             data=json.dumps({'symbol': 'BTC/USDT'}),
                             content_type='text/plain')
        # Should handle gracefully or return appropriate error
        assert response.status_code in [200, 400, 415]

        # Test with correct content type
        response = client.post('/api/sell-position',
                             json={'symbol': 'BTC/USDT'},
                             content_type='application/json')
        assert response.status_code in [200, 400]  # 400 if validation fails, 200 if successful

    def test_data_validation_api(self, client):
        """Test input data validation on API endpoints."""
        # Test with invalid data
        invalid_data = {
            'symbol': '',  # Empty symbol
            'amount': -1,  # Negative amount
            'price': 0     # Zero price
        }

        response = client.post('/api/sell-position',
                             json=invalid_data,
                             content_type='application/json')

        # Should return validation error
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data or 'message' in data
