"""
Unit tests for Pyth price fallback functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import requests

from crypto_bot.utils.pyth import (
    get_pyth_price,
    _get_fallback_price,
    _parse_fallback_response,
    get_price_async,
    _get_single_fallback_price
)


class TestPythFallbackFixes(unittest.TestCase):
    """Test cases for Pyth price fallback improvements."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_symbol = "BTC/USD"

    @patch('crypto_bot.utils.pyth.requests.get')
    def test_pyth_price_with_fallback_success(self, mock_get):
        """Test that Pyth price fetching falls back successfully."""
        # Mock Pyth failure (empty response)
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        # Mock successful fallback
        with patch('crypto_bot.utils.pyth._get_fallback_price') as mock_fallback:
            mock_fallback.return_value = 50000.0

            price = get_pyth_price(self.test_symbol, max_retries=1)

            # Should return fallback price
            self.assertEqual(price, 50000.0)
            mock_fallback.assert_called_once_with(self.test_symbol)

    @patch('crypto_bot.utils.pyth.requests.get')
    def test_fallback_sources_coingecko(self, mock_get):
        """Test CoinGecko fallback source."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "bitcoin": {"usd": 45000.0}
        }
        mock_get.return_value = mock_response

        # Test CoinGecko parsing
        result = _parse_fallback_response("coingecko", mock_response.json(), "BTC/USD")
        self.assertEqual(result, 45000.0)

    @patch('crypto_bot.utils.pyth.requests.get')
    def test_fallback_sources_binance(self, mock_get):
        """Test Binance fallback source."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        # Test with list response
        mock_response.json.return_value = [
            {"symbol": "BTCUSDT", "price": "45000.50"}
        ]
        mock_get.return_value = mock_response

        result = _parse_fallback_response("binance", mock_response.json(), "BTC/USDT")
        self.assertEqual(result, 45000.50)

    @patch('crypto_bot.utils.pyth.requests.get')
    def test_fallback_sources_kraken(self, mock_get):
        """Test Kraken fallback source."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": {
                "XXBTZUSD": {
                    "c": ["45000.00", "100.0"]
                }
            }
        }
        mock_get.return_value = mock_response

        result = _parse_fallback_response("kraken", mock_response.json(), "BTC/USD")
        self.assertEqual(result, 45000.00)

    @patch('crypto_bot.utils.pyth.requests.get')
    def test_fallback_failure_handling(self, mock_get):
        """Test graceful handling of fallback failures."""
        # Mock all requests to fail
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        result = _get_fallback_price(self.test_symbol)
        self.assertIsNone(result)

    def test_parse_fallback_invalid_response(self):
        """Test parsing of invalid fallback responses."""
        # Test with malformed data
        result = _parse_fallback_response("coingecko", {}, "BTC/USD")
        self.assertIsNone(result)

        result = _parse_fallback_response("binance", [], "BTC/USDT")
        self.assertIsNone(result)

        result = _parse_fallback_response("kraken", {"result": {}}, "BTC/USD")
        self.assertIsNone(result)

    @patch('crypto_bot.utils.pyth.requests.get')
    @patch('crypto_bot.utils.pyth.get_pyth_price')
    @patch('asyncio.get_event_loop')
    def test_async_price_fetching(self, mock_loop, mock_pyth_price, mock_get):
        """Test async price fetching with concurrent fallbacks."""
        # Mock event loop
        mock_loop_instance = Mock()
        mock_loop.return_value = mock_loop_instance

        # Mock Pyth price failure
        mock_pyth_price.return_value = None

        # Mock successful fallback
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"bitcoin": {"usd": 45000.0}}
        mock_get.return_value = mock_response

        # Mock future results
        mock_future = Mock()
        mock_future.result.return_value = 45000.0
        mock_loop_instance.run_in_executor.return_value = mock_future

        async def test_async():
            result = await get_price_async(self.test_symbol)
            return result

        # This should work in the test environment
        result = asyncio.run(test_async())
        self.assertIsNotNone(result)

    def test_symbol_format_conversion(self):
        """Test symbol format conversion for different exchanges."""
        from crypto_bot.utils.pyth import FALLBACK_SOURCES

        # Test CoinGecko format
        coingecko_params = FALLBACK_SOURCES["coingecko"]["params"]
        params = coingecko_params("BTC/USD")
        self.assertIn("bitcoin", params.get("ids", ""))

        # Test Binance format
        binance_params = FALLBACK_SOURCES["binance"]["params"]
        params = binance_params("BTC/USD")
        self.assertEqual(params.get("symbol"), "BTCUSD")

        # Test Kraken format
        kraken_params = FALLBACK_SOURCES["kraken"]["params"]
        params = kraken_params("BTC/USD")
        self.assertEqual(params.get("pair"), "BTCUSD")


if __name__ == '__main__':
    unittest.main()
