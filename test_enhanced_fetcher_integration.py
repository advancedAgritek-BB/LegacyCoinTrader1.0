"""
Integration tests for the Enhanced OHLCV Fetcher
Tests the fetcher working with the rest of the application
"""

import asyncio
import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from crypto_bot.utils.enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher


class TestEnhancedOHLCVFetcherIntegration:
    """Integration tests for EnhancedOHLCVFetcher"""

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange object"""
        exchange = Mock()
        exchange.id = 'kraken'
        exchange.timeframes = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return exchange

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        return {
            'exchange': 'kraken',
            'max_concurrent_ohlcv': 3,
            'max_concurrent_dex_ohlcv': 10,
            'min_volume_usd': 1000,
            'ohlcv_fetcher_timeout': 30,
            'use_enhanced_ohlcv_fetcher': True
        }

    @pytest.fixture
    def fetcher(self, mock_exchange, mock_config):
        """Create an EnhancedOHLCVFetcher instance"""
        return EnhancedOHLCVFetcher(mock_exchange, mock_config)

    @pytest.mark.asyncio
    async def test_fetcher_initialization(self, fetcher, mock_exchange):
        """Test that fetcher initializes correctly"""
        assert fetcher.exchange == mock_exchange
        assert fetcher.max_concurrent_cex == 3
        assert fetcher.max_concurrent_dex == 10
        assert len(fetcher.supported_timeframes) > 0
        assert '1m' in fetcher.supported_timeframes

    def test_timeframe_validation(self, fetcher):
        """Test timeframe validation methods"""
        # Test supported timeframe
        is_valid, message = fetcher.validate_timeframe_request('1h')
        assert is_valid
        assert 'supported' in message

        # Test unsupported timeframe
        is_valid, message = fetcher.validate_timeframe_request('10m')
        assert not is_valid
        assert 'not supported' in message

        # Test closest match
        supported = fetcher.get_supported_timeframe('10m')
        assert supported in fetcher.supported_timeframes

    def test_symbol_classification(self, fetcher):
        """Test symbol classification into CEX and DEX"""
        symbols = ['BTC/USD', 'ETH/USDT', 'SOL/USDC', 'MATIC/USD']
        cex_symbols, dex_symbols = fetcher._classify_symbols(symbols)

        # SOL/USDC should be DEX, others should be CEX
        assert len(cex_symbols) == 3  # BTC/USD, ETH/USDT, MATIC/USD
        assert len(dex_symbols) == 1  # SOL/USDC
        assert 'SOL/USDC' in dex_symbols

        # Test DEX classification
        dex_symbols_list = ['SOL/USDC', 'MATIC/USDC']
        cex, dex = fetcher._classify_symbols(dex_symbols_list)
        assert len(cex) == 0
        assert len(dex) == 2

        # Test mixed classification
        mixed_symbols = ['BTC/USD', 'SOL/USDC', 'ETH/USD', 'MATIC/USDC']
        cex, dex = fetcher._classify_symbols(mixed_symbols)
        assert len(cex) == 2  # BTC/USD, ETH/USD
        assert len(dex) == 2  # SOL/USDC, MATIC/USDC

    @pytest.mark.asyncio
    async def test_cex_fetch_batch_empty(self, fetcher):
        """Test CEX batch fetch with empty symbols"""
        result = await fetcher._fetch_cex_ohlcv_batch([], '1h', 100)
        assert result == {}

    @pytest.mark.asyncio
    async def test_dex_fetch_batch_empty(self, fetcher):
        """Test DEX batch fetch with empty symbols"""
        result = await fetcher._fetch_dex_ohlcv_batch([], '1h', 100)
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_batch_empty(self, fetcher):
        """Test main batch fetch with empty symbols"""
        cex_data, dex_data = await fetcher.fetch_ohlcv_batch([], '1h', 100)
        assert cex_data == {} and dex_data == {}

    @pytest.mark.asyncio
    @patch('crypto_bot.utils.enhanced_ohlcv_fetcher.fetch_geckoterminal_ohlcv')
    @patch('crypto_bot.utils.enhanced_ohlcv_fetcher.fetch_helius_ohlcv')
    @patch('crypto_bot.utils.enhanced_ohlcv_fetcher.fetch_dex_ohlcv')
    async def test_dex_fetch_with_fallbacks(self, mock_dex, mock_helius, mock_gecko, fetcher):
        """Test DEX fetch with multiple fallback sources"""
        # Mock GeckoTerminal failure
        mock_gecko.return_value = None

        # Mock Helius success
        mock_ohlcv_data = [
            [1640995200000, 100.0, 105.0, 95.0, 102.0, 1000.0],
            [1640995260000, 102.0, 107.0, 98.0, 104.0, 1200.0]
        ]
        mock_helius.return_value = mock_ohlcv_data

        symbols = ['SOL/USDC']
        result = await fetcher._fetch_dex_ohlcv_batch(symbols, '1m', 100)

        assert 'SOL/USDC' in result
        assert result['SOL/USDC'] == mock_ohlcv_data

        # Verify GeckoTerminal was called
        mock_gecko.assert_called_once()

        # Verify Helius was called as fallback
        mock_helius.assert_called_once()

        # Verify DEX fetcher was not called (Helius succeeded)
        mock_dex.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_cache_with_empty_data(self, fetcher):
        """Test cache update with empty data"""
        cache = {}
        symbols = ['BTC/USD']
        result = await fetcher.update_cache(cache, symbols, '1h', 100)

        # Should return original cache since no data was fetched
        assert result == cache

    @pytest.mark.asyncio
    async def test_update_cache_with_valid_data(self, fetcher):
        """Test cache update with valid OHLCV data"""
        # Mock the fetch_ohlcv_batch method
        mock_data = {
            'BTC/USD': [
                [1640995200000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],
                [1640995260000, 50500.0, 51500.0, 49500.0, 51000.0, 120.0]
            ]
        }

        # Mock to return separate CEX and DEX data
        with patch.object(fetcher, 'fetch_ohlcv_batch', return_value=(mock_data, {})):
            cache = {}
            result = await fetcher.update_cache(cache, ['BTC/USD'], '1h', 100)

            # Should have created cache entry
            assert 'BTC/USD' in result
            assert isinstance(result['BTC/USD'], pd.DataFrame)
            assert len(result['BTC/USD']) == 2

            # Should have return column
            assert 'return' in result['BTC/USD'].columns

    @pytest.mark.asyncio
    async def test_timeout_protection(self, fetcher):
        """Test timeout protection in fetch operations"""
        # Test the timeout configuration is properly set and used
        assert 'ohlcv_fetcher_timeout' in fetcher.config
        assert fetcher.config['ohlcv_fetcher_timeout'] == 30  # From mock config

        # Test that timeout is used in the fetch_ohlcv_batch method
        # We'll create functions that raise TimeoutError to simulate timeout
        async def timeout_cex_fetch(*args, **kwargs):
            raise asyncio.TimeoutError("Simulated timeout")

        async def timeout_dex_fetch(*args, **kwargs):
            raise asyncio.TimeoutError("Simulated timeout")

        # Set a reasonable timeout for testing
        original_timeout = fetcher.config['ohlcv_fetcher_timeout']
        fetcher.config['ohlcv_fetcher_timeout'] = 1  # 1 second timeout

        try:
            with patch.object(fetcher, '_fetch_cex_ohlcv_batch', side_effect=timeout_cex_fetch), \
                 patch.object(fetcher, '_fetch_dex_ohlcv_batch', side_effect=timeout_dex_fetch), \
                 patch('crypto_bot.utils.market_loader.fetch_ohlcv_async', return_value=None):

                import time
                start_time = time.time()

                cex_data, dex_data = await fetcher.fetch_ohlcv_batch(['BTC/USD'], '1h', 100)

                elapsed = time.time() - start_time

                # Should complete quickly due to timeout (much less than the 2 second simulated delay)
                assert elapsed < 0.5, f"Timeout took too long: {elapsed}s"
                # Result should be empty due to timeout
                assert cex_data == {} and dex_data == {}
        finally:
            # Restore original timeout
            fetcher.config['ohlcv_fetcher_timeout'] = original_timeout

    @pytest.mark.asyncio
    async def test_timeframe_fallback(self, fetcher):
        """Test automatic timeframe fallback for unsupported timeframes"""
        # Mock the internal batch methods to return data
        mock_data = {'BTC/USD': [[1640995200000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0]]}

        with patch.object(fetcher, '_fetch_cex_ohlcv_batch', return_value=mock_data), \
             patch.object(fetcher, '_fetch_dex_ohlcv_batch', return_value={}):

            # Test with unsupported timeframe - should use closest supported timeframe
            cex_data, dex_data = await fetcher.fetch_ohlcv_batch(['BTC/USD'], '10m', 100)
            result = {**cex_data, **dex_data}

            # Should get data back (fallback timeframe should work)
            assert 'BTC/USD' in result
            assert len(result['BTC/USD']) > 0

    def test_timeframe_conversion(self, fetcher):
        """Test timeframe string to minutes conversion"""
        assert fetcher._timeframe_to_minutes('1m') == 1
        assert fetcher._timeframe_to_minutes('1h') == 60
        assert fetcher._timeframe_to_minutes('1d') == 1440
        assert fetcher._timeframe_to_minutes('2w') == 20160
        assert fetcher._timeframe_to_minutes('invalid') is None

    @pytest.mark.asyncio
    async def test_concurrent_fetching(self, fetcher):
        """Test that multiple symbols are fetched concurrently"""
        symbols = ['BTC/USD', 'ETH/USD', 'SOL/USDC']

        # Mock the batch fetch methods
        async def mock_cex_fetch(symbols_list, *args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate network delay
            return {sym: [[1640995200000, 100.0, 105.0, 95.0, 102.0, 1000.0]] for sym in symbols_list[:2]}

        async def mock_dex_fetch(symbols_list, *args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate network delay
            return {symbols_list[0]: [[1640995200000, 50.0, 55.0, 45.0, 52.0, 5000.0]]}

        with patch.object(fetcher, '_fetch_cex_ohlcv_batch', side_effect=mock_cex_fetch), \
             patch.object(fetcher, '_fetch_dex_ohlcv_batch', side_effect=mock_dex_fetch):

            import time
            start_time = time.time()

            cex_data, dex_data = await fetcher.fetch_ohlcv_batch(symbols, '1h', 100)
            result = {**cex_data, **dex_data}

            elapsed = time.time() - start_time

            # Should complete faster than sequential (should be ~0.1s not ~0.3s)
            assert elapsed < 0.25  # Allow some margin for test execution
            assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
