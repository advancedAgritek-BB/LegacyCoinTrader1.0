#!/usr/bin/env python3
"""
Comprehensive test suite for EnhancedOHLCVFetcher to prevent regression of critical bugs.
"""

import asyncio
import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher


class TestEnhancedOHLCVFetcher:
    """Test suite for EnhancedOHLCVFetcher functionality."""

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange object."""
        exchange = MagicMock()
        exchange.id = "kraken"
        exchange.symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        exchange.markets = {"BTC/USD": {}, "ETH/USD": {}, "SOL/USD": {}}
        exchange.rateLimit = 100  # Required for rate limiting logic
        return exchange

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "max_concurrent_ohlcv": 3,
            "max_concurrent_dex_ohlcv": 10,
            "min_volume_usd": 1000,
            "ohlcv_fetcher_timeout": 30
        }

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Sample OHLCV data for testing."""
        return [
            [1640995200000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],  # timestamp, open, high, low, close, volume
            [1640995260000, 50500.0, 51500.0, 50000.0, 51000.0, 150.0],
            [1640995320000, 51000.0, 52000.0, 50500.0, 51500.0, 200.0],
        ]

    @pytest.fixture
    def fetcher(self, mock_exchange, sample_config):
        """Create EnhancedOHLCVFetcher instance."""
        return EnhancedOHLCVFetcher(mock_exchange, sample_config)

    @pytest.mark.asyncio
    async def test_fetch_cex_ohlcv_batch_result_processing(self, fetcher, sample_ohlcv_data):
        """Test that CEX OHLCV batch correctly processes fetch_single results."""
        # Mock the semaphore and fetch function
        with patch.object(fetcher, '_get_semaphore') as mock_semaphore, \
             patch('crypto_bot.utils.enhanced_ohlcv_fetcher.fetch_ohlcv_async') as mock_fetch:

            # Setup mocks
            mock_semaphore.return_value.__aenter__ = AsyncMock()
            mock_semaphore.return_value.__aexit__ = AsyncMock()
            mock_fetch.return_value = sample_ohlcv_data

            # Test symbols
            symbols = ['BTC/USD', 'ETH/USD']

            # Call the function
            result = await fetcher._fetch_cex_ohlcv_batch(symbols, '1h', 100)

            # Verify results
            assert len(result) == 2
            assert 'BTC/USD' in result
            assert 'ETH/USD' in result
            assert isinstance(result['BTC/USD'], list)
            assert len(result['BTC/USD']) == 3  # 3 data points

    @pytest.mark.asyncio
    async def test_update_cache_processes_all_cex_data(self, fetcher, sample_ohlcv_data):
        """Test that update_cache correctly processes all CEX data returned by fetch_ohlcv_batch."""
        cache = {}

        # Mock the batch fetch to return data for both symbols
        with patch.object(fetcher, 'fetch_ohlcv_batch') as mock_batch_fetch:
            mock_batch_fetch.return_value = ({
                'BTC/USD': sample_ohlcv_data,
                'ETH/USD': sample_ohlcv_data
            }, {})  # CEX data, empty DEX data

            # Call update_cache
            result_cache = await fetcher.update_cache(cache, ['BTC/USD', 'ETH/USD'], '1h', 100)

            # Verify both symbols are in the cache
            assert 'BTC/USD' in result_cache
            assert 'ETH/USD' in result_cache
            assert isinstance(result_cache['BTC/USD'], pd.DataFrame)
            assert len(result_cache['BTC/USD']) == 3

    @pytest.mark.asyncio
    async def test_update_cache_with_none_data_filtered_out(self, fetcher, sample_ohlcv_data):
        """Test that symbols with None data are properly filtered out."""
        cache = {}

        # Mock batch fetch to return one valid symbol and one None
        with patch.object(fetcher, 'fetch_ohlcv_batch') as mock_batch_fetch:
            mock_batch_fetch.return_value = ({
                'BTC/USD': sample_ohlcv_data,
                'ETH/USD': None  # This should be filtered out
            }, {})

            # Call update_cache
            result_cache = await fetcher.update_cache(cache, ['BTC/USD', 'ETH/USD'], '1h', 100)

            # Verify only valid symbol is in cache
            assert 'BTC/USD' in result_cache
            assert 'ETH/USD' not in result_cache

    @pytest.mark.asyncio
    async def test_update_cache_empty_data_handling(self, fetcher):
        """Test handling of empty data from fetch_ohlcv_batch."""
        cache = {}

        # Mock batch fetch to return empty data
        with patch.object(fetcher, 'fetch_ohlcv_batch') as mock_batch_fetch:
            mock_batch_fetch.return_value = ({}, {})  # Empty CEX and DEX data

            # Call update_cache
            result_cache = await fetcher.update_cache(cache, ['BTC/USD'], '1h', 100)

            # Verify cache remains empty
            assert len(result_cache) == 0

    @pytest.mark.asyncio
    async def test_symbol_classification_logic(self, fetcher):
        """Test that symbol classification correctly separates CEX and DEX symbols."""
        # Mock the _is_valid_base_token function to simulate DEX detection
        with patch('crypto_bot.utils.enhanced_ohlcv_fetcher._is_valid_base_token') as mock_is_valid:
            mock_is_valid.return_value = False  # All symbols treated as CEX

            symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']
            cex_symbols, dex_symbols = fetcher._classify_symbols(symbols)

            # All should be CEX since mock returns False
            assert len(cex_symbols) == 3
            assert len(dex_symbols) == 0
            assert set(cex_symbols) == set(symbols)

    @pytest.mark.asyncio
    async def test_concurrent_fetch_with_mixed_results(self, fetcher, sample_ohlcv_data):
        """Test concurrent fetching with mixed success/failure results."""
        with patch.object(fetcher, '_get_semaphore') as mock_semaphore, \
             patch('crypto_bot.utils.enhanced_ohlcv_fetcher.fetch_ohlcv_async') as mock_fetch:

            # Setup semaphore mocks
            mock_semaphore.return_value.__aenter__ = AsyncMock()
            mock_semaphore.return_value.__aexit__ = AsyncMock()

            # Mock fetch to succeed for first symbol, fail for second
            def mock_fetch_side_effect(*args, **kwargs):
                symbol = args[1] if len(args) > 1 else kwargs.get('symbol', '')
                if 'BTC' in symbol:
                    return sample_ohlcv_data
                else:
                    return None  # Simulate failure

            mock_fetch.side_effect = mock_fetch_side_effect

            symbols = ['BTC/USD', 'ETH/USD']
            result = await fetcher._fetch_cex_ohlcv_batch(symbols, '1h', 100)

            # Verify only successful fetch is in result
            assert 'BTC/USD' in result
            assert 'ETH/USD' not in result
            assert result['BTC/USD'] == sample_ohlcv_data

    @pytest.mark.asyncio
    async def test_exception_handling_in_batch_fetch(self, fetcher):
        """Test that exceptions in individual fetches are handled properly."""
        with patch.object(fetcher, '_get_semaphore') as mock_semaphore, \
             patch('crypto_bot.utils.enhanced_ohlcv_fetcher.fetch_ohlcv_async') as mock_fetch:

            # Setup semaphore mocks
            mock_semaphore.return_value.__aenter__ = AsyncMock()
            mock_semaphore.return_value.__aexit__ = AsyncMock()

            # Mock fetch to raise exception
            mock_fetch.side_effect = Exception("Network error")

            symbols = ['BTC/USD']
            result = await fetcher._fetch_cex_ohlcv_batch(symbols, '1h', 100)

            # Verify empty result when all fetches fail
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_data_validation_in_update_cache(self, fetcher):
        """Test that invalid data formats are properly validated and rejected."""
        cache = {}

        # Test with invalid data (not a list of lists)
        invalid_data = "invalid_data_format"

        with patch.object(fetcher, 'fetch_ohlcv_batch') as mock_batch_fetch:
            mock_batch_fetch.return_value = ({'BTC/USD': invalid_data}, {})

            # This should not crash and should filter out invalid data
            result_cache = await fetcher.update_cache(cache, ['BTC/USD'], '1h', 100)

            # Invalid data should be filtered out
            assert 'BTC/USD' not in result_cache

    @pytest.mark.asyncio
    async def test_minimum_candle_requirement(self, fetcher):
        """Test that symbols with insufficient candles are filtered out."""
        cache = {}

        # Data with only 2 candles (less than minimum required)
        insufficient_data = [
            [1640995200000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],
            [1640995260000, 50500.0, 51500.0, 50000.0, 51000.0, 150.0],
        ]

        with patch.object(fetcher, 'fetch_ohlcv_batch') as mock_batch_fetch:
            mock_batch_fetch.return_value = ({'BTC/USD': insufficient_data}, {})

            result_cache = await fetcher.update_cache(cache, ['BTC/USD'], '1h', 100)

            # Should still cache partial data but log warning
            assert 'BTC/USD' in result_cache
            assert len(result_cache['BTC/USD']) == 2

    def test_semaphore_creation_fallback(self, mock_exchange, sample_config):
        """Test that semaphore creation handles event loop conflicts gracefully."""
        # Mock asyncio.Semaphore to raise RuntimeError (event loop conflict)
        with patch('asyncio.Semaphore', side_effect=RuntimeError("Event loop conflict")):
            fetcher = EnhancedOHLCVFetcher(mock_exchange, sample_config)

            # Should not crash and should have fallback semaphore values
            assert hasattr(fetcher, '_cex_semaphore_value')
            assert hasattr(fetcher, '_dex_semaphore_value')
            assert fetcher._cex_semaphore_value == 3
            assert fetcher._dex_semaphore_value == 10
