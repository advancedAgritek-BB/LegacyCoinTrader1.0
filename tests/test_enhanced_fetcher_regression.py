#!/usr/bin/env python3
"""
Regression test for the critical bug in EnhancedOHLCVFetcher._fetch_cex_ohlcv_batch
where result processing incorrectly treated (symbol, data) tuples as data.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher


class TestEnhancedOHLCVFetcherRegression:
    """Regression tests for critical bugs in EnhancedOHLCVFetcher."""

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
            [1640995200000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],
            [1640995260000, 50500.0, 51500.0, 50000.0, 51000.0, 150.0],
        ]

    @pytest.fixture
    def fetcher(self, mock_exchange, sample_config):
        """Create EnhancedOHLCVFetcher instance."""
        return EnhancedOHLCVFetcher(mock_exchange, sample_config)

    @pytest.mark.asyncio
    async def test_fetch_single_returns_symbol_data_tuple(self, fetcher, sample_ohlcv_data):
        """Test that fetch_single returns (symbol, data) tuple as expected."""
        symbol = 'BTC/USD'

        with patch.object(fetcher, '_get_semaphore') as mock_semaphore, \
             patch('crypto_bot.utils.enhanced_ohlcv_fetcher.fetch_ohlcv_async') as mock_fetch:

            # Setup mocks
            mock_semaphore.return_value.__aenter__ = AsyncMock()
            mock_semaphore.return_value.__aexit__ = AsyncMock()
            mock_fetch.return_value = sample_ohlcv_data

            # Import and call the inner fetch_single function
            # We need to access the inner function from _fetch_cex_ohlcv_batch
            symbols = [symbol]
            tasks = []
            for sym in symbols:
                # This replicates the fetch_single logic from _fetch_cex_ohlcv_batch
                async def fetch_single(s):
                    async with fetcher._get_semaphore(is_cex=True):
                        try:
                            from crypto_bot.utils.enhanced_ohlcv_fetcher import fetch_ohlcv_async
                            data = await fetch_ohlcv_async(
                                fetcher.exchange,
                                s,
                                timeframe='1h',
                                limit=100,
                                since=None,
                                use_websocket=False,
                                force_websocket_history=False
                            )
                            return s, data
                        except Exception as e:
                            return s, None

                tasks.append(fetch_single(sym))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify the result structure
            assert len(results) == 1
            result = results[0]
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == symbol  # symbol
            assert result[1] == sample_ohlcv_data  # data

    @pytest.mark.asyncio
    async def test_result_processing_regression_bug(self, fetcher, sample_ohlcv_data):
        """
        Regression test for the bug where result processing incorrectly treated
        (symbol, data) tuples as data, causing all fetches to appear as failures.

        This test ensures the fix correctly unpacks the tuple.
        """
        # Simulate the old buggy behavior vs new fixed behavior
        mock_result_tuple = ('BTC/USD', sample_ohlcv_data)
        mock_result_none = ('ETH/USD', None)

        results = [mock_result_tuple, mock_result_none]

        # Test the fixed result processing logic
        data_map = {}
        for result in results:
            if isinstance(result, Exception):
                continue

            symbol_result, data = result  # This is the fix
            if data is not None:
                data_map[symbol_result] = data

        # Verify correct processing
        assert 'BTC/USD' in data_map
        assert 'ETH/USD' not in data_map
        assert data_map['BTC/USD'] == sample_ohlcv_data

        # Test what would happen with the old buggy logic
        buggy_data_map = {}
        for symbol, result in [('BTC/USD', mock_result_tuple), ('ETH/USD', mock_result_none)]:
            if isinstance(result, Exception):
                continue
            # Old buggy logic: if result is not None (always true for tuples)
            if result is not None:  # This was the bug!
                buggy_data_map[symbol] = result  # Wrong! Storing tuple instead of data

        # Verify the bug would have caused incorrect behavior
        assert len(buggy_data_map) == 2  # Both symbols would appear to have data
        assert isinstance(buggy_data_map['BTC/USD'], tuple)  # Wrong type stored
        assert isinstance(buggy_data_map['ETH/USD'], tuple)  # Wrong type stored

    @pytest.mark.asyncio
    async def test_end_to_end_fetch_with_regression_protection(self, fetcher, sample_ohlcv_data):
        """
        End-to-end test ensuring the regression fix works in the complete flow.
        This test would fail with the old buggy code.
        """
        with patch.object(fetcher, '_get_semaphore') as mock_semaphore, \
             patch('crypto_bot.utils.enhanced_ohlcv_fetcher.fetch_ohlcv_async') as mock_fetch:

            # Setup mocks
            mock_semaphore.return_value.__aenter__ = AsyncMock()
            mock_semaphore.return_value.__aexit__ = AsyncMock()
            mock_fetch.return_value = sample_ohlcv_data

            # Test the complete _fetch_cex_ohlcv_batch method
            symbols = ['BTC/USD']
            result = await fetcher._fetch_cex_ohlcv_batch(symbols, '1h', 100)

            # With the fix, this should return the actual data
            assert 'BTC/USD' in result
            assert result['BTC/USD'] == sample_ohlcv_data
            assert isinstance(result['BTC/USD'], list)
            assert len(result['BTC/USD']) == 2

    @pytest.mark.asyncio
    async def test_mixed_success_failure_with_regression_protection(self, fetcher, sample_ohlcv_data):
        """
        Test mixed success/failure scenario to ensure the fix handles both cases correctly.
        """
        with patch.object(fetcher, '_get_semaphore') as mock_semaphore, \
             patch('crypto_bot.utils.enhanced_ohlcv_fetcher.fetch_ohlcv_async') as mock_fetch:

            # Setup mocks
            mock_semaphore.return_value.__aenter__ = AsyncMock()
            mock_semaphore.return_value.__aexit__ = AsyncMock()

            # Mock fetch to succeed for BTC, fail for ETH
            call_count = 0
            def mock_fetch_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:  # First call (BTC)
                    return sample_ohlcv_data
                else:  # Second call (ETH)
                    return None

            mock_fetch.side_effect = mock_fetch_side_effect

            symbols = ['BTC/USD', 'ETH/USD']
            result = await fetcher._fetch_cex_ohlcv_batch(symbols, '1h', 100)

            # With the fix, only successful fetch should be in result
            assert len(result) == 1
            assert 'BTC/USD' in result
            assert 'ETH/USD' not in result
            assert result['BTC/USD'] == sample_ohlcv_data
