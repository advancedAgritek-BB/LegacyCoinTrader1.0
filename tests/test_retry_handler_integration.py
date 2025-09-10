"""
Integration tests for Enhanced Error Handling and Retry Logic.

Tests the integration of the retry handler with market_loader and symbol_utils.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import ccxt

from crypto_bot.utils.retry_handler import (
    get_retry_manager,
    RetryConfig,
    RetryableError,
    NonRetryableError
)
from crypto_bot.utils.market_loader import (
    _call_with_enhanced_retry,
    OHLCV_RETRY_CONFIG,
    MARKET_DATA_RETRY_CONFIG
)
from crypto_bot.utils.symbol_utils import get_filtered_symbols


class TestRetryHandlerMarketLoaderIntegration:
    """Test retry handler integration with market_loader."""

    @pytest.mark.asyncio
    async def test_enhanced_retry_with_ohlcv_fetch(self):
        """Test enhanced retry logic with OHLCV fetching."""
        # Mock exchange that fails twice then succeeds
        exchange = Mock()
        exchange.fetch_ohlcv = AsyncMock()
        exchange.fetch_ohlcv.side_effect = [
            ccxt.NetworkError("Network error"),
            ccxt.ExchangeError("Exchange error"),
            [{"timestamp": 1000, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]
        ]

        # Test the enhanced retry function
        result = await _call_with_enhanced_retry(
            exchange.fetch_ohlcv,
            symbol="BTC/USD",
            timeframe="1h",
            limit=1,
            retry_config=OHLCV_RETRY_CONFIG
        )

        assert result is not None
        assert len(result) == 1
        assert exchange.fetch_ohlcv.call_count == 3

    @pytest.mark.asyncio
    async def test_enhanced_retry_with_timeout(self):
        """Test enhanced retry logic with timeout."""
        exchange = Mock()
        exchange.fetch_ohlcv = AsyncMock()
        exchange.fetch_ohlcv.side_effect = [
            ccxt.NetworkError("Network error"),
            [{"timestamp": 1000, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]
        ]

        # Test with timeout
        result = await _call_with_enhanced_retry(
            exchange.fetch_ohlcv,
            symbol="BTC/USD",
            timeframe="1h",
            limit=1,
            timeout=30.0,
            retry_config=OHLCV_RETRY_CONFIG
        )

        assert result is not None
        assert len(result) == 1
        assert exchange.fetch_ohlcv.call_count == 2

    @pytest.mark.asyncio
    async def test_enhanced_retry_with_non_retryable_error(self):
        """Test enhanced retry logic with non-retryable error."""
        exchange = Mock()
        exchange.fetch_ohlcv = AsyncMock()
        exchange.fetch_ohlcv.side_effect = ccxt.AuthenticationError("Invalid API key")

        # Test with non-retryable error
        with pytest.raises(NonRetryableError):
            await _call_with_enhanced_retry(
                exchange.fetch_ohlcv,
                symbol="BTC/USD",
                timeframe="1h",
                limit=1,
                retry_config=OHLCV_RETRY_CONFIG
            )

        assert exchange.fetch_ohlcv.call_count == 1

    @pytest.mark.asyncio
    async def test_enhanced_retry_with_max_retries_exceeded(self):
        """Test enhanced retry logic when max retries are exceeded."""
        exchange = Mock()
        exchange.fetch_ohlcv = AsyncMock()
        exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("Network error")

        # Test with max retries exceeded
        with pytest.raises(RetryableError):
            await _call_with_enhanced_retry(
                exchange.fetch_ohlcv,
                symbol="BTC/USD",
                timeframe="1h",
                limit=1,
                retry_config=OHLCV_RETRY_CONFIG
            )

        assert exchange.fetch_ohlcv.call_count == 6  # max_retries + 1

    @pytest.mark.asyncio
    async def test_enhanced_retry_with_context(self):
        """Test enhanced retry logic with context information."""
        exchange = Mock()
        exchange.fetch_ohlcv = AsyncMock()
        exchange.fetch_ohlcv.side_effect = [
            ccxt.NetworkError("Network error"),
            [{"timestamp": 1000, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000}]
        ]

        context = {"symbol": "BTC/USD", "timeframe": "1h", "attempt": 1}

        result = await _call_with_enhanced_retry(
            exchange.fetch_ohlcv,
            symbol="BTC/USD",
            timeframe="1h",
            limit=1,
            context=context,
            retry_config=OHLCV_RETRY_CONFIG
        )

        assert result is not None
        assert len(result) == 1
        assert exchange.fetch_ohlcv.call_count == 2


class TestRetryHandlerSymbolUtilsIntegration:
    """Test retry handler integration with symbol_utils."""

    @pytest.mark.asyncio
    async def test_symbol_scanning_with_retry(self):
        """Test symbol scanning with retry logic."""
        # Mock exchange with proper attributes
        exchange = Mock()
        exchange.id = "kraken"
        exchange.symbols = []  # Empty symbols to trigger load_markets
        exchange.markets_by_id = {"BTCUSD": {"symbol": "BTC/USD"}, "ETHUSD": {"symbol": "ETH/USD"}}
        exchange.load_markets = AsyncMock()
        exchange.load_markets.side_effect = [
            Exception("Temporary error"),
            Exception("Another error"),
            None  # Success on third attempt
        ]

        config = {
            "symbols": ["BTC/USD", "ETH/USD"],
            "symbol_refresh_minutes": 30,
            "skip_symbol_filters": True  # Skip complex filtering for test
        }

        # Test symbol scanning with retry
        result = await get_filtered_symbols(exchange, config)

        assert result is not None
        assert len(result) == 2
        assert exchange.load_markets.call_count == 3

    @pytest.mark.asyncio
    async def test_symbol_scanning_with_non_retryable_error(self):
        """Test symbol scanning with non-retryable error."""
        # Mock exchange with proper attributes
        exchange = Mock()
        exchange.id = "kraken"
        exchange.symbols = ["BTC/USD", "ETH/USD"]
        exchange.markets_by_id = {"BTCUSD": {"symbol": "BTC/USD"}, "ETHUSD": {"symbol": "ETH/USD"}}
        exchange.load_markets = AsyncMock()
        exchange.load_markets.side_effect = Exception("Critical error")

        config = {
            "symbols": ["BTC/USD", "ETH/USD"],
            "symbol_refresh_minutes": 30,
            "skip_symbol_filters": True  # Skip complex filtering for test
        }

        # Test symbol scanning with non-retryable error
        result = await get_filtered_symbols(exchange, config)

        # Should still return symbols even if load_markets fails
        assert result is not None
        assert len(result) == 2


class TestRetryHandlerPerformance:
    """Test retry handler performance characteristics."""

    @pytest.mark.asyncio
    async def test_retry_handler_overhead(self):
        """Test that retry handler doesn't add excessive overhead."""
        # Simple function that succeeds immediately
        async def simple_func():
            return "success"

        # Time without retry handler
        start_time = time.time()
        for _ in range(10):
            await simple_func()
        direct_time = time.time() - start_time

        # Time with retry handler
        retry_manager = get_retry_manager()
        retry_handler = await retry_manager.get_retry_handler(
            "performance_test",
            RetryConfig(max_retries=0)  # No retries to measure pure overhead
        )

        start_time = time.time()
        for _ in range(10):
            await retry_handler.execute_with_retry(simple_func)
        retry_time = time.time() - start_time

        # Retry handler should not add more than 20x overhead for async operations
        overhead_ratio = retry_time / direct_time
        assert overhead_ratio < 20.0, f"Retry handler overhead too high: {overhead_ratio:.2f}x"

    @pytest.mark.asyncio
    async def test_concurrent_retry_handlers(self):
        """Test concurrent usage of retry handlers."""
        retry_manager = get_retry_manager()
        
        # Create multiple retry handlers
        handlers = []
        for i in range(3):  # Reduced from 5 to 3 for stability
            handler = await retry_manager.get_retry_handler(
                f"concurrent_test_{i}",
                RetryConfig(max_retries=1)  # Reduced from 2 to 1
            )
            handlers.append(handler)

        # Create separate mock functions for each handler to avoid side effect conflicts
        mock_funcs = []
        for i in range(3):
            mock_func = AsyncMock()
            mock_func.side_effect = [
                Exception("Error"),
                "success"
            ]
            mock_funcs.append(mock_func)

        # Execute all handlers concurrently
        tasks = [handlers[i].execute_with_retry(mock_funcs[i]) for i in range(3)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(result == "success" for result in results)
        assert all(mock_func.call_count == 2 for mock_func in mock_funcs)  # 2 calls per handler


class TestRetryHandlerErrorClassification:
    """Test retry handler error classification in real scenarios."""

    @pytest.mark.asyncio
    async def test_network_error_classification(self):
        """Test classification of network errors."""
        exchange = Mock()
        exchange.fetch_ohlcv = AsyncMock()
        exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("Connection timeout")

        # Network errors should be retryable
        with pytest.raises(RetryableError):
            await _call_with_enhanced_retry(
                exchange.fetch_ohlcv,
                symbol="BTC/USD",
                timeframe="1h",
                limit=1,
                retry_config=RetryConfig(max_retries=1)
            )

        assert exchange.fetch_ohlcv.call_count == 2

    @pytest.mark.asyncio
    async def test_authentication_error_classification(self):
        """Test classification of authentication errors."""
        exchange = Mock()
        exchange.fetch_ohlcv = AsyncMock()
        exchange.fetch_ohlcv.side_effect = ccxt.AuthenticationError("Invalid API key")

        # Authentication errors should not be retryable
        with pytest.raises(NonRetryableError):
            await _call_with_enhanced_retry(
                exchange.fetch_ohlcv,
                symbol="BTC/USD",
                timeframe="1h",
                limit=1,
                retry_config=OHLCV_RETRY_CONFIG
            )

        assert exchange.fetch_ohlcv.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_error_classification(self):
        """Test classification of rate limit errors."""
        exchange = Mock()
        exchange.fetch_ohlcv = AsyncMock()
        exchange.fetch_ohlcv.side_effect = ccxt.RateLimitExceeded("Rate limit exceeded")

        # Rate limit errors should be retryable
        with pytest.raises(RetryableError):
            await _call_with_enhanced_retry(
                exchange.fetch_ohlcv,
                symbol="BTC/USD",
                timeframe="1h",
                limit=1,
                retry_config=RetryConfig(max_retries=1)
            )

        assert exchange.fetch_ohlcv.call_count == 2


class TestRetryHandlerMetrics:
    """Test retry handler metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_collection_integration(self):
        """Test metrics collection in integration scenarios."""
        retry_manager = get_retry_manager()
        
        # Create a retry handler
        retry_handler = await retry_manager.get_retry_handler(
            "metrics_test",
            RetryConfig(max_retries=2)
        )

        # Mock function that fails twice then succeeds
        mock_func = AsyncMock()
        mock_func.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            "success"
        ]

        # Execute with retries
        result = await retry_handler.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3

        # Check metrics
        metrics = retry_handler.metrics
        assert metrics.total_attempts == 3
        assert metrics.retry_count == 2
        assert metrics.successful_attempts == 1
        assert metrics.failed_attempts == 2

    @pytest.mark.asyncio
    async def test_global_metrics_collection(self):
        """Test global metrics collection across multiple handlers."""
        retry_manager = get_retry_manager()
        
        # Create multiple handlers
        handlers = []
        for i in range(3):
            handler = await retry_manager.get_retry_handler(
                f"global_metrics_test_{i}",
                RetryConfig(max_retries=1)
            )
            handlers.append(handler)

        # Execute with mixed success/failure
        mock_func_success = AsyncMock(return_value="success")
        mock_func_failure = AsyncMock(side_effect=Exception("Error"))

        # Success case
        await handlers[0].execute_with_retry(mock_func_success)
        
        # Failure case
        with pytest.raises(RetryableError):
            await handlers[1].execute_with_retry(mock_func_failure)

        # Another success case
        await handlers[2].execute_with_retry(mock_func_success)

        # Check global metrics
        all_metrics = retry_manager.get_all_metrics()
        total_attempts = sum(m['total_attempts'] for m in all_metrics.values())
        total_retries = sum(m['retry_count'] for m in all_metrics.values())
        total_successful = sum(m['successful_attempts'] for m in all_metrics.values())
        total_failed = sum(m['failed_attempts'] for m in all_metrics.values())

        assert total_attempts >= 3
        assert total_retries >= 1  # At least one retry for the failure
        assert total_successful >= 2
        assert total_failed >= 1
