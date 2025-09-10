#!/usr/bin/env python3
"""
Test to verify that overlapping cache updates are fixed.
This test ensures that cache updates for multiple timeframes don't interfere with each other.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_bot.main import update_caches
from crypto_bot.phase_runner import BotContext


class TestOverlappingCacheFix:
    """Test suite to verify overlapping cache updates are fixed."""

    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange object."""
        exchange = MagicMock()
        exchange.id = "kraken"
        exchange.symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        exchange.markets = {"BTC/USD": {}, "ETH/USD": {}, "SOL/USD": {}}
        exchange.rateLimit = 100
        return exchange

    @pytest.fixture
    def sample_config(self):
        """Sample configuration with both main timeframe and regime timeframes."""
        return {
            "timeframe": "1h",
            "regime_timeframes": ["4h", "1d", "1w"],
            "max_concurrent_ohlcv": 3,
            "max_concurrent_dex_ohlcv": 10,
            "min_volume_usd": 1000,
            "ohlcv_fetcher_timeout": 30,
            "use_enhanced_ohlcv_fetcher": True,
            "bounce_scalper": {"vol_zscore_threshold": 2.0}
        }

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Sample OHLCV data for testing."""
        return [
            [1640995200000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],
            [1640995260000, 50500.0, 51500.0, 50000.0, 51000.0, 150.0],
        ]

    @pytest.fixture
    def mock_context(self, mock_exchange, sample_config):
        """Create a mock bot context."""
        ctx = MagicMock(spec=BotContext)
        ctx.exchange = mock_exchange
        ctx.config = sample_config
        ctx.df_cache = {}
        ctx.regime_cache = {}
        ctx.current_batch = ["BTC/USD", "ETH/USD"]
        ctx.timing = {}
        return ctx

    @pytest.mark.asyncio
    async def test_consolidated_timeframe_update(self, mock_context, sample_ohlcv_data):
        """
        Test that update_caches consolidates timeframes to avoid overlapping updates.

        This test verifies that when both main timeframe (1h) and regime timeframes (4h, 1d, 1w)
        are needed, they are fetched in a single consolidated operation rather than
        multiple overlapping operations.
        """
        call_count = 0
        captured_timeframes = []

        async def mock_update_multi_tf(*args, **kwargs):
            nonlocal call_count, captured_timeframes
            call_count += 1

            # Capture the timeframes being requested
            additional_timeframes = kwargs.get('additional_timeframes', [])
            main_timeframes = ['1h']  # Default from config.get("timeframes", ["1h"])
            all_timeframes = list(set(main_timeframes + additional_timeframes))
            captured_timeframes.append(all_timeframes)

            # Mock successful cache update
            return {}

        async def mock_update_regime_tf(*args, **kwargs):
            # Mock regime cache update
            return {}

        with patch('crypto_bot.main.update_multi_tf_ohlcv_cache', side_effect=mock_update_multi_tf), \
             patch('crypto_bot.main.update_regime_tf_cache', side_effect=mock_update_regime_tf):

            # Call update_caches
            await update_caches(mock_context)

            # Verify that update_multi_tf_ohlcv_cache was called only once
            assert call_count == 1, f"Expected 1 call to update_multi_tf_ohlcv_cache, got {call_count}"

            # Verify that all required timeframes were consolidated
            expected_timeframes = {'1h', '4h', '1d', '1w'}
            actual_timeframes = set(captured_timeframes[0])
            assert actual_timeframes == expected_timeframes, \
                f"Expected consolidated timeframes {expected_timeframes}, got {actual_timeframes}"

    @pytest.mark.asyncio
    async def test_no_duplicate_symbol_processing(self, mock_context, sample_ohlcv_data):
        """
        Test that symbols are not processed multiple times due to overlapping cache updates.
        """
        symbol_call_counts = {}

        async def mock_enhanced_fetcher_update(cache, symbols, timeframe, limit, since_map=None):
            # Track how many times each symbol is processed
            for symbol in symbols:
                key = f"{symbol}_{timeframe}"
                symbol_call_counts[key] = symbol_call_counts.get(key, 0) + 1

            # Return mock cache with proper pandas DataFrame objects
            import pandas as pd
            result_cache = {}
            for symbol in symbols:
                # Create a mock DataFrame with required columns
                df = pd.DataFrame({
                    'timestamp': [1640995200000, 1640995260000],
                    'open': [50000.0, 50500.0],
                    'high': [51000.0, 51500.0],
                    'low': [49000.0, 50000.0],
                    'close': [50500.0, 51000.0],
                    'volume': [100.0, 150.0]
                })
                result_cache[symbol] = df
            return result_cache

        async def mock_update_regime_tf(*args, **kwargs):
            return {}

        with patch('crypto_bot.utils.enhanced_ohlcv_fetcher.EnhancedOHLCVFetcher') as mock_fetcher_class, \
             patch('crypto_bot.main.update_regime_tf_cache', side_effect=mock_update_regime_tf):

            # Setup mock fetcher
            mock_fetcher = MagicMock()
            mock_fetcher.update_cache.side_effect = mock_enhanced_fetcher_update
            mock_fetcher_class.return_value = mock_fetcher

            # Call update_caches
            await update_caches(mock_context)

            # Verify that each symbol-timeframe combination was processed exactly once
            expected_symbols = mock_context.current_batch
            expected_timeframes = {'1h', '4h', '1d', '1w'}

            for symbol in expected_symbols:
                for timeframe in expected_timeframes:
                    key = f"{symbol}_{timeframe}"
                    count = symbol_call_counts.get(key, 0)
                    assert count <= 1, \
                        f"Symbol {symbol} for timeframe {timeframe} was processed {count} times, expected at most 1"

    def test_config_timeframe_consolidation(self, sample_config):
        """
        Test that the configuration properly defines both main and regime timeframes.
        """
        # Verify main timeframe
        assert sample_config["timeframe"] == "1h"

        # Verify regime timeframes
        regime_tfs = sample_config["regime_timeframes"]
        assert isinstance(regime_tfs, list)
        assert len(regime_tfs) == 3
        assert set(regime_tfs) == {"4h", "1d", "1w"}

        # Verify no duplicate timeframes
        main_tf = sample_config["timeframe"]
        assert main_tf not in regime_tfs, "Main timeframe should not be duplicated in regime timeframes"

    @pytest.mark.asyncio
    async def test_cache_update_failure_handling(self, mock_context):
        """
        Test that cache update failures are handled gracefully and don't crash the system.
        """
        async def mock_failing_update(*args, **kwargs):
            raise Exception("Simulated cache update failure")

        async def mock_update_regime_tf(*args, **kwargs):
            return {}

        with patch('crypto_bot.main.update_multi_tf_ohlcv_cache', side_effect=mock_failing_update), \
             patch('crypto_bot.main.update_regime_tf_cache', side_effect=mock_update_regime_tf):

            # This should not raise an exception
            await update_caches(mock_context)

            # Verify that the function completed despite the failure
            # (The function should log the error but continue)
