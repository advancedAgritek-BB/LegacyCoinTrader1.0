"""
Comprehensive tests for the token discovery system.

Tests cover:
- Basic token discovery from multiple sources
- Enhanced scanner functionality
- Cache management and persistence
- Error handling and fallbacks
- Integration between all components
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
import tempfile
import os
import time
from collections import deque

# Import the modules we need to test
from crypto_bot.solana.scanner import (
    get_solana_new_tokens,
    evaluate_pump_fun_launches,
    evaluate_creator_wallet
)
from crypto_bot.solana.enhanced_scanner import (
    EnhancedSolanaScanner,
    get_enhanced_scanner
)
from crypto_bot.utils.scan_cache_manager import (
    get_scan_cache_manager,
    AdaptiveCacheManager,
    ScanResult,
    CacheEntry
)


class TestTokenDiscoverySystem:
    """Comprehensive test suite for token discovery system."""

    @pytest.fixture
    def mock_config(self) -> Dict[str, Any]:
        """Mock configuration for testing."""
        return {
            "solana_scanner": {
                "enabled": True,
                "limit": 50,
                "url": "https://api.helius.xyz/?api-key=test_key",
                "pump_fun_api_key": "test_pump_key",
                "max_tokens_per_scan": 100,
                "min_score_threshold": 0.3,
                "enable_sentiment": False,
                "enable_pyth_prices": False,
                "min_volume_usd": 10000,
                "max_spread_pct": 2.0,
                "min_liquidity_score": 0.5,
                "min_strategy_fit": 0.6,
                "min_confidence": 0.5
            },
            "enhanced_scanning": {
                "enabled": True,
                "scan_interval": 30,
                "max_tokens_per_scan": 100,
                "min_score_threshold": 0.3,
                "enable_sentiment": False,
                "enable_pyth_prices": False,
                "min_volume_usd": 10000,
                "max_spread_pct": 2.0,
                "min_liquidity_score": 0.5,
                "min_strategy_fit": 0.6,
                "min_confidence": 0.5
            }
        }

    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def temp_cache_dir(self):
        """Temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager."""
        cache = Mock(spec=AdaptiveCacheManager)
        cache.get.return_value = None
        cache.set.return_value = None
        cache.get_stats.return_value = {
            "total_entries": 0,
            "hit_rate": 0.0,
            "total_accesses": 0
        }
        return cache

    # Test basic token discovery APIs
    @pytest.mark.asyncio
    async def test_helius_api_discovery(self, mock_config, mock_session):
        """Test Helius API token discovery."""
        mock_response = {
            "result": [
                {"tokenA": "ABC123", "tokenB": "DEF456"},
                {"tokenA": "GHI789", "tokenB": "JKL012"}
            ]
        }

        with patch('crypto_bot.solana.scanner.aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)

            tokens = await get_solana_new_tokens(mock_config["solana_scanner"])

            assert len(tokens) > 0
            assert "ABC123" in tokens
            assert "DEF456" in tokens
            mock_session.post.assert_called()

    @pytest.mark.asyncio
    async def test_raydium_api_discovery(self, mock_config, mock_session):
        """Test Raydium API token discovery."""
        mock_response = [
            {"pair_id": "ABC123-DEF456", "liquidity": 100000},
            {"pair_id": "GHI789-JKL012", "liquidity": 50000}
        ]

        with patch('crypto_bot.solana.scanner.aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)

            # Mock Helius failure to force Raydium fallback
            mock_session.post.side_effect = Exception("Helius failed")

            tokens = await get_solana_new_tokens(mock_config["solana_scanner"])

            assert len(tokens) > 0
            assert "ABC123" in tokens
            assert "DEF456" in tokens

    @pytest.mark.asyncio
    async def test_pump_fun_api_discovery(self, mock_config, mock_session):
        """Test pump.fun API token discovery."""
        mock_response = [
            {
                "mint": "PUMP123",
                "creator": "CREATOR123",
                "market_cap": 15000,
                "replies": 150,
                "created_timestamp": time.time()
            }
        ]

        with patch('crypto_bot.solana.scanner.aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)

            # Mock other APIs failing
            mock_session.post.side_effect = Exception("Helius failed")
            mock_session.get.side_effect = [
                Exception("Raydium failed"),
                Exception("Orca failed"),
                mock_response  # pump.fun success
            ]

            tokens = await get_solana_new_tokens(mock_config["solana_scanner"])

            assert len(tokens) > 0
            assert "PUMP123" in tokens

    @pytest.mark.asyncio
    async def test_wallet_evaluation(self, mock_session):
        """Test creator wallet evaluation."""
        mock_wallet_response = {"account": "exists"}

        with patch('crypto_bot.solana.scanner.aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_wallet_response)

            launches = [
                {
                    "mint": "TEST123",
                    "creator": "WALLET123",
                    "market_cap": 20000,
                    "replies": 100,
                    "created_timestamp": time.time()
                }
            ]

            evaluated = await evaluate_pump_fun_launches(launches, mock_session)

            assert len(evaluated) == 1
            assert "credibility_score" in evaluated[0]
            assert evaluated[0]["credibility_score"] >= 50

    @pytest.mark.asyncio
    async def test_creator_wallet_analysis(self, mock_session):
        """Test individual creator wallet analysis."""
        mock_wallet_response = {"account": "exists"}

        with patch('crypto_bot.solana.scanner.aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_wallet_response)

            score, factors = await evaluate_creator_wallet("WALLET123", mock_session)

            assert isinstance(score, int)
            assert isinstance(factors, list)
            # Test wallet gets a reasonable score (not extremely negative)
            assert score >= -10

    # Test enhanced scanner functionality
    def test_enhanced_scanner_initialization(self, mock_config):
        """Test enhanced scanner initialization."""
        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager', return_value=Mock()):
            scanner = EnhancedSolanaScanner(mock_config)

            assert scanner.config == mock_config
            assert scanner.scanner_config == mock_config["solana_scanner"]
            assert scanner.max_tokens_per_scan == 100
            assert scanner.min_score_threshold == 0.3

    @pytest.mark.asyncio
    async def test_enhanced_scanner_discovery(self, mock_config):
        """Test enhanced scanner token discovery."""
        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager') as mock_cache, \
             patch('crypto_bot.solana.enhanced_scanner.get_solana_new_tokens') as mock_discovery, \
             patch('crypto_bot.solana.enhanced_scanner.get_sentiment_enhanced_tokens') as mock_sentiment:

            mock_cache.return_value = Mock()
            mock_discovery.return_value = ["TOKEN1", "TOKEN2"]
            mock_sentiment.return_value = []

            scanner = EnhancedSolanaScanner(mock_config)
            tokens = await scanner._discover_tokens()

            assert len(tokens) == 2
            assert "TOKEN1" in tokens
            assert "TOKEN2" in tokens
            mock_discovery.assert_called_once()

    @pytest.mark.asyncio
    async def test_token_analysis(self, mock_config):
        """Test token market condition analysis."""
        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager', return_value=Mock()):
            scanner = EnhancedSolanaScanner(mock_config)

            # Mock all the analysis methods
            with patch.object(scanner, '_get_token_price', return_value=1.0), \
                 patch.object(scanner, '_get_token_volume', return_value=50000.0), \
                 patch.object(scanner, '_get_price_change', return_value=0.02), \
                 patch.object(scanner, '_calculate_atr', return_value=(0.001, 0.05)), \
                 patch.object(scanner, '_get_spread', return_value=0.5), \
                 patch.object(scanner, '_get_sentiment_data', return_value=None):

                conditions = await scanner._analyze_single_token("TEST_TOKEN")

                assert conditions.price == 1.0
                assert conditions.volume_24h == 50000.0
                assert conditions.price_change_24h == 0.02
                assert conditions.atr == 0.001
                assert conditions.atr_percent == 0.05
                assert conditions.spread_pct == 0.5

    def test_token_scoring(self, mock_config):
        """Test token scoring and filtering."""
        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager', return_value=Mock()):
            scanner = EnhancedSolanaScanner(mock_config)

            # Create mock market conditions
            from crypto_bot.solana.enhanced_scanner import MarketConditions
            conditions = MarketConditions(
                price=1.0,
                volume_24h=50000.0,
                volume_ma=45000.0,
                price_change_24h=0.02,
                price_change_7d=0.1,
                atr=0.001,
                atr_percent=0.05,
                spread_pct=0.5,
                liquidity_score=0.8,
                volatility_score=0.6,
                momentum_score=0.7
            )

            analyzed_tokens = {"TEST_TOKEN": conditions}
            scored_tokens = scanner._score_tokens(analyzed_tokens)

            assert len(scored_tokens) == 1
            token, score, regime, data = scored_tokens[0]
            assert token == "TEST_TOKEN"
            assert isinstance(score, float)
            assert 0 <= score <= 1
            assert regime in ["volatile", "trending", "ranging", "neutral"]

    # Test cache manager functionality
    def test_adaptive_cache_initialization(self):
        """Test cache manager initialization."""
        cache = AdaptiveCacheManager()
        assert cache.initial_size == 1000
        assert cache.max_size == 10000
        assert cache.min_size == 100
        assert len(cache.caches) == 0

    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = AdaptiveCacheManager()

        # Test set and get
        cache.set("test_cache", "key1", "value1")
        assert cache.get("test_cache", "key1") == "value1"

        # Test cache miss
        assert cache.get("test_cache", "key2") is None

        # Test invalidation
        assert cache.invalidate("test_cache", "key1") == True
        assert cache.get("test_cache", "key1") is None

    def test_cache_size_management(self):
        """Test cache size limits and eviction."""
        cache = AdaptiveCacheManager(initial_size=5, max_size=10)

        # Fill cache beyond limit
        for i in range(12):
            cache.set("test_cache", f"key{i}", f"value{i}")

        # Should only have max_size entries
        assert len(cache.caches["test_cache"]) <= 10

    def test_cache_statistics(self):
        """Test cache statistics calculation."""
        cache = AdaptiveCacheManager()

        # Add some entries
        cache.set("test_cache", "key1", "value1")
        cache.set("test_cache", "key2", "value2")

        # Get stats
        stats = cache.get_stats("test_cache")
        assert "entries" in stats
        assert "hit_rate" in stats
        assert stats["entries"] == 2

    # Test error handling and fallbacks
    @pytest.mark.asyncio
    async def test_api_failure_fallbacks(self, mock_config, mock_session):
        """Test fallback behavior when APIs fail."""
        with patch('crypto_bot.solana.scanner.aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Make all APIs fail
            mock_session.post.side_effect = Exception("All APIs failed")
            mock_session.get.side_effect = Exception("All APIs failed")

            tokens = await get_solana_new_tokens(mock_config["solana_scanner"])

            # Should return fallback tokens
            assert isinstance(tokens, list)
            assert len(tokens) > 0  # Should have fallback tokens

    @pytest.mark.asyncio
    async def test_partial_api_failures(self, mock_config, mock_session):
        """Test handling of partial API failures."""
        # Mock successful responses for some APIs
        mock_helius_response = {
            "result": [{"tokenA": "TOKEN1", "tokenB": "TOKEN2"}]
        }

        with patch('crypto_bot.solana.scanner.aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_helius_response)

            # Make subsequent APIs fail
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    response = Mock()
                    response.json = AsyncMock(return_value=mock_helius_response)
                    return response
                else:
                    raise Exception("API failed")

            mock_session.post.side_effect = side_effect

            tokens = await get_solana_new_tokens(mock_config["solana_scanner"])

            # Should still get tokens from successful API
            assert "TOKEN1" in tokens
            assert "TOKEN2" in tokens

    # Integration tests
    @pytest.mark.integration
    def test_full_scanner_integration(self, mock_config):
        """Test full scanner integration."""
        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager') as mock_cache, \
             patch('crypto_bot.solana.enhanced_scanner.get_solana_new_tokens') as mock_discovery:

            mock_cache_manager = Mock()
            mock_cache.return_value = mock_cache_manager
            mock_discovery.return_value = ["TOKEN1", "TOKEN2", "TOKEN3"]

            scanner = EnhancedSolanaScanner(mock_config)

            # Mock analysis methods
            async def mock_analysis(token):
                from crypto_bot.solana.enhanced_scanner import MarketConditions
                return MarketConditions(
                    price=1.0,
                    volume_24h=25000.0,
                    volume_ma=22500.0,
                    price_change_24h=0.01,
                    price_change_7d=0.05,
                    atr=0.0005,
                    atr_percent=0.025,
                    spread_pct=0.3,
                    liquidity_score=0.7,
                    volatility_score=0.5,
                    momentum_score=0.6
                )

            scanner._analyze_single_token = mock_analysis

            # Test discovery
            discovered = asyncio.run(scanner._discover_tokens())
            assert len(discovered) == 3

            # Test analysis
            analyzed = asyncio.run(scanner._analyze_tokens(discovered))
            assert len(analyzed) == 3

            # Test scoring
            scored = scanner._score_tokens(analyzed)
            assert len(scored) == 3

            # Test caching
            asyncio.run(scanner._cache_results(scored))
            mock_cache_manager.add_scan_result.assert_called()

    @pytest.mark.integration
    def test_cache_manager_integration(self):
        """Test cache manager integration."""
        cache = AdaptiveCacheManager()

        # Test multiple cache types
        cache.set("ohlcv", "BTC/USDT", {"price": 50000})
        cache.set("orderbook", "BTC/USDT", {"bids": [], "asks": []})
        cache.set("regime", "BTC/USDT", "bullish")

        # Verify entries exist
        assert cache.get("ohlcv", "BTC/USDT") == {"price": 50000}
        assert cache.get("orderbook", "BTC/USDT") == {"bids": [], "asks": []}
        assert cache.get("regime", "BTC/USDT") == "bullish"

        # Test stats
        stats = cache.get_stats()
        assert stats["total_entries"] == 3
        assert "overall_hit_rate_pct" in stats

    # Test edge cases and error conditions
    def test_empty_token_list(self, mock_config):
        """Test handling of empty token lists."""
        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager', return_value=Mock()):
            scanner = EnhancedSolanaScanner(mock_config)

            # Test scoring empty list
            scored = scanner._score_tokens({})
            assert scored == []

    def test_invalid_token_data(self, mock_config):
        """Test handling of invalid token data."""
        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager', return_value=Mock()):
            scanner = EnhancedSolanaScanner(mock_config)

            # Test with None values
            from crypto_bot.solana.enhanced_scanner import MarketConditions
            conditions = MarketConditions(
                price=None,
                volume_24h=0,
                volume_ma=0,
                price_change_24h=0,
                price_change_7d=0,
                atr=0,
                atr_percent=0,
                spread_pct=0,
                liquidity_score=0,
                volatility_score=0,
                momentum_score=0
            )

            analyzed_tokens = {"INVALID_TOKEN": conditions}
            scored = scanner._score_tokens(analyzed_tokens)

            # Should handle gracefully
            assert isinstance(scored, list)

    def test_cache_eviction_under_load(self):
        """Test cache eviction under high load."""
        cache = AdaptiveCacheManager(initial_size=10, max_size=20)

        # Add many entries quickly
        for i in range(50):
            cache.set("test_cache", f"key{i}", f"value{i}")

        # Should maintain size limits
        assert len(cache.caches["test_cache"]) <= 20

        # Should still be able to retrieve recent entries
        recent_entries = [f"key{i}" for i in range(40, 50)]
        for key in recent_entries:
            if key in cache.caches["test_cache"]:
                assert cache.get("test_cache", key) == f"value{key[3:]}"

    # Test configuration validation
    def test_invalid_configuration(self):
        """Test handling of invalid configuration."""
        invalid_config = {
            "solana_scanner": {
                "enabled": True,
                "limit": -1,  # Invalid negative limit
                "max_tokens_per_scan": 0  # Invalid zero
            }
        }

        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager', return_value=Mock()):
            scanner = EnhancedSolanaScanner(invalid_config)

            # Should use defaults for invalid values
            assert scanner.max_tokens_per_scan > 0

    # Performance tests
    def test_cache_performance(self):
        """Test cache performance under load."""
        cache = AdaptiveCacheManager()

        # Test rapid operations
        start_time = time.time()
        for i in range(1000):
            cache.set("perf_test", f"key{i}", f"value{i}")
            cache.get("perf_test", f"key{i}")

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time
        assert duration < 5.0  # Less than 5 seconds for 1000 operations

        # Verify functionality still works
        assert cache.get("perf_test", "key500") == "value500"

    @pytest.mark.asyncio
    async def test_concurrent_token_discovery(self, mock_config):
        """Test concurrent token discovery operations."""
        with patch('crypto_bot.solana.scanner.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock successful response
            mock_response = {"result": [{"tokenA": "TOKEN1", "tokenB": "TOKEN2"}]}
            mock_session.post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)

            # Run multiple discovery operations concurrently
            tasks = [get_solana_new_tokens(mock_config["solana_scanner"]) for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All should return results
            assert all(len(result) > 0 for result in results)
            assert all("TOKEN1" in result for result in results)


if __name__ == "__main__":
    pytest.main([__file__])
